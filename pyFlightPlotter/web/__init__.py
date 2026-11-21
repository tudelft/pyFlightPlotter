"""Browser front end for pyFlightPlotter.

Same idea as FlightPlotterBase + Viewport, rendered in the browser with uPlot
and three.js: synchronized time-series panels with a hover cursor, playback,
zoom/pan, PSD, and a 3D view of the craft. Python provides the signals and the
layout; this module serves them, with the page, from a local HTTP server
(127.0.0.1 only).

    import numpy as np
    from pyFlightPlotter import Quadrotor
    from pyFlightPlotter.web import WebPlotter

    t = np.linspace(0, 10, 2001)
    wp = WebPlotter(t, name="demo")
    wp.add_signals({"roll": np.sin(t), "rollSp": np.sin(t - 0.1)})
    wp.add_panel("Roll", solid=["roll"], dashed=["rollSp"], series_labels=["roll"],
                 style_labels=(None, "estimate", "setpoint"), ylabel="[rad]")
    wp.set_viewport(Quadrotor(), att=quat_wxyz)        # N x 4 array or 4 column names
    wp.show()

Panels go into tabs: add_tab("Name") starts a new one, which later add_panel calls fill. Instead of add_panel /
set_viewport, a layout exported from the page ("Layout ⤓") can be loaded with wp.set_layout("layout.json");
wp.layout() returns the current one in the same format.

Column references anywhere (panels, viewport) are a column name, a
(name, scale) tuple, or an array (registered as a new signal).
"""
import json
import logging
import os
import sys
import threading
import uuid
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

logger = logging.getLogger(__name__)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
TIME = "__time__"
LAYOUT_FORMAT = "flightplotter-layout"  # as layout.js
LAYOUT_VERSION = 2


def _jsonable(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if v is None or isinstance(v, str):
        return v
    return str(v)


def craft_to_json(craft):
    """pyFlightPlotter.Craft3D -> the geometry format of flightplotter/crafts.js"""
    tl = lambda a: np.asarray(a, dtype=float).tolist()
    return {
        "body": [tl(g) for g in craft.geometry],
        "rotors": [{"xyz": tl(r["xyz"]), "axis": tl(r["axis"]), "R": float(r["R"]),
                    "tilt_xyz": tl(r["tilt_xyz"]), "tilt_axis": tl(r["tilt_axis"]),
                    "tilt2_axis": tl(r["tilt2_axis"])} for r in craft.rotors],
        "surfaces": [{"tilt_xyz": tl(s["tilt_xyz"]), "tilt_axis": tl(s["tilt_axis"]),
                      "geometry": [tl(g) for g in s["geometry"]]} for s in craft.surfaces],
    }


class WebPlotter:
    def __init__(self, time, name="Flight Plotter", editable=True, ncols=3, rows=5, theme=None):
        """
        time:     1D array of sample times in seconds
        editable: let the user add panels of any signal in the browser
        ncols:    grid columns for panels without row/col
        rows:     panel rows that fit the window height (0: all)
        theme:    "light" | "dark" | None (follow the browser)
        """
        self.t = np.asarray(time, dtype=float).ravel()
        if self.t.size < 2:
            raise ValueError("time needs at least two samples")
        self.n = self.t.size
        self.name = name
        self.options = {"editable": bool(editable), "ncols": int(ncols), "rows": int(rows), "theme": theme}
        self.signals = {}
        self.dtypes = {}
        self.tabs = [{"name": "Plots", "view": True, "panels": []}]
        self.viewport = None
        self.events = []
        self._anon = 0

    # ------------------------------------------------------------ data
    def add_signal(self, name, values, float64=False):
        """Add a signal (same length as time). float64 keeps full precision,
        e.g. for bit masks or GPS coordinates; the default sends float32."""
        values = np.asarray(values, dtype=float).ravel()
        if values.size != self.n:
            raise ValueError(f"signal '{name}' has {values.size} samples, time has {self.n}")
        self.signals[name] = values
        self.dtypes[name] = "f8" if float64 else "f4"
        return name

    def add_signals(self, data, float64=()):
        """Add all numeric columns of a dict or pandas DataFrame"""
        items = data.items()
        for name, values in items:
            values = np.asarray(values)
            if values.dtype.kind in "biuf":
                self.add_signal(str(name), values, float64=name in float64)
        return self

    def add_event(self, t, name, kind="event"):
        """Vertical marker at time t. kind "mode" draws dotted, "event" solid."""
        self.events.append({"t": float(t), "name": str(name), "kind": kind})
        return self

    def _ref(self, r):
        if r is None:
            return None
        if isinstance(r, tuple):
            return [self._ref(r[0]), float(r[1])]
        if isinstance(r, str):
            if r not in self.signals:
                logger.warning(f"column '{r}' not found")
            return r
        self._anon += 1
        return self.add_signal(f"_signal{self._anon}", r)

    def _refs(self, x, width):
        """list of refs, or an N x width array -> its columns"""
        if x is None:
            return None
        if isinstance(x, np.ndarray) and x.ndim == 2:
            if x.shape != (self.n, width):
                raise ValueError(f"expected an array of shape ({self.n}, {width}), got {x.shape}")
            return [self._ref(x[:, i]) for i in range(width)]
        if len(x) != width:
            raise ValueError(f"expected {width} column references, got {len(x)}")
        return [self._ref(c) for c in x]

    # ------------------------------------------------------------ layout
    @property
    def panels(self):
        """panels of the tab that add_panel currently fills"""
        return self.tabs[-1]["panels"]

    def add_tab(self, name, view=False):
        """Start a new tab; the following add_panel calls fill it. view: show the 3D view in this tab."""
        if len(self.tabs) == 1 and not self.tabs[0]["panels"]:
            self.tabs[0].update(name=str(name), view=bool(view) or self.tabs[0]["view"])
        else:
            self.tabs.append({"name": str(name), "view": bool(view), "panels": []})
        return self

    def add_panel(self, title, solid, light=None, dashed=None, series_labels=None,
                  style_labels=(None, None, None), ylabel="", ylimits=(None, None),
                  row=None, col=None, colors=None):
        """Add a time-series panel; arguments as FlightPlotterBase._plot_timeseries.

        solid/light/dashed: one entry per series (None to skip a style).
        row/col: grid position; None appends after the other panels.
        colors: per series a matplotlib color index or "#rrggbb" (default: index).
        """
        n = len(solid)
        light = [None] * n if light is None else light
        dashed = [None] * n if dashed is None else dashed
        if series_labels is None:
            series_labels = [s if isinstance(s, str) else f"series {i}" for i, s in enumerate(solid)]
        if not (len(light) == len(dashed) == len(series_labels) == n):
            raise ValueError("light, solid, dashed and series_labels must have the same length")
        if len(style_labels) != 3:
            raise ValueError("style_labels must have exactly 3 entries")
        series = []
        for i in range(n):
            s = {"label": series_labels[i], "light": self._ref(light[i]),
                 "solid": self._ref(solid[i]), "dashed": self._ref(dashed[i])}
            if colors is not None:
                s["color"] = colors[i]
            series.append(s)
        self.panels.append({"title": title, "ylabel": ylabel, "ylim": list(ylimits),
                            "styles": list(style_labels), "series": series, "row": row, "col": col})
        return self

    def set_viewport(self, craft, att, attSet=None, attMeas=None,
                     pos=None, posSet=None, posMeas=None,
                     vel=None, velSet=None, velMeas=None,
                     acc=None, accSet=None, accMeas=None,
                     rotor=None, rotorSet=None, surface=None, surfaceSet=None,
                     interpolation="previous"):
        """3D view, arguments as pyFlightPlotter.Viewport.

        att*: N x 4 quaternions [w, x, y, z] (body -> world), or 4 column refs
        pos*, vel*, acc*: N x 3 arrays or 3 column refs
        rotor, rotorSet: N x M (thrust 0..1), N x M x 3 (thrust, tilt1, tilt2),
                         or per rotor a ref or a dict {"thrust", "tilt1", "tilt2"}
        surface, surfaceSet: N x K deflections in rad, or per surface a ref
        """
        def actuators(x, with_tilt):
            if x is None:
                return None
            if isinstance(x, np.ndarray):
                if x.ndim == 3:
                    return [{"thrust": self._ref(x[:, m, 0]), "tilt1": self._ref(x[:, m, 1]),
                             "tilt2": self._ref(x[:, m, 2])} for m in range(x.shape[1])]
                return [self._ref(x[:, m]) for m in range(x.shape[1])]
            out = []
            for r in x:
                if isinstance(r, dict) and with_tilt:
                    out.append({k: self._ref(r.get(k)) for k in ("thrust", "tilt1", "tilt2")})
                else:
                    out.append(self._ref(r))
            return out

        geometry = craft if isinstance(craft, dict) else craft_to_json(craft)
        self.viewport = {
            "craft": geometry,
            "att": self._refs(att, 4), "attSet": self._refs(attSet, 4), "attMeas": self._refs(attMeas, 4),
            "pos": self._refs(pos, 3), "posSet": self._refs(posSet, 3), "posMeas": self._refs(posMeas, 3),
            "vel": self._refs(vel, 3), "velSet": self._refs(velSet, 3), "velMeas": self._refs(velMeas, 3),
            "acc": self._refs(acc, 3), "accSet": self._refs(accSet, 3), "accMeas": self._refs(accMeas, 3),
            "rotor": actuators(rotor, True), "rotorSet": actuators(rotorSet, True),
            "surface": actuators(surface, False), "surfaceSet": actuators(surfaceSet, False),
            "interpolation": interpolation,
        }
        return self

    def layout(self):
        """panels and viewport in the page's layout file format (flightplotter/layout.js)"""
        return _jsonable({"format": LAYOUT_FORMAT, "version": LAYOUT_VERSION,
                          "tabs": self.tabs, "viewport": self.viewport})

    def set_layout(self, layout):
        """Replace the tabs, and the viewport if the layout has a "viewport" key, with a layout
        dict or the path of a JSON file exported from the page. The page checks the layout in
        detail; column references not among the signals are skipped there."""
        if isinstance(layout, (str, os.PathLike)):
            with open(layout, encoding="utf-8") as f:
                layout = json.load(f)
        if isinstance(layout, dict) and "tabs" not in layout and isinstance(layout.get("panels"), list):
            layout = {**layout, "tabs": [{"name": "Plots", "view": True, "panels": layout["panels"]}]}  # version 1
        if not isinstance(layout, dict) or not isinstance(layout.get("tabs"), list) or not layout["tabs"]:
            raise ValueError("a layout is a dict with a non-empty 'tabs' list")
        if layout.get("format", LAYOUT_FORMAT) != LAYOUT_FORMAT:
            raise ValueError(f"not a {LAYOUT_FORMAT} file")
        if layout.get("version", LAYOUT_VERSION) > LAYOUT_VERSION:
            raise ValueError(f"layout version {layout['version']} is newer than this pyFlightPlotter reads")
        self.tabs = [{"name": t.get("name") or f"Tab {i + 1}", "view": t.get("view", i == 0),
                      "panels": list(t.get("panels", []))} for i, t in enumerate(layout["tabs"])]
        if "viewport" in layout:
            self.viewport = layout["viewport"]
        return self

    # ------------------------------------------------------------ serialization
    def to_bytes(self):
        """(meta.json, data.bin) as served to the page"""
        cols = [(TIME, self.t, "f8")] + [(k, v, self.dtypes[k]) for k, v in self.signals.items()]
        cols.sort(key=lambda c: c[2] != "f8")  # float64 first keeps every column aligned
        meta_cols, chunks, off = [], [], 0
        for name, values, dt in cols:
            b = np.ascontiguousarray(values, dtype="<" + dt).tobytes()
            meta_cols.append({"name": name, "dtype": dt, "offset": off})
            chunks.append(b)
            off += len(b)
        meta = {"title": self.name, "n": self.n, "time": TIME, "columns": meta_cols,
                "tabs": self.tabs, "viewport": self.viewport, "events": self.events,
                **self.options}
        return json.dumps(_jsonable(meta)).encode(), b"".join(chunks)

    # ------------------------------------------------------------ showing
    def show(self, open_browser=True, block=None, port=0):
        """Serve the plot and open it in the browser. Returns the URL.

        block: keep Python alive until Ctrl+C. Default: True in scripts,
               False in interactive sessions (IPython / Jupyter / python -i).
        port:  for the first call only; 0 picks a free port.
        """
        server = _Server.get(port)
        sid = uuid.uuid4().hex[:12]
        server.sessions[sid] = self.to_bytes()
        url = f"http://127.0.0.1:{server.port}/index.html?src=/s/{sid}"
        print(f"pyFlightPlotter: {url}", flush=True)
        if open_browser:
            webbrowser.open(url)
        if block is None:
            block = not (hasattr(sys, "ps1") or "IPython" in sys.modules or sys.flags.interactive)
        if block:
            print("Serving, press Ctrl+C to stop.", flush=True)
            try:
                threading.Event().wait()
            except KeyboardInterrupt:
                pass
        return url


# ---------------------------------------------------------------- HTTP server
class _Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map,
                      ".js": "text/javascript", ".css": "text/css", ".json": "application/json"}

    def __init__(self, *args, sessions=None, **kw):
        self.sessions = sessions
        super().__init__(*args, directory=STATIC_DIR, **kw)

    def log_message(self, fmt, *args):
        logger.debug(fmt, *args)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if not path.startswith("/s/"):
            return super().do_GET()
        parts = path.split("/")
        sess = self.sessions.get(parts[2]) if len(parts) == 4 else None
        files = {"meta.json": 0, "data.bin": 1}
        if sess is None or parts[3] not in files:
            return self.send_error(404, "unknown plot (was the Python process restarted?)")
        body = sess[files[parts[3]]]
        self.send_response(200)
        self.send_header("Content-Type", "application/json" if parts[3] == "meta.json" else "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _Server:
    _instance = None

    def __init__(self, port):
        self.sessions = {}
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), partial(_Handler, sessions=self.sessions))
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @classmethod
    def get(cls, port=0):
        if cls._instance is None:
            cls._instance = cls(port)
        return cls._instance


__all__ = ["WebPlotter", "craft_to_json"]
