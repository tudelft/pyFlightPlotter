// Standalone page for pyFlightPlotter.web.WebPlotter: index.html?src=<session>
// loads <session>/meta.json (layout, viewport, column table) and <session>/data.bin.
import { FlightPlotter, attachPopout } from "./flightplotter/index.js";

const msg = document.getElementById("msg");

async function load(src) {
    const meta = await (await fetch(`${src}/meta.json`)).json();
    const res = await fetch(`${src}/data.bin`);
    const total = Number(res.headers.get("Content-Length")) || 0;
    let buf;
    if (res.body && total) {
        const bytes = new Uint8Array(total);
        const reader = res.body.getReader();
        let off = 0;
        for (;;) {
            const { done, value } = await reader.read();
            if (done) {
                break;
            }
            bytes.set(value, off);
            off += value.length;
            msg.textContent = `Loading… ${Math.round((100 * off) / total)} %`;
        }
        buf = bytes.buffer;
    } else {
        buf = await res.arrayBuffer();
    }
    const columns = {};
    for (const c of meta.columns) {
        const T = c.dtype === "f8" ? Float64Array : Float32Array;
        columns[c.name] = new T(buf, c.offset, meta.n);
    }
    const time = columns[meta.time];
    delete columns[meta.time];
    return { meta, time, columns };
}

const src = new URLSearchParams(location.search).get("src");
if (new URLSearchParams(location.search).has("fp-popout")) {
    // a tab popped out of the page that opened this one; it has the data already
    msg.remove();
    attachPopout(document.body);
} else if (!src) {
    msg.textContent = "Open this page through pyFlightPlotter.web.WebPlotter.show().";
} else {
    try {
        const { meta, time, columns } = await load(src);
        document.title = meta.title ? `${meta.title} · Flight Plotter` : "Flight Plotter";
        msg.remove();
        const dark = matchMedia("(prefers-color-scheme: dark)");
        const fp = new FlightPlotter(document.body, {
            data: { time, columns, events: meta.events || [] },
            tabs: meta.tabs || [{ name: "Plots", view: true, panels: meta.panels || [] }],
            viewport: meta.viewport || undefined,
            title: meta.title,
            editable: meta.editable ?? true,
            ncols: meta.ncols ?? 3,
            rows: meta.rows ?? 5,
            theme: meta.theme ?? (dark.matches ? "dark" : "light"),
            popoutUrl: location.pathname,
        });
        if (!meta.theme) {
            dark.addEventListener("change", (e) => fp.setTheme(e.matches ? "dark" : "light"));
        }
        window.flightPlotter = fp;
    } catch (e) {
        console.error(e);
        msg.textContent = `Could not load ${src}: ${e.message || e}. Is the Python process still running?`;
    }
}
