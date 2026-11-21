// Session: what all views of one log share, the data, the layout (tabs and the 3D view's spec) and the time
// state (hover, pinned time, visible range, playback). Each FlightPlotter is a view on a session; the main one
// creates it, and views in popped-out windows attach to it through window.opener.
//
// Views in other windows run their own copy of this library, so a session holds objects from several realms.
// Data written by such a view is copied into this realm when the view detaches (see detach()).
import { layoutToJSON, parseLayout } from "./layout.js";

export class Session {
    /**
     * @param {import("./dataset.js").Dataset} ds
     * @param {{tabs: object[], viewport: object|null}} layout  parsed (parseLayout)
     * @param {object} o  {initialLayout, onLayoutChange, crafts, editable, layoutName, popoutUrl}
     */
    constructor(ds, layout, o) {
        this.ds = ds;
        this.tabs = layout.tabs;
        this.viewportSpec = layout.viewport;
        this.initialLayout = o.initialLayout;
        this.onLayoutChange = o.onLayoutChange || null;
        this.crafts = o.crafts || [];
        this.editable = o.editable;
        this.layoutName = o.layoutName;
        this.popoutUrl = o.popoutUrl || null;
        this.views = new Set();
        /** tab -> its pop-out window */
        this.popouts = new Map();
        this.theme = "light";

        this.hoverT = null;
        this.pinT = null;
        this.playing = false;
        this.playT = 0;
        this.speed = 1;
        this.range = null; // null: full log
    }

    attach(view) {
        this.views.add(view);
        if (view.fixedTab) {
            // a pop-out, also after reloading itself
            this.popouts.set(view.fixedTab, view.popoutWindow);
            this._watchPopouts();
            this._notify("layout", view);
        }
    }

    /** a view goes away; a pop-out's tab returns to the main window */
    detach(view) {
        this.views.delete(view);
        // own the data again: objects made by a closing window's realm shouldn't outlive it. Tabs keep their
        // identity, which other pop-outs hold on to.
        const own = (tab) => (tab.panels = JSON.parse(JSON.stringify(tab.panels)));
        for (const [tab, w] of [...this.popouts]) {
            if (w === view.popoutWindow || w.closed) {
                this.popouts.delete(tab);
                own(tab);
            }
        }
        if (view.fixedTab) {
            own(view.fixedTab);
        }
        this.viewportSpec = JSON.parse(JSON.stringify(this.viewportSpec));
        this._notify("layout");
    }

    /** a crashed or force-closed window may never say goodbye */
    _watchPopouts() {
        if (this._closedPoll) {
            return;
        }
        this._closedPoll = setInterval(() => {
            for (const w of this.popouts.values()) {
                if (w.closed) {
                    this.detach({ popoutWindow: w });
                }
            }
            if (!this.popouts.size) {
                clearInterval(this._closedPoll);
                this._closedPoll = 0;
            }
        }, 1000);
    }

    destroy() {
        this.stopPlay();
        clearInterval(this._closedPoll);
        this._closedPoll = 0;
        for (const w of this.popouts.values()) {
            w.close();
        }
        this.popouts.clear();
        // the registry keeps a pop-out's way back (also across its reloads) until the session ends
        for (const id of this._popoutIds ?? []) {
            delete window.__flightplotterPopouts?.[id];
        }
        this._popoutIds = [];
    }

    _notify(kind, from = null) {
        for (const v of [...this.views]) {
            if (v !== from) {
                try {
                    v.sessionChanged(kind);
                } catch (e) {
                    // a view in a window that's closing
                    console.warn("flightplotter:", e);
                }
            }
        }
    }

    // ------------------------------------------------------------ layout
    getLayout() {
        return layoutToJSON(this.tabs, this.viewportSpec);
    }

    /** replace tabs (and the viewport, if the layout has one); pop-outs whose tab is gone close */
    setLayout(layout, from = null) {
        const l = parseLayout(layout);
        this.tabs = l.tabs;
        if (l.viewport !== undefined) {
            this.viewportSpec = l.viewport;
        }
        for (const [tab, w] of this.popouts) {
            this.popouts.delete(tab);
            w.close();
        }
        this.layoutChanged(from, true);
    }

    /** after any edit; `from` already shows it, unless `all` */
    layoutChanged(from = null, all = false) {
        this.onLayoutChange?.(this.getLayout());
        this._notify("layout", all ? null : from);
    }

    // ------------------------------------------------------------ pop-outs
    /** open a tab in its own window, served from popoutUrl; returns the window or null if blocked */
    popOut(tab) {
        if (!this.popoutUrl || this.popouts.has(tab)) {
            this.popouts.get(tab)?.focus();
            return this.popouts.get(tab) ?? null;
        }
        const id = Math.random().toString(36).slice(2);
        const reg = (window.__flightplotterPopouts ??= {});
        reg[id] = { session: this, tab };
        (this._popoutIds ??= []).push(id);
        const sep = this.popoutUrl.includes("?") ? "&" : "?";
        const w = window.open(`${this.popoutUrl}${sep}fp-popout=${id}`, `fp-${id}`, "popup,width=1200,height=800");
        if (!w) {
            delete reg[id];
            return null;
        }
        this.popouts.set(tab, w);
        this._watchPopouts();
        this._notify("layout");
        return w;
    }

    isPoppedOut(tab) {
        return this.popouts.has(tab);
    }

    // ------------------------------------------------------------ time
    currentT() {
        if (this.playing) {
            return this.playT;
        }
        return this.hoverT ?? this.pinT ?? this.ds.t0;
    }

    /** hover at time t (null: none) in view `from`; the other views draw their cursor there */
    setHover(t, from = null) {
        this.hoverT = t;
        this.hoverFrom = from;
        if (!this.playing) {
            this._notify("time");
        }
    }

    setPin(t) {
        this.stopPlay();
        this.pinT = t;
        this._notify("time");
    }

    /** visible time window; views showing another range follow */
    setRange(a, b, from = null) {
        const r = this.range;
        if (r && Math.abs(r[0] - a) < 1e-9 && Math.abs(r[1] - b) < 1e-9) {
            return;
        }
        this.range = [a, b];
        this._notify("range", from);
    }

    visibleRange() {
        return this.range ?? [this.ds.t0, this.ds.t1];
    }

    setSpeed(s) {
        this.speed = s;
        this._notify("play");
    }

    togglePlay() {
        if (this.playing) {
            this.stopPlay();
            return;
        }
        const [a] = this.visibleRange();
        let t = this.pinT ?? this.hoverT ?? a;
        if (t >= this.ds.t1) {
            t = a;
        }
        this.playT = t;
        this.playing = true;
        this._last = performance.now();
        this._nextFrame();
        this._notify("play");
    }

    stopPlay() {
        if (!this.playing) {
            return;
        }
        cancelAnimationFrame(this._raf);
        clearTimeout(this._timer);
        this.playing = false;
        this.pinT = this.playT;
        this._notify("play");
        this._notify("time");
    }

    _tick(now) {
        if (!this.playing) {
            return;
        }
        // cap the step so a stalled frame does not jump ahead
        const dt = Math.min(0.1, (now - this._last) / 1000);
        this._last = now;
        this.playT += dt * this.speed;
        if (this.playT >= this.ds.t1) {
            this.playT = this.ds.t1;
            this.stopPlay();
            return;
        }
        const [a, b] = this.visibleRange();
        if (this.playT > b || this.playT < a) {
            const span = b - a;
            this.setRange(this.playT - 0.1 * span, this.playT + 0.9 * span);
        }
        this._notify("time");
        this._nextFrame();
    }

    // frames come from this (the main) window, which may be hidden behind a pop-out playing back
    _nextFrame() {
        if (document.hidden) {
            this._timer = setTimeout(() => this._tick(performance.now()), 16);
        } else {
            this._raf = requestAnimationFrame((now) => this._tick(now));
        }
    }

    step(dir, big) {
        this.stopPlay();
        const ds = this.ds;
        const t = this.pinT ?? this.hoverT ?? this.visibleRange()[0];
        if (big) {
            this.pinT = Math.min(ds.t1, Math.max(ds.t0, t + dir * 0.1));
        } else {
            this.pinT = ds.t[Math.min(ds.n - 1, Math.max(0, ds.indexAt(t) + dir))];
        }
        this.hoverT = null;
        this._notify("time");
    }

    unpin() {
        this.stopPlay();
        this.pinT = null;
        this._notify("time");
    }

    setTheme(theme) {
        this.theme = theme;
        this._notify("theme");
    }
}
