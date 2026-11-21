// FlightPlotter: synchronized time-series panels + 3D viewport in one component.
import { Dataset } from "./dataset.js";
import { resolvePanels, placePanels, layoutToJSON, parseLayout } from "./layout.js";
import { PlotGrid } from "./plots.js";
import { renderPsd } from "./psd.js";

const HELP = [
    "<b>Hover</b> a plot to move the time cursor; values show next to each series and the 3D view follows",
    "<b>Click</b> to pin the time, <kbd>Esc</kbd> to unpin. While pinned, the clock shows Δt to the mouse",
    "<b>Drag</b> to zoom into a time window, <b>wheel</b> to zoom around the mouse, <b>double click</b> to reset",
    "<b>Right / middle drag</b>, <kbd>Shift</kbd>+wheel or horizontal scroll to pan",
    "<kbd>Space</kbd> play / pause, <kbd>←</kbd> <kbd>→</kbd> step one sample (<kbd>Shift</kbd>: 0.1 s)",
    "Click a <b>series</b> to hide it, or a <b>line style</b> in the panel header to hide that style",
    "Panel tools on hover: <b>PSD</b> (Welch spectrum of the visible window), <b>PNG</b> export; when editable, <b>+</b> adds a signal and <b>×</b> removes the panel",
    "<b>Layout</b> (when editable): export the panels and 3D view as JSON, import one, or reset to the initial layout",
    "3D view: left drag rotates, right drag pans, wheel zooms; click the axis gizmo to snap the view",
];

function h(tag, attrs = {}, ...children) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
        if (k === "class") {
            el.className = v;
        } else if (k.startsWith("on")) {
            el.addEventListener(k.slice(2), v);
        } else if (v === true) {
            el.setAttribute(k, "");
        } else if (v !== false && v != null) {
            el.setAttribute(k, v);
        }
    }
    el.append(...children);
    return el;
}

export class FlightPlotter {
    /**
     * @param {HTMLElement} container
     * @param {object} o
     * @param {Dataset|{time, columns, events}} o.data
     * @param {object[]} o.panels      panel specs (see layout.js)
     * @param {object} [o.viewport]    viewport spec (see viewport.js); omit for no 3D view
     * @param {object} [o.layout]      layout file object to start with, e.g. a saved one; "reset" still returns to
     *                                 panels/viewport. A malformed one is ignored with a console warning.
     * @param {string} [o.title]
     * @param {boolean} [o.editable]   allow adding/removing panels and signals, and layout import/export (default true)
     * @param {number} [o.ncols]       grid columns for panels without row/col (default 3)
     * @param {number} [o.rows]        panel rows that fit the height; 0 = all (default 5)
     * @param {"light"|"dark"} [o.theme]
     * @param {string} [o.layoutName]  file name for exported layouts (default "layout")
     * @param {(layout: object) => void} [o.onLayoutChange]  after every edit, import or reset, with getLayout()
     */
    constructor(container, o) {
        this.ds = o.data instanceof Dataset ? o.data : new Dataset(o.data);
        // own copies, so edits never mutate the host's objects; Craft3D becomes plain geometry
        this.initialLayout = layoutToJSON(o.panels || [], o.viewport || null);
        let start = parseLayout(this.initialLayout);
        if (o.layout) {
            try {
                const l = parseLayout(o.layout);
                start = { panels: l.panels, viewport: l.viewport === undefined ? start.viewport : l.viewport };
            } catch (e) {
                console.warn(`flightplotter: ignoring the given layout: ${e.message}`);
            }
        }
        this.panelSpecs = start.panels;
        this.viewportSpec = start.viewport;
        this.editable = o.editable ?? true;
        this.ncols = o.ncols ?? 3;
        this.layoutName = o.layoutName || "layout";
        this.onLayoutChange = o.onLayoutChange || null;
        this.hoverT = null;
        this.pinT = null;
        this.playing = false;
        this.playT = 0;
        this.speed = 1;
        this.follow = false;
        this.viewHidden = new Set();
        this._buildDom(container, o);
        this.root.querySelector(".fp-rows").value = String(o.rows ?? 5);
        this.setTheme(o.theme || "light");

        this._onKey = (e) => this._key(e);
        document.addEventListener("keydown", this._onKey);
        this._onResize = () => this._layoutRows();
        window.addEventListener("resize", this._onResize);
    }

    /** element in the toolbar where the host can put its own controls */
    get toolbarSlot() {
        return this.root.querySelector(".fp-slot");
    }

    destroy() {
        this._stopPlay();
        document.removeEventListener("keydown", this._onKey);
        window.removeEventListener("resize", this._onResize);
        this.grid?.destroy();
        this.view?.destroy();
        this.root.remove();
    }

    // ------------------------------------------------------------ dom
    _buildDom(container, o) {
        const btn = (text, title, cls, onclick) => h("button", { class: cls, title, onclick }, text);
        this.$play = btn("▶", "Play / pause (Space)", "fp-play", () => this.togglePlay());
        this.$speed = h(
            "select",
            { title: "Playback speed", onchange: (e) => (this.speed = +e.target.value) },
            ...[0.1, 0.25, 0.5, 1, 2, 5].map((v) => h("option", { value: v, selected: v === 1 }, `${v}×`)),
        );
        this.$clock = h("span", { class: "fp-clock" });
        this.$status = h("span", { class: "fp-status" });
        this.$view3d = btn("3D", "Show / hide the 3D view", "fp-on fp-view-btn", () => this._toggleView());
        this.$layoutFile = h("input", {
            type: "file",
            accept: ".json,application/json",
            hidden: true,
            onchange: (e) => this._importLayoutFile(e.target),
        });
        const layoutMenu = h(
            "span",
            { class: "fp-btngroup" },
            btn("Layout ⤓", "Export the panels and 3D view as JSON", "", () => this.exportLayout()),
            btn("⤒", "Import a layout JSON", "", () => this.$layoutFile.click()),
            btn("↺", "Reset to the initial layout", "", () => this.setLayout(this.initialLayout)),
            this.$layoutFile,
        );
        const rows = h(
            "select",
            { class: "fp-rows", title: "Panel rows that fit the height", onchange: () => this._layoutRows() },
            ...["3", "4", "5"].map((v) => h("option", { value: v }, v)),
            h("option", { value: "0" }, "all"),
        );

        const toolbar = h(
            "div",
            { class: "fp-toolbar" },
            o.title ? h("span", { class: "fp-title", title: o.title }, o.title) : "",
            h("span", { class: "fp-slot" }),
            this.editable ? btn("+ Plot", "Add a plot of any signal", "", () => this._pickSignal()) : "",
            this.editable ? layoutMenu : "",
            h("span", { class: "fp-sep" }),
            this.$play,
            this.$speed,
            this.$clock,
            h("span", { class: "fp-grow" }),
            this.$status,
            h("label", { title: "Panel rows that fit the height" }, "Rows ", rows),
            this.$view3d,
            btn("?", "Mouse and keyboard controls", "", () => this.$help.showModal()),
        );

        this.$grid = h("div", { class: "fp-grid" });
        this.$gridwrap = h("div", { class: "fp-gridwrap" }, this.$grid);
        this.$split = h("div", { class: "fp-split", title: "Drag to resize" });
        this.$view = h("div", { class: "fp-view" });
        const main = h("div", { class: "fp-main" }, this.$gridwrap);
        // built even without a viewport spec, since an imported layout can add one; .fp-noview hides it
        const vbtn = (k, label, color, dashed) => {
            const b = h("button", { "data-k": k, style: `--fp-c:${color}` }, h("i", { class: dashed ? "fp-dash" : "" }), label);
            b.onclick = () => {
                const off = !this.viewHidden.has(k);
                if (off) {
                    this.viewHidden.add(k);
                } else {
                    this.viewHidden.delete(k);
                }
                b.classList.toggle("fp-off", off);
                this.view?.setVisible(k, !off);
            };
            return b;
        };
        this.$follow = btn("Follow", "Camera follows the craft", "", () => {
            this.follow = !this.follow;
            this.$follow.classList.toggle("fp-on", this.follow);
            this.view?.setFollow(this.follow);
        });
        this.$viewbar = h(
            "div",
            { class: "fp-viewbar" },
            vbtn("est", "Estimate", "#1f77b4"),
            vbtn("set", "Setpoint", "#ff7f0e", true),
            vbtn("meas", "Measurement", "#2ca02c", true),
            vbtn("trail", "Path", "#999"),
            this.$follow,
            btn("Reset view", "Reset camera", "", () => {
                this.follow = false;
                this.$follow.classList.remove("fp-on");
                this.view?.setFollow(false);
            }),
        );
        this.$view.append(h("div", { class: "fp-view3d" }), this.$viewbar);
        main.append(this.$split, this.$view);
        this._bindSplitter();

        const close = (d) => h("button", { onclick: () => d.close() }, "Close");
        this.$psd = h("dialog", { class: "fp-dialog" });
        this.$psdSeg = h(
            "select",
            { onchange: () => this._renderPsd() },
            ...[256, 512, 1024, 2048, 4096, 8192].map((v) => h("option", { value: v, selected: v === 1024 }, v)),
        );
        this.$psdTitle = h("h2");
        this.$psdInfo = h("div", { class: "fp-psd-info" });
        this.$psdBody = h("div");
        this.$psd.append(
            h("header", {}, this.$psdTitle, h("label", {}, "Segment ", this.$psdSeg), close(this.$psd)),
            this.$psdInfo,
            this.$psdBody,
        );

        this.$sig = h("dialog", { class: "fp-dialog" });
        this.$sigTitle = h("h2");
        this.$sigSearch = h("input", { placeholder: "Search signals (Enter adds the first match)", autocomplete: "off" });
        this.$sigList = h("div", { class: "fp-siglist" });
        this.$sig.append(h("header", {}, this.$sigTitle, close(this.$sig)), this.$sigSearch, this.$sigList);

        this.$help = h("dialog", { class: "fp-dialog" });
        const ul = h("ul");
        ul.innerHTML = HELP.map((s) => `<li>${s}</li>`).join("");
        this.$help.append(h("header", {}, h("h2", {}, "Controls"), close(this.$help)), ul);

        this.root = h("div", { class: "fp-root" }, toolbar, main, this.$psd, this.$sig, this.$help);
        container.append(this.root);
    }

    _bindSplitter() {
        const split = this.$split;
        split.addEventListener("pointerdown", (e) => {
            split.setPointerCapture(e.pointerId);
            split.classList.add("fp-drag");
            const rect = this.root.getBoundingClientRect();
            const move = (ev) => {
                const w = Math.min(rect.width * 0.7, Math.max(200, rect.right - ev.clientX));
                this.root.style.setProperty("--fp-view-w", `${w}px`);
            };
            split.addEventListener("pointermove", move);
            split.addEventListener(
                "pointerup",
                () => {
                    split.classList.remove("fp-drag");
                    split.removeEventListener("pointermove", move);
                },
                { once: true },
            );
        });
    }

    // ------------------------------------------------------------ theme
    /** "light" | "dark"; rebuilds the plots since canvas colors are baked in */
    setTheme(theme) {
        this.root.classList.toggle("fp-dark", theme === "dark");
        const cs = getComputedStyle(this.root);
        const v = (name) => cs.getPropertyValue(name).trim();
        this.theme = {
            bg: v("--fp-panel"),
            text: v("--fp-text"),
            muted: v("--fp-muted"),
            grid: v("--fp-grid"),
            tick: v("--fp-tick"),
            mode: v("--fp-event-mode"),
            event: v("--fp-event"),
            gridLines: v("--fp-grid3d"),
        };
        this._build(true);
    }

    // ------------------------------------------------------------ build
    _build(keepRange = false) {
        const range = keepRange && this.grid ? this.grid.range : null;
        const panels = placePanels(resolvePanels(this.ds, this.panelSpecs), this.ncols);
        this.root.classList.toggle("fp-noview", !this.viewportSpec);
        this.grid?.destroy();
        this.grid = new PlotGrid(this.$grid, this.ds, panels, this.theme, {
            editable: this.editable,
            onHover: (t) => {
                this.hoverT = t;
                if (!this.playing) {
                    this._showTime();
                }
            },
            onClick: (t) => {
                this._stopPlay();
                this.pinT = t;
                this._showTime();
            },
            onPsd: (P) => {
                this._psdPanel = P;
                this.$psd.showModal();
                this._renderPsd();
            },
            onRemove: (P) => {
                this.panelSpecs = this.panelSpecs.filter((c) => c !== P.spec.src);
                this._build(true);
                this._layoutChanged();
            },
            onAddSignal: (P) => this._pickSignal(P.spec.src),
        });
        this._layoutRows();
        if (range) {
            this.grid.setRange(...range);
        }
        this._buildView();
        const d = this.ds;
        this.$status.textContent = `${d.n.toLocaleString()} samples × ${d.cols.size} signals, ${(d.t1 - d.t0).toFixed(1)} s`;
        this._showTime();
    }

    async _buildView() {
        this.view?.destroy();
        this.view = null;
        if (!this.viewportSpec || this.$view.classList.contains("fp-hidden")) {
            return;
        }
        const el = this.$view.querySelector(".fp-view3d");
        try {
            // three.js loads on demand, so plots appear before the 3D view
            const { Viewport } = await import("./viewport.js");
            if (this.view || !this.root.isConnected) {
                return;
            }
            this.view = new Viewport(el, this.ds, this.viewportSpec, { bg: this.theme.bg, grid: this.theme.gridLines });
            for (const b of this.$viewbar.querySelectorAll("[data-k]")) {
                b.hidden = !this.view.has(b.dataset.k);
            }
            for (const k of this.viewHidden) {
                this.view.setVisible(k, false);
            }
            if (this.follow) {
                this.view.setFollow(true);
            }
            this._showTime();
        } catch (e) {
            console.error(e);
            el.textContent = `3D view unavailable: ${e.message || e}`;
        }
    }

    _toggleView() {
        const hide = !this.$view.classList.contains("fp-hidden");
        this.$view.classList.toggle("fp-hidden", hide);
        this.$split.hidden = hide;
        this.$view3d.classList.toggle("fp-on", !hide);
        this._buildView();
    }

    _layoutRows() {
        if (!this.grid) {
            return;
        }
        const want = +this.root.querySelector(".fp-rows").value;
        const avail = this.$gridwrap.clientHeight - 12;
        const rows = want === 0 ? this.grid.nrows : Math.min(want, this.grid.nrows);
        const gap = 6;
        const hgt = Math.max(150, Math.floor((avail - gap * (rows - 1)) / rows));
        this.$grid.style.setProperty("--fp-row-h", `${hgt}px`);
    }

    // ------------------------------------------------------------ time
    _currentT() {
        if (this.playing) {
            return this.playT;
        }
        return this.hoverT ?? this.pinT ?? this.ds.t0;
    }

    /** move the pinned cursor to time t (seconds) */
    setTime(t) {
        this._stopPlay();
        this.pinT = t;
        this._showTime();
    }

    _showTime() {
        const t = this._currentT();
        this.view?.setTime(t);
        const head = this.playing ? this.playT : this.pinT;
        this.grid?.setPlayhead(head);
        const c = this.$clock;
        if (this.pinT != null && this.hoverT != null && !this.playing) {
            const dt = this.hoverT - this.pinT;
            c.innerHTML = `t ${this.hoverT.toFixed(3)} s <span class="fp-pin">Δ ${dt >= 0 ? "+" : ""}${dt.toFixed(3)}</span>`;
        } else if (head != null) {
            c.innerHTML = `<span class="fp-pin">t ${head.toFixed(3)} s</span>`;
        } else {
            c.textContent = `t ${t.toFixed(3)} s`;
        }
    }

    togglePlay() {
        if (this.playing) {
            this._stopPlay();
            return;
        }
        let t = this.pinT ?? this.hoverT ?? this.grid.range[0];
        if (t >= this.ds.t1) {
            t = this.grid.range[0];
        }
        this.playT = t;
        this.playing = true;
        this._last = performance.now();
        this.$play.textContent = "❚❚";
        this.$play.classList.add("fp-on");
        this._raf = requestAnimationFrame((now) => this._tick(now));
    }

    _stopPlay() {
        if (!this.playing) {
            return;
        }
        cancelAnimationFrame(this._raf);
        this.playing = false;
        this.pinT = this.playT;
        this.$play.textContent = "▶";
        this.$play.classList.remove("fp-on");
        this._showTime();
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
            this._stopPlay();
            return;
        }
        const [a, b] = this.grid.range;
        if (this.playT > b || this.playT < a) {
            const span = b - a;
            this.grid.setRange(this.playT - 0.1 * span, this.playT + 0.9 * span);
        }
        this._showTime();
        this._raf = requestAnimationFrame((n) => this._tick(n));
    }

    _step(dir, big) {
        this._stopPlay();
        const ds = this.ds;
        const t = this.pinT ?? this.hoverT ?? this.grid.range[0];
        if (big) {
            this.pinT = Math.min(ds.t1, Math.max(ds.t0, t + dir * 0.1));
        } else {
            this.pinT = ds.t[Math.min(ds.n - 1, Math.max(0, ds.indexAt(t) + dir))];
        }
        this.hoverT = null;
        this._showTime();
    }

    _key(e) {
        // only while visible, and not while typing
        if (!this.root.isConnected || this.root.offsetParent === null) {
            return;
        }
        if (e.target.closest?.("input, select, textarea, dialog, [contenteditable]")) {
            return;
        }
        if (e.code === "Space") {
            e.preventDefault();
            this.togglePlay();
        } else if (e.key === "Escape") {
            this._stopPlay();
            this.pinT = null;
            this._showTime();
        } else if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
            e.preventDefault();
            this._step(e.key === "ArrowRight" ? 1 : -1, e.shiftKey);
        }
    }

    // ------------------------------------------------------------ layout
    /** the current panels and viewport as plain JSON (see layout.js), including the user's edits */
    getLayout() {
        return layoutToJSON(this.panelSpecs, this.viewportSpec);
    }

    /**
     * Replace the panels and, if the layout has a "viewport" key, the 3D view. The data stays.
     * Throws an Error naming the offending entry if the layout is malformed.
     */
    setLayout(layout) {
        const l = parseLayout(layout);
        this.panelSpecs = l.panels;
        if (l.viewport !== undefined) {
            this.viewportSpec = l.viewport;
        }
        this._build(true);
        this._layoutChanged();
    }

    /** download getLayout() as <layoutName>.json */
    exportLayout() {
        const blob = new Blob([JSON.stringify(this.getLayout(), null, 2)], { type: "application/json" });
        const a = h("a", { href: URL.createObjectURL(blob), download: `${this.layoutName}.json` });
        a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 0);
    }

    async _importLayoutFile(input) {
        const file = input.files?.[0];
        input.value = "";
        if (!file) {
            return;
        }
        try {
            this.setLayout(JSON.parse(await file.text()));
        } catch (e) {
            alert(`Could not import ${file.name}: ${e.message || e}`);
        }
    }

    _layoutChanged() {
        this.onLayoutChange?.(this.getLayout());
    }

    // ------------------------------------------------------------ dialogs
    _renderPsd() {
        const P = this._psdPanel;
        const [a, b] = this.grid.range;
        this.$psdTitle.textContent = `${P.spec.title}: PSD of ${a.toFixed(2)} – ${b.toFixed(2)} s`;
        this.$psdInfo.textContent = renderPsd(this.$psdBody, this.ds, P, this.grid.range, +this.$psdSeg.value, this.theme);
    }

    _pickSignal(target = null) {
        this.$sigTitle.textContent = target ? `Add signal to "${target.title}"` : "New plot";
        const names = this.ds.names();
        const choose = (n) => {
            this.$sig.close();
            if (target) {
                target.series.push({ label: n, solid: n, color: target.series.length });
                if (target.custom && target.series.length === 2) {
                    target.title = "Custom";
                }
            } else {
                this.panelSpecs.push({ title: n, custom: true, styles: [null, null, null], series: [{ label: n, solid: n }] });
            }
            this._build(true);
            this._layoutChanged();
            requestAnimationFrame(() => this.$gridwrap.scrollTo({ top: this.$gridwrap.scrollHeight, behavior: "smooth" }));
        };
        const render = () => {
            const q = this.$sigSearch.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
            this.$sigList.replaceChildren();
            let shown = 0;
            for (const n of names) {
                const l = n.toLowerCase();
                if (q.length && !q.every((w) => l.includes(w))) {
                    continue;
                }
                this.$sigList.append(h("div", { onclick: () => choose(n) }, n));
                if (++shown > 400) {
                    break;
                }
            }
            this.$sigList.firstChild?.classList.add("fp-sel");
        };
        this.$sigSearch.oninput = render;
        this.$sigSearch.onkeydown = (e) => {
            if (e.key === "Enter") {
                const f = this.$sigList.querySelector(".fp-sel");
                if (f) {
                    choose(f.textContent);
                }
            }
        };
        this.$sigSearch.value = "";
        render();
        this.$sig.showModal();
        this.$sigSearch.focus();
    }
}
