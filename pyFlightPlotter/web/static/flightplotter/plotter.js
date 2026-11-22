// FlightPlotter: synchronized time-series panels + 3D viewport in one component.
import { Dataset } from "./dataset.js";
import { Session } from "./session.js";
import { STYLE_KEYS, TAB_FORMAT, resolvePanels, placePanels, layoutToJSON, tabToJSON, parseLayout, parseLayoutOrTab, refToText, refFromText } from "./layout.js";
import { PlotGrid } from "./plots.js";
import { renderPsd } from "./psd.js";

// "download" and "upload" from Lucide (lucide.dev, ISC license), the set Nuxt UI hosts tend to use
const ICONS = {
    download: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5"/><path d="M12 15V3"/>',
    upload: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m17 8-5-5-5 5"/><path d="M12 3v12"/>',
};
const iconSvg = (name) =>
    `<svg class="fp-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name]}</svg>`;
/** an inline icon element, sized to the text */
function icon(name) {
    const t = document.createElement("template");
    t.innerHTML = iconSvg(name);
    return t.content.firstChild;
}

const HELP = [
    "<b>Hover</b> a plot to move the time cursor; values show next to each series and the 3D view follows",
    "<b>Click</b> to pin the time, <kbd>Esc</kbd> to unpin. While pinned, the clock shows Δt to the mouse",
    "<b>Drag</b> to zoom into a time window, <b>wheel</b> to zoom around the mouse, <b>double click</b> to reset",
    "<b>Right / middle drag</b>, <kbd>Shift</kbd>+wheel or horizontal scroll to pan",
    "<kbd>Space</kbd> play / pause, <kbd>←</kbd> <kbd>→</kbd> step one sample (<kbd>Shift</kbd>: 0.1 s)",
    "<kbd>Shift</kbd>+<kbd>Space</kbd> or <kbd>Ctrl</kbd>+<kbd>Space</kbd> (or <kbd>Shift</kbd>/<kbd>Ctrl</kbd>+click ▶) replays: the pinned time stays put while a dashed line plays. <kbd>Space</kbd> stops it, leaving the dashed line where it stopped, and plays again from the pinned time, until you pin another time or press <kbd>Esc</kbd>",
    "Click a <b>series</b> to hide it, or a <b>line style</b> in the panel header to hide that style",
    "Panel tools on hover: <b>PSD</b> (Welch spectrum of the visible window), <b>PNG</b> export; when editable, <b>+</b> adds a signal and <b>×</b> removes the panel",
    `<b>Layout</b> (when editable): export the panels and 3D view as JSON (${iconSvg("download")}), import one (${iconSvg("upload")}), or reset to the initial layout (↺)`,
    `Tabs (when editable): ${iconSvg("download")} on the active tab saves just that tab as a <code>.tab.json</code> file, ${iconSvg("upload")} next to <b>+</b> adds the tabs of such files (or all tabs of a layout file)`,
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
     * @param {object[]} [o.panels]    panel specs (see layout.js), for a single tab
     * @param {{name: string, panels: object[], view?: boolean}[]} [o.tabs]  instead of panels: tabs of panels;
     *                                 view shows the 3D view in that tab (default: first tab only)
     * @param {object} [o.viewport]    viewport spec (see viewport.js), shared by all tabs; omit for no 3D view
     * @param {object} [o.layout]      layout file object to start with, e.g. a saved one; "reset" still returns to
     *                                 tabs/panels/viewport. A malformed one is ignored with a console warning.
     * @param {string} [o.title]
     * @param {boolean} [o.editable]   allow editing panels, tabs and the craft, and layout import/export (default true)
     * @param {number} [o.ncols]       grid columns for panels without row/col (default 3)
     * @param {number} [o.rows]        panel rows that fit the height; 0 = all (default 5)
     * @param {"light"|"dark"} [o.theme]
     * @param {string} [o.layoutName]  file name for exported layouts (default "layout")
     * @param {(layout: object) => void} [o.onLayoutChange]  after every edit, import or reset, with getLayout()
     * @param {{label: string, viewport: object}[]} [o.crafts]  choices for the 3D view's craft selector; each
     *                                 viewport is merged into the viewport spec (typically craft, rotor, rotorSet,
     *                                 surface, surfaceSet), and the craft's name becomes the label
     * @param {{label: string, viewport: object}[]} [o.signals]  choices for the 3D view's signal selector, e.g.
     *                                 which signals drive the rotors; each viewport is merged into the viewport spec
     *                                 (typically rotor, rotorSet) and outlasts a change of craft. The spec's
     *                                 "signals" key names the chosen one.
     * @param {string} [o.popoutUrl]   page that shows a tab in its own window: it loads this library and calls
     *                                 attachPopout(). Without it, tabs can't be popped out.
     * @param {Session} [o.session]    (attachPopout) join the session of a view in another window ...
     * @param {object} [o.tab]         ... showing this one of its tabs
     */
    constructor(container, o) {
        if (o.session) {
            this.session = o.session;
            this.fixedTab = o.tab;
            this.popoutWindow = window;
        } else {
            const ds = o.data instanceof Dataset ? o.data : new Dataset(o.data);
            // own copies, so edits never mutate the host's objects; Craft3D becomes plain geometry
            const initialLayout = layoutToJSON(o.tabs || [{ name: "Plots", view: true, panels: o.panels || [] }], o.viewport || null);
            let start = parseLayout(initialLayout);
            if (o.layout) {
                try {
                    const l = parseLayout(o.layout);
                    start = { tabs: l.tabs, viewport: l.viewport === undefined ? start.viewport : l.viewport };
                } catch (e) {
                    console.warn(`flightplotter: ignoring the given layout: ${e.message}`);
                }
            }
            this.session = new Session(ds, start, {
                initialLayout,
                onLayoutChange: o.onLayoutChange,
                crafts: (o.crafts || []).map((c) => ({ label: c.label, viewport: JSON.parse(JSON.stringify(c.viewport)) })),
                signals: (o.signals || []).map((c) => ({ label: c.label, viewport: JSON.parse(JSON.stringify(c.viewport)) })),
                editable: o.editable ?? true,
                layoutName: o.layoutName || "layout",
                popoutUrl: o.popoutUrl,
            });
            this.session.theme = o.theme || "light";
        }
        const S = this.session;
        this.ds = S.ds;
        this.editable = S.editable;
        this.ncols = o.ncols ?? 3;
        this.activeTab = 0;
        this.follow = false;
        this.viewHidden = new Set();
        S.attach(this);
        this._buildDom(container, o);
        this.root.querySelector(".fp-rows").value = String(o.rows ?? 5);
        this._applyTheme();

        this._onKey = (e) => this._key(e);
        document.addEventListener("keydown", this._onKey);
        this._onResize = () => this._layoutRows();
        window.addEventListener("resize", this._onResize);
        if (!this.fixedTab) {
            // pop-outs can't live without this window
            this._onPageHide = () => this.session.destroy();
            window.addEventListener("pagehide", this._onPageHide);
        }
    }

    /** element in the toolbar where the host can put its own controls */
    get toolbarSlot() {
        return this.root.querySelector(".fp-slot");
    }

    /** the tab this view shows: its fixed one in a pop-out, else the active one that isn't popped out */
    get tab() {
        if (this.fixedTab) {
            return this.fixedTab;
        }
        const S = this.session;
        let i = Math.max(0, Math.min(this.activeTab, S.tabs.length - 1));
        if (S.isPoppedOut(S.tabs[i])) {
            const j = S.tabs.findIndex((t) => !S.isPoppedOut(t));
            if (j >= 0) {
                i = j;
            }
        }
        this.activeTab = i;
        return S.tabs[i];
    }

    get panelSpecs() {
        return this.tab.panels;
    }

    set panelSpecs(v) {
        this.tab.panels = v;
    }

    get viewportSpec() {
        return this.session.viewportSpec;
    }

    set viewportSpec(v) {
        this.session.viewportSpec = v;
    }

    get crafts() {
        return this.session.crafts;
    }

    get signals() {
        return this.session.signals;
    }

    get initialLayout() {
        return this.session.initialLayout;
    }

    destroy() {
        document.removeEventListener("keydown", this._onKey);
        window.removeEventListener("resize", this._onResize);
        window.removeEventListener("pagehide", this._onPageHide);
        if (this.fixedTab) {
            this.session.detach(this);
        } else {
            this.session.views.delete(this);
            this.session.destroy();
        }
        this.grid?.destroy();
        this.view?.destroy();
        this.root.remove();
    }

    /** the session changed: "time" | "range" | "play" | "layout" | "theme" */
    sessionChanged(kind) {
        if (kind === "time") {
            this._showTime();
        } else if (kind === "range") {
            this.grid?.setRange(...this.session.visibleRange());
        } else if (kind === "play") {
            this._showPlay();
        } else if (kind === "layout") {
            if (this.fixedTab && !this.session.tabs.includes(this.fixedTab)) {
                window.close(); // its tab was closed or replaced
                return;
            }
            this._build();
        } else if (kind === "theme") {
            this._applyTheme();
        }
    }

    // ------------------------------------------------------------ dom
    _buildDom(container, o) {
        const btn = (text, title, cls, onclick) => h("button", { class: cls, title, onclick }, ...[text].flat());
        this.$play = btn("▶", "Play / pause (Space); Shift/Ctrl+click: replay from the pinned time, which stays (Shift/Ctrl+Space)", "fp-play", (e) =>
            this.togglePlay(e.shiftKey || e.ctrlKey || e.metaKey),
        );
        this.$speed = h(
            "select",
            { title: "Playback speed", onchange: (e) => this.session.setSpeed(+e.target.value) },
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
        this.$tabFile = h("input", {
            type: "file",
            accept: ".json,application/json",
            multiple: true,
            hidden: true,
            onchange: (e) => this._importTabFiles(e.target),
        });
        const layoutMenu = h(
            "span",
            { class: "fp-btngroup" },
            btn(["Layout ", icon("download")], "Export the panels and 3D view as JSON", "", () => this.exportLayout()),
            btn(icon("upload"), "Import a layout JSON", "", () => this.$layoutFile.click()),
            btn("↺", "Reset to the initial layout", "", () => this.setLayout(this.initialLayout)),
            this.$layoutFile,
            this.$tabFile,
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
            (this.$title = h("span", { class: "fp-title", title: o.title || "" }, o.title || "")),
            h("span", { class: "fp-slot" }),
            this.editable ? btn("+ Plot", "Add a plot of any signal", "", () => this._pickSignal()) : "",
            this.editable && !this.fixedTab ? layoutMenu : "",
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
            vbtn("labels", "Rotor no.", "#999"),
            this.$follow,
            btn("Reset view", "Reset camera", "", () => {
                this.follow = false;
                this.$follow.classList.remove("fp-on");
                this.view?.setFollow(false);
            }),
        );
        this.$craftSel = h("select", { title: "Craft drawn in the 3D view", onchange: (e) => this._chooseCraft(+e.target.value) });
        this.$signalSel = h("select", { title: "Signals driving the rotors", onchange: (e) => this._chooseSignals(+e.target.value) });
        this.$rotorsBtn = btn("Rotors…", "Spin direction and signals per rotor", "", () => this._editRotors());
        this.$viewbar2 = h("div", { class: "fp-viewbar fp-viewbar2" }, this.$craftSel, this.$signalSel, this.$rotorsBtn);
        // top-down view of the craft, drawn by the Viewport over this box
        this.$inset = h("div", { class: "fp-inset", hidden: true });
        this.$view.append(h("div", { class: "fp-view3d" }), h("div", { class: "fp-viewtop" }, this.$viewbar, this.$inset), this.$viewbar2);
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

        this.$edit = h("dialog", { class: "fp-dialog fp-edit" });
        this.$signals = h("datalist", { id: `fp-signals-${Math.random().toString(36).slice(2)}` });

        this.$help = h("dialog", { class: "fp-dialog" });
        const ul = h("ul");
        ul.innerHTML = HELP.map((s) => `<li>${s}</li>`).join("");
        this.$help.append(h("header", {}, h("h2", {}, "Controls"), close(this.$help)), ul);

        this.$tabs = h("div", { class: "fp-tabs" });
        this.root = h("div", { class: "fp-root" }, toolbar, this.$tabs, main, this.$psd, this.$sig, this.$edit, this.$signals, this.$help);
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
    /** "light" | "dark", for every view of this log; rebuilds the plots since canvas colors are baked in */
    setTheme(theme) {
        this.session.setTheme(theme);
    }

    _applyTheme() {
        const theme = this.session.theme;
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
        this._build();
    }

    // ------------------------------------------------------------ build
    _build() {
        const S = this.session;
        this._renderTabs();
        this._applyTabView();
        const panels = placePanels(resolvePanels(this.ds, this.panelSpecs), this.ncols);
        this.root.classList.toggle("fp-noview", !this.viewportSpec);
        this._syncCraftSel();
        this.grid?.destroy();
        this.grid = new PlotGrid(this.$grid, this.ds, panels, this.theme, {
            editable: this.editable,
            onHover: (t) => S.setHover(t, this),
            onClick: (t) => S.setPin(t),
            onRange: (a, b) => S.setRange(a, b, this),
            onPsd: (P) => {
                this._psdPanel = P;
                this.$psd.showModal();
                this._renderPsd();
            },
            onRemove: (P) => {
                this.panelSpecs = this.panelSpecs.filter((c) => c !== P.spec.src);
                this._build();
                this._layoutChanged();
            },
            onAddSignal: (P) => this._pickSignal(P.spec.src),
            onEdit: (P) => this._editPanel(P.spec.src),
            onMove: (P, to) => {
                // pin every shown panel where it is, then move this one
                for (const Q of this.grid.panels) {
                    Q.spec.src.row = Q.spec.row;
                    Q.spec.src.col = Q.spec.col;
                }
                const a = P.spec.src;
                if (to.panel) {
                    const b = to.panel.spec.src;
                    [a.row, a.col, b.row, b.col] = [b.row, b.col, a.row, a.col];
                } else {
                    a.row = to.row;
                    a.col = to.col;
                }
                this._build();
                this._layoutChanged();
            },
        });
        this._layoutRows();
        if (S.range) {
            this.grid.setRange(...S.range);
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
            this.$inset.hidden = !this.view.setInset(this.$inset);
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

    /** show or hide the 3D view as the shown tab wants */
    _applyTabView() {
        const hide = !this.tab.view;
        this.$view.classList.toggle("fp-hidden", hide);
        this.$split.hidden = hide;
        this.$view3d.classList.toggle("fp-on", !hide);
    }

    _toggleView() {
        this.tab.view = !this.tab.view;
        this._applyTabView();
        this._buildView();
        this._layoutChanged();
    }

    // ------------------------------------------------------------ tabs
    _renderTabs() {
        const S = this.session;
        if (this.fixedTab) {
            this.$tabs.hidden = true;
            this.$title.textContent = this.$title.title = this.fixedTab.name;
            document.title = this.fixedTab.name;
            return;
        }
        const shown = this.tab;
        this.$tabs.hidden = S.tabs.length < 2 && !this.editable && !S.popoutUrl;
        const btn = (text, title, onclick, cls = "") => h("button", { class: cls, title, onclick }, text);
        this.$tabs.replaceChildren(
            ...S.tabs.map((t, i) => {
                const popped = S.isPoppedOut(t);
                const name = btn(
                    t.name,
                    popped ? "In its own window; click to bring it to the front" : this.editable ? "Double click to rename" : "",
                    () => (popped ? S.popOut(t) : this.showTab(i)),
                    "fp-tabname",
                );
                if (this.editable) {
                    name.addEventListener("dblclick", () => this._renameTab(t, name));
                }
                return h(
                    "div",
                    { class: `fp-tab${t === shown && !popped ? " fp-active" : ""}${popped ? " fp-popped" : ""}` },
                    name,
                    this.editable && t === shown && !popped ? btn(icon("download"), "Save this tab as a .tab.json file", () => this.exportTab(t)) : "",
                    S.popoutUrl && !popped ? btn("⧉", "Open this tab in a new window", () => this._popOut(t)) : "",
                    this.editable && S.tabs.length > 1 ? btn("×", "Close this tab", () => this._closeTab(t)) : "",
                );
            }),
            this.editable ? btn("+", "New tab", () => this._addTab(), "fp-tabadd") : "",
            this.editable ? btn(icon("upload"), "Add tabs from .tab.json files (or all tabs of a layout file)", () => this.$tabFile.click(), "fp-tabadd") : "",
        );
    }

    /** show tab i (main view) */
    showTab(i) {
        // no rebuild for the shown tab: it would replace the tab button under a double click
        if (this.session.tabs[i] === this.tab) {
            return;
        }
        this.activeTab = i;
        this._build();
    }

    _addTab() {
        const S = this.session;
        let n = S.tabs.length + 1;
        while (S.tabs.some((t) => t.name === `Tab ${n}`)) {
            n++;
        }
        S.tabs.push({ name: `Tab ${n}`, view: false, panels: [] });
        this.activeTab = S.tabs.length - 1;
        this._build();
        this._layoutChanged();
    }

    /**
     * Append tabs, e.g. from parseTab(); names already taken get a " (2)", " (3)", ... suffix.
     * Shows the first of them.
     */
    addTabs(tabs) {
        const S = this.session;
        if (!tabs.length) {
            return;
        }
        const first = S.tabs.length;
        for (const t of tabs) {
            let name = t.name;
            for (let n = 2; S.tabs.some((u) => u.name === name); n++) {
                name = `${t.name} (${n})`;
            }
            S.tabs.push({ name, view: !!t.view, panels: structuredClone(t.panels) });
        }
        this.activeTab = first;
        this._build();
        this._layoutChanged();
    }

    _closeTab(t) {
        const S = this.session;
        if (t.panels.length && !confirm(`Close "${t.name}" and its ${t.panels.length} plot(s)?`)) {
            return;
        }
        S.popouts.get(t)?.close();
        S.popouts.delete(t);
        const i = S.tabs.indexOf(t);
        S.tabs.splice(i, 1);
        if (this.activeTab >= i) {
            this.activeTab = Math.max(0, this.activeTab - 1);
        }
        this._build();
        this._layoutChanged();
    }

    _renameTab(t, el) {
        const input = h("input", { value: t.name, class: "fp-tabinput" });
        const done = (commit) => {
            if (commit && input.value.trim()) {
                t.name = input.value.trim();
                this._layoutChanged();
            }
            this._renderTabs();
        };
        input.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                done(true);
            } else if (e.key === "Escape") {
                done(false);
            }
        });
        input.addEventListener("blur", () => done(true), { once: true });
        el.replaceWith(input);
        input.select();
    }

    _popOut(t) {
        if (!this.session.popOut(t)) {
            alert("The browser blocked the new window. Allow pop-ups for this site and try again.");
        }
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
    /** move the pinned cursor to time t (seconds) */
    setTime(t) {
        this.session.setPin(t);
    }

    /** start or stop playback; replay: keep the pinned time (Session.togglePlay) */
    togglePlay(replay = false) {
        this.session.togglePlay(replay);
    }

    _showPlay() {
        const S = this.session;
        this.$play.textContent = S.playing ? "❚❚" : "▶";
        this.$play.classList.toggle("fp-on", S.playing);
        this.$speed.value = String(S.speed);
    }

    _showTime() {
        const S = this.session;
        const t = S.currentT();
        this.view?.setTime(t);
        const head = S.playing && S.replayT == null ? S.playT : S.pinT;
        this.grid?.setPlayhead(head, S.replayT);
        if (S.hoverFrom !== this) {
            this.grid?.showCursor(S.playing ? null : S.hoverT);
        }
        const c = this.$clock;
        // shown relative to the pin
        const rel = S.playing ? S.replayT : (S.hoverT ?? S.replayT);
        if (S.pinT != null && rel != null) {
            const dt = rel - S.pinT;
            c.innerHTML = `t ${rel.toFixed(3)} s <span class="fp-pin">Δ ${dt >= 0 ? "+" : ""}${dt.toFixed(3)}</span>`;
        } else if (head != null) {
            c.innerHTML = `<span class="fp-pin">t ${head.toFixed(3)} s</span>`;
        } else {
            c.textContent = `t ${t.toFixed(3)} s`;
        }
    }

    _key(e) {
        // only while visible, and not while typing
        if (!this.root.isConnected || this.root.offsetParent === null) {
            return;
        }
        if (e.target.closest?.("input, select, textarea, dialog, [contenteditable], [role=combobox], [role=listbox]")) {
            return;
        }
        const S = this.session;
        if (e.code === "Space") {
            e.preventDefault();
            S.togglePlay(e.shiftKey || e.ctrlKey || e.metaKey);
        } else if (e.key === "Escape") {
            S.unpin();
        } else if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
            e.preventDefault();
            S.step(e.key === "ArrowRight" ? 1 : -1, e.shiftKey);
        }
    }

    // ------------------------------------------------------------ layout
    /** the current tabs and viewport as plain JSON (see layout.js), including the user's edits */
    getLayout() {
        return this.session.getLayout();
    }

    /**
     * Replace the tabs and, if the layout has a "viewport" key, the 3D view. The data stays.
     * Throws an Error naming the offending entry if the layout is malformed.
     */
    setLayout(layout) {
        parseLayout(layout); // throws before anything changes
        this.activeTab = 0;
        this.session.setLayout(layout);
    }

    /** download getLayout() as <layoutName>.json */
    exportLayout() {
        this._download(this.getLayout(), `${this.session.layoutName}.json`);
    }

    /** download one tab as <layoutName>-<tab name>.tab.json */
    exportTab(tab = this.tab) {
        const name = tab.name.replace(/[^\p{L}\p{N}._-]+/gu, "_").replace(/^_+|_+$/g, "") || "tab";
        this._download(tabToJSON(tab), `${this.session.layoutName}-${name}.tab.json`);
    }

    _download(obj, filename) {
        const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
        const a = h("a", { href: URL.createObjectURL(blob), download: filename });
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
            const x = JSON.parse(await file.text());
            // a single tab joins the others rather than replacing them all
            if (x?.format === TAB_FORMAT) {
                this.addTabs(parseLayoutOrTab(x).tabs);
            } else {
                this.setLayout(x);
            }
        } catch (e) {
            alert(`Could not import ${file.name}: ${e.message || e}`);
        }
    }

    /** add the tabs of every chosen file; one bad file doesn't stop the others */
    async _importTabFiles(input) {
        const files = [...(input.files || [])];
        input.value = "";
        const tabs = [];
        const errors = [];
        for (const file of files) {
            try {
                tabs.push(...parseLayoutOrTab(JSON.parse(await file.text())).tabs);
            } catch (e) {
                errors.push(`${file.name}: ${e.message || e}`);
            }
        }
        this.addTabs(tabs);
        if (errors.length) {
            alert(`Could not import\n${errors.join("\n")}`);
        }
    }

    _layoutChanged() {
        this.session.layoutChanged(this);
    }

    /** typed text -> column reference; a signal named like "a*2" stays a name */
    _parseRef(t) {
        return this.ds.has(t.trim()) ? t.trim() : refFromText(t);
    }

    /** text input for a column reference, with signal suggestions and a mark when the log lacks it */
    _refInput(r) {
        const names = this.ds.names();
        if (this.$signals.childElementCount !== names.length) {
            this.$signals.replaceChildren(...names.map((n) => h("option", { value: n, label: this.ds.describe(n) || undefined })));
        }
        const i = h("input", { value: refToText(r), autocomplete: "off", list: this.$signals.id, class: "fp-ref", placeholder: "—" });
        const mark = () => {
            const ref = this._parseRef(i.value);
            const col = Array.isArray(ref) ? ref[0] : ref;
            const missing = col != null && !this.ds.has(col);
            i.classList.toggle("fp-missing", missing);
            i.title = missing ? `"${col}" is not in this log` : 'A signal name, optionally "* scale"';
        };
        i.addEventListener("input", mark);
        mark();
        return i;
    }

    // ------------------------------------------------------------ craft
    _syncCraftSel() {
        const sel = this.$craftSel;
        const name = this.viewportSpec?.craft?.name;
        const idx = this.crafts.findIndex((c) => c.label === name);
        sel.replaceChildren(
            ...(idx < 0 ? [h("option", { value: -1, selected: true }, name || "Custom craft")] : []),
            ...this.crafts.map((c, i) => h("option", { value: i, selected: i === idx }, c.label)),
        );
        sel.hidden = this.crafts.length === 0;
        const ssel = this.$signalSel;
        const sname = this.viewportSpec?.signals;
        const sidx = this.signals.findIndex((c) => c.label === sname);
        ssel.replaceChildren(
            ...(sidx < 0 ? [h("option", { value: -1, selected: true }, "Custom signals")] : []),
            ...this.signals.map((c, i) => h("option", { value: i, selected: i === sidx }, c.label)),
        );
        ssel.hidden = this.signals.length === 0 || !(this.viewportSpec?.craft?.rotors || []).length;
        this.$rotorsBtn.hidden = !this.editable || !(this.viewportSpec?.craft?.rotors || []).length;
        this.$viewbar2.hidden = sel.hidden && ssel.hidden && this.$rotorsBtn.hidden;
    }

    _chooseCraft(i) {
        const c = this.crafts[i];
        if (!c) {
            return;
        }
        const v = structuredClone(c.viewport);
        // the chosen signals stay, also if the craft choice brings its own
        const s = this.signals.find((x) => x.label === this.viewportSpec?.signals);
        this.viewportSpec = { ...this.viewportSpec, ...v, ...(s ? structuredClone(s.viewport) : {}), craft: { ...v.craft, name: c.label } };
        this._syncCraftSel();
        this._buildView();
        this._layoutChanged();
    }

    _chooseSignals(i) {
        const c = this.signals[i];
        if (!c) {
            return;
        }
        this.viewportSpec = { ...this.viewportSpec, ...structuredClone(c.viewport), signals: c.label };
        this._syncCraftSel();
        this._buildView();
        this._layoutChanged();
    }

    /** dialog per drawn rotor: label, spin direction, and the estimate / setpoint thrust signals */
    _editRotors() {
        const spec = this.viewportSpec;
        const rotors = spec?.craft?.rotors || [];
        // an actuator entry is a thrust ref or {thrust, tilt1, tilt2}
        const isTilted = (e) => e && typeof e === "object" && !Array.isArray(e);
        const thrustOf = (e) => (isTilted(e) ? e.thrust : e);
        const withThrust = (e, r) => (isTilted(e) ? { ...e, thrust: r } : r);
        const fmt = (v) => (Math.abs(v) < 5e-4 ? "0" : v.toFixed(3));
        const rows = rotors.map((r, i) => {
            const spin = h(
                "select",
                {},
                ...[["", "—"], ["cw", "CW"], ["ccw", "CCW"]].map(([v, t]) => h("option", { value: v, selected: (r.spin || "") === v }, t)),
            );
            const label = h("input", { value: r.label ?? "", placeholder: String(i + 1), class: "fp-label" });
            const est = this._refInput(thrustOf(spec.rotor?.[i]));
            const set = this._refInput(thrustOf(spec.rotorSet?.[i]));
            const tr = h(
                "tr",
                {},
                h("td", {}, String(i + 1)),
                h("td", { class: "fp-muted" }, r.xyz.map(fmt).join(", ")),
                h("td", {}, label),
                h("td", {}, spin),
                h("td", {}, est),
                h("td", {}, set),
            );
            return { tr, spin, label, est, set };
        });
        const apply = () => {
            const craft = {
                ...spec.craft,
                rotors: rotors.map((r, i) => ({ ...r, spin: rows[i].spin.value || undefined, label: rows[i].label.value.trim() || undefined })),
            };
            delete craft.name; // edited, so no longer the chosen preset
            const pad = (a) => Array.from({ length: rotors.length }, (_, i) => a?.[i] ?? null);
            const next = {
                ...spec,
                craft,
                rotor: pad(spec.rotor).map((e, i) => withThrust(e, this._parseRef(rows[i].est.value))),
                rotorSet: pad(spec.rotorSet).map((e, i) => withThrust(e, this._parseRef(rows[i].set.value))),
            };
            if (JSON.stringify([next.rotor, next.rotorSet]) !== JSON.stringify([pad(spec.rotor), pad(spec.rotorSet)])) {
                delete next.signals; // edited, so no longer the chosen signal set
            }
            this.viewportSpec = next;
            this.$edit.close();
            this._syncCraftSel();
            this._buildView();
            this._layoutChanged();
        };
        this.$edit.replaceChildren(
            h(
                "header",
                {},
                h("h2", {}, "Rotors"),
                h("button", { type: "button", onclick: () => this.$edit.close() }, "Cancel"),
                h("button", { type: "button", class: "fp-on", onclick: apply }, "Apply"),
            ),
            h(
                "table",
                { class: "fp-series-table" },
                h("thead", {}, h("tr", {}, ...["#", "Position (body x, y, z)", "Label", "Spin", "Estimate (0..1)", "Setpoint (0..1)"].map((t) => h("th", {}, t)))),
                h("tbody", {}, ...rows.map((r) => r.tr)),
            ),
            h("div", { class: "fp-edit-foot" }, h("span", { class: "fp-muted" }, 'Spin as seen from above. Thrust signals as "name * scale", scaled to 0..1.')),
        );
        this.$edit.showModal();
    }

    /** dialog editing a panel spec in place: titles, limits, style labels and the series table */
    _editPanel(spec) {
        const text = (value, attrs = {}) => h("input", { value: value ?? "", autocomplete: "off", ...attrs });
        const refInput = (r) => this._refInput(r);
        const parseRef = (t) => this._parseRef(t);

        const $title = text(spec.title);
        const $ylabel = text(spec.ylabel);
        const $ylo = text(spec.ylim?.[0], { type: "number", step: "any", placeholder: "auto" });
        const $yhi = text(spec.ylim?.[1], { type: "number", step: "any", placeholder: "auto" });
        const $styles = [0, 1, 2].map((i) => text(spec.styles?.[i], { placeholder: "no label" }));
        const $rows = h("tbody");
        const addRow = (s = {}) => {
            const tr = h(
                "tr",
                {},
                h("td", {}, text(s.label, { class: "fp-label", placeholder: "label" })),
                ...STYLE_KEYS.map((k) => h("td", {}, refInput(s[k]))),
                h(
                    "td",
                    { class: "fp-rowtools" },
                    h("button", { type: "button", title: "Move up", onclick: () => tr.previousSibling?.before(tr) }, "↑"),
                    h("button", { type: "button", title: "Move down", onclick: () => tr.nextSibling?.after(tr) }, "↓"),
                    h("button", { type: "button", title: "Remove this series", onclick: () => tr.remove() }, "×"),
                ),
            );
            // colours default to the series index; pin it so reordering keeps them
            tr._color = s.color ?? $rows.childElementCount;
            $rows.append(tr);
            return tr;
        };
        (spec.series || []).forEach((s) => addRow(s));
        const $error = h("div", { class: "fp-edit-error" });

        const apply = () => {
            const num = (i) => (i.value.trim() === "" ? null : +i.value);
            const series = [...$rows.children].map((tr) => {
                const [label, light, solid, dashed] = [...tr.querySelectorAll("input")].map((i) => i.value);
                const s = { label: label.trim(), light: parseRef(light), solid: parseRef(solid), dashed: parseRef(dashed) };
                if (tr._color != null) {
                    s.color = tr._color;
                }
                return s;
            });
            const blank = series.findIndex((s) => !s.light && !s.solid && !s.dashed);
            if (blank >= 0) {
                $error.textContent = `Series ${blank + 1} has no signal; give it one or remove it.`;
                return;
            }
            if (!series.length) {
                $error.textContent = "A panel needs a series. To get rid of the panel, use × in its tools.";
                return;
            }
            if ($ylo.value.trim() && $yhi.value.trim() && +$ylo.value >= +$yhi.value) {
                $error.textContent = "The lower end of the y range must be below the upper end.";
                return;
            }
            Object.assign(spec, {
                title: $title.value,
                ylabel: $ylabel.value,
                ylim: [num($ylo), num($yhi)],
                styles: $styles.map((i) => i.value.trim() || null),
                series: series.map((s, i) => ({ ...s, label: s.label || `series ${i + 1}` })),
            });
            this.$edit.close();
            this._build();
            this._layoutChanged();
        };

        const field = (label, ...inputs) => h("label", {}, h("span", {}, label), ...inputs);
        this.$edit.replaceChildren(
            h("header", {}, h("h2", {}, "Edit panel"), h("button", { type: "button", onclick: () => this.$edit.close() }, "Cancel"),
                h("button", { type: "button", class: "fp-on", onclick: apply }, "Apply")),
            h("div", { class: "fp-form" },
                field("Title", $title),
                field("Y label", $ylabel),
                field("Y range", $ylo, h("span", { class: "fp-muted" }, "to"), $yhi),
                field("Line styles", ...$styles),
            ),
            h("table", { class: "fp-series-table" },
                h("thead", {}, h("tr", {}, h("th", {}, "Series"),
                    ...STYLE_KEYS.map((k, i) => h("th", {}, k, h("small", {}, spec.styles?.[i] ? ` (${spec.styles[i]})` : ""))),
                    h("th"))),
                $rows,
            ),
            h("div", { class: "fp-edit-foot" },
                h("button", { type: "button", onclick: () => addRow().querySelector("input").focus() }, "+ Series"),
                h("span", { class: "fp-muted" }, 'Signals by name; "name * 10" scales.'),
                $error,
            ),
        );
        this.$edit.showModal();
        $title.focus();
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
            this._build();
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
                const d = this.ds.describe(n);
                this.$sigList.append(
                    h("div", { onclick: () => choose(n), "data-name": n, title: d || undefined }, n, d ? h("span", { class: "fp-desc" }, d) : ""),
                );
                if (++shown > 400) {
                    break;
                }
            }
            this.$sigList.firstChild?.classList.add("fp-sel");
        };
        this.$sigSearch.oninput = render;
        this.$sigSearch.onkeydown = (e) => {
            if (e.key === "Enter") {
                // else the keypress re-clicks "+ Plot", which gets the focus back when the dialog closes
                e.preventDefault();
                const f = this.$sigList.querySelector(".fp-sel");
                if (f) {
                    choose(f.dataset.name);
                }
            }
        };
        this.$sigSearch.value = "";
        render();
        this.$sig.showModal();
        this.$sigSearch.focus();
    }
}

/**
 * For the page a pop-out window loads (FlightPlotter's popoutUrl): show the tab it was opened for, attached to
 * the opener's session. Returns the view, or null if this page wasn't opened as a pop-out.
 * @param {HTMLElement} container
 */
export function attachPopout(container) {
    const id = new URLSearchParams(location.search).get("fp-popout");
    if (!id) {
        return null;
    }
    let reg = null;
    try {
        reg = window.opener?.__flightplotterPopouts?.[id];
    } catch {
        // opener navigated to another origin
    }
    if (!reg || !reg.session.tabs.includes(reg.tab)) {
        container.textContent = "This window has lost its main window. Close it and open the tab again from there.";
        return null;
    }
    const fp = new FlightPlotter(container, { session: reg.session, tab: reg.tab, rows: 0 });
    window.addEventListener("pagehide", () => fp.destroy());
    return fp;
}
