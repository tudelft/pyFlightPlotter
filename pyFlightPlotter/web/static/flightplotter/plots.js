// Grid of synchronized uPlot time-series panels.
//
// All panels share one x range and one cursor:
//   drag               zoom to the selected time window
//   wheel              zoom around the mouse; shift+wheel or horizontal scroll pans
//   right/middle drag  pan
//   double click       reset to the full time range
//   click              pin the time cursor (callback)
import uPlot from "uplot";
import { STYLE_KEYS } from "./layout.js";

// matplotlib "tab10", the default colors of the matplotlib pyFlightPlotter
export const COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"];

export function seriesColor(s) {
    return typeof s.color === "string" ? s.color : COLORS[s.color % COLORS.length];
}

export function rgba(hex, a) {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

export const LINE_STYLE = {
    light: (c) => ({ stroke: rgba(c, 0.3), width: 1 }),
    solid: (c) => ({ stroke: rgba(c, 0.85), width: 1.25 }),
    dashed: (c) => ({ stroke: c, width: 1.5, dash: [6, 4] }),
};

// at most 9 characters ("-1.23e-45"), the fixed width of .fp-v in style.css
export function fmt(v) {
    if (v == null || !Number.isFinite(v)) {
        return "–";
    }
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e5)) {
        // three-digit exponents only occur in float64 extremes; drop digits to stay in width
        const s = v.toExponential(2);
        return s.length > 9 ? v.toExponential(0) : s;
    }
    if (a >= 100) {
        return v.toFixed(1);
    }
    return v.toPrecision(3);
}

function axisSize(self, values, axisIdx, cycleNum) {
    const axis = self.axes[axisIdx];
    if (cycleNum > 1) {
        return axis._size;
    }
    let size = axis.ticks.size + axis.gap;
    const longest = (values ?? []).reduce((acc, v) => (v.length > acc.length ? v : acc), "");
    if (longest !== "") {
        self.ctx.font = axis.font[0];
        size += self.ctx.measureText(longest).width / devicePixelRatio;
    }
    return Math.ceil(size);
}

export class PlotGrid {
    /**
     * @param {HTMLElement} root    container for the panels (CSS grid)
     * @param {Dataset} ds
     * @param {object[]} panels     resolved and placed panel specs
     * @param {object} theme        {text, muted, grid, tick, mode, event}
     * @param {object} cb           {editable, onHover(t|null), onClick(t), onRange(min, max), onPsd(P), onRemove(P), onAddSignal(P),
     *                               onEdit(P), onMove(P, {panel} | {row, col})}
     */
    constructor(root, ds, panels, theme, cb) {
        this.root = root;
        this.ds = ds;
        this.theme = theme;
        this.cb = cb;
        this.panels = [];
        this.t0 = ds.t0;
        this.t1 = ds.t1;
        this.range = [this.t0, this.t1];
        this.playhead = null;
        this.hovered = null;
        this.showEvents = true;
        this._pendingRange = null;

        root.replaceChildren();
        this.ncols = Math.max(1, ...panels.map((p) => p.col + 1));
        this.nrows = Math.max(1, ...panels.map((p) => p.row + 1));
        root.style.setProperty("--fp-ncols", this.ncols);
        const lastRow = new Map();
        for (const p of panels) {
            lastRow.set(p.col, Math.max(lastRow.get(p.col) ?? -1, p.row));
        }
        for (const p of panels) {
            this._build(p, lastRow.get(p.col) === p.row);
        }

        this._ro = new ResizeObserver((entries) => {
            for (const e of entries) {
                const P = e.target._panel;
                const { width, height } = e.contentRect;
                if (width > 10 && height > 10) {
                    P.u.setSize({ width: Math.floor(width), height: Math.floor(height) });
                }
            }
        });
        for (const P of this.panels) {
            this._ro.observe(P.plotEl);
        }
        if (cb.editable) {
            this._bindDrop();
        }
    }

    destroy() {
        cancelAnimationFrame(this._rangeFrame);
        this._ro.disconnect();
        for (const P of this.panels) {
            P.u.destroy();
        }
        this.root.replaceChildren();
    }

    // ------------------------------------------------------------ building
    _build(spec, isLastRow) {
        const ds = this.ds;
        const th = this.theme;
        const el = document.createElement("section");
        el.className = "fp-panel";
        el.style.gridRow = spec.row + 1;
        el.style.gridColumn = spec.col + 1;

        const P = { spec, el, groupOn: spec.series.map(() => true), styleOn: [true, true, true] };

        // header: title and line style key (click toggles that style)
        const head = document.createElement("header");
        const title = document.createElement("h3");
        title.textContent = spec.title;
        title.title = spec.title;
        head.append(title);
        if (this.cb.editable) {
            // drag the header onto another panel (swap) or an empty cell (move)
            head.draggable = true;
            head.classList.add("fp-drag-handle");
            title.title = `${spec.title}\nDrag to move, double click to edit`;
            head.addEventListener("dragstart", (e) => {
                this._dragging = P;
                e.dataTransfer.effectAllowed = "move";
                e.dataTransfer.setData("text/plain", spec.title);
                el.classList.add("fp-dragged");
            });
            head.addEventListener("dragend", () => {
                this._dragging = null;
                el.classList.remove("fp-dragged");
                this._clearDropMarks();
            });
            title.addEventListener("dblclick", () => this.cb.onEdit?.(P));
        }
        const styleKey = document.createElement("div");
        styleKey.className = "fp-stylekey";
        STYLE_KEYS.forEach((k, si) => {
            const label = spec.styles[si];
            if (!label || !spec.series.some((s) => s[k])) {
                return;
            }
            const b = document.createElement("button");
            b.className = `fp-sk fp-sk-${k}`;
            b.innerHTML = "<i></i>";
            b.append(label);
            b.title = `Show/hide all "${label}" lines`;
            b.onclick = () => {
                P.styleOn[si] = !P.styleOn[si];
                b.classList.toggle("fp-off", !P.styleOn[si]);
                this._applyVisibility(P);
            };
            styleKey.append(b);
        });
        head.append(styleKey);
        el.append(head);

        // chips: one per series, with live values; click toggles the series
        const chips = document.createElement("div");
        chips.className = "fp-chips";
        el.append(chips);

        const data = [ds.t];
        const series = [{}];
        P.groups = spec.series.map((s, g) => {
            const color = seriesColor(s);
            const idx = {};
            for (const k of STYLE_KEYS) {
                const r = s[k];
                if (!r) {
                    continue;
                }
                let arr = ds.get(r.col);
                if (r.scale !== 1) {
                    const src = arr;
                    arr = new Float32Array(src.length);
                    for (let i = 0; i < src.length; i++) {
                        arr[i] = src[i] * r.scale;
                    }
                }
                idx[k] = data.length;
                data.push(arr);
                series.push({ label: `${s.label} ${k}`, ...LINE_STYLE[k](color), points: { show: false } });
            }
            const chip = document.createElement("button");
            chip.className = "fp-chip";
            chip.style.setProperty("--fp-c", color);
            chip.title = STYLE_KEYS.filter((k) => s[k])
                .map((k) => {
                    const r = s[k];
                    const line = `${spec.styles[STYLE_KEYS.indexOf(k)] || k}: ${r.col}${r.scale !== 1 ? ` × ${+r.scale.toPrecision(4)}` : ""}`;
                    const d = ds.describe(r.col);
                    return d ? `${line}\n    ${d}` : line;
                })
                .join("\n");
            chip.innerHTML = "<i></i><span class=\"fp-lbl\"></span>";
            chip.querySelector(".fp-lbl").textContent = s.label;
            const vals = {};
            for (const k of STYLE_KEYS) {
                if (s[k]) {
                    const v = document.createElement("span");
                    v.className = `fp-v fp-v-${k}`;
                    v.textContent = "–";
                    chip.append(v);
                    vals[k] = v;
                }
            }
            chip.onclick = () => {
                P.groupOn[g] = !P.groupOn[g];
                chip.classList.toggle("fp-off", !P.groupOn[g]);
                this._applyVisibility(P);
            };
            chips.append(chip);
            return { idx, chip, vals, data: Object.fromEntries(Object.entries(idx).map(([k, i]) => [k, data[i]])) };
        });

        const plotEl = document.createElement("div");
        plotEl.className = "fp-plot";
        plotEl._panel = P;
        P.plotEl = plotEl;

        // per-panel tools, shown on hover
        const tools = document.createElement("div");
        tools.className = "fp-tools";
        const tool = (txt, tip, fn) => {
            const b = document.createElement("button");
            b.textContent = txt;
            b.title = tip;
            b.onclick = fn;
            tools.append(b);
        };
        tool("PSD", "Power spectral density of the visible window", () => this.cb.onPsd?.(P));
        tool("PNG", "Save this panel as PNG", () => this._savePng(P));
        if (this.cb.editable) {
            tool("✎", "Edit this panel", () => this.cb.onEdit?.(P));
            tool("+", "Add a signal to this panel", () => this.cb.onAddSignal?.(P));
            tool("×", "Remove this panel", () => this.cb.onRemove?.(P));
        }
        plotEl.append(tools);
        el.append(plotEl);
        this.root.append(el);

        const [ylo, yhi] = spec.ylim;
        const axisCommon = {
            stroke: th.muted,
            grid: { stroke: th.grid, width: 1, dash: [3, 3] },
            ticks: { stroke: th.tick, size: 4 },
            font: "11px system-ui, sans-serif",
            labelFont: "11px system-ui, sans-serif",
        };
        const opts = {
            width: 400,
            height: 150,
            pxAlign: 0,
            legend: { show: false },
            cursor: {
                y: false,
                points: { show: false },
                sync: { key: this.syncKey, setSeries: false },
                drag: { x: true, y: false, setScale: false },
                bind: { dblclick: () => () => {} },
            },
            select: { show: true },
            scales: {
                x: { time: false, min: this.range[0], max: this.range[1], auto: false },
                y: {
                    range: (u, min, max) => {
                        if (min == null || !Number.isFinite(min)) {
                            min = ylo ?? 0;
                            max = yhi ?? 1;
                        }
                        const [a, b] = uPlot.rangeNum(min, max, 0.1, true);
                        const lo = ylo ?? a;
                        const hi = yhi ?? b;
                        return lo < hi ? [lo, hi] : [lo - 1, lo + 1];
                    },
                },
            },
            axes: [
                {
                    ...axisCommon,
                    size: isLastRow ? 40 : 26,
                    gap: 3,
                    label: isLastRow ? "Time [s]" : undefined,
                    labelSize: isLastRow ? 14 : 0,
                    labelGap: 0,
                },
                { ...axisCommon, size: axisSize, gap: 4, label: spec.ylabel, labelSize: 16, labelGap: 2 },
            ],
            series,
            hooks: {
                setCursor: [(u) => this._onCursor(P, u)],
                setSelect: [(u) => this._onSelect(P, u)],
                draw: [(u) => this._drawEvents(P, u)],
            },
        };
        P.u = new uPlot(opts, data, plotEl);

        P.playheadEl = document.createElement("div");
        P.playheadEl.className = "fp-playhead";
        P.playheadEl.hidden = true;
        P.u.over.append(P.playheadEl);
        this._bindMouse(P);
        this.panels.push(P);
    }

    // ------------------------------------------------------------ drag and drop
    _bindDrop() {
        const root = this.root;
        // grid cell under the pointer, from the grid's track sizes
        const cellAt = (e) => {
            const r = root.getBoundingClientRect();
            const cs = getComputedStyle(root);
            const gap = parseFloat(cs.columnGap) || 0;
            const colW = (r.width - parseFloat(cs.paddingLeft || 0) - parseFloat(cs.paddingRight || 0) - gap * (this.ncols - 1)) / this.ncols;
            const rowH = parseFloat(cs.getPropertyValue("--fp-row-h")) || 200;
            return {
                col: Math.max(0, Math.min(this.ncols - 1, Math.floor((e.clientX - r.left) / (colW + gap)))),
                row: Math.max(0, Math.floor((e.clientY - r.top) / (rowH + gap))),
            };
        };
        const target = (e) => {
            const el = e.target.closest?.(".fp-panel");
            const P = el && this.panels.find((x) => x.el === el);
            return P ? { panel: P } : cellAt(e);
        };
        root.addEventListener("dragover", (e) => {
            if (!this._dragging) {
                return;
            }
            e.preventDefault();
            e.dataTransfer.dropEffect = "move";
            this._clearDropMarks();
            const t = target(e);
            if (t.panel) {
                if (t.panel !== this._dragging) {
                    t.panel.el.classList.add("fp-drop");
                }
            } else {
                this._dropGhost ??= Object.assign(document.createElement("div"), { className: "fp-drop-ghost" });
                this._dropGhost.style.gridRow = t.row + 1;
                this._dropGhost.style.gridColumn = t.col + 1;
                root.append(this._dropGhost);
            }
        });
        root.addEventListener("dragleave", (e) => {
            if (!root.contains(e.relatedTarget)) {
                this._clearDropMarks();
            }
        });
        root.addEventListener("drop", (e) => {
            const P = this._dragging;
            if (!P) {
                return;
            }
            e.preventDefault();
            this._clearDropMarks();
            const t = target(e);
            if (t.panel !== P) {
                this.cb.onMove?.(P, t);
            }
        });
    }

    _clearDropMarks() {
        for (const el of this.root.querySelectorAll(".fp-drop")) {
            el.classList.remove("fp-drop");
        }
        this._dropGhost?.remove();
    }

    get syncKey() {
        // one cursor sync group per grid, so several plotters on a page stay independent
        this._syncKey ??= `fp-${Math.random().toString(36).slice(2)}`;
        return this._syncKey;
    }

    _applyVisibility(P) {
        P.u.batch(() => {
            P.groups.forEach((G, g) => {
                STYLE_KEYS.forEach((k, si) => {
                    const i = G.idx[k];
                    if (i == null) {
                        return;
                    }
                    const show = P.groupOn[g] && P.styleOn[si];
                    if (P.u.series[i].show !== show) {
                        P.u.setSeries(i, { show });
                    }
                });
            });
        });
    }

    // ------------------------------------------------------------ mouse
    _bindMouse(P) {
        const over = P.u.over;
        over.addEventListener("mouseenter", () => {
            this.hovered = P;
        });
        over.addEventListener("mouseleave", () => {
            if (this.hovered === P) {
                this.hovered = null;
            }
            this.cb.onHover?.(null);
        });
        over.addEventListener(
            "wheel",
            (e) => {
                e.preventDefault();
                const [min, max] = this._pendingRange || this.range;
                const span = max - min;
                const w = over.clientWidth;
                if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
                    const d = ((e.shiftKey ? e.deltaY || e.deltaX : e.deltaX) / w) * span;
                    this.setRange(min + d, max + d);
                } else {
                    const x = min + (e.offsetX / w) * span;
                    const f = Math.exp(e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015));
                    this.setRange(x - (x - min) * f, x + (max - x) * f);
                }
            },
            { passive: false },
        );
        over.addEventListener("dblclick", () => this.setRange(this.t0, this.t1));
        over.addEventListener("contextmenu", (e) => e.preventDefault());
        let down = null;
        over.addEventListener("pointerdown", (e) => {
            down = { x: e.clientX, y: e.clientY, button: e.button, range: [...this.range] };
            if (e.button === 1 || e.button === 2) {
                over.setPointerCapture(e.pointerId);
                e.preventDefault();
            }
        });
        over.addEventListener("pointermove", (e) => {
            if (!down || (down.button !== 1 && down.button !== 2)) {
                return;
            }
            const [min, max] = down.range;
            const d = (-(e.clientX - down.x) / over.clientWidth) * (max - min);
            this.setRange(min + d, max + d);
        });
        over.addEventListener("pointerup", (e) => {
            if (down && down.button === 0 && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 4) {
                this.cb.onClick?.(P.u.posToVal(e.offsetX, "x"));
            }
            down = null;
        });
    }

    _onSelect(P, u) {
        if (u.select.width < 3) {
            return;
        }
        // cursor sync mirrors the drag box into every panel and fires this hook in each of them
        // on mouseup; only the dragged panel zooms, but all of them have to clear their box
        const hide = () => u.setSelect({ left: 0, top: 0, width: 0, height: 0 }, false);
        if (this.hovered !== P) {
            hide();
            return;
        }
        const a = u.posToVal(u.select.left, "x");
        const b = u.posToVal(u.select.left + u.select.width, "x");
        hide();
        this.setRange(a, b);
    }

    /**
     * Draw the cursor at time t (null: hide it) for hover in another window, without reporting it as hover here.
     * uPlot's cursor sync covers one window only: each window runs its own copy of uPlot.
     */
    showCursor(t) {
        const P = this.panels[0];
        if (!P || this.hovered || t === this._remoteT) {
            return;
        }
        this._remoteT = t;
        const u = P.u;
        // the other panels follow through uPlot's sync
        u.setCursor(t == null ? { left: -10, top: -10 } : { left: u.valToPos(t, "x"), top: u.over.clientHeight / 2 });
    }

    _onCursor(P, u) {
        const idx = u.cursor.idx;
        const inside = idx != null && u.cursor.left >= 0;
        if (this.hovered === P) {
            this._remoteT = undefined;
            this.cb.onHover?.(inside ? u.posToVal(u.cursor.left, "x") : null);
        }
        if (inside) {
            this._writeChips(P, idx);
        } else if (this.playhead != null) {
            this._writeChips(P, this.ds.indexAt(this.playhead));
        }
    }

    _writeChips(P, idx) {
        for (const G of P.groups) {
            for (const k in G.vals) {
                G.vals[k].textContent = fmt(G.data[k][idx]);
            }
        }
    }

    // ------------------------------------------------------------ range
    setRange(min, max) {
        const full = this.t1 - this.t0;
        const span = Math.max(max - min, 1e-4);
        if (span >= full) {
            min = this.t0;
            max = this.t1;
        } else {
            if (min < this.t0) {
                min = this.t0;
                max = min + span;
            }
            if (max > this.t1) {
                max = this.t1;
                min = max - span;
            }
        }
        this._pendingRange = [min, max];
        if (this._rangeFrame) {
            return;
        }
        // coalesce wheel / drag bursts into one redraw per frame
        this._rangeFrame = requestAnimationFrame(() => {
            this._rangeFrame = 0;
            const [a, b] = this._pendingRange;
            this._pendingRange = null;
            this.range = [a, b];
            for (const P of this.panels) {
                P.u.setScale("x", { min: a, max: b });
            }
            this._placePlayheads();
            this.cb.onRange?.(a, b);
        });
    }

    // ------------------------------------------------------------ playhead
    setPlayhead(t) {
        this.playhead = t;
        this._placePlayheads();
        if (t != null && !this.hovered) {
            const idx = this.ds.indexAt(t);
            for (const P of this.panels) {
                this._writeChips(P, idx);
            }
        }
    }

    _placePlayheads() {
        const t = this.playhead;
        for (const P of this.panels) {
            if (t == null || t < this.range[0] || t > this.range[1]) {
                P.playheadEl.hidden = true;
                continue;
            }
            P.playheadEl.hidden = false;
            P.playheadEl.style.transform = `translateX(${P.u.valToPos(t, "x")}px)`;
        }
    }

    // ------------------------------------------------------------ events
    setShowEvents(on) {
        this.showEvents = on;
        for (const P of this.panels) {
            P.u.redraw(false, false);
        }
    }

    _drawEvents(P, u) {
        const ev = this.ds.events;
        if (!ev.length || !this.showEvents) {
            return;
        }
        const { ctx } = u;
        const { top, height } = u.bbox;
        const min = u.scales.x.min;
        const max = u.scales.x.max;
        const pr = devicePixelRatio;
        ctx.save();
        ctx.lineWidth = pr;
        ctx.font = `${10 * pr}px system-ui, sans-serif`;
        ctx.textBaseline = "top";
        ctx.textAlign = "left"; // uPlot leaves this at "center" after drawing axes
        let labelEnd = -1e9;
        for (const e of ev) {
            if (e.t < min || e.t > max) {
                continue;
            }
            const x = Math.round(u.valToPos(e.t, "x", true)) + 0.5;
            const mode = e.kind === "mode";
            ctx.strokeStyle = rgba(mode ? this.theme.mode : this.theme.event, 0.6);
            ctx.setLineDash(mode ? [2 * pr, 2 * pr] : []);
            ctx.beginPath();
            ctx.moveTo(x, top);
            ctx.lineTo(x, top + height);
            ctx.stroke();
            if (P.spec.row === 0 && x > labelEnd) {
                ctx.fillStyle = mode ? this.theme.mode : this.theme.event;
                ctx.fillText(e.name, x + 3 * pr, top + 2 * pr);
                labelEnd = x + ctx.measureText(e.name).width + 10 * pr;
            }
        }
        ctx.restore();
    }

    _savePng(P) {
        const src = P.u.ctx.canvas;
        const pr = devicePixelRatio;
        const headH = 24 * pr;
        const c = document.createElement("canvas");
        c.width = src.width;
        c.height = src.height + headH;
        const ctx = c.getContext("2d");
        ctx.fillStyle = this.theme.bg;
        ctx.fillRect(0, 0, c.width, c.height);
        ctx.fillStyle = this.theme.text;
        ctx.font = `600 ${13 * pr}px system-ui, sans-serif`;
        ctx.fillText(P.spec.title, 8 * pr, 16 * pr);
        let x = ctx.measureText(P.spec.title).width + 24 * pr;
        ctx.font = `${11 * pr}px system-ui, sans-serif`;
        P.spec.series.forEach((s, g) => {
            if (!P.groupOn[g]) {
                return;
            }
            ctx.fillStyle = seriesColor(s);
            ctx.fillRect(x, 9 * pr, 10 * pr, 3 * pr);
            ctx.fillStyle = this.theme.text;
            ctx.fillText(s.label, x + 14 * pr, 15 * pr);
            x += ctx.measureText(s.label).width + 28 * pr;
        });
        ctx.drawImage(src, 0, headH);
        const a = document.createElement("a");
        a.download = `${P.spec.title}.png`.replace(/[\\/:*?"<>|]/g, "_");
        a.href = c.toDataURL("image/png");
        a.click();
    }
}
