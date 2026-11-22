// Panel specs and grid placement.
//
// A panel spec is plain JSON (the Python WebPlotter sends the same format):
//   {
//     title, ylabel, ylim: [lo|null, hi|null], row, col,       // row/col optional
//     styles: [lightLabel|null, solidLabel|null, dashedLabel|null],
//     series: [{ label, color?, light, solid, dashed }],      // each: "column" | ["column", scale] | null
//   }
// This mirrors FlightPlotterBase._plot_timeseries: per series up to three line
// styles; light = thin and faint, solid = main, dashed = setpoint/reference.
//
// A layout file holds tabs of panels and the 3D view (viewport.js), without data:
//   { format: "flightplotter-layout", version: 2,
//     tabs: [{ name, view: bool, panels: [panel spec, ...] }],   view: show the 3D view in this tab
//     viewport: {...} | null }                                     shared by all tabs
// Version 1 files ({ panels, viewport }) read as one tab.
// A tab file (<name>.tab.json) holds one tab, without the 3D view, which belongs to the whole layout:
//   { format: "flightplotter-tab", version: 2, name, view, panels: [panel spec, ...] }
// Column references that a dataset lacks are skipped, so one layout can serve logs with different signals.

export const STYLE_KEYS = ["light", "solid", "dashed"];

/** "col" | ["col", scale] -> {col, scale}, or null if missing from the dataset */
export function resolveRef(ds, r) {
    if (r == null) {
        return null;
    }
    const [col, scale] = Array.isArray(r) ? r : [r, 1];
    return ds.has(col) ? { col, scale: scale ?? 1 } : null;
}

/** column reference -> "name" or "name * scale", as typed in the edit dialogs */
export function refToText(r) {
    if (r == null) {
        return "";
    }
    if (Array.isArray(r)) {
        return r[1] == null || r[1] === 1 ? r[0] : `${r[0]} * ${r[1]}`;
    }
    return r;
}

/** "name" | "name * scale" | "" -> column reference */
export function refFromText(t) {
    t = t.trim();
    if (!t) {
        return null;
    }
    const m = t.match(/^(.*\S)\s*\*\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?)$/i);
    return m && Number.isFinite(+m[2]) ? [m[1], +m[2]] : t;
}

/** resolve column references; drop series styles whose column is missing and panels left empty */
export function resolvePanels(ds, panels) {
    const out = [];
    for (const p of panels) {
        const series = (p.series || [])
            .map((s, i) => ({
                label: s.label ?? `series ${i}`,
                color: s.color ?? i,
                light: resolveRef(ds, s.light),
                solid: resolveRef(ds, s.solid),
                dashed: resolveRef(ds, s.dashed),
            }))
            .filter((s) => s.light || s.solid || s.dashed);
        if (!series.length) {
            continue;
        }
        out.push({
            title: p.title || "",
            ylabel: p.ylabel || "",
            ylim: p.ylim || [null, null],
            styles: p.styles || [null, null, null],
            row: p.row ?? null,
            col: p.col ?? null,
            series,
            custom: !!p.custom,
            src: p, // the spec this came from, which edits change
        });
    }
    return out;
}

/** give panels without row/col (or on a cell taken already) the next free slot (row-major), then drop empty rows
 *  and columns */
export function placePanels(panels, ncols = 3) {
    const taken = new Set();
    for (const p of panels) {
        if (p.row != null && p.col != null) {
            if (taken.has(`${p.row},${p.col}`)) {
                p.row = p.col = null;
            } else {
                taken.add(`${p.row},${p.col}`);
            }
        }
    }
    let r = 0;
    let c = 0;
    for (const p of panels) {
        if (p.row != null && p.col != null) {
            continue;
        }
        while (taken.has(`${r},${c}`)) {
            c++;
            if (c >= ncols) {
                c = 0;
                r++;
            }
        }
        p.row = r;
        p.col = c;
        taken.add(`${r},${c}`);
    }
    for (const key of ["row", "col"]) {
        const used = [...new Set(panels.map((p) => p[key]))].sort((a, b) => a - b);
        const remap = new Map(used.map((v, i) => [v, i]));
        for (const p of panels) {
            p[key] = remap.get(p[key]);
        }
    }
    return panels;
}

export const LAYOUT_FORMAT = "flightplotter-layout";
export const TAB_FORMAT = "flightplotter-tab";
export const LAYOUT_VERSION = 2; // of both file formats

const stripTab = (t) => ({
    name: t.name,
    view: !!t.view,
    panels: t.panels.map((p) => {
        const { src, ...rest } = p; // eslint-disable-line no-unused-vars
        return rest;
    }),
});

/** tabs + viewport as a layout file object; plain JSON, so Craft3D becomes its geometry */
export function layoutToJSON(tabs, viewport) {
    return JSON.parse(
        JSON.stringify({
            format: LAYOUT_FORMAT,
            version: LAYOUT_VERSION,
            tabs: tabs.map(stripTab),
            viewport: viewport ?? null,
        }),
    );
}

/** one tab as a tab file object */
export function tabToJSON(tab) {
    return JSON.parse(JSON.stringify({ format: TAB_FORMAT, version: LAYOUT_VERSION, ...stripTab(tab) }));
}

const VIEW_VECTORS = ["att", "attSet", "attMeas", "pos", "posSet", "posMeas", "vel", "velSet", "velMeas", "acc", "accSet", "accMeas"];
const VIEW_ACTUATORS = ["rotor", "rotorSet", "surface", "surfaceSet"];

const fail = (path, msg) => {
    throw new Error(`${path}: ${msg}`);
};
const isObj = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
const optional = (v, ok, path, what) => {
    if (v !== undefined && v !== null && !ok(v)) {
        fail(path, `expected ${what}`);
    }
};
const isStr = (v) => typeof v === "string";
const isNum = (v) => typeof v === "number" && Number.isFinite(v);
const isVec3 = (v) => Array.isArray(v) && v.length === 3 && v.every(isNum);
const isPolyline = (v) => Array.isArray(v) && v.every(isVec3);
const checkRef = (r, path) => {
    if (r == null || isStr(r) || (Array.isArray(r) && r.length === 2 && isStr(r[0]) && (r[1] == null || isNum(r[1])))) {
        return;
    }
    fail(path, 'expected a column name, ["name", scale] or null');
};

function checkHeader(x, format) {
    if (!isObj(x)) {
        fail(format === TAB_FORMAT ? "tab" : "layout", "expected an object");
    }
    if (x.format !== undefined && x.format !== format) {
        const hint = { [LAYOUT_FORMAT]: "a whole layout", [TAB_FORMAT]: "a single tab" }[x.format];
        fail("format", `expected "${format}"${hint ? `, but this file holds ${hint}` : ""}`);
    }
    if (x.version !== undefined && !(isNum(x.version) && x.version <= LAYOUT_VERSION)) {
        fail("version", `this flightplotter reads layout version ${LAYOUT_VERSION} or older`);
    }
}

function checkPanels(panels, base) {
    panels.forEach((p, i) => {
        const at = `${base}[${i}]`;
        if (!isObj(p)) {
            fail(at, "expected an object");
        }
        optional(p.title, isStr, `${at}.title`, "a string");
        optional(p.ylabel, isStr, `${at}.ylabel`, "a string");
        optional(
            p.ylim,
            (v) => Array.isArray(v) && v.length === 2 && v.every((y) => y == null || isNum(y)) && !(isNum(v[0]) && isNum(v[1]) && v[0] >= v[1]),
            `${at}.ylim`,
            "[lo|null, hi|null] with lo < hi",
        );
        optional(p.styles, (v) => Array.isArray(v) && v.length === 3 && v.every((y) => y == null || isStr(y)), `${at}.styles`, "three labels or nulls");
        for (const k of ["row", "col"]) {
            optional(p[k], (v) => Number.isInteger(v) && v >= 0, `${at}.${k}`, "a non-negative integer");
        }
        if (!Array.isArray(p.series)) {
            fail(`${at}.series`, "expected a list of series");
        }
        p.series.forEach((s, j) => {
            const sat = `${at}.series[${j}]`;
            if (!isObj(s)) {
                fail(sat, "expected an object");
            }
            optional(s.label, isStr, `${sat}.label`, "a string");
            optional(s.color, (v) => (isStr(v) && /^#[0-9a-f]{6}$/i.test(v)) || (Number.isInteger(v) && v >= 0), `${sat}.color`, 'a color index (0, 1, ...) or "#rrggbb"');
            STYLE_KEYS.forEach((k) => checkRef(s[k], `${sat}.${k}`));
        });
    });
}

/** check a tab object; at: its path in error messages, panelsAt: that of its panels */
function checkTab(t, at, panelsAt = `${at}.panels`) {
    if (!isObj(t)) {
        fail(at, "expected an object");
    }
    optional(t.name, isStr, `${at}.name`, "a string");
    optional(t.view, (v) => typeof v === "boolean", `${at}.view`, "true or false");
    if (!Array.isArray(t.panels)) {
        fail(panelsAt, "expected a list of panels");
    }
    checkPanels(t.panels, panelsAt);
}

/**
 * Check a tab file object (or one tab of a layout) and return a copy: {name, view, panels}.
 * A missing name becomes fallbackName, a missing view false. Throws like parseLayout().
 */
export function parseTab(x, fallbackName = "Tab") {
    checkHeader(x, TAB_FORMAT);
    checkTab(x, "tab", "panels");
    return { name: x.name?.trim() || fallbackName, view: x.view ?? false, panels: structuredClone(x.panels) };
}

/** file object of either kind -> {tabs, viewport?}: a tab file reads as a layout with that one tab and no viewport key */
export function parseLayoutOrTab(x) {
    return isObj(x) && x.format === TAB_FORMAT ? { tabs: [parseTab(x)], viewport: undefined } : parseLayout(x);
}

/**
 * Check a layout file object and return a copy: {tabs, viewport}, where viewport is undefined if the
 * file has no "viewport" key (keep the current one) and null if it is null (no 3D view).
 * Throws an Error that names the offending entry, e.g. "panels[2].series[0].solid: ...".
 */
export function parseLayout(x) {
    checkHeader(x, LAYOUT_FORMAT);
    let tabs;
    if (x.tabs !== undefined) {
        if (!Array.isArray(x.tabs) || !x.tabs.length) {
            fail("tabs", "expected a non-empty list of tabs");
        }
        tabs = x.tabs;
    } else if (Array.isArray(x.panels)) {
        tabs = [{ name: "Plots", view: true, panels: x.panels }];
    } else {
        fail("tabs", "expected a list of tabs (or, in version 1 files, panels)");
    }
    tabs.forEach((t, ti) => (x.tabs ? checkTab(t, `tabs[${ti}]`) : checkTab(t, "layout", "panels")));

    const v = x.viewport;
    if (v !== undefined && v !== null) {
        if (!isObj(v)) {
            fail("viewport", "expected an object or null");
        }
        if (v.craft !== undefined) {
            if (!isObj(v.craft)) {
                fail("viewport.craft", "expected {body, rotors, surfaces}");
            }
            for (const k of ["body", "rotors", "surfaces"]) {
                optional(v.craft[k], Array.isArray, `viewport.craft.${k}`, "a list");
            }
            optional(v.craft.name, isStr, "viewport.craft.name", "a string");
            (v.craft.body || []).forEach((pl, i) => optional(pl, isPolyline, `viewport.craft.body[${i}]`, "a polyline [[x, y, z], ...]"));
            (v.craft.surfaces || []).forEach((sf, i) => {
                const at = `viewport.craft.surfaces[${i}]`;
                if (!isObj(sf) || !isVec3(sf.tilt_xyz) || !isVec3(sf.tilt_axis) || !Array.isArray(sf.geometry) || !sf.geometry.every(isPolyline)) {
                    fail(at, "expected {tilt_xyz: [x, y, z], tilt_axis: [x, y, z], geometry: [polyline, ...]}");
                }
            });
            (v.craft.rotors || []).forEach((r, i) => {
                if (!isObj(r) || !isVec3(r.xyz) || !isVec3(r.axis) || !(isNum(r.R) && r.R > 0)) {
                    fail(`viewport.craft.rotors[${i}]`, "expected {xyz: [x, y, z], axis: [x, y, z], R > 0, ...}");
                }
                for (const k of ["tilt_xyz", "tilt_axis", "tilt2_axis"]) {
                    optional(r[k], isVec3, `viewport.craft.rotors[${i}].${k}`, "[x, y, z]");
                }
                optional(r.spin, (y) => y === "cw" || y === "ccw", `viewport.craft.rotors[${i}].spin`, '"cw", "ccw" or null');
                optional(r.label, isStr, `viewport.craft.rotors[${i}].label`, "a string");
            });
        }
        for (const k of VIEW_VECTORS) {
            optional(v[k], Array.isArray, `viewport.${k}`, "a list of column references");
            (v[k] || []).forEach((r, i) => checkRef(r, `viewport.${k}[${i}]`));
        }
        for (const k of VIEW_ACTUATORS) {
            optional(v[k], Array.isArray, `viewport.${k}`, "a list, one entry per actuator");
            (v[k] || []).forEach((r, i) => {
                if (isObj(r)) {
                    ["thrust", "tilt1", "tilt2"].forEach((t) => checkRef(r[t], `viewport.${k}[${i}].${t}`));
                } else {
                    checkRef(r, `viewport.${k}[${i}]`);
                }
            });
        }
        optional(v.interpolation, (y) => y === "previous" || y === "linear", "viewport.interpolation", '"previous" or "linear"');
    }
    return {
        tabs: tabs.map((t, i) => ({ name: t.name?.trim() || `Tab ${i + 1}`, view: t.view ?? i === 0, panels: structuredClone(t.panels) })),
        viewport: v === undefined ? undefined : structuredClone(v),
    };
}
