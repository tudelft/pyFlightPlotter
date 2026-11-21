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
// A layout file holds the panels and the 3D view (viewport.js), without data:
//   { format: "flightplotter-layout", version: 1, panels: [panel spec, ...], viewport: {...} | null }
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

/** give panels without row/col the next free slot (row-major), then drop empty rows and columns */
export function placePanels(panels, ncols = 3) {
    const taken = new Set(panels.filter((p) => p.row != null && p.col != null).map((p) => `${p.row},${p.col}`));
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
export const LAYOUT_VERSION = 1;

/** panels + viewport as a layout file object; plain JSON, so Craft3D becomes its geometry */
export function layoutToJSON(panels, viewport) {
    const strip = (p) => {
        const { src, ...rest } = p; // eslint-disable-line no-unused-vars
        return rest;
    };
    return JSON.parse(
        JSON.stringify({ format: LAYOUT_FORMAT, version: LAYOUT_VERSION, panels: panels.map(strip), viewport: viewport ?? null }),
    );
}

const VIEW_VECTORS = ["att", "attSet", "attMeas", "pos", "posSet", "posMeas", "vel", "velSet", "velMeas", "acc", "accSet", "accMeas"];
const VIEW_ACTUATORS = ["rotor", "rotorSet", "surface", "surfaceSet"];

/**
 * Check a layout file object and return a copy: {panels, viewport}, where viewport is undefined if the
 * file has no "viewport" key (keep the current one) and null if it is null (no 3D view).
 * Throws an Error that names the offending entry, e.g. "panels[2].series[0].solid: ...".
 */
export function parseLayout(x) {
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
    const checkRef = (r, path) => {
        if (r == null || isStr(r) || (Array.isArray(r) && r.length === 2 && isStr(r[0]) && (r[1] == null || isNum(r[1])))) {
            return;
        }
        fail(path, 'expected a column name, ["name", scale] or null');
    };

    if (!isObj(x)) {
        fail("layout", "expected an object");
    }
    if (x.format !== undefined && x.format !== LAYOUT_FORMAT) {
        fail("format", `expected "${LAYOUT_FORMAT}"`);
    }
    if (x.version !== undefined && !(isNum(x.version) && x.version <= LAYOUT_VERSION)) {
        fail("version", `this flightplotter reads layout version ${LAYOUT_VERSION} or older`);
    }
    if (!Array.isArray(x.panels)) {
        fail("panels", "expected a list of panels");
    }
    x.panels.forEach((p, i) => {
        const at = `panels[${i}]`;
        if (!isObj(p)) {
            fail(at, "expected an object");
        }
        optional(p.title, isStr, `${at}.title`, "a string");
        optional(p.ylabel, isStr, `${at}.ylabel`, "a string");
        optional(p.ylim, (v) => Array.isArray(v) && v.length === 2 && v.every((y) => y == null || isNum(y)), `${at}.ylim`, "[lo|null, hi|null]");
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
            optional(s.color, (v) => isStr(v) || Number.isInteger(v), `${sat}.color`, 'a color index or "#rrggbb"');
            STYLE_KEYS.forEach((k) => checkRef(s[k], `${sat}.${k}`));
        });
    });

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
    return { panels: structuredClone(x.panels), viewport: v === undefined ? undefined : structuredClone(v) };
}
