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
            src: p.src ?? null,
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
