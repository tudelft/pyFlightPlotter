// CSV export of a Dataset, with the signals the host computed kept apart from the ones it was given.

const ROWS_PER_PART = 2000;

/** a cell: empty for NaN and inf; for float32, the shortest text that reads back as the same float32 */
function cell(v, single) {
    if (!Number.isFinite(v)) {
        return "";
    }
    if (single) {
        for (let digits = 6; digits < 9; digits++) {
            const x = Number(v.toPrecision(digits));
            if (Math.fround(x) === v) {
                return String(x);
            }
        }
        return String(Number(v.toPrecision(9)));
    }
    return String(v);
}

/** a CSV field, quoted if it needs to be */
function field(s) {
    return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/**
 * The dataset as CSV, as a list of text parts (for `new Blob(parts)`, which keeps a large table out of one string).
 *
 * Layout: an optional `#` comment block, a header row, then one row per sample. The first column is `time` [s].
 * The signals the host gave are next, under their own names, then the computed ones (`dataset.derived`) under
 * `<derivedPrefix><name>`, so the origin is in the column name itself, and the comment block explains each
 * computed column. NaN becomes an empty cell.
 *
 * @param {import("./dataset.js").Dataset} ds
 * @param {object} [o]
 * @param {string[]} [o.header]  lines for the top of the comment block (file, units, ...)
 * @param {boolean} [o.comments=true]  false leaves out the comment block, for readers that don't skip `#` lines
 * @param {string} [o.derivedPrefix="calc:"]  marks the computed columns
 * @returns {string[]}
 */
export function datasetToCsv(ds, { header = [], comments = true, derivedPrefix = "calc:" } = {}) {
    const given = ds.names().filter((n) => !ds.derived.has(n));
    const computed = ds.names().filter((n) => ds.derived.has(n));
    const names = [...given, ...computed];
    const cols = names.map((n) => ds.get(n));
    const single = cols.map((c) => c instanceof Float32Array);
    const heads = ["time", ...given, ...computed.map((n) => derivedPrefix + n)];

    const parts = [];
    if (comments) {
        const lines = [...header];
        lines.push(
            "time is in seconds. An empty cell is a missing value.",
            `${given.length} column(s) without a "${derivedPrefix}" prefix are the signals as they came in, ` +
                "with only the unit or scaling changed.",
        );
        if (computed.length) {
            lines.push(
                `${computed.length} column(s) prefixed "${derivedPrefix}" were computed by the exporting app ` +
                    "from those signals; they are not part of the original data:",
            );
            // columns that share a description (the axes of a vector) share a line
            const byText = new Map();
            for (const n of computed) {
                const text = ds.describe(n).replace(/\s*\n\s*/g, " ");
                byText.set(text, [...(byText.get(text) ?? []), derivedPrefix + n]);
            }
            for (const [text, group] of byText) {
                lines.push(`  ${group.join(", ")}: ${text}`);
            }
        }
        parts.push(lines.flatMap((l) => l.split("\n")).map((l) => `# ${l}\n`).join(""));
    }
    parts.push(`${heads.map(field).join(",")}\n`);
    for (let i0 = 0; i0 < ds.n; i0 += ROWS_PER_PART) {
        const rows = [];
        for (let i = i0; i < Math.min(ds.n, i0 + ROWS_PER_PART); i++) {
            const row = [String(ds.t[i])];
            for (let c = 0; c < cols.length; c++) {
                row.push(cell(cols[c][i], single[c]));
            }
            rows.push(row.join(","));
        }
        parts.push(`${rows.join("\n")}\n`);
    }
    return parts;
}
