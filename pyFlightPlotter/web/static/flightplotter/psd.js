// Welch power spectral density.
import uPlot from "uplot";
import { STYLE_KEYS } from "./layout.js";
import { LINE_STYLE, seriesColor } from "./plots.js";

function fftInPlace(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
        let bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            [re[i], re[j]] = [re[j], re[i]];
            [im[i], im[j]] = [im[j], im[i]];
        }
    }
    for (let len = 2; len <= n; len <<= 1) {
        const ang = (-2 * Math.PI) / len;
        const wr = Math.cos(ang);
        const wi = Math.sin(ang);
        for (let i = 0; i < n; i += len) {
            let cr = 1;
            let ci = 0;
            for (let k = 0; k < len / 2; k++) {
                const a = i + k;
                const b = a + len / 2;
                const tr = re[b] * cr - im[b] * ci;
                const ti = re[b] * ci + im[b] * cr;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                const nr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = nr;
            }
        }
    }
}

/**
 * One-sided Welch PSD: Hann window, 50 % overlap, mean removed per segment,
 * segments containing non-finite samples skipped.
 * @returns {{f: Float64Array, psd: Float64Array, count: number, seg: number}} f without the DC bin
 */
export function welch(y, fs, nperseg = 1024) {
    const n = y.length;
    const seg = Math.min(nperseg, 1 << Math.floor(Math.log2(n)));
    const step = seg >> 1;
    const win = new Float64Array(seg);
    let wss = 0;
    for (let i = 0; i < seg; i++) {
        win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / seg);
        wss += win[i] * win[i];
    }
    const nf = seg / 2 + 1;
    const psd = new Float64Array(nf);
    const re = new Float64Array(seg);
    const im = new Float64Array(seg);
    let count = 0;
    for (let s = 0; s + seg <= n; s += step) {
        let mean = 0;
        let bad = false;
        for (let i = 0; i < seg; i++) {
            const v = y[s + i];
            if (!Number.isFinite(v)) {
                bad = true;
                break;
            }
            mean += v;
        }
        if (bad) {
            continue;
        }
        mean /= seg;
        for (let i = 0; i < seg; i++) {
            re[i] = (y[s + i] - mean) * win[i];
            im[i] = 0;
        }
        fftInPlace(re, im);
        for (let k = 0; k < nf; k++) {
            psd[k] += re[k] * re[k] + im[k] * im[k];
        }
        count++;
    }
    const scale = 1 / (fs * wss * Math.max(count, 1));
    for (let k = 0; k < nf; k++) {
        psd[k] *= scale * (k === 0 || k === nf - 1 ? 1 : 2);
    }
    const f = new Float64Array(nf);
    for (let k = 0; k < nf; k++) {
        f[k] = (k * fs) / seg;
    }
    return { f: f.subarray(1), psd: psd.subarray(1), count, seg };
}

/**
 * Render the PSD of a panel's visible series over [a, b] into `body`.
 * @returns {string} info line
 */
export function renderPsd(body, ds, P, [a, b], nperseg, theme) {
    const i0 = ds.indexAt(a);
    const i1 = ds.indexAt(b) + 1;
    const n = i1 - i0;
    body.replaceChildren();
    if (n < 64) {
        body.append("Zoom out: the visible window has fewer than 64 samples.");
        return "";
    }
    const fs = (n - 1) / (ds.t[i1 - 1] - ds.t[i0]);
    const data = [];
    const series = [{ label: "f [Hz]", value: (u, v) => (v == null ? "–" : v.toFixed(1)) }];
    let fx = null;
    let info = "";
    P.groups.forEach((G, g) => {
        if (!P.groupOn[g]) {
            return;
        }
        const s = P.spec.series[g];
        STYLE_KEYS.forEach((k, si) => {
            if (!G.data[k] || !P.styleOn[si]) {
                return;
            }
            const r = welch(G.data[k].subarray(i0, i1), fs, nperseg);
            fx = r.f;
            info = `fs = ${fs.toFixed(1)} Hz, ${n} samples, segment ${r.seg} (Δf = ${(fs / r.seg).toFixed(2)} Hz), ${r.count} averages, Hann window`;
            data.push(Array.from(r.psd, (v) => (v > 0 ? v : null))); // log scale: no zeros
            series.push({
                label: `${s.label} (${P.spec.styles[si] || k})`,
                ...LINE_STYLE[k](seriesColor(s)),
                points: { show: false },
                value: (u, v) => (v == null ? "–" : v.toExponential(2)),
            });
        });
    });
    if (!fx) {
        body.append("No visible series.");
        return "";
    }
    const axis = {
        stroke: theme.muted,
        grid: { stroke: theme.grid, dash: [3, 3] },
        ticks: { stroke: theme.tick },
    };
    new uPlot(
        {
            width: Math.min(window.innerWidth - 80, 1100),
            height: Math.min(window.innerHeight - 240, 560),
            scales: { x: { time: false, distr: 3 }, y: { distr: 3 } },
            axes: [
                { ...axis, label: "Frequency [Hz]" },
                { ...axis, label: "PSD [unit²/Hz]", size: 70, values: (u, vals) => vals.map((v) => (v == null ? "" : v.toExponential(0))) },
            ],
            cursor: { drag: { x: true, y: true, uni: 20 }, points: { show: false } },
            series,
        },
        [fx, ...data],
        body,
    );
    return info;
}
