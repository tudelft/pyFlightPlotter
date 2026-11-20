// Small signal helpers for computing derived columns before plotting.

/**
 * Rotate vectors by unit quaternions, sample by sample.
 * @param {ArrayLike<number>[]} q  [w, x, y, z] arrays; normalized here, near-zero norm -> identity
 * @param {ArrayLike<number>[]} v  [x, y, z] arrays
 * @param {boolean} [inverse]      rotate by the conjugate (e.g. world -> body for a body -> world quaternion)
 * @returns {Float32Array[]} [x, y, z]
 */
export function rotateByQuaternion(q, v, inverse = false) {
    const n = v[0].length;
    const out = [new Float32Array(n), new Float32Array(n), new Float32Array(n)];
    const [qw, qx, qy, qz] = q;
    for (let i = 0; i < n; i++) {
        let w = qw[i];
        let x = qx[i];
        let y = qy[i];
        let z = qz[i];
        const nrm = Math.hypot(w, x, y, z);
        if (nrm > 1e-6) {
            w /= nrm;
            x /= nrm;
            y /= nrm;
            z /= nrm;
        } else {
            w = 1;
            x = y = z = 0;
        }
        if (inverse) {
            x = -x;
            y = -y;
            z = -z;
        }
        const vx = v[0][i];
        const vy = v[1][i];
        const vz = v[2][i];
        // v' = v + w t + q.xyz × t,  t = 2 q.xyz × v
        const tx = 2 * (y * vz - z * vy);
        const ty = 2 * (z * vx - x * vz);
        const tz = 2 * (x * vy - y * vx);
        out[0][i] = vx + w * tx + (y * tz - z * ty);
        out[1][i] = vy + w * ty + (z * tx - x * tz);
        out[2][i] = vz + w * tz + (x * ty - y * tx);
    }
    return out;
}

/** mean sample rate of a time vector in Hz */
export function sampleRate(t) {
    return (t.length - 1) / (t[t.length - 1] - t[0]);
}

/**
 * Zero-phase 2nd order Butterworth lowpass (filtered forward and backward).
 * Starts each pass in steady state at the first sample, so there is no start-up transient.
 */
export function lowpassFiltfilt(y, fs, fc) {
    const K = Math.tan((Math.PI * Math.min(fc, 0.45 * fs)) / fs);
    const s2 = Math.SQRT2;
    const norm = 1 / (1 + s2 * K + K * K);
    const b0 = K * K * norm;
    const b1 = 2 * b0;
    const b2 = b0;
    const a1 = 2 * (K * K - 1) * norm;
    const a2 = (1 - s2 * K + K * K) * norm;
    const n = y.length;
    const pass = (src, dst, forward) => {
        const step = forward ? 1 : -1;
        let i = forward ? 0 : n - 1;
        let x1 = src[i];
        let x2 = src[i];
        let y1 = src[i];
        let y2 = src[i];
        for (let k = 0; k < n; k++, i += step) {
            const x0 = src[i];
            const y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            x2 = x1;
            x1 = x0;
            y2 = y1;
            y1 = y0;
            dst[i] = y0;
        }
    };
    const tmp = new Float64Array(n);
    const out = new Float64Array(n);
    pass(y, tmp, true);
    pass(tmp, out, false);
    return out;
}

/**
 * First or second backward difference divided by dt^order. Samples with zero
 * time step give 0; the first `order` samples repeat the first valid one.
 */
export function derivative(t, y, order = 1) {
    const n = y.length;
    const out = new Float32Array(n);
    for (let i = order; i < n; i++) {
        const d = order === 1 ? y[i] - y[i - 1] : y[i] - 2 * y[i - 1] + y[i - 2];
        const h = i < n - 1 ? t[i + 1] - t[i] : t[n - 1] - t[n - 2];
        out[i] = h > 0 ? d / h ** order : 0;
    }
    for (let i = 0; i < order && i < n; i++) {
        out[i] = out[Math.min(order, n - 1)];
    }
    return out;
}

/**
 * Turn changes of a bit-mask signal into events, e.g. flight mode flags.
 * @param {ArrayLike<number>} t
 * @param {ArrayLike<number>} flags  32 bit masks
 * @param {string[]} bitNames        name per bit; missing names print as bitN
 * @param {string} [kind]
 */
export function bitChangeEvents(t, flags, bitNames, kind = "mode") {
    const events = [];
    let last = 0;
    for (let i = 0; i < flags.length; i++) {
        const cur = flags[i] >>> 0;
        if (cur === last) {
            continue;
        }
        const parts = [];
        for (let b = 0; b < 32; b++) {
            const m = (1 << b) >>> 0;
            if ((cur & m) && !(last & m)) {
                parts.push(`+${bitNames[b] ?? `bit${b}`}`);
            } else if (!(cur & m) && (last & m)) {
                parts.push(`−${bitNames[b] ?? `bit${b}`}`);
            }
        }
        events.push({ t: t[i], name: parts.join(" "), kind });
        last = cur;
    }
    return events;
}
