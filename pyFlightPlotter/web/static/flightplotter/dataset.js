// Columnar time-series data: one shared time vector, named signal arrays.

export class Dataset {
    /**
     * @param {object} o
     * @param {ArrayLike<number>} o.time  sample times in seconds, monotonic
     * @param {Object<string, ArrayLike<number>>|Map<string, ArrayLike<number>>} o.columns  signals, same length as time
     * @param {{t: number, name: string, kind?: string}[]} [o.events]  time markers
     */
    constructor({ time, columns = {}, events = [] }) {
        this.t = time instanceof Float64Array ? time : Float64Array.from(time);
        this.n = this.t.length;
        if (this.n < 2) {
            throw new Error("need at least two samples");
        }
        this.cols = new Map();
        const entries = columns instanceof Map ? columns.entries() : Object.entries(columns);
        for (const [name, arr] of entries) {
            this.add(name, arr);
        }
        this.events = [...events].map((e) => ({ kind: "event", ...e })).sort((a, b) => a.t - b.t);
    }

    has(name) {
        return this.cols.has(name);
    }

    get(name) {
        return this.cols.get(name);
    }

    add(name, arr) {
        if (arr.length !== this.n) {
            throw new Error(`signal "${name}" has ${arr.length} samples, time has ${this.n}`);
        }
        this.cols.set(name, arr);
    }

    names() {
        return [...this.cols.keys()];
    }

    get t0() {
        return this.t[0];
    }

    get t1() {
        return this.t[this.n - 1];
    }

    /** index of the last sample with t[i] <= time, clamped to the data */
    indexAt(time) {
        const t = this.t;
        if (!(time > t[0])) {
            return 0;
        }
        if (time >= t[this.n - 1]) {
            return this.n - 1;
        }
        let lo = 0;
        let hi = this.n - 1;
        while (hi - lo > 1) {
            const mid = (lo + hi) >> 1;
            if (t[mid] <= time) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return lo;
    }
}
