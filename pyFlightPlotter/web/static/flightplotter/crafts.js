// Craft geometry for the 3D viewport, the JS counterpart of pyFlightPlotter.Craft3D.
//
// Plain-object format (what Craft3D.toJSON() returns and Python's WebPlotter sends):
//   { body: [polyline, ...],                         polyline = [[x, y, z], ...], body frame
//     rotors: [{ xyz, axis, R, tilt_xyz, tilt_axis, tilt2_axis }],
//     surfaces: [{ tilt_xyz, tilt_axis, geometry: [polyline, ...] }] }
// Rotor tilt follows Craft3D.generate(): the rotor rotates about the point
// xyz - tilt_xyz, first about tilt_axis, then about tilt2_axis = axis × tilt_axis.

const norm = (v) => {
    const l = Math.hypot(...v);
    return v.map((c) => c / l);
};
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

export class Craft3D {
    /** @param {number[][][]} body  list of polylines */
    constructor(body = [[[0, 0, 0]]]) {
        this.body = body;
        this.rotors = [];
        this.surfaces = [];
    }

    addRotor({ xyz = [0, 0, 0], axis = [0, 0, -1], tilt_xyz = [0, 0, 0], tilt_axis = [1, 0, 0], R = 0.1 } = {}) {
        const a = norm(axis);
        const ta = norm(tilt_axis);
        const t2 = cross(a, ta);
        if (Math.hypot(...t2) < 1e-4) {
            throw new Error("tilt_axis cannot be parallel to axis");
        }
        this.rotors.push({ xyz, axis: a, R, tilt_xyz, tilt_axis: ta, tilt2_axis: norm(t2) });
        return this;
    }

    addSurface({ tilt_xyz = [0, 0, 0], tilt_axis = [0, 1, 0], geometry = [] } = {}) {
        this.surfaces.push({ tilt_xyz, tilt_axis: norm(tilt_axis), geometry });
        return this;
    }

    toJSON() {
        return { body: this.body, rotors: this.rotors, surfaces: this.surfaces };
    }
}

/** forward-pointing triangle at x = l, as in pyFlightPlotter.Quadrotor */
export function noseArrow(l) {
    return [[l, -l / 4, 0], [l, l / 4, 0], [l + l / 4, 0, 0], [l, -l / 4, 0]];
}

/** pyFlightPlotter.Quadrotor */
export function quadrotor(l = 0.2, R = 0.1) {
    const c = new Craft3D([noseArrow(l)]);
    for (const xyz of [[-l, l, 0], [l, l, 0], [-l, -l, 0], [l, -l, 0]]) {
        c.addRotor({ xyz, axis: [0, 0, -1], R });
    }
    return c;
}

/** pyFlightPlotter.Tailsitter */
export function tailsitter(l = 0.3, R = 0.1) {
    const c = new Craft3D([[[0, 2 * l, l], [0, 2 * l, 0], [0, 0, -l], [l, 0, -l], [0, 0, -l], [0, -2 * l, 0], [0, -2 * l, l], [0, 2 * l, l]]]);
    c.addRotor({ xyz: [0, -l, -l], axis: [0, 0, -1], R });
    c.addRotor({ xyz: [0, l, -l], axis: [0, 0, -1], R });
    const surf = [[0, -2 * l, l], [0, -2 * l, 1.5 * l], [0, -0.5 * l, 1.5 * l], [0, -0.5 * l, l]];
    c.addSurface({ tilt_xyz: [0, 0, l], tilt_axis: [0, 1, 0], geometry: [surf] });
    c.addSurface({ tilt_xyz: [0, 0, l], tilt_axis: [0, 1, 0], geometry: [surf.map(([x, y, z]) => [x, -y, z])] });
    return c;
}

/** n rotors evenly spaced on a ring of radius l */
export function multirotor(n, l = 0.3, R = 0.1) {
    const c = new Craft3D([noseArrow(l)]);
    for (let i = 0; i < n; i++) {
        const a = (2 * Math.PI * i) / n + Math.PI / n;
        c.addRotor({ xyz: [l * Math.cos(a), l * Math.sin(a), 0], axis: [0, 0, -1], R });
    }
    return c;
}
