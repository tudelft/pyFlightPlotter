// three.js 3D view of a craft at the cursor time, the counterpart of pyFlightPlotter.Viewport.
//
// Spec (column references as in panel specs: "col" | ["col", scale]):
//   {
//     craft: Craft3D | {body, rotors, surfaces},
//     att, attSet, attMeas: [w, x, y, z],          attitude quaternions, body -> world
//     pos, posSet, posMeas: [x, y, z],              world frame (NED-like: z down)
//     vel, velSet, velMeas, acc, accSet, accMeas: [x, y, z]
//     rotor, rotorSet: per rotor a thrust ref (0..1), or {thrust, tilt1, tilt2} (tilts in rad)
//     surface, surfaceSet: per surface a deflection ref (rad)
//     interpolation: "previous" | "linear",
//     signals: label of the host's signal choice the rotor signals came from (FlightPlotter's signals option)
//   }
// The estimate (att) is drawn solid, the setpoint (attSet) and the measurement (attMeas) dashed. Rotor thrust shows
// on the estimate as filled arcs that grow with thrust from the nose side, clockwise seen from above: the state
// (rotor) fills the rotor disc, the command (rotorSet) a ring around it. Rotor tilts show on the estimate (rotor) and the setpoint (rotorSet).
// Renders on demand only, so an idle view costs nothing.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { ViewHelper } from "three/addons/helpers/ViewHelper.js";
import { Line2 } from "three/addons/lines/Line2.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";
import { COLORS } from "./plots.js";
import { resolveRef } from "./layout.js";

const DEG = Math.PI / 180;
const TRAIL_S = 2.0;
// line width [px] of the dashed setpoint / measurement ghosts and vectors
const DASHED_WIDTH = 1;
// thrust arcs: steps per full circle, opacity, and the setpoint ring's width as a fraction of the rotor radius
const ARC_LEVELS = 96;
const ARC_OPACITY = 0.4;
const RING_WIDTH = 0.2;

export class Viewport {
    /**
     * @param {HTMLElement} container
     * @param {Dataset} ds
     * @param {object} spec  see above
     * @param {object} theme {bg, grid, gridCenter}
     */
    constructor(container, ds, spec, theme) {
        this.container = container;
        this.ds = ds;
        this.follow = false;
        this.linear = spec.interpolation === "linear";
        this.materials = [];
        this.time = ds.t0;

        // -------------------------------------------------- data
        const ref = (r) => {
            const x = resolveRef(ds, r);
            return x && { a: ds.get(x.col), k: x.scale };
        };
        const vec = (refs) => {
            if (!Array.isArray(refs)) {
                return null;
            }
            const v = refs.map(ref);
            return v.every(Boolean) ? v : null;
        };
        // rotor entry: thrust ref, or {thrust, tilt1, tilt2} with tilt refs in rad
        const perActuator = (refs) =>
            Array.isArray(refs)
                ? refs.map((r) => (r && typeof r === "object" && !Array.isArray(r) ? [ref(r.thrust), ref(r.tilt1), ref(r.tilt2)] : [ref(r)]))
                : null;
        this.d = {
            att: vec(spec.att), attSet: vec(spec.attSet), attMeas: vec(spec.attMeas),
            pos: vec(spec.pos), posSet: vec(spec.posSet), posMeas: vec(spec.posMeas),
            vel: vec(spec.vel), velSet: vec(spec.velSet), velMeas: vec(spec.velMeas),
            acc: vec(spec.acc), accSet: vec(spec.accSet), accMeas: vec(spec.accMeas),
            rotor: perActuator(spec.rotor), rotorSet: perActuator(spec.rotorSet),
            surface: Array.isArray(spec.surface) ? spec.surface.map(ref) : null,
            surfaceSet: Array.isArray(spec.surfaceSet) ? spec.surfaceSet.map(ref) : null,
        };
        this.geometry = typeof spec.craft?.toJSON === "function" ? spec.craft.toJSON() : spec.craft || { body: [], rotors: [], surfaces: [] };

        // -------------------------------------------------- three
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setPixelRatio(devicePixelRatio);
        renderer.autoClear = false;
        container.append(renderer.domElement);
        this.renderer = renderer;

        const scene = new THREE.Scene();
        this.scene = scene;
        const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 5000);
        camera.up.set(0, 0, -1);
        this.camera = camera;

        // flight extent, from the position estimate and setpoint
        const box = new THREE.Box3();
        const p3 = new THREE.Vector3();
        for (const p of [this.d.pos, this.d.posSet]) {
            if (!p) {
                continue;
            }
            const stride = Math.max(1, Math.floor(ds.n / 5000));
            for (let i = 0; i < ds.n; i += stride) {
                p3.set(p[0].a[i] * p[0].k, p[1].a[i] * p[1].k, p[2].a[i] * p[2].k);
                if (Number.isFinite(p3.x + p3.y + p3.z)) {
                    box.expandByPoint(p3);
                }
            }
        }
        if (box.isEmpty()) {
            box.set(new THREE.Vector3(-0.5, -0.5, -0.5), new THREE.Vector3(0.5, 0.5, 0.5));
        }
        box.expandByScalar(0.5);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const extent = Math.max(size.x, size.y, size.z);
        this.home = { center, extent };

        // ground grid at z = 0, spacing 1/2/5 × 10^k
        const raw = Math.max(size.x, size.y) / 12;
        const p10 = 10 ** Math.floor(Math.log10(Math.max(raw, 1e-3)));
        const gstep = Math.max(1, [1, 2, 5, 10].map((m) => m * p10).find((s) => s >= raw));
        const gdiv = Math.max(2, Math.ceil(Math.max(size.x, size.y) / gstep) + 2);
        this.grid = new THREE.GridHelper(gdiv * gstep, gdiv);
        this.grid.rotation.x = Math.PI / 2;
        this.grid.position.set(Math.round(center.x / gstep) * gstep, Math.round(center.y / gstep) * gstep, 0);
        scene.add(this.grid);
        // big flights: draw the craft enlarged in the overview, true size when following
        this.craftScale = Math.min(25, Math.max(1, extent / 12));
        scene.add(new THREE.AxesHelper(0.5 * this.craftScale));

        // -------------------------------------------------- craft ghosts
        const [C0, C1, C2, C3, C4, C5] = COLORS;
        this.ghosts = {
            est: this.d.att && this._craft({ color: C0, width: 2.5, arcs: [C0, C1], labels: true }),
            set: this.d.attSet && this._craft({ color: C1, width: DASHED_WIDTH, dashed: true }),
            meas: this.d.attMeas && this._craft({ color: C2, width: DASHED_WIDTH, dashed: true }),
        };
        for (const g of Object.values(this.ghosts)) {
            if (g) {
                g.root.scale.setScalar(this.craftScale);
                scene.add(g.root);
            }
        }

        this.markers = {
            pos: this.d.pos && this._sprite(C0, "dot", 9),
            posSet: this.d.posSet && this._sprite(C1, "ring", 14),
            posMeas: this.d.posMeas && this._sprite(C2, "ring", 20),
        };
        this.vectors = {
            vel: this.d.vel && this._vector(C0, 2),
            velSet: this.d.velSet && this._vector(C1, DASHED_WIDTH, true),
            velMeas: this.d.velMeas && this._vector(C2, DASHED_WIDTH, true),
            acc: this.d.acc && this._vector(C3, 2),
            accSet: this.d.accSet && this._vector(C4, DASHED_WIDTH, true),
            accMeas: this.d.accMeas && this._vector(C5, DASHED_WIDTH, true),
        };
        for (const o of [...Object.values(this.markers), ...Object.values(this.vectors)]) {
            if (o) {
                scene.add(o);
            }
        }

        // trajectory: whole flight (faint) + trail of the last seconds
        this.trails = [];
        for (const [p, c] of [[this.d.pos, C0], [this.d.posSet, C1]]) {
            if (!p) {
                continue;
            }
            const arr = new Float32Array(ds.n * 3);
            for (let i = 0; i < ds.n; i++) {
                for (let j = 0; j < 3; j++) {
                    arr[3 * i + j] = p[j].a[i] * p[j].k;
                }
            }
            // separate geometries over one attribute: the draw range is per geometry
            const attr = new THREE.BufferAttribute(arr, 3);
            const fullGeo = new THREE.BufferGeometry().setAttribute("position", attr);
            const trailGeo = new THREE.BufferGeometry().setAttribute("position", attr);
            const full = new THREE.Line(fullGeo, new THREE.LineBasicMaterial({ color: c, transparent: true, opacity: 0.25 }));
            const trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color: c, transparent: true, opacity: 0.9 }));
            full.frustumCulled = trail.frustumCulled = false;
            scene.add(full, trail);
            this.trails.push({ full, trail });
        }

        // -------------------------------------------------- controls
        this.controls = new OrbitControls(camera, renderer.domElement);
        this.controls.addEventListener("change", () => this.requestRender());
        this.viewHelper = new ViewHelper(camera, renderer.domElement);
        this.viewHelper.center = this.controls.target;
        renderer.domElement.addEventListener("pointerup", (e) => {
            if (this.viewHelper.handleClick(e)) {
                this.requestRender();
            }
        });
        this._lastFrame = performance.now();
        this.setTheme(theme);
        this.resetView();

        this._ro = new ResizeObserver(() => this.resize());
        this._ro.observe(container);
        this.resize();
        this.setTime(ds.t0);
    }

    /**
     * Draw a top-down view of the craft in its body frame (nose up) over the area of el, which lies on top of the
     * view: outline, rotor numbers and thrust arcs, whatever setVisible() hides in the main view except the numbers.
     * @returns {boolean} whether there is one, i.e. the craft has rotors
     */
    setInset(el) {
        const geo = this.geometry;
        if (!(geo.rotors || []).length) {
            return false;
        }
        let half = 1e-3;
        for (const pl of geo.body || []) {
            for (const p of pl) {
                half = Math.max(half, Math.abs(p[0]), Math.abs(p[1]));
            }
        }
        for (const r of geo.rotors) {
            const reach = r.R * (1 + RING_WIDTH);
            half = Math.max(half, Math.abs(r.xyz[0]) + reach, Math.abs(r.xyz[1]) + reach);
        }
        const n0 = this.materials.length;
        const craft = this._craft({ color: COLORS[0], width: 2, arcs: [COLORS[0], COLORS[1]], labels: true });
        // own line materials: their resolution is the inset's size, not the canvas's
        const materials = this.materials.splice(n0);
        const scene = new THREE.Scene();
        scene.add(craft.root);
        // looking down the body z axis (FRD: z down), x up and y right on screen; the frustum is set per size
        const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 100 * half);
        camera.position.set(0, 0, -10 * half);
        camera.up.set(1, 0, 0);
        camera.lookAt(0, 0, 0);
        this.inset = { el, craft, materials, scene, camera, half, px: 0 };
        this._insetRo = new ResizeObserver(() => this.requestRender());
        this._insetRo.observe(el);
        this.setTime(this.time);
        return true;
    }

    _renderInset() {
        const I = this.inset;
        const c = this.renderer.domElement.getBoundingClientRect();
        const r = I.el.getBoundingClientRect();
        const w = Math.min(r.width, c.right - r.left);
        const h = Math.min(r.height, c.bottom - r.top);
        if (I.el.offsetParent === null || w < 10 || h < 10) {
            return;
        }
        for (const m of I.materials) {
            m.resolution.set(w, h);
        }
        const s = Math.min(w, h);
        if (s !== I.px) {
            this._fitInset(s);
        }
        const x = r.left - c.left + (w - s) / 2;
        const y = c.height - (r.top - c.top) - (h + s) / 2; // from the bottom
        const R = this.renderer;
        R.setViewport(x, y, s, s);
        R.setScissor(r.left - c.left, c.height - (r.top - c.top) - h, w, h);
        R.setScissorTest(true);
        R.setClearColor(this.scene.background);
        R.clear();
        R.render(I.scene, I.camera);
        R.setScissorTest(false);
        R.setViewport(0, 0, c.width, c.height);
    }

    /** frame the inset craft for s pixels, with rotor numbers of a fixed pixel height outside the setpoint rings */
    _fitInset(s) {
        const I = this.inset;
        I.px = s;
        const LABEL_PX = 26;
        // the numbers take LABEL_PX of the s pixels across 2 * half, which includes them: solve for half
        const half = I.half / Math.max(0.3, 1 - (2.2 * LABEL_PX) / s);
        const lh = (LABEL_PX * 2 * half) / s;
        this.geometry.rotors.forEach((r, k) => {
            const label = I.craft.rotors[k].label;
            label.scale.set(2 * lh, lh, 1);
            label.position.setLength(r.R * (1 + RING_WIDTH) + 0.55 * lh);
        });
        Object.assign(I.camera, { left: -half, right: half, top: half, bottom: -half });
        I.camera.updateProjectionMatrix();
    }

    destroy() {
        this._insetRo?.disconnect();
        this._ro.disconnect();
        cancelAnimationFrame(this._raf);
        this.controls.dispose();
        this.renderer.dispose();
        this.renderer.domElement.remove();
    }

    setTheme(theme) {
        this.scene.background = new THREE.Color(theme.bg);
        const mats = Array.isArray(this.grid.material) ? this.grid.material : [this.grid.material];
        // GridHelper uses vertex colors; tint through the material instead
        for (const m of mats) {
            m.vertexColors = false;
            m.color = new THREE.Color(theme.grid);
            m.needsUpdate = true;
        }
        this.requestRender();
    }

    // -------------------------------------------------- builders
    _mat(color, width, dashed = false, opacity = 1) {
        const m = new LineMaterial({ color, linewidth: width, dashed, dashSize: 0.03, gapSize: 0.02, transparent: opacity < 1, opacity });
        this.materials.push(m);
        return m;
    }

    _line(points, mat) {
        const g = new LineGeometry();
        g.setPositions(points.flat());
        const l = new Line2(g, mat);
        if (mat.dashed) {
            l.computeLineDistances();
        }
        return l;
    }

    /** unit-radius thrust arc geometry, from angle 0 counter-clockwise: a pie ("pie") or the setpoint ring ("ring") */
    _arcGeometry(kind, level) {
        this._arcGeo ??= { pie: [], ring: [] };
        const cache = this._arcGeo[kind];
        if (!cache[level]) {
            const theta = (2 * Math.PI * level) / ARC_LEVELS;
            const segs = Math.max(1, Math.ceil((64 * level) / ARC_LEVELS));
            cache[level] =
                kind === "pie"
                    ? new THREE.CircleGeometry(1, segs, 0, theta)
                    : new THREE.RingGeometry(1, 1 + RING_WIDTH, segs, 1, 0, theta);
        }
        return cache[level];
    }

    /**
     * arcs: [state color, command color] to draw thrust arcs with (the estimate only)
     */
    _craft({ color, width, dashed = false, arcs = null, labels = false }) {
        const geo = this.geometry;
        const root = new THREE.Group();
        const mat = this._mat(color, width, dashed);
        for (const pl of geo.body || []) {
            root.add(this._line(pl, mat));
        }

        const rot = [];
        (geo.rotors || []).forEach((r, i) => {
            const axis = new THREE.Vector3(...r.axis).normalize();
            const a = Math.abs(axis.x) < 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);
            const e1 = a.clone().cross(axis).normalize();
            const e2 = axis.clone().cross(e1);
            const tiltXyz = new THREE.Vector3(...(r.tilt_xyz || [0, 0, 0]));
            // pivot at xyz - tilt_xyz, rotor centre at tilt_xyz from the pivot
            const pivot = new THREE.Group();
            pivot.position.set(...r.xyz).sub(tiltXyz);
            const body = new THREE.Group();
            body.position.copy(tiltXyz);
            pivot.add(body);
            const pts = [];
            for (let k = 0; k <= 24; k++) {
                const u = (2 * Math.PI * k) / 24;
                const p = e1.clone().multiplyScalar(Math.cos(u) * r.R).add(e2.clone().multiplyScalar(Math.sin(u) * r.R));
                pts.push(p.toArray());
            }
            body.add(this._line(pts, mat));
            if (r.spin === "cw" || r.spin === "ccw") {
                // e1 -> e2 runs counter-clockwise seen from where the axis points
                const sgn = r.spin === "ccw" ? 1 : -1;
                for (const u of [Math.PI / 4, (5 * Math.PI) / 4]) {
                    const radial = e1.clone().multiplyScalar(Math.cos(u)).addScaledVector(e2, Math.sin(u));
                    const tang = e1.clone().multiplyScalar(-Math.sin(u)).addScaledVector(e2, Math.cos(u)).multiplyScalar(sgn);
                    const head = radial.clone().multiplyScalar(r.R);
                    const back = head.clone().addScaledVector(tang, -0.4 * r.R);
                    body.add(this._line([
                        back.clone().addScaledVector(radial, 0.2 * r.R).toArray(),
                        head.toArray(),
                        back.clone().addScaledVector(radial, -0.2 * r.R).toArray(),
                    ], mat));
                }
            }
            let label = null;
            if (labels) {
                // outside the disc, away from the craft's centre
                const out = new THREE.Vector3(...r.xyz).projectOnPlane(axis);
                if (out.lengthSq() < 1e-9) {
                    out.copy(e1);
                }
                label = this._label(r.label ?? String(i + 1), color, r.R);
                label.position.copy(out.normalize().multiplyScalar(1.6 * r.R));
                body.add(label);
            }
            let arc = null;
            if (arcs) {
                // in the rotor plane, starting towards the nose, the same way round for every rotor: clockwise
                // seen from where the axis points (the y mirror turns the geometry's counter-clockwise around)
                const x = new THREE.Vector3(1, 0, 0).projectOnPlane(axis);
                if (x.lengthSq() < 1e-9) {
                    x.copy(e1);
                }
                x.normalize();
                const disc = new THREE.Group();
                disc.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, axis.clone().cross(x), axis));
                disc.scale.set(r.R, -r.R, r.R);
                const mesh = (c) => {
                    const m = new THREE.Mesh(
                        this._arcGeometry("pie", 0),
                        new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: ARC_OPACITY, side: THREE.DoubleSide, depthWrite: false }),
                    );
                    m.visible = false;
                    disc.add(m);
                    return m;
                };
                arc = { state: mesh(arcs[0]), command: mesh(arcs[1]) };
                body.add(disc);
            }
            root.add(pivot);
            rot.push({
                pivot, arc, label,
                a1: new THREE.Vector3(...(r.tilt_axis || [1, 0, 0])).normalize(),
                a2: new THREE.Vector3(...(r.tilt2_axis || [0, 1, 0])).normalize(),
            });
        });

        const surf = [];
        for (const s of geo.surfaces || []) {
            const pivot = new THREE.Group();
            pivot.position.set(...s.tilt_xyz);
            for (const pl of s.geometry) {
                pivot.add(this._line(pl.map((p) => [p[0] - s.tilt_xyz[0], p[1] - s.tilt_xyz[1], p[2] - s.tilt_xyz[2]]), mat));
            }
            root.add(pivot);
            surf.push({ pivot, axis: new THREE.Vector3(...s.tilt_axis).normalize() });
        }
        return { root, rotors: rot, surfaces: surf };
    }

    /** text sprite, height h in craft units */
    _label(text, color, h) {
        const c = document.createElement("canvas");
        c.width = 128;
        c.height = 64;
        const g = c.getContext("2d");
        g.font = "bold 44px system-ui, sans-serif";
        g.textAlign = "center";
        g.textBaseline = "middle";
        g.fillStyle = color;
        g.fillText(text, 64, 34);
        const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false, transparent: true }));
        s.scale.set(2 * h, h, 1);
        s.renderOrder = 11;
        s.userData.rotorLabel = true;
        return s;
    }

    _sprite(color, kind, px) {
        const c = document.createElement("canvas");
        c.width = c.height = 64;
        const g = c.getContext("2d");
        g.strokeStyle = g.fillStyle = color;
        g.lineWidth = 7;
        g.beginPath();
        g.arc(32, 32, kind === "dot" ? 26 : 27, 0, 2 * Math.PI);
        if (kind === "dot") {
            g.fill();
        } else {
            g.stroke();
        }
        const mat = new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), sizeAttenuation: false, depthTest: false, transparent: true });
        const s = new THREE.Sprite(mat);
        s.renderOrder = 10;
        s.userData.px = px; // diameter in pixels
        return s;
    }

    _vector(color, width, dashed = false) {
        // unit line along +x, oriented and scaled per frame
        const l = this._line([[0, 0, 0], [1, 0, 0]], this._mat(color, width, dashed));
        l.userData.enabled = true;
        return l;
    }

    // -------------------------------------------------- update
    _sampler(t) {
        const ds = this.ds;
        const i = ds.indexAt(t);
        let f = 0;
        if (this.linear && i < ds.n - 1) {
            const dt = ds.t[i + 1] - ds.t[i];
            f = dt > 0 ? Math.min(1, Math.max(0, (t - ds.t[i]) / dt)) : 0;
        }
        return (r) => {
            if (!r) {
                return 0;
            }
            const v = f > 0 ? r.a[i] + f * (r.a[i + 1] - r.a[i]) : r.a[i];
            return v * r.k;
        };
    }

    setTime(t) {
        this.time = t;
        const val = this._sampler(t);
        const d = this.d;
        const v3 = (v, out = new THREE.Vector3()) => out.set(val(v[0]), val(v[1]), val(v[2]));
        const q4 = (q, out) => {
            out.set(val(q[1]), val(q[2]), val(q[3]), val(q[0]));
            return out.length() > 1e-6 ? out.normalize() : out.identity();
        };
        const pos = d.pos ? v3(d.pos) : new THREE.Vector3();
        const posMeas = d.posMeas ? v3(d.posMeas) : pos;

        const setArc = (m, kind, c, on = true) => {
            const level = c?.[0] ? Math.round(Math.min(1, Math.max(0, val(c[0]))) * ARC_LEVELS) : 0;
            m.visible = on && level > 0;
            if (m.visible) {
                m.geometry = this._arcGeometry(kind, level);
            }
        };
        const setRotors = (g, src) => {
            if (!src) {
                return;
            }
            g.rotors.forEach((r, k) => {
                const c = src[k];
                if (!c) {
                    return;
                }
                if (c[1] || c[2]) {
                    const q1 = new THREE.Quaternion().setFromAxisAngle(r.a1, val(c[1]));
                    const q2 = new THREE.Quaternion().setFromAxisAngle(r.a2, val(c[2]));
                    r.pivot.quaternion.copy(q2.multiply(q1));
                }
            });
        };
        const setSurfaces = (g, src) => g.surfaces.forEach((s, k) => s.pivot.quaternion.setFromAxisAngle(s.axis, src?.[k] ? val(src[k]) : 0));

        const G = this.ghosts;
        if (G.est) {
            G.est.root.position.copy(pos);
            q4(d.att, G.est.root.quaternion);
            setRotors(G.est, d.rotor);
            G.est.rotors.forEach((r, k) => {
                setArc(r.arc.state, "pie", d.rotor?.[k]);
                setArc(r.arc.command, "ring", d.rotorSet?.[k], !this.commandArcsHidden);
            });
        }
        if (this.inset) {
            const g = this.inset.craft;
            setRotors(g, d.rotor);
            setSurfaces(g, d.surface);
            g.rotors.forEach((r, k) => {
                setArc(r.arc.state, "pie", d.rotor?.[k]);
                setArc(r.arc.command, "ring", d.rotorSet?.[k]);
            });
            setSurfaces(G.est, d.surface);
        }
        if (G.set) {
            G.set.root.position.copy(pos);
            q4(d.attSet, G.set.root.quaternion);
            setRotors(G.set, d.rotorSet);
            setSurfaces(G.set, d.surfaceSet);
        }
        if (G.meas) {
            G.meas.root.position.copy(posMeas);
            q4(d.attMeas, G.meas.root.quaternion);
        }

        const M = this.markers;
        for (const k of ["pos", "posSet", "posMeas"]) {
            if (M[k]) {
                v3(d[k], M[k].position);
            }
        }

        const V = this.vectors;
        const unitX = new THREE.Vector3(1, 0, 0);
        const setVec = (line, origin, src, k) => {
            if (!line || !line.userData.enabled) {
                if (line) {
                    line.visible = false;
                }
                return;
            }
            const dir = v3(src).multiplyScalar(k);
            const len = dir.length();
            line.visible = len > 1e-4;
            if (!line.visible) {
                return;
            }
            line.position.copy(origin);
            line.quaternion.setFromUnitVectors(unitX, dir.divideScalar(len));
            line.scale.setScalar(len);
        };
        // vectors as in pyFlightPlotter.Viewport: 0.2 s of velocity, 0.1 s² of acceleration
        setVec(V.vel, pos, d.vel, 0.2);
        setVec(V.velSet, pos, d.velSet, 0.2);
        setVec(V.velMeas, posMeas, d.velMeas, 0.2);
        setVec(V.acc, pos, d.acc, 0.1);
        setVec(V.accSet, pos, d.accSet, 0.1);
        setVec(V.accMeas, posMeas, d.accMeas, 0.1);

        const i = this.ds.indexAt(t);
        const i0 = this.ds.indexAt(this.ds.t[i] - TRAIL_S);
        for (const tr of this.trails) {
            tr.trail.geometry.setDrawRange(i0, i - i0 + 1);
        }

        if (this.follow) {
            const delta = pos.clone().sub(this.controls.target);
            this.controls.target.add(delta);
            this.camera.position.add(delta);
            this.controls.update();
        }
        this.requestRender();
    }

    setFollow(on) {
        this.follow = on;
        for (const g of Object.values(this.ghosts)) {
            if (g) {
                g.root.scale.setScalar(on ? 1 : this.craftScale);
            }
        }
        if (on) {
            const val = this._sampler(this.time);
            const p = this.d.pos ? new THREE.Vector3(...this.d.pos.map(val)) : new THREE.Vector3();
            const off = this.camera.position.clone().sub(this.controls.target).setLength(2.0);
            this.controls.target.copy(p);
            this.camera.position.copy(p).add(off);
            this.controls.update();
        } else {
            this.resetView();
        }
        this.setTime(this.time);
    }

    /** show/hide "est" | "set" | "meas" (craft, marker, vectors) or "trail" */
    setVisible(key, on) {
        const g = this.ghosts[key];
        if (g) {
            g.root.visible = on;
        }
        if (key === "set") {
            // the command arcs belong to the setpoint, though drawn on the estimate
            this.commandArcsHidden = !on;
        }
        const extra = { est: ["pos", "vel", "acc"], set: ["posSet", "velSet", "accSet"], meas: ["posMeas", "velMeas", "accMeas"] }[key] || [];
        for (const k of extra) {
            if (this.markers[k]) {
                this.markers[k].visible = on;
            }
            if (this.vectors[k]) {
                this.vectors[k].userData.enabled = on;
            }
        }
        if (key === "trail") {
            for (const tr of this.trails) {
                tr.full.visible = tr.trail.visible = on;
            }
        }
        if (key === "labels") {
            for (const g of [this.ghosts.est, this.inset?.craft]) {
                g?.root.traverse((o) => {
                    if (o.userData.rotorLabel) {
                        o.visible = on;
                    }
                });
            }
        }
        this.setTime(this.time);
    }

    has(key) {
        if (key === "trail") {
            return this.trails.length > 0;
        }
        if (key === "labels") {
            return !!this.ghosts.est && (this.geometry.rotors || []).length > 0;
        }
        const marker = { est: "pos", set: "posSet", meas: "posMeas" }[key];
        return !!(this.ghosts[key] || this.markers[marker]);
    }

    resetView() {
        const { center, extent } = this.home;
        // fit the flight into the narrower of the two fields of view
        const tanV = Math.tan((this.camera.fov * DEG) / 2);
        const tanFit = Math.min(tanV, tanV * this.camera.aspect);
        const dist = Math.max(2.5, (0.6 * extent) / tanFit);
        this.controls.target.copy(center);
        this.camera.position.copy(center).add(new THREE.Vector3(-1, 0.8, -0.55).setLength(dist));
        this.controls.update();
        this.requestRender();
    }

    resize() {
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        if (w < 10 || h < 10) {
            return;
        }
        this.renderer.setSize(w, h, false);
        this.renderer.domElement.style.width = `${w}px`;
        this.renderer.domElement.style.height = `${h}px`;
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        for (const m of this.materials) {
            m.resolution.set(w, h);
        }
        if (!this._homed) {
            // first real size: frame the flight for this aspect ratio
            this._homed = true;
            if (!this.follow) {
                this.resetView();
            }
        }
        for (const s of Object.values(this.markers)) {
            if (s) {
                s.scale.setScalar((s.userData.px / h) * 2 * Math.tan((this.camera.fov * DEG) / 2));
            }
        }
        this.requestRender();
    }

    requestRender() {
        if (this._raf) {
            return;
        }
        this._raf = requestAnimationFrame(() => {
            this._raf = 0;
            const now = performance.now();
            const dt = (now - this._lastFrame) / 1000;
            this._lastFrame = now;
            if (this.viewHelper.animating) {
                this.viewHelper.update(dt);
                this.requestRender();
            }
            this.renderer.clear();
            this.renderer.render(this.scene, this.camera);
            this.viewHelper.render(this.renderer);
            if (this.inset) {
                this._renderInset();
            }
        });
    }
}
