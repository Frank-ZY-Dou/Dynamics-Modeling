"""G5 with MuJoCo: settle a scene and measure it, instead of trusting the rest state.

Every body becomes a MuJoCo mesh geom (MuJoCo convex-hulls mesh geoms; pass
`pieces` to supply convex pieces per body, e.g. from CoACD, for a PhysX-like
proxy). Fixed bodies are static geoms in the world body. Reports peak body
speed, peak displacement, peak tilt and the final displacement per free body,
together with the number of integration steps actually taken and which bodies
fell back from CoACD to a single hull.
"""
from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass, field

import numpy as np

from ..scene.model import Scene


def _write_obj(path, verts, faces):
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in faces:
            f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")


def _quat_wxyz(R):
    from scipy.spatial.transform import Rotation as Rot
    q = Rot.from_matrix(R).as_quat()   # xyzw
    return (q[3], q[0], q[1], q[2])


@dataclass
class SettleReport:
    peak_speed: float
    peak_disp: float
    final_disp: dict = field(default_factory=dict)
    per_body_peak_speed: dict = field(default_factory=dict)
    steps: int = 0                                      # integration steps actually taken
    dt: float = 0.0
    left_support: list = field(default_factory=list)   # free bodies that ended below / outside their support slab
    trajectory: list = field(default_factory=list)     # optional: (time, {body: (xpos, xquat_wxyz)}) every record_every steps
    sim_seconds: float = 0.0                            # steps * dt
    peak_tilt_deg: float = 0.0                          # largest rotation of any body's up axis from its start
    per_body_peak_tilt_deg: dict = field(default_factory=dict)
    proxy_fallback: list = field(default_factory=list)  # bodies whose CoACD decomposition failed and used one hull
    engine_warnings: dict = field(default_factory=dict) # MuJoCo warning counters that fired (the engine reset the state), non-finite readings

    def summary(self):
        worst = max(self.final_disp.items(), key=lambda kv: kv[1])[0] if self.final_disp else "-"
        return (f"peak_speed={self.peak_speed:.3f} m/s  peak_disp={self.peak_disp:.4f} m  peak_tilt={self.peak_tilt_deg:.1f} deg  "
                f"final_disp max={max(self.final_disp.values(), default=0):.4f} m ({worst})  steps={self.steps}"
                + (f"  left_support={self.left_support}" if self.left_support else "")
                + (f"  hull_fallback={self.proxy_fallback}" if self.proxy_fallback else ""))


_COACD_CACHE: dict = {}


def coacd_pieces_info(b, threshold: float = 0.05, max_hulls: int = 32):
    """(convex pieces of a body's mesh in the model frame via CoACD, used_fallback), cached on
    the mesh content. When CoACD fails the single convex hull stands in and used_fallback is True."""
    import coacd, trimesh, hashlib
    key = (hashlib.sha1(np.ascontiguousarray(b.verts).tobytes() + np.ascontiguousarray(b.faces).tobytes()).hexdigest(), threshold, max_hulls)
    if key in _COACD_CACHE:
        return _COACD_CACHE[key]
    m = trimesh.Trimesh(b.verts, b.faces, process=False)
    try:
        parts = coacd.run_coacd(coacd.Mesh(np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)),
                                threshold=threshold, max_convex_hull=max_hulls, preprocess_mode="auto")
        out = ([(np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int32)) for v, f in parts], False)
    except Exception:  # noqa: BLE001
        h = m.convex_hull; out = ([(np.asarray(h.vertices), np.asarray(h.faces))], True)
    _COACD_CACHE[key] = out
    return out


def coacd_pieces(b, threshold: float = 0.05, max_hulls: int = 32):
    """Convex pieces of a body's mesh (model frame) via CoACD; see coacd_pieces_info."""
    return coacd_pieces_info(b, threshold, max_hulls)[0]


def _tilt_deg(xquat_wxyz: np.ndarray, z0: np.ndarray) -> np.ndarray:
    """Angle (degrees) between each body's current up axis and its initial one."""
    from scipy.spatial.transform import Rotation as Rot
    q = np.asarray(xquat_wxyz, dtype=np.float64)
    z = Rot.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()[:, :, 2]
    c = np.clip(np.einsum("ij,ij->i", z, z0), -1.0, 1.0)
    return np.degrees(np.arccos(c))


def settle_and_measure(scene: Scene, seconds: float = 2.0, timestep: float = 2e-3, gravity: bool = True,
                       pieces: dict | None = None, max_faces: int = 4000, support_tops: dict | None = None,
                       decompose_free: bool = False, max_hull_verts: int = 256, verbose: bool = False,
                       record_every: int = 0) -> SettleReport:
    """support_tops: support name -> plate height; such fixed bodies become a 5 cm slab
    whose top is that height (the convex hull of a table with legs and a rim is not its plate).
    decompose_free: CoACD convex pieces for free bodies instead of one convex hull.
    Mass properties come from the collision proxy itself (uniform density, hull inertia): an explicit
    inertial at the AABB-centred body origin sits above the centroid of round objects (a stem, a handle)
    and makes them top-heavy, so they roll off the table. A floor plane below the lowest support bounds
    a fall; bodies that end outside/below their support slab are listed in `left_support`.
    max_hull_verts caps every collision hull like PhysX does (256): a 4000-face hull of a round object
    has facets so small that contact-normal noise makes it roll."""
    for key, value in (("seconds", seconds), ("timestep", timestep)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{key} must be a positive finite number, got {value!r}")
    if seconds < timestep:
        raise ValueError(f"seconds={seconds} is shorter than one integration step ({timestep}); nothing would be simulated")
    n_steps = int(round(seconds / timestep))
    if n_steps > 10_000_000:
        raise ValueError(f"{n_steps} integration steps requested; shorten `seconds` or raise `timestep`")
    if not scene.bodies:
        raise ValueError("cannot settle an empty scene")
    import mujoco
    import trimesh
    from ..scene.model import slab_proxy
    fallback = []
    with tempfile.TemporaryDirectory(prefix="simready_mj_") as tmp:
        assets, bodies_xml = [], []
        for k, b in enumerate(scene.bodies):
            geoms = []
            parts = pieces.get(b.name) if pieces else None
            if parts is None and b.fixed and support_tops and b.name in support_tops:
                px = slab_proxy(b, top=support_tops[b.name])
                # express the slab in the body's frame: verts_local = R^T (world - center)
                wv = px.world_vertices()
                parts = [((b.rotation.T @ (wv - b.center).T).T, px.faces)]
            if parts is None and b.fixed and ("ground" in b.tags or np.linalg.matrix_rank(b.verts - b.verts.mean(0), tol=1e-6) < 3):
                continue        # a ground plane or a flat sheet: no convex hull (the floor plane below the supports stands in)
            if parts is None and decompose_free and not b.fixed:
                parts, used_hull = coacd_pieces_info(b)
                if used_hull:
                    fallback.append(b.name)
            if parts is None:
                v, f = b.verts, b.faces
                if len(f) > max_faces:
                    try:
                        m = trimesh.Trimesh(v, f, process=False).simplify_quadric_decimation(face_count=max_faces)
                        v, f = np.asarray(m.vertices), np.asarray(m.faces)
                    except Exception:  # noqa: BLE001
                        pass
                parts = [(v, f)]
            for j, (v, f) in enumerate(parts):
                name = f"m{k}_{j}"
                path = os.path.join(tmp, name + ".obj")
                _write_obj(path, v, f)
                assets.append(f'<mesh name="{name}" file="{path}" inertia="convex" maxhullvert="{max_hull_verts}"/>')
                geoms.append(f'<geom type="mesh" mesh="{name}" condim="3" friction="1 0.005 0.0001" density="500"/>')
            w, x, y, z = _quat_wxyz(b.rotation)
            pos = f'{b.center[0]:.6f} {b.center[1]:.6f} {b.center[2]:.6f}'
            if b.fixed:
                bodies_xml.append(f'<body name="{b.name}" pos="{pos}" quat="{w:.6f} {x:.6f} {y:.6f} {z:.6f}">{"".join(geoms)}</body>')
            else:
                bodies_xml.append(f'<body name="{b.name}" pos="{pos}" quat="{w:.6f} {x:.6f} {y:.6f} {z:.6f}"><freejoint/>'
                                  f'{"".join(geoms)}</body>')
        g = "0 0 -9.81" if gravity else "0 0 0"
        fixed = [b for b in scene.bodies if b.fixed]
        z_floor = (min(float(b.world_aabb()[0][2]) for b in fixed) - 0.005) if fixed else \
                  (min(float(b.world_aabb()[0][2]) for b in scene.bodies) - 1.0)
        bodies_xml.append(f'<geom name="floor" type="plane" size="0 0 1" pos="0 0 {z_floor:.6f}" condim="3" friction="1 0.005 0.0001"/>')
        xml = f"""<mujoco><compiler angle="radian" boundmass="0.001" boundinertia="1e-7"/><option timestep="{timestep}" gravity="{g}"/>
<asset>{''.join(assets)}</asset><worldbody>{''.join(bodies_xml)}</worldbody></mujoco>"""
        model = mujoco.MjModel.from_xml_string(xml)      # meshes are read here; the files can go afterwards
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    free = [b.name for b in scene.free()]
    bid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in free}
    ids = np.array([bid[n] for n in free], dtype=int)
    p0 = {n: data.xpos[bid[n]].copy() for n in free}
    z0 = np.zeros((len(free), 3))
    if free:
        from scipy.spatial.transform import Rotation as Rot
        q0 = np.asarray(data.xquat[ids], dtype=np.float64)
        z0 = Rot.from_quat(q0[:, [1, 2, 3, 0]]).as_matrix()[:, :, 2]
    peak_v, peak_d = 0.0, 0.0
    per_v = {n: 0.0 for n in free}
    per_tilt = np.zeros(len(free))
    traj = []
    if record_every:
        traj.append((0.0, {n: (data.xpos[bid[n]].copy(), data.xquat[bid[n]].copy()) for n in free}))
    done = 0
    nonfinite = 0
    for step in range(n_steps):
        mujoco.mj_step(model, data)
        done += 1
        if record_every and (step + 1) % record_every == 0:
            traj.append((float(data.time), {n: (data.xpos[bid[n]].copy(), data.xquat[bid[n]].copy()) for n in free}))
        for n in free:
            v = float(np.linalg.norm(data.cvel[bid[n]][3:]))   # linear part of the body velocity
            d = float(np.linalg.norm(data.xpos[bid[n]] - p0[n]))
            if not (math.isfinite(v) and math.isfinite(d)):
                nonfinite += 1
                continue
            per_v[n] = max(per_v[n], v); peak_v = max(peak_v, v); peak_d = max(peak_d, d)
        if free:
            tilt_now = _tilt_deg(data.xquat[ids], z0)
            per_tilt = np.maximum(per_tilt, np.where(np.isfinite(tilt_now), tilt_now, per_tilt))
    # MuJoCo answers a bad state (acceleration, position or velocity out of range) by resetting the
    # simulation and counting a warning; such a run is not a settle and is reported as one that failed
    warnings = {mujoco.mjtWarning(i).name: int(data.warning[i].number) for i in range(len(data.warning)) if data.warning[i].number}
    if nonfinite:
        warnings["nonfinite_readings"] = nonfinite
    final = {n: float(np.linalg.norm(data.xpos[bid[n]] - p0[n])) for n in free}
    left = []
    if support_tops:
        rects = {}
        for b in fixed:
            if b.name in support_tops:
                lo, hi = b.world_aabb(); rects[b.name] = (lo[:2], hi[:2], support_tops[b.name])
        for n in free:
            x = data.xpos[bid[n]]
            on_any = any((lo[0] - 1e-3 <= x[0] <= hi[0] + 1e-3) and (lo[1] - 1e-3 <= x[1] <= hi[1] + 1e-3) and x[2] > top - 0.02
                         for lo, hi, top in rects.values())
            if not on_any:
                left.append(n)
    else:
        left = [n for n in free if data.xpos[bid[n]][2] < p0[n][2] - 0.3]
    tilt = {n: float(t) for n, t in zip(free, per_tilt)}
    rep = SettleReport(peak_v, peak_d, final, per_v, done, timestep, left, traj, done * timestep,
                       float(per_tilt.max()) if free else 0.0, tilt, fallback, warnings)
    if verbose:
        print("[settle]", rep.summary())
    return rep
