"""S4R-QP: progressive scaling with QP displacement correction (3- or 6-DOF).

Every scale step detects the contacts that the next inflation would
activate, solves one QP for the minimum-norm translation (optionally with
rotation) that keeps each active pair at least d_hat apart after the
inflation, applies it and inflates. No CCD, no Newton: one QP per step.

Result contract of ``solve_s4r_qp`` (all keys documented at the return
statement): the returned poses are scored at FULL scale; ``status``,
``continuation_complete`` and ``native_converged`` state whether the
continuation reached full scale and whether the tail refinement found a
penetration-free state on its own. A run that stops early (step budget,
QP failure, infeasible container) is reported as such and never as a
success.
"""
import os
import sys
import time

import numpy as np
import trimesh
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import osqp
from scipy.spatial.transform import Rotation as RotLib

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from mesh_collision import _contains_points as _mesh_contains_points  # noqa: E402
from mesh_collision import _vertex_components, _nested_component_contact, _piece_layout, _piece_boxes, _any_piece_nested  # noqa: E402
from mesh_collision import _signed_volume  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# OSQP helpers
# ─────────────────────────────────────────────────────────────────────

def qp_status_ok(result) -> bool:
    """True when OSQP returned a usable primal solution.

    OSQP reports ``'solved'`` and ``'solved inaccurate'`` (with a space; the
    enum is OSQP_SOLVED_INACCURATE). Everything else (maximum iterations,
    primal/dual infeasible, non-convex, interrupted) has no solution the
    caller may apply.
    """
    status = str(getattr(result.info, 'status', ''))
    status = status.strip().lower().replace('_', ' ')
    return status in ('solved', 'solved inaccurate')


def osqp_solve(P, q, A, l, u, x0=None, **settings):
    """Set up and solve one OSQP problem; returns the OSQP result object.

    Accepts the 1.x setting names (``polishing``, ``warm_starting``) and
    falls back to the 0.6 names (``polish``, ``warm_start``) when the
    installed OSQP rejects them, so either version in requirements.txt works.
    """
    solver = osqp.OSQP()
    try:
        solver.setup(P, q, A, l, u, **settings)
    except Exception:
        legacy = {'polishing': 'polish', 'warm_starting': 'warm_start'}
        solver = osqp.OSQP()
        solver.setup(P, q, A, l, u,
                     **{legacy.get(k, k): v for k, v in settings.items()})
    if x0 is not None:
        solver.warm_start(x=np.asarray(x0, dtype=np.float64))
    try:
        return solver.solve(raise_error=False)
    except TypeError:
        return solver.solve()


_OSQP_STEP = dict(verbose=False, eps_abs=1e-6, eps_rel=1e-6,
                  max_iter=4000, polishing=True, warm_starting=True)
_OSQP_TAIL = dict(verbose=False, eps_abs=1e-7, eps_rel=1e-7,
                  max_iter=8000, polishing=True)


# ─────────────────────────────────────────────────────────────────────
# Starting-scale admissibility (coincident or near-coincident centroids)
# ─────────────────────────────────────────────────────────────────────

def _spread_direction(k: int) -> np.ndarray:
    """Deterministic unit vector for the k-th exactly coincident pair.

    Consecutive k give well-separated directions (golden-angle spiral on
    the sphere), so three or more bodies sharing one centroid fan out
    instead of being stacked along a single axis.
    """
    if k == 0:
        return np.array([1.0, 0.0, 0.0])
    z = 1.0 - 2.0 * ((k * 0.6180339887498949) % 1.0)
    r = np.sqrt(max(0.0, 1.0 - z * z))
    a = 2.399963229728653 * k
    v = np.array([r * np.cos(a), r * np.sin(a), z])
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])


def _admissibility_violations(C, target_fn, block=1024):
    """Pairs (i, j, d_ij) with ||c_i - c_j|| < target(i, j), lexicographic,
    computed block-wise so the memory stays O(block * N)."""
    N = len(C)
    out = []
    for a in range(0, N, block):
        b = min(N, a + block)
        d = np.linalg.norm(C[a:b, None, :] - C[None, :, :], axis=2)  # (b-a, N)
        tgt = target_fn(np.arange(a, b)[:, None], np.arange(N)[None, :])
        rows, cols = np.nonzero(d < tgt)
        for r, c in zip(rows.tolist(), cols.tolist()):
            i = a + r
            if c > i:
                out.append((i, c, float(d[r, c])))
    return out


def separate_coincident_centroids(centers, radii, d_hat, s_min,
                                  eps=1e-6, max_passes=100, verbose=False):
    """Move centroid pairs apart until the starting scale is admissible.

    The continuation may only start at a scale s_min at which no two bodies
    are already within d_hat of each other, i.e. for every pair

        ||c_i - c_j|| >= d_hat + s_min (r_i + r_j) + eps,

    with r the bounding-sphere radius about the scaling centre. Violating
    pairs (typically exactly coincident centroids) are split symmetrically
    along their centre line, or along a deterministic per-pair direction
    when the centroids coincide. Passes repeat until no pair is violated,
    because splitting one pair can shorten another one that was already
    fine; a single pass does not establish the condition.

    Modifies ``centers`` in place. Returns (passes_used, remaining_violations).
    """
    C = np.asarray(centers, dtype=np.float64)
    R = np.asarray(radii, dtype=np.float64)
    N = len(C)
    if N < 2:
        return 0, 0

    def target(ii, jj):
        return d_hat + s_min * (R[ii] + R[jj]) + eps

    # Bodies that share a centroid exactly first receive a distinct
    # infinitesimal offset each (one direction per body, spread over the
    # sphere), so the pair pushes below act in three dimensions. Without
    # this, the first split defines the only axis of a coincident cluster
    # and every later push stays on that line, which converges slowly.
    tie = _admissibility_violations(C, lambda ii, jj: np.full(np.broadcast(ii, jj).shape, 1e-12))
    if tie:
        tied = sorted({b for (i, j, _) in tie for b in (i, j)})
        seed = 1e-6 * float(d_hat + s_min * R.max() + eps)
        for b in tied:
            C[b] = C[b] + seed * _spread_direction(b)

    k_coincident = 0
    for p in range(max_passes):
        viol = _admissibility_violations(C, target)
        if not viol:
            return p, 0
        for (i, j, d) in viol:
            axis = C[j] - C[i]
            n = float(np.linalg.norm(axis))
            if n < 1e-12:
                axis = _spread_direction(k_coincident)
                k_coincident += 1
            else:
                axis = axis / n
            # Overshoot the target by a hair so rounding cannot leave the
            # pair a few ulp below it and flag it again next pass.
            gap = float(target(i, j)) - n + 1e-9
            if gap <= 0.0:
                continue
            C[j] = C[j] + 0.5 * gap * axis
            C[i] = C[i] - 0.5 * gap * axis
            if verbose:
                print(f"    jitter pair ({i},{j}) by ±{0.5 * gap:.5f}")
    return max_passes, len(_admissibility_violations(C, target))


def _one_sided_support(model_verts, rot, direction) -> float:
    """max_v n^T R v over the (scale-1) model vertices, clamped at zero:
    how far the body reaches from its scaling centre along ``direction``.
    A further conservative bound; the closure coefficient only needs the
    support to dominate the true reach."""
    projs = model_verts @ (rot.T @ direction)
    return float(max(projs.max(), 0.0))


def _projected_width(model_verts, rot, direction) -> float:
    projs = model_verts @ (rot.T @ direction)
    return float(projs.max() - projs.min())


# ─────────────────────────────────────────────────────────────────────
# Exact FCL contact oracle on unit-scale BVHs (the benchmark backend)
# ─────────────────────────────────────────────────────────────────────

class PrebuiltFCLOracle:
    """FCL contact oracle that builds one unit-scale BVH per body and reuses
    it at every scale.

    Change of variable u = x / s: at scale s a body with model vertices
    M_i = nf_i * v_i and centre c_i occupies s * R_i M_i + c_i in world
    space, which is R_i M_i + c_i / s in u-space. One BVH built on M_i
    therefore serves every scale; only the FCL transform (R_i, c_i / s)
    changes. u-space distances and points scale by s back to world units;
    directions are scale-invariant.

    Per candidate pair the query is exact for closed triangle meshes:

    * separated: FCL ``distance`` with nearest points; the witness direction
      is the segment between the two nearest points.
    * surfaces crossing: FCL ``collide``; the deepest triangle pair's normal
      is the outward normal of the crossed triangle, i.e. the local exit
      direction of the crossing body. That direction is also right inside
      the cavity of a non-convex body, where the centroid line points the
      wrong way, so it is used as reported (the BVHs are built with outward
      winding: a mesh with negative signed volume is flipped once here).
    * one body entirely inside the other: neither FCL query reports it (no
      surface intersection, positive distance), so pairs whose u-space AABBs
      nest are checked with a point-containment test and reported as a
      penetration whose depth is the surface gap plus the inner body's width
      along the exit direction (a depth for the correction rows, not the
      minimal freeing translation).

    Broadphase: pairs are candidates when the Euclidean gap between their
    u-space AABBs is at most (ds (E_i + E_j) + d_hat) / s, with E the
    bounding-sphere radius about the scaling centre. The AABB gap bounds
    the true distance from below and the narrow filter keeps a pair when
    its signed distance is below ds (e_i + e_j) + d_hat with e <= E the
    one-sided support along the contact normal, so no pair the filter would
    keep can be pruned, and the candidate set depends only on the world
    geometry (not on how a mesh splits its size between vertices and
    normalize_factor). A SOI cull removes pairs whose bounding spheres
    cannot touch before scale (s + ds) / 0.9.
    """

    def __init__(self, nfs, mverts, mfaces, d_hat):
        import fcl
        self._fcl = fcl
        self.N = len(nfs)
        self.d_hat = float(d_hat)
        self.nfs = [float(v) for v in nfs]
        self.model_verts = []
        self.faces = []
        self.bvh = []
        self.max_extents = np.zeros(self.N)
        self._model_mesh = [None] * self.N
        self._pieces = [None] * self.N
        self._layout = [None] * self.N
        self._inc = None
        for i in range(self.N):
            Mi = (self.nfs[i] * np.asarray(mverts[i], dtype=np.float64))
            Fi = np.asarray(mfaces[i], dtype=np.int32)
            if len(Fi) and _signed_volume(Mi, Fi) < 0.0:
                Fi = np.ascontiguousarray(Fi[:, ::-1])
            self.model_verts.append(Mi)
            self.faces.append(Fi)
            self.max_extents[i] = float(np.max(np.linalg.norm(Mi, axis=1))) if len(Mi) else 0.0
            m = fcl.BVHModel()
            m.beginModel(len(Mi), len(Fi))
            m.addSubModel(Mi, Fi)
            m.endModel()
            self.bvh.append(m)

    def support(self, i, rot, direction) -> float:
        return _one_sided_support(self.model_verts[i], rot, direction)

    def _model_trimesh(self, i):
        if self._model_mesh[i] is None:
            self._model_mesh[i] = trimesh.Trimesh(
                vertices=self.model_verts[i], faces=self.faces[i], process=False)
        return self._model_mesh[i]

    def _components(self, i):
        """Vertex index arrays of body i's closed pieces (one for a connected mesh)."""
        if self._pieces[i] is None:
            self._pieces[i] = _vertex_components(self.faces[i], len(self.model_verts[i]))
        return self._pieces[i]

    def _piece_layout(self, i):
        if self._layout[i] is None:
            self._layout[i] = _piece_layout(self._components(i))
        return self._layout[i]

    def _contains_model_point(self, i, p_local) -> bool:
        """Ray-parity containment of a point given in body i's model frame."""
        return bool(_mesh_contains_points(self._model_trimesh(i),
                                          np.asarray(p_local, dtype=np.float64)[None, :])[0])

    def find_contacts(self, s, ds, centers, rots, extra_margin=0.0,
                      all_contacts=False, incremental=False):
        """Contacts at scale s that the inflation by ds may activate.

        Returns a list of (i, j, d_signed, n, e_i, e_j, cp_on_i, cp_on_j) in
        world units: d_signed < 0 is a penetration depth, n points so that
        moving j along +n (and i along -n) separates the pair, e are the
        one-sided supports along n at scale 1, cp are the witness points.

        ``incremental=True`` reuses the previous call when it had the same
        s, ds and margin and the rotations are unchanged (the tail
        refinement: translation-only steps at fixed scale). A body's
        distance to any other body changes by at most the length of its
        move, so a pair whose cached distance minus the two moves is still
        above the pair's margin cannot be a contact and is not queried; a
        pair of two unmoved bodies keeps its previous result. The candidate
        set is refreshed only for pairs involving a moved body. The result
        is identical to a full detection.
        """
        fcl = self._fcl
        N = self.N
        inv_s = 1.0 / s
        d_hat = self.d_hat
        C = np.asarray(centers, dtype=np.float64)
        contacts = []
        if N < 2:
            return contacts

        lo = np.empty((N, 3)); hi = np.empty((N, 3))
        piece_boxes = {}      # u-space AABB per closed piece, for bodies made of several pieces
        for i in range(N):
            Vw = (rots[i] @ self.model_verts[i].T).T + C[i] * inv_s
            lo[i] = Vw.min(axis=0); hi[i] = Vw.max(axis=0)
            if len(self._components(i)) > 1:
                piece_boxes[i] = _piece_boxes(Vw, self._piece_layout(i))

        fcl_objs = [fcl.CollisionObject(
            self.bvh[i],
            fcl.Transform(np.asarray(rots[i], dtype=np.float64),
                          (C[i] * inv_s).astype(np.float64))) for i in range(N)]

        E = self.max_extents
        key = (float(s), float(ds), float(extra_margin), bool(all_contacts))
        state = self._incremental_state(key, rots) if incremental else None

        def _pair_margin_world(ei, ej):
            return ds * (ei + ej) + d_hat + extra_margin

        if state is None:
            ii, jj = np.triu_indices(N, k=1)
            ci, cj = self._broadphase_pairs(ii, jj, C, lo, hi, s, ds, extra_margin)
            move = np.zeros(N)
            d_cache = {}
        else:
            C_prev = state['centers']
            move = np.linalg.norm(C - C_prev, axis=1)
            moved = move > 0.0
            ci_prev, cj_prev = state['cand']
            keep = ~moved[ci_prev] & ~moved[cj_prev]
            parts_i = [ci_prev[keep]]
            parts_j = [cj_prev[keep]]
            others = np.arange(N)
            for m in np.nonzero(moved)[0]:
                js = others[others != m]
                a = np.minimum(m, js); b = np.maximum(m, js)
                pi, pj = self._broadphase_pairs(a, b, C, lo, hi, s, ds, extra_margin)
                parts_i.append(pi); parts_j.append(pj)
            ci = np.concatenate(parts_i); cj = np.concatenate(parts_j)
            if len(ci):
                codes = np.unique(ci.astype(np.int64) * N + cj.astype(np.int64))
                ci = (codes // N).astype(np.int64); cj = (codes % N).astype(np.int64)
            d_cache = state['d']

        new_cache = {}
        for k in range(len(ci)):
            i = int(ci[k]); j = int(cj[k])
            if state is not None:
                prev = d_cache.get((i, j))
                if prev is not None:
                    d_old, tup = prev
                    delta = move[i] + move[j]
                    if delta == 0.0:
                        new_cache[(i, j)] = prev
                        if tup is not None:
                            contacts.append(tup)
                        continue
                    bound = d_old - delta
                    if bound >= _pair_margin_world(E[i], E[j]):
                        new_cache[(i, j)] = (bound, None)
                        continue

            req = fcl.DistanceRequest(enable_nearest_points=True,
                                      enable_signed_distance=True)
            res = fcl.DistanceResult()
            d_u = fcl.distance(fcl_objs[i], fcl_objs[j], req, res)

            if d_u > 0.0:
                cp_i_u = np.asarray(res.nearest_points[0], dtype=np.float64)
                cp_j_u = np.asarray(res.nearest_points[1], dtype=np.float64)
                nested = self._nested_contact(i, j, cp_i_u, cp_j_u, d_u, s, lo, hi, C, rots, piece_boxes)
                if nested is not None:
                    d_signed, n_raw, cp_on_a, cp_on_b = nested
                else:
                    n_raw = cp_j_u - cp_i_u
                    d_signed = d_u * s
                    cp_on_a = cp_i_u * s
                    cp_on_b = cp_j_u * s
            else:
                creq = fcl.CollisionRequest(num_max_contacts=16,
                                            enable_contact=True)
                cres = fcl.CollisionResult()
                fcl.collide(fcl_objs[i], fcl_objs[j], creq, cres)
                if not cres.is_collision or not cres.contacts:
                    new_cache[(i, j)] = (0.0, None)
                    continue
                if all_contacts:
                    # Diagnostic ablation: every reported triangle contact
                    # becomes its own constraint row instead of the single
                    # deepest witness.
                    for c_k in cres.contacts:
                        depth_world = float(c_k.penetration_depth) * s
                        n_raw = np.asarray(c_k.normal, dtype=np.float64)
                        nn = np.linalg.norm(n_raw)
                        if nn < 1e-12:
                            continue
                        n_k = n_raw / nn
                        pos_world = np.asarray(c_k.pos, dtype=np.float64) * s
                        contacts.append((i, j, -depth_world, n_k,
                                         self.support(i, rots[i], n_k),
                                         self.support(j, rots[j], -n_k),
                                         pos_world.copy(),
                                         pos_world + n_k * depth_world))
                    continue
                c_best = max(cres.contacts, key=lambda c: c.penetration_depth)
                depth_world = float(c_best.penetration_depth) * s
                n_raw = np.asarray(c_best.normal, dtype=np.float64)
                pos_world = np.asarray(c_best.pos, dtype=np.float64) * s
                cp_on_a = pos_world.copy()
                cp_on_b = pos_world + n_raw * depth_world
                d_signed = -depth_world

            nn = np.linalg.norm(n_raw)
            if nn < 1e-12:
                n_raw = C[j] - C[i]
                nn = np.linalg.norm(n_raw)
            if nn < 1e-12:
                new_cache[(i, j)] = (d_signed, None)
                continue
            n = n_raw / nn
            ext_i = self.support(i, rots[i], n)
            ext_j = self.support(j, rots[j], -n)
            if d_signed < ds * (ext_i + ext_j) + d_hat:
                tup = (i, j, d_signed, n, ext_i, ext_j, cp_on_a, cp_on_b)
                contacts.append(tup)
                new_cache[(i, j)] = (d_signed, tup)
            else:
                new_cache[(i, j)] = (d_signed, None)

        if incremental and not all_contacts:
            self._inc = {'key': key, 'centers': C.copy(),
                         'rots': [np.array(R, dtype=np.float64, copy=True) for R in rots],
                         'cand': (np.asarray(ci, dtype=np.int64), np.asarray(cj, dtype=np.int64)),
                         'd': new_cache}
        else:
            self._inc = None
        return contacts

    def _broadphase_pairs(self, ii, jj, C, lo, hi, s, ds, extra_margin):
        """Candidate pairs among the given (ii, jj) index arrays: SOI cull on
        bounding spheres, then Euclidean u-space AABB gap against the
        conservative margin."""
        E = self.max_extents
        inv_s = 1.0 / s
        d_ij = np.linalg.norm(C[ii] - C[jj], axis=1)
        s_contact = np.maximum(0.0, (d_ij - self.d_hat) / (E[ii] + E[jj] + 1e-12))
        soi_keep = (s + ds) >= (s_contact * 0.9)
        margin_u = (ds * (E[ii] + E[jj]) + self.d_hat + extra_margin) * inv_s
        gap_axis = np.maximum(np.maximum(lo[ii] - hi[jj], lo[jj] - hi[ii]), 0.0)
        near = np.einsum('ij,ij->i', gap_axis, gap_axis) <= margin_u * margin_u
        keep = soi_keep & near
        return np.asarray(ii)[keep], np.asarray(jj)[keep]

    def _incremental_state(self, key, rots):
        st = getattr(self, '_inc', None)
        if st is None or st['key'] != key or len(st['rots']) != self.N:
            return None
        for i in range(self.N):
            if not np.array_equal(st['rots'][i], rots[i]):
                return None
        return st

    def _nested_contact(self, i, j, cp_i_u, cp_j_u, d_u, s, lo, hi, C, rots, piece_boxes):
        """Containment check for a separated-by-FCL pair: the whole body when
        the AABBs nest, or one closed piece of a multi-piece body inside the
        other body (the bodies' AABBs need not nest then).

        Returns (d_signed, n, cp_on_i, cp_on_j) in world units, else None.
        n follows the i->j convention: moving j along +n (i along -n)
        carries the inner body toward the outer surface; |d_signed| is the
        surface gap plus the inner body's (or piece's) width along n.
        """
        inv_s = 1.0 / s
        j_in_i = bool(np.all(lo[j] >= lo[i] - 1e-12) and np.all(hi[j] <= hi[i] + 1e-12))
        i_in_j = bool(np.all(lo[i] >= lo[j] - 1e-12) and np.all(hi[i] <= hi[j] + 1e-12))
        if j_in_i or i_in_j:
            if j_in_i:
                outer, inner, q_u = i, j, cp_j_u
            else:
                outer, inner, q_u = j, i, cp_i_u
            p_local = rots[outer].T @ (q_u - C[outer] * inv_s)
            if self._contains_model_point(outer, p_local):
                n_raw = cp_i_u - cp_j_u
                nn = np.linalg.norm(n_raw)
                if nn < 1e-12:
                    return None
                n = n_raw / nn
                width = s * _projected_width(self.model_verts[inner], rots[inner], n)
                d_signed = -(d_u * s + width)
                return d_signed, n, cp_i_u * s, cp_j_u * s
        # a piece can only be inside the other body when its box nests in that body's box
        if not (_any_piece_nested(piece_boxes.get(j), lo[i], hi[i]) or _any_piece_nested(piece_boxes.get(i), lo[j], hi[j])):
            return None
        comps_i = self._components(i)
        comps_j = self._components(j)
        Vi = (rots[i] @ self.model_verts[i].T).T + C[i] * inv_s
        Vj = (rots[j] @ self.model_verts[j].T).T + C[j] * inv_s

        def contains(k):
            return lambda p: self._contains_model_point(k, rots[k].T @ (p - C[k] * inv_s))

        def closest(k):
            def f(pts):
                p_local = (rots[k].T @ (pts - C[k] * inv_s).T).T
                q, dist, _ = trimesh.proximity.closest_point(self._model_trimesh(k), p_local)
                return (rots[k] @ np.asarray(q, dtype=np.float64).T).T + C[k] * inv_s, dist
            return f

        hit = _nested_component_contact(Vi, comps_i, Vj, comps_j,
                                        contains(i), contains(j), closest(i), closest(j))
        if hit is None:
            return None
        d_u2, n, cpi_u, cpj_u = hit
        if n is None:
            n_raw = cp_i_u - cp_j_u
            nn = np.linalg.norm(n_raw)
            if nn < 1e-12:
                return None
            n = n_raw / nn
        return d_u2 * s, n, cpi_u * s, cpj_u * s


# ─────────────────────────────────────────────────────────────────────
# Per-step constraint assembly shared by the main step, the rotation-locked
# retry and the tail refinement, so every solve path carries the same
# hard constraints (contacts, rotation clamp, container walls, joints).
# ─────────────────────────────────────────────────────────────────────

def _skew(v):
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def _assemble_constraints(active, ds, d_hat, centers, rots, active_bodies,
                          body_map, dof, max_omega, box, joints, s_next,
                          lock_rotation=False):
    """Stack the hard constraints of one solve over the variables
    (Δp_b, [ω_b]) of ``active_bodies``.

    Rows in order:
      contacts   n·(Δp_j − Δp_i) + (y_j×n)·ω_j − (y_i×n)·ω_i ≥ ds(e_i+e_j) + d_hat − d
      rotation   −max_omega ≤ ω ≤ max_omega  (ω = 0 when lock_rotation)
      walls      lo + s_next·sup⁻ ≤ c + Δp ≤ hi − s_next·sup⁺ per axis
      joints     Δp_i − Δp_j − [a_i]×ω_i + [a_j]×ω_j = (p_j − p_i) + (a_j − a_i)

    Returns (A csc, l, u, b_contact, walls_infeasible). ``walls_infeasible``
    is True when some wall bound inverts (l > u): no position keeps that
    body inside the container at s_next, so the QP is genuinely infeasible.
    """
    n_b = len(active_bodies)
    n_vars = dof * n_b
    rows, cols, vals = [], [], []
    l_parts, u_parts = [], []
    r = 0

    n_c = len(active)
    b_c = np.zeros(n_c)
    for ci, (i, j, d_curr, n_ij, ext_i, ext_j, cp_i, cp_j) in enumerate(active):
        ii = body_map[i]; jj = body_map[j]
        for a in range(3):
            rows += [ci, ci]
            cols += [dof * ii + a, dof * jj + a]
            vals += [-float(n_ij[a]), float(n_ij[a])]
        if dof == 6:
            cross_i = np.cross(cp_i - centers[i], n_ij)
            cross_j = np.cross(cp_j - centers[j], n_ij)
            for a in range(3):
                rows += [ci, ci]
                cols += [dof * ii + 3 + a, dof * jj + 3 + a]
                vals += [-float(cross_i[a]), float(cross_j[a])]
        b_c[ci] = ds * (ext_i + ext_j) + d_hat - d_curr
    l_parts.append(b_c); u_parts.append(np.full(n_c, np.inf))
    r = n_c

    if dof == 6:
        bound = 0.0 if lock_rotation else float(max_omega)
        for idx in range(n_b):
            for a in range(3):
                rows.append(r); cols.append(dof * idx + 3 + a); vals.append(1.0)
                r += 1
        l_parts.append(np.full(3 * n_b, -bound)); u_parts.append(np.full(3 * n_b, bound))

    walls_infeasible = False
    if box is not None:
        box_lo, box_hi, sup_plus, sup_minus = box
        lw = np.empty(3 * n_b); uw = np.empty(3 * n_b)
        for idx, b_orig in enumerate(active_bodies):
            for a in range(3):
                rows.append(r); cols.append(dof * idx + a); vals.append(1.0)
                r += 1
                c_a = centers[b_orig][a]
                uw[3 * idx + a] = box_hi[a] - s_next * sup_plus[b_orig][a] - c_a
                lw[3 * idx + a] = box_lo[a] + s_next * sup_minus[b_orig][a] - c_a
        if np.any(lw > uw + 1e-9):
            walls_infeasible = True
        l_parts.append(lw); u_parts.append(uw)

    if joints:
        for (bi, bj, a_i, a_j) in joints:
            if bi not in body_map or bj not in body_map:
                continue
            ii = body_map[bi]; jj = body_map[bj]
            a_iw = rots[bi] @ np.asarray(a_i, dtype=np.float64)
            a_jw = rots[bj] @ np.asarray(a_j, dtype=np.float64)
            rhs = (centers[bj] - centers[bi]) + (a_jw - a_iw)
            Si = _skew(a_iw); Sj = _skew(a_jw)
            for a in range(3):
                rows += [r, r]
                cols += [dof * ii + a, dof * jj + a]
                vals += [1.0, -1.0]
                if dof == 6:
                    for c in range(3):
                        if Si[a, c] != 0.0:
                            rows.append(r); cols.append(dof * ii + 3 + c); vals.append(-float(Si[a, c]))
                        if Sj[a, c] != 0.0:
                            rows.append(r); cols.append(dof * jj + 3 + c); vals.append(float(Sj[a, c]))
                r += 1
            l_parts.append(rhs); u_parts.append(rhs)

    A = sp.csc_matrix((np.asarray(vals, dtype=np.float64),
                       (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
                      shape=(r, n_vars))
    l = np.concatenate(l_parts) if l_parts else np.zeros(0)
    u = np.concatenate(u_parts) if u_parts else np.zeros(0)
    return A, l, u, b_c, walls_infeasible


# ─────────────────────────────────────────────────────────────────────
# Main solver
# ─────────────────────────────────────────────────────────────────────

_BACKENDS = ('trimesh', 'fcl', 'fcl_prebuilt', 'warp')


def solve_s4r_qp(objects, d_hat=0.02, ds_max=0.05, max_steps=200, verbose=True,
                  enable_rotation=False, rotation_weight=None, max_omega=0.1,
                  adaptive_ds=False, contact_sparsity=False, use_dual=False,
                  revalidate_interval=3, audit=False,
                  contact_backend='fcl',
                  trajectory_dumper=None, dump_every=1,
                  target_centers=None, attraction_alpha=None,
                  perturb_rot_deg=0.0, perturb_seed=None,
                  box_bounds=None, joints=None,
                  profile=False):
    """S4R with QP-based displacement correction (3-DOF, or 6-DOF with
    ``enable_rotation``).

    At each scale step:
    1. find the pairs that would come within d_hat after inflating by ds;
    2. solve  min ||Δp||² (+ β||ω||²)  s.t. the linearised contact rows and
       every other hard constraint of the scene (container walls, joints);
    3. apply the displacement, inflate.

    Hard constraints are assembled by one builder for every solve path
    (main step, rotation-locked retry, tail refinement), so no path drops
    them. ``use_dual`` solves the contact-only translation QP in contact
    force space and therefore rejects rotation, walls, joints and attraction.

    Contact backends: ``'fcl'`` and ``'fcl_prebuilt'`` are the same exact
    oracle (:class:`PrebuiltFCLOracle`, unit-scale BVHs built once);
    ``'warp'`` is the GPU oracle in ``s4r_gpu``; ``'trimesh'`` is a
    dependency-free vertex-sampling fallback that can miss crossings
    between samples and is not a certificate.

    Returns a dict. Poses (``final_centers``, ``final_rotations``) are
    always scored at full scale: ``pen``/``max_pen`` come from the shared
    mesh evaluator at s = 1 regardless of where the continuation stopped.
    ``continuation_complete`` says whether scale 1 was reached;
    ``native_converged`` whether the tail's own feasibility test passed at
    full scale (``tail_stop_reason`` gives its exit: ``'feasible'``,
    ``'tail_stagnation'``, ``'tail_iter_cap'``, ``'tail_qp_failure'``,
    ``'tail_disabled'``, ``'not_run'``); ``status`` is ``'converged'`` only
    when the continuation is complete, the poses are finite, the evaluator
    finds no penetrating pair on the full-size bodies and the walls/joints
    hold, otherwise it names the failure (``'max_steps'``,
    ``'qp_failure'``, ``'numerical_failure'``, ``'container_infeasible'``,
    ``'residual_penetration'``, ``'wall_violation'``,
    ``'joint_violation'``). The same rule is used by the GPU driver.
    """
    if contact_backend not in _BACKENDS:
        raise ValueError(f"contact_backend must be one of {_BACKENDS}, got {contact_backend!r}")
    if not (np.isfinite(ds_max) and ds_max > 0.0):
        raise ValueError(f"ds_max must be a positive finite scale step, got {ds_max}")
    if not (np.isfinite(d_hat) and d_hat >= 0.0):
        raise ValueError(f"d_hat must be a non-negative finite distance, got {d_hat}")
    if int(max_steps) < 1:
        raise ValueError("max_steps must be at least 1")
    if int(revalidate_interval) < 1:
        raise ValueError("revalidate_interval must be at least 1")
    if use_dual and (enable_rotation or box_bounds is not None or joints
                     or target_centers is not None):
        raise ValueError(
            "use_dual solves the contact-only translation QP; rotation, "
            "container walls, joints and attraction need the primal solver "
            "(use_dual=False)")
    if joints and not enable_rotation:
        raise ValueError(
            "joints need enable_rotation=True: with translation only a "
            "joint row welds the two links instead of letting them swing")

    def _sync_backend():
        if contact_backend == 'warp':
            import warp as wp
            wp.synchronize()

    # Count all per-scene method setup; module imports are excluded.
    _sync_backend()
    method_t0 = time.perf_counter()
    N = len(objects)
    centers = np.array([o.center for o in objects], dtype=np.float64).reshape(N, 3)
    centers0 = centers.copy()
    rots = [np.asarray(o.rotation, dtype=np.float64).copy() for o in objects]
    nfs = [float(o.normalize_factor) for o in objects]
    mverts = [np.asarray(o.collision_verts_model, dtype=np.float64).copy() for o in objects]
    mfaces = [np.asarray(o.collision_faces, dtype=np.int32).copy() for o in objects]
    if N and not np.all(np.isfinite(centers)):
        raise ValueError("object centers contain NaN or Inf")
    for i in range(N):
        if rots[i].shape != (3, 3) or not np.all(np.isfinite(rots[i])):
            raise ValueError(f"object {i}: rotation must be a finite 3x3 matrix")
        if not np.all(np.isfinite(mverts[i])) or not np.isfinite(nfs[i]):
            raise ValueError(f"object {i}: mesh vertices / normalize_factor must be finite")
    notes = []

    # Phase-time accumulators (only populated when profile=True).
    _phase_times = {'contact': 0.0, 'qp': 0.0, 'tail': 0.0}
    # Initial scale s_min: must satisfy eq:smin_safe,
    #     s_min < min_{i≠j} (||c_i - c_j|| - d_hat) / (R_i + R_j).
    # The fixed default 0.01 is checked against this bound below and the
    # centroids are separated deterministically when it is violated.
    scale = 0.01

    # ── Optional one-time stochastic SO(3) re-orientation at s_min ───────
    # At the shrunk, collision-free state, randomly perturb each body's
    # orientation and let the 6-DOF QP re-optimise rotation as bodies
    # re-inflate. Rotation-only about the body's own pivot, so the RMSD
    # reference centers0 is untouched. Separate rng: never desyncs the seed.
    if enable_rotation and perturb_rot_deg and perturb_rot_deg > 0.0:
        _prng = np.random.default_rng(perturb_seed)
        _ang = np.deg2rad(float(perturb_rot_deg))
        for i in range(N):
            axis = _prng.normal(size=3)
            nrm = np.linalg.norm(axis)
            if nrm < 1e-12:
                continue
            rots[i] = RotLib.from_rotvec((axis / nrm) * _ang).as_matrix() @ rots[i]

    # Bounding-sphere radius about the scaling centre, at scale 1.
    max_extents = np.array([nfs[i] * (np.max(np.linalg.norm(mverts[i], axis=1)) if len(mverts[i]) else 0.0)
                            for i in range(N)])
    max_extent_all = float(max_extents.max()) if N else 0.0

    # ── Optional box-confinement (container) constraints ──────────────────
    # box_bounds=((lo_x,lo_y,lo_z),(hi_x,hi_y,hi_z)): every body must stay
    # inside the axis-aligned container at each scale step. The walls are
    # hard rows of every QP (main step, retry, tail), including steps without
    # any pair contact. box_bounds=None leaves the solver unchanged.
    walls = None
    if box_bounds is not None:
        box_lo = np.asarray(box_bounds[0], dtype=np.float64)
        box_hi = np.asarray(box_bounds[1], dtype=np.float64)
        # Per-body, per-axis support half-extents at full scale under the
        # body's INITIAL rotation. Exact for the translation-only path; with
        # enable_rotation a rotating body could protrude, which the final
        # full-scale wall check reports.
        box_sup_plus = np.zeros((N, 3))
        box_sup_minus = np.zeros((N, 3))
        for i in range(N):
            wv = nfs[i] * (mverts[i] @ rots[i].T)
            box_sup_plus[i] = wv.max(axis=0)
            box_sup_minus[i] = -wv.min(axis=0)
        if enable_rotation:
            import warnings as _w
            _w.warn("box_bounds support extents are fixed at the initial rotation; "
                    "exact only for translation-only. The final wall check is exact.")
        walls = (box_lo, box_hi, box_sup_plus, box_sup_minus)

    joints = [(int(bi), int(bj), np.asarray(a_i, dtype=np.float64),
               np.asarray(a_j, dtype=np.float64)) for (bi, bj, a_i, a_j) in (joints or [])]
    joint_bodies = set()
    for (bi, bj, _, _) in joints:
        joint_bodies.add(bi); joint_bodies.add(bj)

    if rotation_weight is None:
        # β = R_max² so that β||ω||² matches ||Δp||² in length² units
        # (surface arc displacement from rotation ω is R||ω||).
        rotation_weight = (max_extent_all ** 2)

    # ── Starting-scale admissibility and SOI event schedule ──────────────
    # Vectorised pairwise first-contact scales (bounding spheres). The
    # events feed the adaptive schedule; the minimum is the eq:smin_safe
    # bound the fixed s_min must respect.
    if N >= 2:
        _ii, _jj = np.triu_indices(N, k=1)
        _d = np.linalg.norm(centers[_ii] - centers[_jj], axis=1)
        _sc = np.minimum(np.maximum(0.0, (_d - d_hat) / (max_extents[_ii] + max_extents[_jj] + 1e-12)), 2.0)
        s_min_safe = float(_sc.min())
        event_scales = np.unique(np.minimum(_sc[_sc <= 1.0 + 0.1], 1.0)).tolist()
        del _ii, _jj, _d
    else:
        _sc = np.zeros(0)
        s_min_safe = float('inf')
        event_scales = []
    if verbose:
        print(f"  [s_min check] eq:smin_safe bound = {s_min_safe:.4f}, "
              f"using s_min = {scale:.4f}  "
              f"({'OK' if scale < s_min_safe else 'TOO LARGE; separating centroids'})")
    if scale >= s_min_safe:
        passes, remaining = separate_coincident_centroids(
            centers, max_extents, d_hat, scale, verbose=verbose)
        notes.append(f"starting-scale admissibility restored in {passes} passes "
                     f"({remaining} pairs still violating)")
        if remaining:
            raise RuntimeError("could not separate coincident centroids to an "
                               "admissible starting scale")
        # Event schedule from the separated centroids.
        _ii, _jj = np.triu_indices(N, k=1)
        _d = np.linalg.norm(centers[_ii] - centers[_jj], axis=1)
        _sc = np.minimum(np.maximum(0.0, (_d - d_hat) / (max_extents[_ii] + max_extents[_jj] + 1e-12)), 2.0)
        event_scales = np.unique(np.minimum(_sc[_sc <= 1.0 + 0.1], 1.0)).tolist()
        del _ii, _jj, _d
    if not event_scales or event_scales[-1] < 1.0:
        event_scales.append(1.0)

    def world_verts(i, s=None):
        if s is None:
            s = scale
        return s * nfs[i] * (rots[i] @ mverts[i].T).T + centers[i]

    def build_mesh(i, s=None):
        return trimesh.Trimesh(vertices=world_verts(i, s), faces=mfaces[i], process=False)

    def compute_extent(i, direction):
        return _one_sided_support(nfs[i] * mverts[i], rots[i], direction)

    # ── Contact oracles ───────────────────────────────────────────────────
    _fcl_oracle = [None]

    def _get_fcl_oracle():
        if _fcl_oracle[0] is None:
            _fcl_oracle[0] = PrebuiltFCLOracle(nfs, mverts, mfaces, d_hat)
        return _fcl_oracle[0]

    _warp_oracle = [None]

    def _init_warp_oracle():
        if _warp_oracle[0] is None:
            # The Warp GPU oracle lives in ``s4r_gpu``; imported lazily so the
            # CPU backends never pay the warp import.
            _gpu_dir = os.path.join(os.path.dirname(_HERE), "s4r_gpu")
            if _gpu_dir not in sys.path:
                sys.path.insert(0, _gpu_dir)
            from warp_pair_contact_v3 import S4RWarpContactOracleV3

            class _ObjShim:
                __slots__ = ("normalize_factor", "collision_verts_model", "collision_faces")

                def __init__(self, nf, v, f):
                    self.normalize_factor = nf
                    self.collision_verts_model = v
                    self.collision_faces = f
            shims = [_ObjShim(nfs[i], mverts[i], mfaces[i]) for i in range(N)]
            _warp_oracle[0] = S4RWarpContactOracleV3(shims, d_hat=d_hat)
        return _warp_oracle[0]

    _all_contacts = os.environ.get('S4R_ALL_CONTACTS', '0') == '1'

    def find_contacts(s, ds, bidirectional=True):
        """Pairs needing attention at scale s for an inflation by ds:
        (i, j, d_signed, normal, extent_i, extent_j, cp_on_i, cp_on_j).
        d_signed < 0 is a penetration depth; ``normal`` points so that
        moving j along +normal separates the pair (QP row n·(Δp_j-Δp_i) ≥ b).
        ``bidirectional`` is kept for backwards compatibility (ignored).
        """
        if contact_backend == 'warp':
            return _init_warp_oracle().find_contacts(s, ds, centers, rots)
        if contact_backend in ('fcl', 'fcl_prebuilt'):
            return _get_fcl_oracle().find_contacts(s, ds, centers, rots,
                                                   all_contacts=_all_contacts)
        return _find_contacts_trimesh(s, ds)

    def find_contacts_margin(s, ds, extra_margin, incremental=False):
        """find_contacts with an extra broadphase margin (world units), for a
        tail that targets a clearance above zero. extra_margin=0 is
        identical to find_contacts. ``incremental`` lets the FCL oracle
        reuse its previous call (same scale, translation-only moves)."""
        if contact_backend in ('fcl', 'fcl_prebuilt'):
            return _get_fcl_oracle().find_contacts(s, ds, centers, rots,
                                                   extra_margin=extra_margin,
                                                   all_contacts=_all_contacts,
                                                   incremental=incremental)
        if extra_margin <= 0.0:
            return find_contacts(s, ds)
        raise NotImplementedError(
            "S4R_TAIL_TARGET_MARGIN requires the FCL contact backend")

    def _find_contacts_trimesh(s, ds):
        """Dependency-free fallback: bidirectional vertex sampling with a
        ray-cast inside test. A crossing between samples is not seen, so
        this backend is a contact generator, not a certificate."""
        contacts = []
        meshes = [build_mesh(i, s) for i in range(N)]
        for i in range(N):
            vi = np.asarray(meshes[i].vertices)
            ai0, ai1 = vi.min(0), vi.max(0)
            for j in range(i + 1, N):
                d_ij_now = float(np.linalg.norm(centers[i] - centers[j]))
                s_contact_now = max(
                    0.0,
                    (d_ij_now - d_hat) / (max_extents[i] + max_extents[j] + 1e-12),
                )
                if s + ds < s_contact_now * 0.9:
                    continue
                vj = np.asarray(meshes[j].vertices)
                aj0, aj1 = vj.min(0), vj.max(0)
                margin = ds * (max_extents[i] + max_extents[j]) + d_hat
                if not (np.all(ai0 - margin <= aj1) and np.all(aj0 - margin <= ai1)):
                    continue

                cl_ji, d_ji, _ = trimesh.proximity.closest_point(meshes[i], vj)
                cl_ij, d_ij, _ = trimesh.proximity.closest_point(meshes[j], vi)

                inside_ji = np.zeros(len(vj), dtype=bool)
                mask_j_in_ai = np.all((vj >= ai0 - d_hat) & (vj <= ai1 + d_hat), axis=1)
                if mask_j_in_ai.any():
                    inside_ji[mask_j_in_ai] = _mesh_contains_points(meshes[i], vj[mask_j_in_ai])
                inside_ij = np.zeros(len(vi), dtype=bool)
                mask_i_in_aj = np.all((vi >= aj0 - d_hat) & (vi <= aj1 + d_hat), axis=1)
                if mask_i_in_aj.any():
                    inside_ij[mask_i_in_aj] = _mesh_contains_points(meshes[j], vi[mask_i_in_aj])

                sd_ji = np.where(inside_ji, -d_ji, d_ji)
                sd_ij = np.where(inside_ij, -d_ij, d_ij)
                min_ji = sd_ji.min()
                min_ij = sd_ij.min()
                if min_ji <= min_ij:
                    k = int(np.argmin(sd_ji))
                    d_signed = float(sd_ji[k])
                    if inside_ji[k]:
                        n = cl_ji[k] - vj[k]
                        cp_on_a, cp_on_b = vj[k].copy(), cl_ji[k].copy()
                    else:
                        n = vj[k] - cl_ji[k]
                        cp_on_a, cp_on_b = cl_ji[k].copy(), vj[k].copy()
                else:
                    k = int(np.argmin(sd_ij))
                    d_signed = float(sd_ij[k])
                    if inside_ij[k]:
                        n = -(cl_ij[k] - vi[k])
                        cp_on_a, cp_on_b = cl_ij[k].copy(), vi[k].copy()
                    else:
                        n = -(vi[k] - cl_ij[k])
                        cp_on_a, cp_on_b = vi[k].copy(), cl_ij[k].copy()

                nn = np.linalg.norm(n)
                if nn < 1e-12:
                    n = centers[j] - centers[i]
                    nn = np.linalg.norm(n)
                if nn < 1e-12:
                    continue
                n = n / nn
                ext_i = compute_extent(i, n)
                ext_j = compute_extent(j, -n)
                if d_signed < ds * (ext_i + ext_j) + d_hat:
                    contacts.append((i, j, d_signed, n, ext_i, ext_j, cp_on_a, cp_on_b))
        return contacts

    # Unified timing policy: everything from function entry to here (event
    # schedule, backend construction) is setup_time; the continuation, tail
    # and cleanup are solve_time; the reported `time` is their sum.
    if contact_backend in ('fcl', 'fcl_prebuilt'):
        _get_fcl_oracle()
    elif contact_backend == 'warp':
        _init_warp_oracle()
    _sync_backend()
    setup_time = time.perf_counter() - method_t0
    solve_t0 = time.perf_counter()
    diagnostics = []
    warm_cache = {}
    prev_sensitivity = None
    prev_active_bodies = None
    prev_dof_per_body = None

    # ── Audit: evaluator-equivalent check at every iteration ─────────
    audit_log = []
    if audit:
        from mesh_collision import evaluate_world_collision_meshes

        def _audit_pairs_at(s_now):
            ms = [build_mesh(i, s_now) for i in range(N)]
            stats = evaluate_world_collision_meshes(ms)
            pen_set = set()
            bounds = [m.bounds for m in ms]
            for i in range(N):
                amin, amax = bounds[i]
                for j in range(i + 1, N):
                    bmin, bmax = bounds[j]
                    if not (np.all(amin <= bmax) and np.all(bmin <= amax)):
                        continue
                    vj = np.asarray(ms[j].vertices, dtype=np.float64)
                    vi = np.asarray(ms[i].vertices, dtype=np.float64)
                    cl_i, d_ji, fi = trimesh.proximity.closest_point(ms[i], vj)
                    cl_j, d_ij, fj = trimesh.proximity.closest_point(ms[j], vi)
                    n_i = ms[i].face_normals[fi]
                    n_j = ms[j].face_normals[fj]
                    inside_ji = np.sum((vj - cl_i) * n_i, axis=1) < 0
                    inside_ij = np.sum((vi - cl_j) * n_j, axis=1) < 0
                    if np.any(inside_ji) or np.any(inside_ij):
                        pen_set.add((i, j))
            return stats.pen_pairs, stats.max_penetration, stats.min_signed_distance, pen_set

    # ── Frozen-witness cache state ────────────────────────────────────
    # cached_contacts holds the last detection, advanced analytically on the
    # steps in between by the displacement that was actually applied
    # (last_dp) and the inflation that actually happened (last_ds_applied):
    #     d~^{k+1} = d~^k + n·(Δp_j − Δp_i) − ds_k (e_i + e_j).
    # Every accepted step records its own (Δp, ds), including steps that
    # moved nothing, so the update never reuses an older displacement. A
    # rotation update invalidates the cache (the update above is
    # translation-only) and forces a fresh detection.
    cached_contacts = None
    cache_propagated = False
    ds_retry_cap = float('inf')
    steps_since_detection = 999
    last_dp = np.zeros((N, 3))
    last_ds_applied = 0.0
    cache_err_max = 0.0
    truth_set = set()
    missed = set()
    truth_pen = truth_maxp = truth_minsd = 0

    def _snapshot_verts():
        return [world_verts(i, scale).astype(np.float32) for i in range(N)]

    def _snapshot_verts_at(s_now):
        return [world_verts(i, s_now).astype(np.float32) for i in range(N)]

    if trajectory_dumper is not None:
        trajectory_dumper.set_bodies(
            faces_list=[f.astype(np.int32) for f in mfaces],
            verts_list=_snapshot_verts_at(1.0),
        )
        trajectory_dumper.add_frame(step=-2, sub=0, scale=1.0, verts_list=_snapshot_verts_at(1.0))
        trajectory_dumper.add_frame(step=-1, sub=0, scale=float(scale), verts_list=_snapshot_verts())

    def _dump_frame(step_idx: int):
        if trajectory_dumper is None:
            return
        if step_idx != 0 and step_idx % dump_every != 0 and step_idx != max_steps - 1:
            return
        trajectory_dumper.add_frame(step=step_idx, sub=0, scale=float(scale), verts_list=_snapshot_verts())

    def _apply_attraction_only(max_frac=1.0):
        """Attraction-only predictor when no contact pushes: pull bodies
        toward target_centers. Closed-form minimiser of ½‖Δ‖² + ½α‖c+Δ−t‖²
        is Δ = −α/(1+α)·(c−t), capped by a fraction of the scale step.
        Returns the applied displacement (N,3) or None."""
        if target_centers is None or attraction_alpha is None:
            return None
        raw = attraction_alpha(scale) if callable(attraction_alpha) else attraction_alpha
        a = float(np.asarray(raw).max())
        if a <= 1e-8:
            return None
        err = centers - target_centers
        delta = -a / (1.0 + a) * err
        cap = max_frac * ds * float(max_extent_all + 1e-9)
        norms = np.linalg.norm(delta, axis=1, keepdims=True)
        delta = delta * np.minimum(1.0, cap / np.clip(norms, 1e-9, None))
        centers[:] += delta
        return delta

    # Set when the container walls become mutually infeasible at some scale:
    # no translation keeps every body inside the box. The walls are HARD, so
    # the run reports infeasibility instead of relaxing them.
    container_infeasible = False
    stop_reason = None
    steps_used = 0
    ds = ds_max

    for step in range(max_steps):
        if scale >= 1.0 - 1e-6:
            break
        steps_used = step + 1
        # The QP-failure retry halves ds; the cap keeps the halving alive
        # across the schedule below.
        ds = min(ds_max, 1.0 - scale, ds_retry_cap)

        # ── Event-driven scheduling ──────────────────────────────────
        if adaptive_ds:
            next_events = [s for s in event_scales if s > scale + 1e-6]
            if next_events:
                ds_to_event = next_events[0] - scale
                if ds_to_event > ds_max:
                    ds = min(ds_to_event, 3.0 * ds_max)
            if step > 0 and diagnostics and not diagnostics[-1].get('had_contacts', True):
                ds = min(2.0 * ds_max, 1.0 - scale)
            ds = min(ds, ds_retry_cap)

        # ── Full detection or analytical update ──────────────────────
        # interval=M detects every M steps (interval=1: every step).
        if cached_contacts is None or steps_since_detection >= revalidate_interval - 1:
            is_final = (scale + ds >= 1.0 - 1e-6)
            _t_contact = time.time() if profile else 0.0
            contacts = find_contacts(scale, ds, bidirectional=is_final)
            if profile:
                _phase_times['contact'] += time.time() - _t_contact
            step_cache_err = None
            if cache_propagated and cached_contacts:
                prop = {(c[0], c[1]): c[2] for c in cached_contacts}
                errs = [abs(c[2] - prop[(c[0], c[1])]) for c in contacts if (c[0], c[1]) in prop]
                if errs:
                    step_cache_err = float(max(errs))
                    cache_err_max = max(cache_err_max, step_cache_err)
            if audit:
                truth_pen, truth_maxp, truth_minsd, truth_set = _audit_pairs_at(scale)
                solver_set = set((min(i, j), max(i, j)) for (i, j, *_) in contacts)
                missed = truth_set - solver_set
                print(f"[audit] step={step:3d} scale={scale:.4f} "
                      f"solver_contacts={len(contacts):4d}  "
                      f"evaluator_pen={truth_pen:4d}  "
                      f"missed_by_solver={len(missed):3d}  "
                      f"max_pen={truth_maxp:.4f}  "
                      f"min_sd={truth_minsd:.4f}")
                if missed:
                    print(f"        missed pairs: {sorted(missed)[:5]}"
                          f"{'...' if len(missed) > 5 else ''}")
            cached_contacts = contacts
            cache_propagated = False
            steps_since_detection = 0
        else:
            if cached_contacts:
                cached_contacts = [
                    (i, j, d + n.dot(last_dp[j] - last_dp[i]) - last_ds_applied * (ei + ej),
                     n, ei, ej, cpi + last_dp[i], cpj + last_dp[j])
                    for i, j, d, n, ei, ej, cpi, cpj in cached_contacts
                ]
                cache_propagated = True
            contacts = cached_contacts
            steps_since_detection += 1
            step_cache_err = None

        # Pairs that need pushing this step.
        active = [c for c in contacts if ds * (c[4] + c[5]) + d_hat - c[2] > 0]
        n_pen = sum(1 for c in contacts if c[2] < -1e-6)

        # A step without active contacts needs no solve unless container
        # walls are present (a body next to a wall inflates into it even
        # without any pair contact, so the wall rows are solved every step)
        # or an attraction term would move jointed bodies (the QP carries
        # the joint rows; the closed-form attraction step does not).
        attraction_on = target_centers is not None and attraction_alpha is not None
        if not active and walls is None and not (joints and attraction_on):
            delta = _apply_attraction_only()
            last_dp = delta if delta is not None else np.zeros((N, 3))
            last_ds_applied = ds
            scale += ds
            ds_retry_cap = float('inf')
            diagnostics.append({'step': step, 'scale': scale, 'ds': ds,
                                'n_active': 0, 'n_pen': n_pen,
                                'had_contacts': bool(contacts)})
            _dump_frame(step)
            continue

        n_active = len(active)

        # ── Active bodies (contact-graph sparsity) ────────────────────
        # With walls every body carries a wall row every step, so sparsity
        # is disabled; joint endpoints are always in the QP so a joint row is
        # never dropped because one end had no contact.
        if contact_sparsity and walls is None:
            active_set = {b for c in active for b in (c[0], c[1])} | joint_bodies
            active_bodies = sorted(active_set)
        else:
            active_bodies = list(range(N))
        body_map = {b: idx for idx, b in enumerate(active_bodies)}
        n_bodies_qp = len(active_bodies)
        dof_per_body = 6 if enable_rotation else 3
        n_vars = dof_per_body * n_bodies_qp

        # ── Objective ────────────────────────────────────────────────
        # min ½ xᵀPx + qᵀx; optional attraction α/2‖c + Δp − t‖² adds α to
        # the translation diagonal and α(c − t) to q.
        alpha_tr = None
        if target_centers is not None and attraction_alpha is not None:
            raw = attraction_alpha(scale) if callable(attraction_alpha) else attraction_alpha
            raw = np.asarray(raw, dtype=np.float64)
            alpha_tr = np.full(N, float(raw)) if raw.ndim == 0 else raw.reshape(-1)
        diag = []
        q = np.zeros(n_vars)
        for idx, b in enumerate(active_bodies):
            a_b = float(alpha_tr[b]) if alpha_tr is not None else 0.0
            diag.extend([1.0 + a_b] * 3)
            if enable_rotation:
                diag.extend([rotation_weight] * 3)
            if a_b > 0.0:
                q[dof_per_body * idx: dof_per_body * idx + 3] = a_b * (centers[b] - target_centers[b])
        P = sp.diags(diag, format='csc')

        # ── Constraints ──────────────────────────────────────────────
        A_full, l_full, u_full, b_c, walls_infeasible = _assemble_constraints(
            active, ds, d_hat, centers, rots, active_bodies, body_map,
            dof_per_body, max_omega, walls, joints, scale + ds)
        if walls_infeasible:
            container_infeasible = True
            stop_reason = 'container_infeasible'
            if verbose:
                print(f"  Step {step + 1}: container infeasible at "
                      f"scale={scale + ds:.3f} (walls cannot bound all "
                      f"bodies inside the box). Stopping.")
            break

        # ── Predictor: ODE sensitivity from the previous step ─────────
        x0 = np.zeros(n_vars)
        used_predictor = False
        if (prev_sensitivity is not None and prev_active_bodies is not None
                and dof_per_body == prev_dof_per_body
                and len(prev_sensitivity) == n_vars
                and prev_active_bodies == set(active_bodies)):
            x0 = ds * prev_sensitivity
            used_predictor = True
        elif (prev_sensitivity is not None and prev_active_bodies is not None
                and dof_per_body == prev_dof_per_body
                and len(prev_sensitivity) == n_vars
                and len(prev_active_bodies & set(active_bodies)) > len(active_bodies) * 0.5):
            x0 = ds * prev_sensitivity * 0.5
            used_predictor = True
        else:
            for idx, b in enumerate(active_bodies):
                if b in warm_cache:
                    cached = warm_cache[b]
                    copy_len = min(len(cached), dof_per_body)
                    x0[dof_per_body * idx: dof_per_body * idx + copy_len] = cached[:copy_len]

        # ── Solve ────────────────────────────────────────────────────
        t_qp = time.time()
        qp_iters = 0
        result_x = None
        fallback_used = False
        _t_qp = time.time() if profile else 0.0
        if use_dual:
            dual_result = solve_dual_qp(A_full, l_full, np.ones(n_vars), n_vars)
            qp_solved = dual_result is not None
            if qp_solved:
                result_x, qp_iters = dual_result[0], dual_result[2]
        else:
            result = osqp_solve(P, q, A_full, l_full, u_full, x0=x0, **_OSQP_STEP)
            qp_solved = qp_status_ok(result)
            if qp_solved:
                result_x, qp_iters = result.x, result.info.iter
            elif enable_rotation:
                # Rotation-locked retry: identical rows (contacts, walls,
                # joints) with ω fixed at zero, so the fallback keeps every
                # hard constraint of the requested problem.
                A_lock, l_lock, u_lock, _, _ = _assemble_constraints(
                    active, ds, d_hat, centers, rots, active_bodies, body_map,
                    dof_per_body, max_omega, walls, joints, scale + ds,
                    lock_rotation=True)
                result = osqp_solve(P, q, A_lock, l_lock, u_lock, x0=None, **_OSQP_STEP)
                qp_solved = qp_status_ok(result)
                if qp_solved:
                    result_x, qp_iters = result.x, result.info.iter
                    fallback_used = True
        if profile:
            _phase_times['qp'] += time.time() - _t_qp
        qp_time = time.time() - t_qp

        if not qp_solved or result_x is None or not np.all(np.isfinite(result_x)):
            # Reduce ds and retry with a fresh detection: the cache was
            # already propagated for this failed attempt.
            ds *= 0.5
            ds_retry_cap = ds
            steps_since_detection = 999
            cache_propagated = False
            if ds < 1e-6:
                stop_reason = 'qp_failure'
                if verbose:
                    print(f"  Step {step + 1}: QP infeasible, ds too small. Stopping.")
                break
            continue

        x = result_x.reshape(n_bodies_qp, dof_per_body)
        dp = x[:, :3]
        for idx, b in enumerate(active_bodies):
            warm_cache[b] = x[idx].copy()

        # ── Sensitivity dq*/ds for the predictor ─────────────────────
        # Binding contact rows: dq*/ds = −A_bindᵀ (A_bind A_bindᵀ)⁻¹ ∂b/∂s,
        # with ∂d/∂s = −(e_i + e_j) per contact.
        prev_sensitivity = None
        if n_active:
            dd_ds = np.array([-(c[4] + c[5]) for c in active])
            A_c = A_full.tocsr()[:n_active]
            residuals = A_c @ result_x - b_c
            binding = residuals < 1e-4
            if binding.any():
                A_bind = A_c[binding]
                dd_bind = dd_ds[binding]
                try:
                    G = (A_bind @ A_bind.T).tocsc() + 1e-8 * sp.identity(int(binding.sum()), format='csc')
                    d_lambda = spla.spsolve(G, dd_bind)
                    sens = -(A_bind.T @ d_lambda)
                    if np.all(np.isfinite(sens)):
                        prev_sensitivity = np.asarray(sens).reshape(-1)
                except Exception:
                    prev_sensitivity = None
        prev_active_bodies = set(active_bodies)
        prev_dof_per_body = dof_per_body

        # ── Apply translation ────────────────────────────────────────
        dp_full = np.zeros((N, 3))
        for idx, b in enumerate(active_bodies):
            centers[b] += dp[idx]
            dp_full[b] = dp[idx]
        last_dp = dp_full
        last_ds_applied = ds

        # ── Apply rotation ───────────────────────────────────────────
        max_rot_applied = 0.0
        rot_total = 0.0
        if enable_rotation:
            omega = x[:, 3:6]
            for idx, b in enumerate(active_bodies):
                w = omega[idx]
                nw = np.linalg.norm(w)
                max_rot_applied = max(max_rot_applied, nw)
                if nw > 1e-12:
                    rots[b] = RotLib.from_rotvec(w).as_matrix() @ rots[b]
            # Step B: SO(3) refinement. Skipped when walls or joints are
            # present: it rotates bodies about their own centroids without
            # those rows and would break them.
            if n_active >= 3 and walls is None and not joints:
                oracle = _get_fcl_oracle() if contact_backend in ('fcl', 'fcl_prebuilt') else None
                rot_total = optimize_rotations_on_manifold(
                    centers, rots, nfs, mverts, mfaces, N, d_hat, scale + ds,
                    contact_backend=contact_backend,
                    bvh_cache=oracle.bvh if oracle is not None else None,
                    model_verts_cache=oracle.model_verts if oracle is not None else None)
            if max_rot_applied > 1e-12 or rot_total > 0.0:
                # The analytic cache update is translation-only.
                steps_since_detection = 999

        scale += ds
        ds_retry_cap = float('inf')
        _dump_frame(step)

        diag_entry = {
            'step': step, 'scale': scale, 'ds': ds,
            'n_active': n_active, 'n_pen': n_pen,
            'had_contacts': True,
            'qp_time_ms': qp_time * 1000,
            'qp_iters': qp_iters,
            'max_disp': float(np.max(np.linalg.norm(dp, axis=1))) if n_bodies_qp else 0.0,
            'max_rot': max_rot_applied,
            'n_vars': n_vars,
            'used_predictor': used_predictor,
        }
        if fallback_used:
            diag_entry['fallback'] = True
        if step_cache_err is not None:
            diag_entry['cache_err'] = step_cache_err
        diagnostics.append(diag_entry)

        if audit:
            post_pen, post_maxp, post_minsd, post_set = _audit_pairs_at(scale)
            new_pen = post_set - truth_set
            print(f"        post-step: scale={scale:.4f}  "
                  f"pen_after={post_pen:3d}  max_pen={post_maxp:.4f}  "
                  f"new_pen_this_step={len(new_pen)}")
            audit_log.append({
                'step': step, 'scale_before': scale - ds, 'scale_after': scale,
                'solver_contacts': n_active,
                'evaluator_pen_before': truth_pen,
                'evaluator_pen_after': post_pen,
                'max_pen_before': truth_maxp,
                'max_pen_after': post_maxp,
                'missed_by_solver_before': len(missed),
                'new_pen_this_step': len(new_pen),
            })

        if verbose and (step + 1) % 5 == 0:
            rmsd = float(np.sqrt(np.mean(np.sum((centers - centers0) ** 2, axis=1))))
            rot_str = f" max_rot={max_rot_applied:.4f}" if enable_rotation else ""
            print(f"  Step {step + 1}: scale={scale:.3f} contacts={n_active} "
                  f"pen={n_pen} max_push={diag_entry['max_disp']:.4f} "
                  f"RMSD={rmsd:.4f}{rot_str}")

    continuation_complete = bool(scale >= 1.0 - 1e-6)
    if not continuation_complete and stop_reason is None:
        stop_reason = 'max_steps'
    if continuation_complete:
        scale = 1.0

    # ── Tail refinement: correction QPs at scale 1 with ds = 0 ────────
    # Residual linearisation error can leave a few pairs with d < 0 at full
    # scale. Correction QPs use the same active set as the main loop
    # (penetrating and near-contact pairs) plus the walls and joints, so
    # pushing one pair does not flip a neighbour or break a constraint.
    # Only run when the continuation actually reached full scale: a tail at
    # a shrunken scale would say nothing about the delivered scene.
    max_tail_iters = int(os.environ.get('S4R_MAX_TAIL_ITERS', 20))
    stagnation_cap = int(os.environ.get('S4R_TAIL_STAGNATION_CAP', 3))
    tail_margin = float(os.environ.get('S4R_TAIL_TARGET_MARGIN', '0.0'))
    if tail_margin > d_hat:
        raise ValueError("S4R_TAIL_TARGET_MARGIN must not exceed d_hat: the "
                         "correction rows push pairs to the d_hat clearance")
    tail_stop_reason = 'not_run'
    _t_tail_start = time.time() if profile else 0.0
    if continuation_complete:
        tail_stop_reason = 'tail_iter_cap' if max_tail_iters > 0 else 'tail_disabled'
        prev_max_pen = float('inf')
        prev_pen_pairs = None
        stagnation = 0
        def _joint_residual_now():
            res = 0.0
            for (bi, bj, a_i, a_j) in joints:
                gap = (centers[bi] + rots[bi] @ a_i) - (centers[bj] + rots[bj] @ a_j)
                res = max(res, float(np.linalg.norm(gap)))
            return res

        for tail in range(max_tail_iters):
            _t_td = time.time() if profile else 0.0
            contacts_tail = find_contacts_margin(1.0, 0.0, tail_margin, incremental=True)
            if profile:
                _phase_times['tail_contact'] = _phase_times.get('tail_contact', 0.0) + time.time() - _t_td
            pen_pairs_tail = [c for c in contacts_tail if c[2] < tail_margin]
            # The 6-DOF steps satisfy the joint rows to first order in ω; the
            # translation-only correction below restores them exactly, so a
            # residual above tolerance is corrected even without penetration.
            joint_off = _joint_residual_now() > 1e-9 if joints else False
            if not pen_pairs_tail and not joint_off:
                tail_stop_reason = 'feasible'
                if audit:
                    print(f"[TAIL] iter={tail} pen=0 (margin target {tail_margin}), converged")
                break

            if pen_pairs_tail:
                max_pen_now = max(tail_margin - c[2] for c in pen_pairs_tail)
                n_pen_now = len(pen_pairs_tail)
                pair_dropped = prev_pen_pairs is None or n_pen_now < prev_pen_pairs
                if max_pen_now >= prev_max_pen - 1e-5 and not pair_dropped:
                    stagnation += 1
                    if stagnation >= stagnation_cap:
                        tail_stop_reason = 'tail_stagnation'
                        if audit:
                            print(f"[TAIL] iter={tail} stagnated at max_pen={max_pen_now:.4f} pairs={n_pen_now}, stopping")
                        break
                else:
                    stagnation = 0
                prev_max_pen = max_pen_now
                prev_pen_pairs = n_pen_now
            else:
                max_pen_now = 0.0
                n_pen_now = 0
            if audit:
                print(f"[TAIL] iter={tail} scale={scale:.4f} pen_pairs={n_pen_now} "
                      f"active_total={len(contacts_tail)} max_pen={max_pen_now:.4f}")

            if contact_sparsity and walls is None:
                active_bodies_t = sorted({b for c in contacts_tail for b in (c[0], c[1])} | joint_bodies)
            else:
                active_bodies_t = list(range(N))
            body_map_t = {b: idx for idx, b in enumerate(active_bodies_t)}
            n_b_t = len(active_bodies_t)
            n_vars_t = 3 * n_b_t
            A_t, l_t, u_t, _, walls_infeasible_t = _assemble_constraints(
                contacts_tail, 0.0, d_hat, centers, rots, active_bodies_t,
                body_map_t, 3, max_omega, walls, joints, 1.0)
            if walls_infeasible_t:
                container_infeasible = True
                tail_stop_reason = 'container_infeasible'
                if verbose:
                    print("  Tail: container infeasible at s=1 (walls cannot bound all bodies).")
                break
            _t_tq = time.time() if profile else 0.0
            res_t = osqp_solve(sp.identity(n_vars_t, format='csc'), np.zeros(n_vars_t),
                               A_t, l_t, u_t, **_OSQP_TAIL)
            if profile:
                _phase_times['tail_qp'] = _phase_times.get('tail_qp', 0.0) + time.time() - _t_tq
            if not qp_status_ok(res_t) or not np.all(np.isfinite(res_t.x)):
                tail_stop_reason = 'tail_qp_failure'
                if audit:
                    print(f"[TAIL] iter={tail} QP status={res_t.info.status}, stopping")
                break

            # Damp the step to the linearisation radius: no body moves more
            # than d_hat per iteration.
            dp_t = res_t.x.reshape(n_b_t, 3)
            max_disp_pred = float(np.max(np.linalg.norm(dp_t, axis=1)))
            alpha = 1.0 if max_disp_pred <= d_hat else (d_hat / max_disp_pred)
            # Bodies with only inactive rows get a numerically-zero step
            # from OSQP; leaving them exactly in place lets the next
            # detection reuse their pair results.
            for idx, b in enumerate(active_bodies_t):
                step_b = alpha * dp_t[idx]
                if float(np.linalg.norm(step_b)) > 1e-10:
                    centers[b] += step_b
            diagnostics.append({
                'step': 'tail', 'tail_iter': tail,
                'pen_pairs': n_pen_now,
                'active_total': len(contacts_tail),
                'max_pen_before': max_pen_now,
                'max_disp_pred': max_disp_pred,
                'alpha': alpha,
            })
    if profile:
        _phase_times['tail'] = time.time() - _t_tail_start

    # ── Penalty cleanup pass ────────────────────────────────────────────
    # Pairwise mass-balanced pushes (|depth| + ε along the contact normal)
    # for residual pairs on which the tail QP stagnates; applied only to
    # those pairs so the global RMSD barely moves. It maintains no wall or
    # joint rows, so it is skipped when those constraints are present, and
    # it is disabled in every result reported in the paper.
    # cleanup_iters_used: number of iterations that applied a push (0 = the
    # pass ran but the scene was already free; None = disabled or skipped).
    cleanup_iters_used = None
    cleanup_enabled = not int(os.environ.get('S4R_DISABLE_PENALTY_CLEANUP', 0))
    if cleanup_enabled and continuation_complete and (walls is not None or joints):
        notes.append("penalty cleanup skipped: container walls / joints are hard constraints it cannot maintain")
        cleanup_enabled = False
    if cleanup_enabled and continuation_complete:
        cleanup_iters_used = 0
        max_cleanup_iters = int(os.environ.get('S4R_MAX_CLEANUP_ITERS', 200))
        cleanup_eps = float(os.environ.get('S4R_CLEANUP_EPS', d_hat * 0.1))
        prev_pen_n = None
        stagnant = 0
        gauss_seidel = False
        for ci in range(max_cleanup_iters):
            cleanup_contacts = find_contacts(1.0, 0.0, bidirectional=True)
            cleanup_pen = [c for c in cleanup_contacts if c[2] < 0.0]
            if not cleanup_pen:
                if audit:
                    print(f"[CLEANUP] iter={ci} pen=0, converged")
                break
            cleanup_iters_used += 1
            n_pen_now = len(cleanup_pen)
            if prev_pen_n is not None:
                if n_pen_now > prev_pen_n:
                    cleanup_eps *= 0.5
                    stagnant = 0
                elif n_pen_now == prev_pen_n:
                    stagnant += 1
                    if stagnant >= 2:
                        cleanup_eps = min(cleanup_eps * 1.5, d_hat * 2.0)
                        gauss_seidel = True
                else:
                    stagnant = 0
            prev_pen_n = n_pen_now
            if audit:
                print(f"[CLEANUP] iter={ci} pen_pairs={n_pen_now} "
                      f"eps={cleanup_eps:.5f} mode={'GS' if gauss_seidel else 'J'}")
            if gauss_seidel:
                cleanup_pen.sort(key=lambda c: c[2])
            push = np.zeros_like(centers)
            for (i, j, d_signed, n_ij, _ei, _ej, _cpa, _cpb) in cleanup_pen:
                magnitude = (-d_signed) + cleanup_eps
                if gauss_seidel:
                    centers[i] -= 0.5 * magnitude * n_ij
                    centers[j] += 0.5 * magnitude * n_ij
                else:
                    push[i] -= 0.5 * magnitude * n_ij
                    push[j] += 0.5 * magnitude * n_ij
            if not gauss_seidel:
                centers += push

    _sync_backend()
    solve_time = time.perf_counter() - solve_t0
    method_total_time = setup_time + solve_time

    # ── Final evaluation at FULL scale ─────────────────────────────────
    # The delivered scene is the full-size bodies at the final poses, so the
    # score is taken at s = 1 whatever scale the continuation reached.
    eval_t0 = time.perf_counter()
    from mesh_collision import evaluate_world_collision_meshes
    meshes = [build_mesh(i, 1.0) for i in range(N)]
    stats = evaluate_world_collision_meshes(meshes)
    rmsd = float(np.sqrt(np.mean(np.sum((centers - centers0) ** 2, axis=1)))) if N else 0.0
    poses_finite = bool(np.all(np.isfinite(centers)) and all(np.all(np.isfinite(R)) for R in rots))
    wall_violation = 0.0
    if walls is not None:
        for m in meshes:
            v = np.asarray(m.vertices)
            wall_violation = max(wall_violation,
                                 float(np.max(box_lo - v.min(axis=0))),
                                 float(np.max(v.max(axis=0) - box_hi)))
        wall_violation = max(0.0, wall_violation)
    joint_residual = 0.0
    for (bi, bj, a_i, a_j) in joints:
        gap = (centers[bi] + rots[bi] @ a_i) - (centers[bj] + rots[bj] @ a_j)
        joint_residual = max(joint_residual, float(np.linalg.norm(gap)))
    evaluation_time = time.perf_counter() - eval_t0

    native_converged = bool(continuation_complete and tail_stop_reason == 'feasible'
                            and not container_infeasible)
    if not continuation_complete:
        status = stop_reason
    elif not poses_finite:
        status = 'numerical_failure'
    elif container_infeasible:
        status = 'container_infeasible'
    elif stats.pen_pairs > 0:
        status = 'residual_penetration'
    elif wall_violation > 1e-6:
        status = 'wall_violation'
    elif joint_residual > 1e-6:
        status = 'joint_violation'
    else:
        status = 'converged'

    if profile:
        _phase_times['other'] = max(0.0, solve_time - _phase_times['contact']
                                    - _phase_times['qp'] - _phase_times['tail'])

    if trajectory_dumper is not None:
        trajectory_dumper.add_frame(step=steps_used + 1, sub=0, scale=float(scale),
                                    verts_list=_snapshot_verts())
        out_path = trajectory_dumper.write()
        print(f"  Trajectory dumped to {out_path}  "
              f"({len(trajectory_dumper._frames)} frames)")

    return {
        # Scored at full scale by the shared mesh evaluator.
        "pen": stats.pen_pairs,
        "max_pen": stats.max_penetration,
        "evaluated_at_scale": 1.0,
        "rmsd": rmsd,
        "timing_policy": "per_scene_setup_plus_solve_v1",
        "setup_time": setup_time,
        "solve_time": solve_time,
        "method_total_time": method_total_time,
        "evaluation_time": evaluation_time,
        "solver_internal_time": solve_time,
        "time": method_total_time,
        "scale": scale,
        "steps": steps_used,
        # Outcome contract.
        "status": status,
        "continuation_complete": continuation_complete,
        "stop_reason": tail_stop_reason if continuation_complete else stop_reason,
        "tail_stop_reason": tail_stop_reason,
        "native_converged": native_converged,
        "container_infeasible": container_infeasible,
        "wall_violation": wall_violation,
        "joint_residual": joint_residual,
        "poses_finite": poses_finite,
        "cleanup_iters_used": cleanup_iters_used,
        # Largest discrepancy between an analytically advanced cached
        # distance and the following fresh detection (cache accuracy).
        "cache_error_max": cache_err_max,
        "notes": notes,
        "final_centers": centers.copy(),
        "final_rotations": np.stack(rots, axis=0) if N else np.zeros((0, 3, 3)),
        "diagnostics": diagnostics,
        "audit_log": audit_log,
        "phase_times": _phase_times if profile else None,
    }


# ─────────────────────────────────────────────────────────────────────
# SO(3) refinement (6-DOF path)
# ─────────────────────────────────────────────────────────────────────

def optimize_rotations_on_manifold(centers, rots, nfs, mverts, mfaces, N,
                                     d_hat, scale, max_rot_iters=3,
                                     contact_backend='fcl',
                                     bvh_cache=None,
                                     model_verts_cache=None):
    """Step B of the alternating minimisation: rotate each body on SO(3) to
    open its near contacts while staying close to its current rotation.

    Geodesic gradient step R_new = exp(ω) R_old with ω the summed
    repulsive torque. For a contact on body i at lever arm r (contact point
    minus centroid) with unit direction n from i's surface toward the
    neighbour, rotating by ω moves the contact point by ω × r, so the gap
    along n changes by −n·(ω × r) = −ω·(r × n). Repulsion means the point
    moves away from the neighbour, i.e. along −n, so the torque is
    r × (−f n) with f = max(0, d_hat − d). A neighbour point lying inside
    body i (negative distance) is treated with the reversed direction, so
    the surface sweeps past it.

    Backends: 'fcl' / 'fcl_prebuilt' reuse the unit-scale BVHs through
    ``bvh_cache``/``model_verts_cache`` (populated on first use);
    'trimesh' is the dependency-free closest-point path.
    """
    use_fcl = contact_backend in ('fcl', 'fcl_prebuilt')
    if use_fcl:
        try:
            import fcl  # noqa: F401
        except ImportError:
            use_fcl = False
    if use_fcl:
        return _optimize_rotations_fcl(centers, rots, nfs, mverts, mfaces, N,
                                        d_hat, scale, max_rot_iters,
                                        bvh_cache=bvh_cache,
                                        model_verts_cache=model_verts_cache)
    return _optimize_rotations_trimesh(centers, rots, nfs, mverts, mfaces, N,
                                        d_hat, scale, max_rot_iters)


def _repulsive_torque(r_i, n_ij, force_mag):
    """Torque on body i that moves its contact point (lever arm r_i) away
    from the neighbour along −n_ij (see optimize_rotations_on_manifold)."""
    return np.cross(r_i, -force_mag * n_ij)


def _optimize_rotations_fcl(centers, rots, nfs, mverts, mfaces, N,
                             d_hat, scale, max_rot_iters,
                             bvh_cache=None, model_verts_cache=None):
    """FCL-backed rotation refinement in u-space (u = x / s): one
    unit-scale BVH per body, rotation applied through the FCL transform."""
    import fcl

    inv_s = 1.0 / scale

    if bvh_cache is not None and len(bvh_cache) == N:
        bvh_models = bvh_cache
        model_verts_scaled = model_verts_cache
    else:
        bvh_models = [] if bvh_cache is None else bvh_cache
        model_verts_scaled = [] if model_verts_cache is None else model_verts_cache
        bvh_models.clear()
        model_verts_scaled.clear()
        for i in range(N):
            Vi = (nfs[i] * mverts[i]).astype(np.float64)
            faces_i = np.asarray(mfaces[i], dtype=np.int32)
            if len(faces_i) and _signed_volume(Vi, faces_i) < 0.0:
                faces_i = np.ascontiguousarray(faces_i[:, ::-1])
            model_verts_scaled.append(Vi)
            m = fcl.BVHModel()
            m.beginModel(len(Vi), len(faces_i))
            m.addSubModel(Vi, faces_i)
            m.endModel()
            bvh_models.append(m)

    def _world_aabb_u(i):
        Vw = (rots[i] @ model_verts_scaled[i].T).T + centers[i] * inv_s
        v_min = Vw.min(axis=0); v_max = Vw.max(axis=0)
        return (float(v_min[0]), float(v_min[1]), float(v_min[2]),
                float(v_max[0]), float(v_max[1]), float(v_max[2]))

    def _make_fcl_obj(i):
        return fcl.CollisionObject(
            bvh_models[i],
            fcl.Transform(rots[i].astype(np.float64),
                          (centers[i] * inv_s).astype(np.float64)),
        )

    fcl_objs = [_make_fcl_obj(i) for i in range(N)]
    aabbs = [_world_aabb_u(i) for i in range(N)]
    near_thresh_u = d_hat * 1.5 * inv_s
    margin = 2.0 * d_hat * inv_s

    total_rot = 0.0
    for _ in range(max_rot_iters):
        any_updated = False
        for i in range(N):
            ax0, ay0, az0, ax1, ay1, az1 = aabbs[i]
            torque = np.zeros(3)
            n_contacts = 0
            for j in range(N):
                if j == i:
                    continue
                bx0, by0, bz0, bx1, by1, bz1 = aabbs[j]
                if (ax0 - margin > bx1 or ay0 - margin > by1 or az0 - margin > bz1 or
                        bx0 - margin > ax1 or by0 - margin > ay1 or bz0 - margin > az1):
                    continue
                req = fcl.DistanceRequest(enable_nearest_points=True,
                                          enable_signed_distance=True)
                res = fcl.DistanceResult()
                d_u = fcl.distance(fcl_objs[i], fcl_objs[j], req, res)
                if d_u > near_thresh_u:
                    continue
                if d_u > 0.0:
                    cp_i_u = np.asarray(res.nearest_points[0], dtype=np.float64)
                    cp_j_u = np.asarray(res.nearest_points[1], dtype=np.float64)
                    n_raw = cp_j_u - cp_i_u          # from i's surface toward j
                    cl_world = cp_i_u * scale
                    d_min_world = d_u * scale
                else:
                    creq = fcl.CollisionRequest(num_max_contacts=8, enable_contact=True)
                    cres = fcl.CollisionResult()
                    fcl.collide(fcl_objs[i], fcl_objs[j], creq, cres)
                    if not cres.is_collision or not cres.contacts:
                        continue
                    c_best = max(cres.contacts, key=lambda c: c.penetration_depth)
                    # The reported normal is the exit direction of body j
                    # (second argument) through the crossed triangle.
                    n_raw = np.asarray(c_best.normal, dtype=np.float64)
                    cl_world = np.asarray(c_best.pos, dtype=np.float64) * scale
                    d_min_world = -float(c_best.penetration_depth) * scale
                nn = np.linalg.norm(n_raw)
                if nn < 1e-12:
                    continue
                n_ij = n_raw / nn
                force_mag = max(0.0, d_hat - d_min_world)
                if force_mag <= 0.0:
                    continue
                torque += _repulsive_torque(cl_world - centers[i], n_ij, force_mag)
                n_contacts += 1

            if n_contacts > 0 and np.linalg.norm(torque) > 1e-8:
                omega = 0.5 * torque / n_contacts
                nw = np.linalg.norm(omega)
                max_step = 0.05  # ~3 degrees
                if nw > max_step:
                    omega *= max_step / nw
                    nw = max_step
                rots[i] = RotLib.from_rotvec(omega).as_matrix() @ rots[i]
                total_rot += nw
                any_updated = True
                fcl_objs[i] = _make_fcl_obj(i)
                aabbs[i] = _world_aabb_u(i)
        if not any_updated:
            break
    return total_rot


def _optimize_rotations_trimesh(centers, rots, nfs, mverts, mfaces, N,
                                  d_hat, scale, max_rot_iters):
    """Dependency-free closest-point path (vertex samples of the neighbour
    against body i's surface)."""
    import trimesh

    def world_verts_i(i):
        return scale * nfs[i] * (rots[i] @ mverts[i].T).T + centers[i]

    total_rot = 0.0
    meshes = [trimesh.Trimesh(vertices=world_verts_i(i), faces=mfaces[i], process=False)
              for i in range(N)]
    for _ in range(max_rot_iters):
        any_updated = False
        for i in range(N):
            vi = np.asarray(meshes[i].vertices)
            ai0, ai1 = vi.min(0), vi.max(0)
            torque = np.zeros(3)
            n_contacts = 0
            for j in range(N):
                if j == i:
                    continue
                vj = np.asarray(meshes[j].vertices)
                aj0, aj1 = vj.min(0), vj.max(0)
                if not (np.all(ai0 - d_hat * 2 <= aj1) and np.all(aj0 - d_hat * 2 <= ai1)):
                    continue
                cl, dists, _ = trimesh.proximity.closest_point(meshes[i], vj)
                near = dists < d_hat * 1.5
                if not near.any():
                    continue
                inside = np.zeros(len(vj), dtype=bool)
                inside[near] = _mesh_contains_points(meshes[i], vj[near])
                sd = np.where(inside, -dists, dists)
                k = int(np.argmin(sd))
                d_min = float(sd[k])
                if d_min < d_hat * 1.5:
                    n_ij = vj[k] - cl[k]            # from i's surface toward j
                    if inside[k]:
                        n_ij = -n_ij                # neighbour point inside i
                    nn = np.linalg.norm(n_ij)
                    if nn < 1e-12:
                        continue
                    n_ij = n_ij / nn
                    force_mag = max(0.0, d_hat - d_min)
                    if force_mag <= 0.0:
                        continue
                    torque += _repulsive_torque(cl[k] - centers[i], n_ij, force_mag)
                    n_contacts += 1
            if n_contacts > 0 and np.linalg.norm(torque) > 1e-8:
                omega = 0.5 * torque / n_contacts
                nw = np.linalg.norm(omega)
                max_step = 0.05
                if nw > max_step:
                    omega *= max_step / nw
                    nw = max_step
                rots[i] = RotLib.from_rotvec(omega).as_matrix() @ rots[i]
                total_rot += nw
                any_updated = True
                meshes[i] = trimesh.Trimesh(vertices=world_verts_i(i), faces=mfaces[i], process=False)
        if not any_updated:
            break
    return total_rot


# ─────────────────────────────────────────────────────────────────────
# Dual (contact-force space) QP
# ─────────────────────────────────────────────────────────────────────

def solve_dual_qp(A_contact, b_contact, P_diag, n_vars):
    """Solve the contact-only QP in dual (contact force) space.

    Primal: min ½ xᵀPx  s.t.  A x ≥ b
    Dual:   min ½ λᵀGλ − bᵀλ  s.t.  λ ≥ 0,   G = A P⁻¹ Aᵀ,   x* = P⁻¹ Aᵀ λ*.

    Only one-sided contact rows are representable here; the caller rejects
    walls, joints, rotation clamps and attraction terms before reaching this
    path. Returns (x_primal, lambda, qp_iters) or None when OSQP fails.
    """
    A = sp.csr_matrix(A_contact)
    n_constraints = A.shape[0]
    P_inv = sp.diags(1.0 / np.asarray(P_diag, dtype=np.float64), format='csr')
    G = (A @ P_inv @ A.T).tocsc()
    q_dual = -np.asarray(b_contact, dtype=np.float64)
    A_dual = sp.identity(n_constraints, format='csc')
    l_dual = np.zeros(n_constraints)
    u_dual = np.full(n_constraints, np.inf)
    result = osqp_solve(G, q_dual, A_dual, l_dual, u_dual, verbose=False,
                        eps_abs=1e-6, eps_rel=1e-6, max_iter=4000, polishing=True)
    if not qp_status_ok(result) or not np.all(np.isfinite(result.x)):
        return None
    lam = np.maximum(result.x, 0.0)
    x_primal = P_inv @ (A.T @ lam)
    return np.asarray(x_primal).reshape(-1), lam, result.info.iter


if __name__ == "__main__":
    import argparse

    # The canonical scene generators live in ../scenes (kubric/hy3d/thingi):
    #   python s4r/s4r_qp.py --dataset kubric --n-objects 40 --seed 42
    sys.path.insert(0, _HERE)
    sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "scenes"))

    parser = argparse.ArgumentParser(
        description="S4R-QP: Progressive Scaling + QP Solver")
    parser.add_argument('--dataset', choices=['kubric', 'hy3d', 'thingi'],
                        default='kubric',
                        help='Benchmark mesh pool (see ../data and the README)')
    parser.add_argument('--n-objects', '--N', dest='n_objects', type=int,
                        default=40, help='Number of objects')
    parser.add_argument('--seed', type=int, nargs='+', default=[42, 123, 456],
                        help='Random seed(s); the paper benchmark seeds are 42, 123, 456')
    parser.add_argument('--d-hat', type=float, default=0.02, help='Safety distance')
    parser.add_argument('--ds-max', type=float, default=0.05, help='Max scale step')
    parser.add_argument('--contact-backend', choices=['trimesh', 'fcl', 'fcl_prebuilt', 'warp'],
                        default='fcl_prebuilt',
                        help='Collision backend (fcl/fcl_prebuilt = exact FCL oracle; warp = NVIDIA Warp GPU)')
    parser.add_argument('--M', type=int, default=3,
                        help='Revalidation interval: full collision detection every M steps '
                             '(benchmark default M=3; M=1 detects every step)')
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--rotation', action='store_true',
                        help='Enable rotation DOFs (6-DOF QP + SO(3) refinement, slower)')
    parser.add_argument('--dual', action='store_true', help='Use the dual QP formulation')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    from make_scene import make_scene

    rot_str = "6DOF" if args.rotation else "3DOF"
    dual_str = "+dual" if args.dual else ""
    print(f"=== S4R-QP {rot_str}{dual_str} M={args.M} | "
          f"{args.dataset}(N={args.n_objects}) ===")
    print(f"{'seed':<7} {'pen':>4} {'RMSD':>8} {'time':>7} {'scale':>6}  status")
    print("-" * 50)

    for seed in args.seed:
        objects, _sxz, _sy = make_scene(args.dataset, args.n_objects, seed, 1.0)
        r = solve_s4r_qp(
            objects, d_hat=args.d_hat, ds_max=args.ds_max,
            max_steps=args.max_steps, verbose=args.verbose,
            enable_rotation=args.rotation,
            contact_sparsity=True, adaptive_ds=True,
            use_dual=args.dual,
            revalidate_interval=args.M,
            contact_backend=args.contact_backend)
        print(f"{seed:<7} {r['pen']:>4} {r['rmsd']:>8.4f} "
              f"{r['time']:>6.2f}s {r['scale']:>6.4f}  {r['status']}")
