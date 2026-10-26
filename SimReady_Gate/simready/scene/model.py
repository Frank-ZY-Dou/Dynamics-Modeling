"""Simulator-agnostic scene layer.

A Body carries a triangle mesh in its own model frame, centred at the mesh's
AABB centre (the S4R reference centre), a world pose, and semantic tags.
Fixed bodies (fixtures, containers declared fixed) act as obstacles only.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np


def aabb_of(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return v.min(axis=0), v.max(axis=0)


@dataclass
class Body:
    name: str
    verts: np.ndarray            # (V,3) model frame, AABB-centred
    faces: np.ndarray            # (F,3) int
    center: np.ndarray           # (3,) world position of the reference centre
    rotation: np.ndarray         # (3,3) world rotation
    fixed: bool = False
    tags: set = field(default_factory=set)
    source: str = ""
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_mesh(cls, name, verts, faces, center=None, rotation=None, **kw):
        verts = np.asarray(verts, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int32)
        lo, hi = aabb_of(verts)
        c_model = 0.5 * (lo + hi)
        verts = verts - c_model
        rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64)
        # A mesh given in world coordinates has its reference centre at R c_model + t.
        center = (rotation @ c_model) if center is None else np.asarray(center, dtype=np.float64)
        return cls(name=name, verts=verts, faces=faces, center=center, rotation=rotation, **kw)

    def world_vertices(self, scale: float = 1.0) -> np.ndarray:
        return (self.rotation @ (scale * self.verts).T).T + self.center

    def world_aabb(self, scale: float = 1.0):
        return aabb_of(self.world_vertices(scale))

    def extent_along(self, n: np.ndarray, scale: float = 1.0) -> float:
        """Full extent of the (scaled, rotated) body along unit direction n."""
        p = (self.rotation @ (scale * self.verts).T).T @ n
        return float(p.max() - p.min())

    def support_offset(self, up: np.ndarray, scale: float = 1.0) -> float:
        """Distance from the reference centre down to the lowest vertex along up."""
        p = (self.rotation @ (scale * self.verts).T).T @ up
        return -float(p.min())

    @property
    def diag(self) -> float:
        lo, hi = aabb_of(self.verts)
        return float(np.linalg.norm(hi - lo))


@dataclass
class Scene:
    bodies: list
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    units: str = "m"
    meta: dict = field(default_factory=dict)

    def __getitem__(self, name: str) -> Body:
        for b in self.bodies:
            if b.name == name:
                return b
        raise KeyError(name)

    def names(self):
        return [b.name for b in self.bodies]

    def free(self):
        return [b for b in self.bodies if not b.fixed]

    def fixed_bodies(self):
        return [b for b in self.bodies if b.fixed]

    def support_height(self, support: Body, at=None, radius: float = 0.05, footprint=None) -> float:
        """Height of the support SURFACE under a body, along `up` (z up assumed).

        With `footprint` (an (m, 2) array of xy sample points, normally the xy of the body's
        lowest vertices), vertical lines through those points meet the support and the highest
        crossing is the seat height: the surface a body's bottom actually meets, on a coarse top
        face (no vertices under the body), on a rim-less shelf, or on the floor of a container.
        Without crossings the vertex rule is used: the highest support vertex within `radius` of
        `at`, else the global maximum."""
        if footprint is not None and len(footprint):
            hits = self._surface_hits(support, np.asarray(footprint, dtype=np.float64)[:, :2])
            if hits.size:
                return float(hits.max())
        if at is not None:
            hits = self._surface_hits(support, np.asarray(at, dtype=np.float64)[None, :2])
            if hits.size:
                return float(hits.max())
        v = support.world_vertices()
        h = v @ self.up
        if at is not None:
            d = np.linalg.norm(v[:, :2] - np.asarray(at, dtype=np.float64)[:2], axis=1)
            m = d <= radius
            if m.any():
                return float(h[m].max())
        return float(h.max())

    def _surface_hits(self, support: Body, xy: np.ndarray) -> np.ndarray:
        """Highest z of the support's surface on the vertical line through each xy, for the points whose line meets
        it (vertical_hits; empty if none does)."""
        z = self.vertical_hits(support, xy)
        return z[np.isfinite(z)]

    def crossings(self, body: Body, xy: np.ndarray, facing: bool = False):
        """(index of the point, world z) of every crossing of the vertical lines through `xy` with the body's surface,
        and with `facing` whether the face crossed there faces up.

        No rays are cast: the triangles are binned on a grid of the plane (_VerticalIndex) and each point is tested
        against the triangles of its cell. A body turned about z alone, as every body that a repair moves is, is
        indexed once in its own frame and the points are brought into it, so a new pose costs nothing; a body tilted
        otherwise is indexed in the world frame, again for each pose."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        R = body.rotation
        if abs(float(R[2, 2]) - 1.0) < 1e-9:
            index = vertical_index(body.verts, body.faces)
            out = index.crossings((xy - body.center[:2]) @ R[:2, :2], facing=facing)
            return (out[0], out[1] + float(body.center[2])) + out[2:]
        key = (tuple(np.round(body.center, 9)), tuple(np.round(R.ravel(), 9)), id(body.verts), body.verts.shape,
               id(body.faces), body.faces.shape)
        cache = body.meta.setdefault("_vertical_cache", {})
        if key not in cache:
            cache.clear()
            cache[key] = (_VerticalIndex(body.world_vertices(), body.faces), body.verts, body.faces)
        return cache[key][0].crossings(xy, facing=facing)

    def vertical_hits(self, body: Body, xy: np.ndarray, upward: bool = False, below: float | None = None) -> np.ndarray:
        """Per point of `xy`, the z where a vertical line through it meets the body's surface: the highest crossing
        (only those under `below` when given), or with `upward` the lowest; NaN where the line misses the body
        (see crossings)."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        out = np.full(len(xy), np.nan)
        if not len(xy):
            return out
        pi, z = self.crossings(body, xy)
        if not upward and below is not None:
            keep = z < below
            pi, z = pi[keep], z[keep]
        if len(pi):
            acc = np.full(len(xy), np.inf if upward else -np.inf)
            (np.minimum if upward else np.maximum).at(acc, pi, z)
            hit = np.isfinite(acc)
            out[hit] = acc[hit]
        return out


class _VerticalIndex:
    """A mesh's triangles binned on a grid of the plane by their xy bounding boxes, to find where vertical lines meet
    the mesh: a point is tested only against the triangles of its cell, by its barycentric coordinates in the plane
    (inclusive within 1e-9 on the edges, so that a line through an edge or a vertex meets it); a triangle seen edge-on
    from above has no area to meet and is left out."""

    def __init__(self, verts: np.ndarray, faces: np.ndarray):
        T = np.asarray(verts, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
        a, b, c = T[:, 0], T[:, 1], T[:, 2]
        d = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
        ext = np.ptp(T[:, :, :2], axis=1).max(axis=1)
        keep = np.abs(d) > 1e-12 * np.maximum(ext * ext, 1e-30)
        self.a, self.b, self.c, self.d = a[keep], b[keep], c[keep], d[keep]
        self.n = int(keep.sum())
        if not self.n:
            return
        lo, hi = T[keep][:, :, :2].min(axis=1), T[keep][:, :, :2].max(axis=1)
        self.origin = lo.min(axis=0)
        span = np.maximum(hi.max(axis=0) - self.origin, 1e-9)
        # a cell about twice the typical triangle, and at most 1024 cells along the longer side
        self.size = max(2.0 * float(np.median(hi - lo, axis=0).max()), float(span.max()) / 1024.0, 1e-9)
        self.shape = (int(span[0] / self.size) + 1, int(span[1] / self.size) + 1)
        i0 = ((lo - self.origin) / self.size).astype(np.int64)
        i1 = np.minimum(((hi - self.origin) / self.size).astype(np.int64), np.array(self.shape) - 1)
        nx, ny = i1[:, 0] - i0[:, 0] + 1, i1[:, 1] - i0[:, 1] + 1
        counts = nx * ny
        tri = np.repeat(np.arange(self.n), counts)
        k = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)
        cell = (i0[tri, 0] + k % nx[tri]) * self.shape[1] + (i0[tri, 1] + k // nx[tri])
        order = np.argsort(cell, kind="stable")
        self.tri = tri[order]
        self.ptr = np.searchsorted(cell[order], np.arange(self.shape[0] * self.shape[1] + 1))

    def crossings(self, xy: np.ndarray, facing: bool = False):
        """(index of the point, z) of every crossing of the vertical lines through `xy` (model frame) with the mesh, and
        with `facing` whether the triangle crossed there faces up (counter-clockwise seen from above)."""
        empty = (np.zeros(0, dtype=np.int64), np.zeros(0)) + ((np.zeros(0, dtype=bool),) if facing else ())
        if not self.n or not len(xy):
            return empty
        g = np.floor((xy - self.origin) / self.size).astype(np.int64)
        on = (g[:, 0] >= 0) & (g[:, 1] >= 0) & (g[:, 0] < self.shape[0]) & (g[:, 1] < self.shape[1])
        pts = np.nonzero(on)[0]
        cell = g[pts, 0] * self.shape[1] + g[pts, 1]
        first, cnt = self.ptr[cell], self.ptr[cell + 1] - self.ptr[cell]
        if not int(cnt.sum()):
            return empty
        pi = np.repeat(pts, cnt)
        ti = self.tri[np.repeat(first, cnt) + np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)]
        p, a, b, c, d = xy[pi], self.a[ti], self.b[ti], self.c[ti], self.d[ti]
        wa = ((b[:, 0] - p[:, 0]) * (c[:, 1] - p[:, 1]) - (c[:, 0] - p[:, 0]) * (b[:, 1] - p[:, 1])) / d
        wb = ((c[:, 0] - p[:, 0]) * (a[:, 1] - p[:, 1]) - (a[:, 0] - p[:, 0]) * (c[:, 1] - p[:, 1])) / d
        wc = 1.0 - wa - wb
        hit = (wa >= -1e-9) & (wb >= -1e-9) & (wc >= -1e-9)
        out = (pi[hit], (wa * a[:, 2] + wb * b[:, 2] + wc * c[:, 2])[hit])
        return out + ((d[hit] > 0),) if facing else out


_VINDEX: OrderedDict = OrderedDict()   # (identity of the vertex and face arrays) -> (index, verts, faces)
_VINDEX_FACES = 0
_VINDEX_MAX_FACES = 2_000_000          # least recently used out beyond this many triangles in all


def vertical_index(verts: np.ndarray, faces: np.ndarray) -> _VerticalIndex:
    """The _VerticalIndex of a mesh in its own frame, kept for the identity of its arrays (rebound, never edited in
    place, in this package; the entry keeps them alive so that the identity cannot be recycled)."""
    global _VINDEX_FACES
    key = (id(verts), id(faces), verts.shape, faces.shape)
    hit = _VINDEX.get(key)
    if hit is not None:
        _VINDEX.move_to_end(key)
        return hit[0]
    while _VINDEX and _VINDEX_FACES + len(faces) > _VINDEX_MAX_FACES:
        _, (_, _, old_faces) = _VINDEX.popitem(last=False)
        _VINDEX_FACES -= len(old_faces)
    index = _VerticalIndex(verts, faces)
    _VINDEX[key] = (index, verts, faces)
    _VINDEX_FACES += len(faces)
    return index


LEVEL_NORMAL = 0.99         # a face whose unit normal has at least this upward component can carry a body
LEVEL_MERGE = 0.004         # such faces within this height of each other are one level
LEVEL_MIN_AREA = 0.10       # a level covers at least this fraction of the fixture's footprint box
LEVEL_MIN_HEADROOM = 0.02   # and leaves at least this much room under the fixture's surface above it
LEVEL_GRID = 64             # a level is measured on a LEVEL_GRID x LEVEL_GRID grid over the box of its faces


@dataclass(frozen=True)
class Level:
    """A surface of a fixture that bodies can rest on (a shelf board, the floor of a compartment, the top), in the
    scene frame: its height, the bounding rectangle of its faces and the room above it up to the fixture's next
    surface over most of it (inf when nothing of the fixture is above most of it). A lip or a retaining rail over a
    strip of the level does not set its headroom: the contacts keep a body out from under it."""
    index: int
    z: float
    x0: float
    x1: float
    y0: float
    y1: float
    headroom: float


def support_levels(scene: Scene, body: Body) -> list:
    """The levels of a fixture, numbered from 1 at the bottom: its upward faces (LEVEL_NORMAL) grouped by height
    (LEVEL_MERGE), each group covering at least LEVEL_MIN_AREA of the fixture's footprint box. On a grid over the
    group's box (LEVEL_GRID), the level is where its surface is and the fixture does not cover the point just above it.
    It covers it when the first face above faces up (a plinth's top under a cabinet's bottom board: the board's top is
    the first face, as the board's bottom coincides with the plinth's top, and this holds for an open shell too), or
    when the faces crossed above, counted +1 facing up and -1 facing down, sum to one or more (a wall standing on the
    board, among parts that overlap each other). A group most of which is so covered is no level; a level's rectangle
    is the box of its free points, half a cell out, and its headroom their median room above, at least
    LEVEL_MIN_HEADROOM. Every level but the highest is under another part of the fixture over most of it. A table has
    its top (and the shelf under it, if it has one), a bookcase one level per shelf and its top. Wire decks, whose
    upward faces are the tops of wires, are not found. Kept on the body for its pose."""
    key = (tuple(np.round(body.center, 9)), tuple(np.round(body.rotation.ravel(), 9)), id(body.verts), id(body.faces))
    hit = body.meta.get("_levels")
    if hit is not None and hit[0] == key:
        return hit[1]
    V = body.world_vertices()
    T = V[np.asarray(body.faces, dtype=np.int64)]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    area = 0.5 * np.linalg.norm(n, axis=1)
    up = (area > 0) & (n[:, 2] >= LEVEL_NORMAL * 2.0 * np.maximum(area, 1e-300))
    T, area = T[up], area[up]
    z = T[:, :, 2].mean(axis=1)
    lo, hi = V.min(axis=0), V.max(axis=0)
    foot = float((hi[0] - lo[0]) * (hi[1] - lo[1]))
    out = []
    order = np.argsort(z, kind="stable")
    breaks = np.nonzero(np.diff(z[order]) > LEVEL_MERGE)[0] + 1
    for g in np.split(order, breaks) if len(order) else []:
        a = float(area[g].sum())
        if a < LEVEL_MIN_AREA * foot:
            continue
        zz = float(np.average(z[g], weights=area[g]))
        P = T[g][:, :, :2].reshape(-1, 2)
        lo2, hi2 = P.min(axis=0), P.max(axis=0)
        cell = (hi2 - lo2) / LEVEL_GRID
        gx, gy = np.meshgrid(lo2[0] + (np.arange(LEVEL_GRID) + 0.5) * cell[0], lo2[1] + (np.arange(LEVEL_GRID) + 0.5) * cell[1])
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        pi, zc, up_face = scene.crossings(body, pts, facing=True)
        on = np.zeros(len(pts), dtype=bool)
        on[pi[up_face & (np.abs(zc - zz) <= LEVEL_MERGE)]] = True       # the level's surface is at the point
        above = zc > zz + 1e-4
        pi, zc, up_face = pi[above], zc[above], up_face[above]
        # a line through an edge two triangles share meets both: one crossing of the surface
        _, keep = np.unique(np.column_stack([pi, np.round(zc * 1e7), up_face]), axis=0, return_index=True)
        pi, zc, up_face = pi[keep], zc[keep], up_face[keep]
        first = np.full(len(pts), math.inf)
        np.minimum.at(first, pi, zc)
        winding = np.zeros(len(pts))
        np.add.at(winding, pi, np.where(up_face, 1.0, -1.0))     # the solids of the fixture the point is in
        covered = winding > 0.5
        k = np.nonzero(zc == first[pi])[0]                       # the first crossing above each point
        covered[pi[k[up_face[k]]]] = True
        free = on & ~covered
        if not free.any() or covered[on].sum() * 2 > on.sum():
            continue
        room = float(np.median(first[free])) - zz
        if room < LEVEL_MIN_HEADROOM:
            continue
        (x0, y0), (x1, y1) = np.maximum(pts[free].min(axis=0) - 0.5 * cell, lo2), np.minimum(pts[free].max(axis=0) + 0.5 * cell, hi2)
        out.append((zz, float(x0), float(x1), float(y0), float(y1), room))
    # below the top, a level is under another part of the fixture (a shelf under a board): the bottom boards of a pallet,
    # open above between the slats of its deck, are none
    out = [v for k, v in enumerate(out) if k == len(out) - 1 or math.isfinite(v[5])]
    levels = [Level(k + 1, *v) for k, v in enumerate(out)]
    body.meta["_levels"] = (key, levels)
    return levels


def slab_proxy(body: Body, thickness: float = 0.05, up=None, top: float | None = None) -> Body:
    """A box spanning a support body's top face, for cheap repair-time contact.

    Verification should still use the full mesh; this proxy only carries the
    top surface and its edges, which is what tabletop repair needs.
    """
    import trimesh
    up = np.array([0.0, 0.0, 1.0]) if up is None else np.asarray(up, dtype=np.float64)
    v = body.world_vertices()
    lo, hi = v.min(axis=0), v.max(axis=0)
    top = float(hi[2]) if top is None else float(top)
    box = trimesh.creation.box(extents=(hi[0] - lo[0], hi[1] - lo[1], thickness))
    center = np.array([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), top - 0.5 * thickness])
    return Body.from_mesh(body.name, np.asarray(box.vertices), np.asarray(box.faces), center=center,
                          rotation=np.eye(3), fixed=True, tags=set(body.tags) | {"proxy"}, source=body.source,
                          meta=dict(body.meta, proxy_of=body.name))


SLAB_LEVEL_TOL = 0.005      # bodies seated this close to a support's top level rest on its top, and a slab stands in for it
PROXY_BOX_TOL_M = 0.01      # a decimated copy keeps every side of the body's box within this,
PROXY_BOX_TOL_FRAC = 0.05   # or within this fraction of the box's diagonal, whichever is larger


class MeshProxy:
    """Context manager: swap free bodies' meshes for decimated copies (and supports whose bodies
    rest on their top level for slabs) during repair, then restore the full meshes for
    verification. A decimated body keeps
    its own vertices in meta['full_verts'] meanwhile: the contacts run on the copy, footprints are
    measured on the mesh the predicates will see (decimation moves a silhouette by millimetres).

    A copy must keep the body's bounding box (PROXY_BOX_TOL_M, PROXY_BOX_TOL_FRAC): quadric
    decimation can collapse a whole flat part of a body made of several (a carton's sides under a
    detailed cap), and the repair seats a body by its copy's lowest point. A copy that loses part
    of the body is made again with twice the faces; a body no copy keeps runs on its own mesh.
    `notes` says which bodies needed either."""

    def __init__(self, scene: Scene, faces: int = 2000, slab_supports: bool = True, slab_tops: dict | None = None):
        self.scene, self.faces, self.slab_supports = scene, faces, slab_supports
        self.slab_tops = slab_tops or {}     # support name -> plate height (else the mesh max)
        self.saved = {}
        self.notes = []

    def __enter__(self):
        import trimesh
        for b in self.scene.bodies:
            self.saved[b.name] = (b.verts, b.faces, b.center.copy(), b.rotation.copy())
            if b.fixed:
                if self.slab_supports and "support" in b.tags:
                    top = self.slab_tops.get(b.name)
                    levels = support_levels(self.scene, b)
                    # a slab stands in for the top level only: a body resting on a lower one (a shelf of a bookcase)
                    # needs the boards and walls around it, so that fixture keeps its mesh
                    if top is None or not levels or top >= levels[-1].z - SLAB_LEVEL_TOL:
                        px = slab_proxy(b, top=top); b.verts, b.faces, b.center, b.rotation = px.verts, px.faces, px.center, px.rotation
                continue
            if len(b.faces) <= self.faces:
                continue
            full = trimesh.Trimesh(b.verts, b.faces, process=False)
            lo, hi = b.verts.min(0), b.verts.max(0)
            tol = max(PROXY_BOX_TOL_M, PROXY_BOX_TOL_FRAC * float(np.linalg.norm(hi - lo)))
            budget, copy = self.faces, None
            while budget < len(b.faces):
                try:
                    m = full.simplify_quadric_decimation(face_count=budget)
                except ImportError as e:             # trimesh decimates through fast_simplification
                    raise ImportError(f"mesh decimation needs fast-simplification ({e}): pip install fast-simplification") from e
                v = np.asarray(m.vertices, dtype=np.float64); f = np.asarray(m.faces, dtype=np.int32)
                used = v[f.reshape(-1)]                  # the box of what the faces keep, not of every vertex left
                if len(f) and max(float(np.abs(used.min(0) - lo).max()), float(np.abs(used.max(0) - hi).max())) <= tol:
                    copy = (v, f)
                    break
                budget *= 2
            if copy is None:
                self.notes.append(f"{b.name}: no decimated copy kept its bounding box; the repair ran on its {len(b.faces)} faces")
                continue
            if budget > self.faces:
                self.notes.append(f"{b.name}: a copy of {self.faces} faces lost part of the body; the repair ran on one of {budget}")
            b.meta["full_verts"] = b.verts      # what footprints are measured on meanwhile
            b.verts, b.faces = copy             # same frame: decimation keeps coordinates
        return self.scene

    def __exit__(self, *exc):
        for b in self.scene.bodies:
            v, f, c, R = self.saved[b.name]
            b.verts, b.faces = v, f
            b.meta.pop("full_verts", None)
            if b.fixed:
                b.center, b.rotation = c, R   # free bodies keep their repaired poses
        return False
