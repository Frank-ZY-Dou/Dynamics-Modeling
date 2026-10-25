"""Simulator-agnostic scene layer.

A Body carries a triangle mesh in its own model frame, centred at the mesh's
AABB centre (the S4R reference centre), a world pose, and semantic tags.
Fixed bodies (fixtures, containers declared fixed) act as obstacles only.
"""
from __future__ import annotations

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
        """Height of the support SURFACE under a body, along `up` (z up assumed for rays).

        With `footprint` (an (m, 2) array of xy sample points, normally the xy of the body's
        lowest vertices), downward rays are cast at those points and the highest hit is the seat
        height: the surface a body's bottom actually meets, on a coarse top face (no vertices
        under the body), on a rim-less shelf, or on the floor of a container. Without hits the
        vertex rule is used: the highest support vertex within `radius` of `at`, else the global
        maximum."""
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

    def vertical_hits(self, body: Body, xy: np.ndarray, upward: bool = False, below: float | None = None) -> np.ndarray:
        """Per point of `xy`, the z where a vertical line through it meets the body's surface: the highest crossing
        (only those under `below` when given), or with `upward` the lowest; NaN where the line misses the body.

        No rays are cast: the triangles are binned on a grid of the plane (_VerticalIndex) and each point is tested
        against the triangles of its cell. A body turned about z alone, as every body that a repair moves is, is
        indexed once in its own frame and the points are brought into it, so a new pose costs nothing; a body tilted
        otherwise is indexed in the world frame, again for each pose."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        out = np.full(len(xy), np.nan)
        if not len(xy):
            return out
        R = body.rotation
        if abs(float(R[2, 2]) - 1.0) < 1e-9:
            index = vertical_index(body.verts, body.faces)
            pi, z = index.crossings((xy - body.center[:2]) @ R[:2, :2])
            z = z + float(body.center[2])
        else:
            key = (tuple(np.round(body.center, 9)), tuple(np.round(R.ravel(), 9)), id(body.verts), body.verts.shape,
                   id(body.faces), body.faces.shape)
            cache = body.meta.setdefault("_vertical_cache", {})
            if key not in cache:
                cache.clear()
                cache[key] = (_VerticalIndex(body.world_vertices(), body.faces), body.verts, body.faces)
            pi, z = cache[key][0].crossings(xy)
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

    def crossings(self, xy: np.ndarray):
        """(index of the point, z) of every crossing of the vertical lines through `xy` (model frame) with the mesh."""
        empty = (np.zeros(0, dtype=np.int64), np.zeros(0))
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
        return pi[hit], (wa * a[:, 2] + wb * b[:, 2] + wc * c[:, 2])[hit]


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


_WARNED = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        import warnings
        warnings.warn(message, RuntimeWarning, stacklevel=3)


class MeshProxy:
    """Context manager: swap free bodies' meshes for decimated copies (and supports for
    slabs) during repair, then restore the full meshes for verification. A decimated body keeps
    its own vertices in meta['full_verts'] meanwhile: the contacts run on the copy, footprints are
    measured on the mesh the predicates will see (decimation moves a silhouette by millimetres)."""

    def __init__(self, scene: Scene, faces: int = 2000, slab_supports: bool = True, slab_tops: dict | None = None):
        self.scene, self.faces, self.slab_supports = scene, faces, slab_supports
        self.slab_tops = slab_tops or {}     # support name -> plate height (else the mesh max)
        self.saved = {}

    def __enter__(self):
        import trimesh
        for b in self.scene.bodies:
            self.saved[b.name] = (b.verts, b.faces, b.center.copy(), b.rotation.copy())
            if b.fixed:
                if self.slab_supports and "support" in b.tags:
                    px = slab_proxy(b, top=self.slab_tops.get(b.name)); b.verts, b.faces, b.center, b.rotation = px.verts, px.faces, px.center, px.rotation
                continue
            if len(b.faces) > self.faces:
                try:
                    m = trimesh.Trimesh(b.verts, b.faces, process=False).simplify_quadric_decimation(face_count=self.faces)
                    v = np.asarray(m.vertices, dtype=np.float64); f = np.asarray(m.faces, dtype=np.int32)
                    if len(f) > 20:
                        b.meta["full_verts"] = b.verts      # what footprints are measured on meanwhile
                        b.verts, b.faces = v, f       # same frame: decimation keeps coordinates
                except ImportError as e:             # trimesh decimates through fast_simplification
                    _warn_once("decimation", f"mesh decimation is unavailable ({e}); the repair runs on the full meshes, "
                                             "slower and with more memory (pip install fast-simplification)")
                except Exception:  # noqa: BLE001
                    pass
        return self.scene

    def __exit__(self, *exc):
        for b in self.scene.bodies:
            v, f, c, R = self.saved[b.name]
            b.verts, b.faces = v, f
            b.meta.pop("full_verts", None)
            if b.fixed:
                b.center, b.rotation = c, R   # free bodies keep their repaired poses
        return False
