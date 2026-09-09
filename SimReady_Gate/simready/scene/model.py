"""Simulator-agnostic scene layer.

A Body carries a triangle mesh in its own model frame, centred at the mesh's
AABB centre (the S4R reference centre), a world pose, and semantic tags.
Fixed bodies (fixtures, containers declared fixed) act as obstacles only.
"""
from __future__ import annotations

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
        """Highest z of the support surface hit by a downward ray at each xy (empty if none).

        The world-frame mesh and its ray intersector are cached on the support body. The cache
        key covers the pose (centre and rotation) and the identity of the vertex and face arrays,
        which the cache keeps alive so that the identity cannot be recycled: replacing a body's
        mesh (a proxy, a decimated copy) or moving it invalidates the entry. Arrays are rebound,
        never edited in place, everywhere in this package.
        """
        import trimesh
        from ..errors import GeometryQueryError
        key = (id(support), tuple(np.round(support.center, 9)), tuple(np.round(support.rotation.ravel(), 9)),
               id(support.verts), support.verts.shape, id(support.faces), support.faces.shape)
        cache = support.meta.setdefault("_ray_cache", {})
        if key not in cache:
            cache.clear()
            m = trimesh.Trimesh(support.world_vertices(), support.faces, process=False)
            cache[key] = (m, m.ray, support.verts, support.faces)
        m, ray, _, _ = cache[key]
        top = float(m.bounds[1][2]) + 1.0
        origins = np.column_stack([xy, np.full(len(xy), top)])
        dirs = np.tile(np.array([[0.0, 0.0, -1.0]]), (len(xy), 1))
        try:
            loc, idx_ray, _ = ray.intersects_location(origins, dirs, multiple_hits=True)
        except Exception as exc:  # a failed query is not a miss
            raise GeometryQueryError(f"downward ray query on {support.name} failed: {type(exc).__name__}: {exc}") from exc
        if len(loc) == 0:
            return np.zeros(0)
        best = {}
        for pnt, r in zip(loc, idx_ray):
            best[r] = max(best.get(r, -np.inf), float(pnt[2]))
        return np.array(list(best.values()))


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


class MeshProxy:
    """Context manager: swap free bodies' meshes for decimated copies (and supports for
    slabs) during repair, then restore the full meshes for verification."""

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
                        b.verts, b.faces = v, f       # same frame: decimation keeps coordinates
                except Exception:  # noqa: BLE001
                    pass
        return self.scene

    def __exit__(self, *exc):
        for b in self.scene.bodies:
            v, f, c, R = self.saved[b.name]
            b.verts, b.faces = v, f
            if b.fixed:
                b.center, b.rotation = c, R   # free bodies keep their repaired poses
        return False
