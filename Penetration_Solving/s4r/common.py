from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np


@dataclass
class SphereObject:
    """Toy convex proxy used by the included runnable examples.

    Replace this with your own mesh / convex-proxy representation when you plug
    in Drake, FCL, MIND-FCL, Bullet, or another collision backend.
    """

    name: str
    center: np.ndarray
    radius: float
    inv_mass: float = 1.0

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "center": [float(x) for x in self.center],
            "radius": float(self.radius),
            "inv_mass": float(self.inv_mass),
        }

    @staticmethod
    def from_json(data: dict) -> "SphereObject":
        return SphereObject(
            name=str(data["name"]),
            center=np.asarray(data["center"], dtype=float),
            radius=float(data["radius"]),
            inv_mass=float(data.get("inv_mass", 1.0)),
        )


@dataclass
class PairSignedDistance:
    i: int
    j: int
    signed_distance: float
    direction_ij: np.ndarray


@dataclass
class PairPenetration:
    i: int
    j: int
    depth: float
    direction_ij: np.ndarray


class SignedDistanceOracle(Protocol):
    def signed_distances(self, centers: np.ndarray) -> list[PairSignedDistance]:
        """Return all relevant pairwise signed distances.

        centers: (N, 3) array of candidate object translations.
        """


class PenetrationDepthOracle(Protocol):
    def penetrations(self, centers: np.ndarray) -> list[PairPenetration]:
        """Return penetrations for currently overlapping pairs only.

        direction_ij should point from object i toward object j, i.e. moving
        object i by -direction_ij and object j by +direction_ij reduces overlap.
        """


class SphereOracle(SignedDistanceOracle, PenetrationDepthOracle):
    """Runnable toy oracle for the provided examples.

    This is *not* the target production backend. It exists so the baseline code
    can run end-to-end without external collision libraries.
    """

    def __init__(self, objects: list[SphereObject]):
        self.objects = objects
        self.radii = np.asarray([obj.radius for obj in objects], dtype=float)

    def signed_distances(self, centers: np.ndarray) -> list[PairSignedDistance]:
        pairs: list[PairSignedDistance] = []
        n = len(self.objects)
        for i in range(n):
            for j in range(i + 1, n):
                delta = centers[j] - centers[i]
                dist = float(np.linalg.norm(delta))
                if dist < 1e-12:
                    # Arbitrary but deterministic fallback direction.
                    direction = np.array([1.0, 0.0, 0.0], dtype=float)
                else:
                    direction = delta / dist
                signed = dist - (self.radii[i] + self.radii[j])
                pairs.append(
                    PairSignedDistance(
                        i=i,
                        j=j,
                        signed_distance=float(signed),
                        direction_ij=direction,
                    )
                )
        return pairs

    def penetrations(self, centers: np.ndarray) -> list[PairPenetration]:
        out: list[PairPenetration] = []
        for pair in self.signed_distances(centers):
            if pair.signed_distance < 0.0:
                out.append(
                    PairPenetration(
                        i=pair.i,
                        j=pair.j,
                        depth=float(-pair.signed_distance),
                        direction_ij=pair.direction_ij,
                    )
                )
        return out


def flatten_centers(objects: Iterable[SphereObject]) -> np.ndarray:
    return np.stack([obj.center for obj in objects], axis=0)


def apply_centers(objects: list[SphereObject], centers: np.ndarray) -> list[SphereObject]:
    copied: list[SphereObject] = []
    for obj, center in zip(objects, centers):
        copied.append(
            SphereObject(
                name=obj.name,
                center=np.asarray(center, dtype=float),
                radius=obj.radius,
                inv_mass=obj.inv_mass,
            )
        )
    return copied


def scene_stats_from_signed_distances(pairs: Iterable[PairSignedDistance]) -> dict:
    signed = np.asarray([p.signed_distance for p in pairs], dtype=float)
    if signed.size == 0:
        return {
            "num_pairs": 0,
            "min_signed_distance": float("inf"),
            "num_violating_pairs": 0,
            "max_penetration": 0.0,
        }
    violations = np.maximum(-signed, 0.0)
    return {
        "num_pairs": int(signed.size),
        "min_signed_distance": float(np.min(signed)),
        "num_violating_pairs": int(np.sum(signed < 0.0)),
        "max_penetration": float(np.max(violations)),
    }


def load_sphere_scene(path: str | Path) -> list[SphereObject]:
    data = json.loads(Path(path).read_text())
    return [SphereObject.from_json(item) for item in data["objects"]]


def save_sphere_scene(path: str | Path, objects: Iterable[SphereObject]) -> None:
    payload = {"objects": [obj.to_json() for obj in objects]}
    Path(path).write_text(json.dumps(payload, indent=2))


def summarize_scene(objects: list[SphereObject]) -> dict:
    centers = flatten_centers(objects)
    oracle = SphereOracle(objects)
    stats = scene_stats_from_signed_distances(oracle.signed_distances(centers))
    stats.update(
        {
            "num_objects": len(objects),
            "bbox_min": [float(x) for x in np.min(centers, axis=0)],
            "bbox_max": [float(x) for x in np.max(centers, axis=0)],
        }
    )
    return stats
