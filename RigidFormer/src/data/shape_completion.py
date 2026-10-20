"""Learned shape completion for the partial point clouds of data.partial_view (data.partial_view.completion).

A depth camera sees one side of an object, so contacts computed from the observed points miss the hidden side.
CompletionNet predicts points on the whole surface from the observed ones, using only what the camera provides: the
observed points in their view frame (origin at the centroid, z up, x horizontal towards the camera) and three
viewing-geometry values (centroid height above the floor, camera elevation and distance), never the object's pose,
shape or size. The predicted points are added to the observed ones and the union is farthest-point subsampled to
data.partial_view.max_points.

The network is trained on observations of training scenes (scripts/make_completion_data.py), with the same camera
augmentations as the training samples, against points on the objects' collision surfaces (scripts/train_completion.py).
"""
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .partial_view import view_frame

N_EXTRA = 3                         # view_frame's extra inputs


class CompletionNet(nn.Module):
    """PointNet encoder (two max-pooled stages, as in PCN) and an MLP decoder of n_out points, in the view frame."""

    def __init__(self, n_out: int = 512, latent: int = 512):
        super().__init__()
        self.n_out = n_out
        self.mlp1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.ReLU(), nn.Conv1d(64, 128, 1))
        self.mlp2 = nn.Sequential(nn.Conv1d(256, 256, 1), nn.ReLU(), nn.Conv1d(256, latent, 1))
        self.dec = nn.Sequential(nn.Linear(latent + N_EXTRA, 512), nn.ReLU(), nn.Linear(512, 1024), nn.ReLU(),
                                 nn.Linear(1024, 3 * n_out))

    def forward(self, pts: torch.Tensor, mask: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        """pts (B, N, 3), mask (B, N) of the valid points, extra (B, N_EXTRA) -> (B, n_out, 3)."""
        f = self.mlp1(pts.transpose(1, 2))
        low = torch.finfo(f.dtype).min
        g = f.masked_fill(~mask[:, None], low).amax(dim=-1, keepdim=True)
        f = self.mlp2(torch.cat([f, g.expand_as(f)], dim=1))
        g = f.masked_fill(~mask[:, None], low).amax(dim=-1)
        return self.dec(torch.cat([g, extra], dim=1)).view(-1, self.n_out, 3)


def chamfer(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean distances (B,) from every predicted point to the target cloud and from every target point to the
    prediction (L1 Chamfer terms)."""
    d = torch.cdist(pred, target)
    return d.min(dim=2).values.mean(dim=1), d.min(dim=1).values.mean(dim=1)


def surface_points(objects_dir: str, shape: str, k: int, seed: int = 0) -> np.ndarray:
    """k points uniformly distributed (by area) on a KuBasic collision mesh, asset units: the completion target."""
    import trimesh
    m = trimesh.load(Path(objects_dir) / shape / 'collision_geometry.obj', force='mesh', process=False)
    V, F = np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
    rng = np.random.default_rng(seed)
    f = rng.choice(len(F), size=k, p=area / area.sum())
    r1, r2 = rng.random(k), rng.random(k)
    s = np.sqrt(r1)
    return (1 - s)[:, None] * A[f] + (s * (1 - r2))[:, None] * B[f] + (s * r2)[:, None] * C[f]


def load_completion(path: str, device='cpu') -> Tuple[CompletionNet, dict]:
    """CompletionNet of a checkpoint, in eval mode on `device`, and the checkpoint dict."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    net = CompletionNet(**ck['net'])
    net.load_state_dict(ck['state_dict'])
    return net.to(device).eval(), ck


class Completer:
    """Completes observed clouds with a trained CompletionNet (CPU by default; loaded once per process)."""

    def __init__(self, path: str, input_points: int = 256, device: str = 'cpu'):
        self.path, self.input_points, self.device = str(path), int(input_points), device
        self._net, self._pid = None, None

    @property
    def net(self) -> CompletionNet:
        if self._net is None or self._pid != os.getpid():       # a forked DataLoader worker loads its own copy
            self._net, _ = load_completion(self.path, self.device)
            self._pid = os.getpid()
        return self._net

    def complete(self, clouds: Sequence[np.ndarray], camera: np.ndarray) -> List[np.ndarray]:
        """Predicted surface points (n_out, 3; world) of every observed cloud (n_i, 3; world, FPS order, so the first
        input_points form an FPS subset), seen from camera centre (3,)."""
        if not len(clouds):
            return []
        frames = [view_frame(X, camera) for X in clouds]
        n = max(min(len(X), self.input_points) for X in clouds)
        pts = np.zeros((len(clouds), n, 3), np.float32)
        mask = np.zeros((len(clouds), n), bool)
        for i, (X, (c, R, _)) in enumerate(zip(clouds, frames)):
            m = min(len(X), self.input_points)
            pts[i, :m] = (np.asarray(X[:m], np.float64) - c) @ R
            mask[i, :m] = True
        extra = np.stack([e for _, _, e in frames]).astype(np.float32)
        with torch.no_grad():
            out = self.net(torch.from_numpy(pts).to(self.device), torch.from_numpy(mask).to(self.device),
                           torch.from_numpy(extra).to(self.device)).cpu().numpy().astype(np.float64)
        return [c + Y @ R.T for Y, (c, R, _) in zip(out, frames)]
