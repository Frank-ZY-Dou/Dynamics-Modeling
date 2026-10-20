"""Tests of the training augmentations (src/data/gpu_augmentation.py) and of the orientation loss's error term
(src/utils/anchor_rotation.py). CPU only.

    python -m pytest tests/test_gpu_augmentation.py -q
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.gpu_augmentation import GPUAugmentation, rotvec_to_matrix  # noqa: E402
from src.utils.anchor_rotation import centered_anchor_error  # noqa: E402

CPU = torch.device('cpu')


def make_config(**aug):
    return {'data': {'augmentation': aug}}


def kabsch(P, Q):
    """Proper rotation R and translation t with Q ~ P @ R.T + t (P, Q: (N, 3) float64)."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, qc - pc @ R.T


def random_quat_rotations(n, rng):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=1).reshape(n, 3, 3)


def make_batch(B=6, T=8, Nobj=3, Nv=12, seed=0, step_sizes=None):
    """Rigid objects (reference shapes moved by random poses per frame), padded: object 2 of every sample has only 7
    valid vertices, and the last object of sample 0 is padding."""
    rng = np.random.default_rng(seed)
    ref = rng.normal(scale=0.3, size=(B, Nobj, Nv, 3))
    pos = np.zeros((B, T, Nobj, Nv, 3))
    for b in range(B):
        for t in range(T):
            R = random_quat_rotations(Nobj, rng)
            p = rng.normal(scale=2.0, size=(Nobj, 3)) + np.array([0, 0, 1.0])
            pos[b, t] = np.einsum('oij,onj->oni', R, ref[b]) + p[:, None]
    vmask = np.ones((B, T, Nobj, Nv), dtype=bool)
    vmask[:, :, 2, 7:] = False
    omask = np.ones((B, Nobj), dtype=bool)
    omask[0, -1] = False
    vmask[0, :, -1] = False
    pos[~vmask] = 0.0                                   # PADDING_VALUE
    ref[~vmask[:, 0]] = 0.0
    batch = {
        'obj_positions': torch.tensor(pos, dtype=torch.float32),
        'ref_obj_positions': torch.tensor(ref, dtype=torch.float32),
        'valid_vtx_mask': torch.tensor(vmask),
        'valid_obj_mask': torch.tensor(omask),
        'skip_step_size': torch.tensor(step_sizes if step_sizes is not None else [1.0] * B, dtype=torch.float32),
    }
    return batch


def clone(batch):
    return {k: v.clone() for k, v in batch.items()}


def pairwise(x):
    return torch.cdist(x.double(), x.double())


def test_rotvec_to_matrix():
    rv = torch.tensor([[0.0, 0.0, math.pi / 2], [1e-12, 0.0, 0.0], [0.3, -0.2, 0.9]], dtype=torch.float64)
    R = rotvec_to_matrix(rv)
    assert torch.allclose(R[0], torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64),
                          atol=1e-12)
    assert torch.allclose(R[1], torch.eye(3, dtype=torch.float64), atol=1e-12)
    eye = torch.eye(3, dtype=torch.float64).expand(3, 3, 3)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-12)
    assert torch.allclose(torch.linalg.det(R), torch.ones(3, dtype=torch.float64), atol=1e-12)
    # the rotation axis is left fixed and the angle is the vector's norm
    axis = rv[2] / rv[2].norm()
    assert torch.allclose(R[2] @ axis, axis, atol=1e-12)
    angle = torch.arccos((torch.trace(R[2]) - 1) / 2)
    assert abs(angle.item() - rv[2].norm().item()) < 1e-9


def test_per_sample_rotation_is_a_yaw_per_sample():
    torch.manual_seed(0)
    aug = GPUAugmentation(make_config(rotation_augmentation={'enable': True, 'per_sample': True,
                                                             'prob_use_rotation': 1.0}), CPU)
    batch = make_batch()
    out = aug(clone(batch), split='train')
    x0, x1 = batch['obj_positions'], out['obj_positions']
    B = x0.shape[0]
    # heights unchanged, all distances inside a sample unchanged (one rigid motion of the whole scene)
    assert torch.allclose(x0[..., 2], x1[..., 2], atol=1e-6)
    for b in range(B):
        a = x0[b].reshape(-1, 3)
        c = x1[b].reshape(-1, 3)
        assert torch.allclose(pairwise(a), pairwise(c), atol=1e-4)
    angles = out['rotation_angles_deg_per_sample']
    assert angles.shape == (B,) and len(set(np.round(angles.numpy(), 3))) == B   # one angle per sample
    # every sample is turned about z by its own angle, positions and reference shapes alike
    for b in range(B):
        th = math.radians(angles[b].item())
        Rz = torch.tensor([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]])
        assert torch.allclose(x0[b] @ Rz.T, x1[b], atol=1e-5)
        assert torch.allclose(batch['ref_obj_positions'][b] @ Rz.T, out['ref_obj_positions'][b], atol=1e-5)
    # not applied outside training
    ev = aug(clone(batch), split='test')
    assert torch.equal(ev['obj_positions'], batch['obj_positions'])


def test_augmented_tensors_are_contiguous():
    """The model reshapes the augmented positions with .view(), which needs contiguous tensors."""
    aug = GPUAugmentation(make_config(rotation_augmentation={'enable': True, 'per_sample': True, 'prob_use_rotation': 1.0},
                                      reflection_augmentation={'enable': True, 'prob': 0.5},
                                      rigid_noise={'enable': True, 'pos_std': 0.001, 'rot_std_deg': 0.1}), CPU)
    out = aug(clone(make_batch()), split='train')
    for k in ('obj_positions', 'ref_obj_positions', 'obj_positions_input'):
        assert out[k].is_contiguous(), k
        out[k].view(-1, 3)


def test_per_sample_rotation_probability():
    torch.manual_seed(1)
    aug = GPUAugmentation(make_config(rotation_augmentation={'enable': True, 'per_sample': True,
                                                             'prob_use_rotation': 0.0}), CPU)
    batch = make_batch()
    out = aug(clone(batch), split='train')
    assert torch.allclose(out['obj_positions'], batch['obj_positions'])


def test_reflection_keeps_every_object_rigid_with_proper_rotations():
    torch.manual_seed(2)
    aug = GPUAugmentation(make_config(reflection_augmentation={'enable': True, 'prob': 1.0}), CPU)
    batch = make_batch()
    out = aug(clone(batch), split='train')
    x0, x1 = batch['obj_positions'], out['obj_positions']
    assert out['reflected'].all()
    assert torch.allclose(x1[..., 0], -x0[..., 0]) and torch.allclose(x1[..., 1:], x0[..., 1:])
    assert torch.allclose(out['ref_obj_positions'][..., 0], -batch['ref_obj_positions'][..., 0])
    # each mirrored frame is a proper rigid motion (rotation, det +1) of the mirrored reference shape: the mirrored
    # sequence is again a sequence of rigid-body poses
    ref = out['ref_obj_positions'].double().numpy()
    pos = x1.double().numpy()
    vm = batch['valid_vtx_mask'].numpy()
    om = batch['valid_obj_mask'].numpy()
    B, T, Nobj = pos.shape[:3]
    for b in range(B):
        for o in range(Nobj):
            if not om[b, o]:
                continue
            m = vm[b, 0, o]
            for t in range(T):
                R, tr = kabsch(ref[b, o, m], pos[b, t, o, m])
                assert np.linalg.det(R) > 0.999
                assert np.abs(ref[b, o, m] @ R.T + tr - pos[b, t, o, m]).max() < 1e-5


def test_reflection_probability_zero_and_eval():
    aug = GPUAugmentation(make_config(reflection_augmentation={'enable': True, 'prob': 0.0}), CPU)
    batch = make_batch()
    assert torch.equal(aug(clone(batch), split='train')['obj_positions'], batch['obj_positions'])
    aug1 = GPUAugmentation(make_config(reflection_augmentation={'enable': True, 'prob': 1.0}), CPU)
    assert torch.equal(aug1(clone(batch), split='val')['obj_positions'], batch['obj_positions'])


def test_rigid_noise_perturbs_inputs_rigidly_and_keeps_targets():
    torch.manual_seed(3)
    aug = GPUAugmentation(make_config(rigid_noise={'enable': True, 'pos_std': 0.01, 'rot_std_deg': 2.0}), CPU)
    batch = make_batch()
    out = aug(clone(batch), split='train')
    clean, noisy = batch['obj_positions'], out['obj_positions_input']
    vm = batch['valid_vtx_mask']
    assert torch.equal(out['obj_positions'], clean)                     # the loss targets stay ground truth
    assert torch.equal(noisy[:, -1], clean[:, -1])                      # the last frame is only a target
    assert torch.equal(noisy[~vm], clean[~vm])                          # padding untouched
    assert not torch.allclose(noisy[:, :-1][vm[:, :-1]], clean[:, :-1][vm[:, :-1]])
    # every perturbed object is a rigid, proper copy of the clean one
    B, T, Nobj = clean.shape[:3]
    c, n = clean.double().numpy(), noisy.double().numpy()
    for b in range(B):
        for t in range(T - 1):
            for o in range(Nobj):
                m = vm[b, t, o].numpy()
                if m.sum() < 3:
                    continue
                R, tr = kabsch(c[b, t, o, m], n[b, t, o, m])
                assert np.linalg.det(R) > 0.999
                assert np.abs(c[b, t, o, m] @ R.T + tr - n[b, t, o, m]).max() < 1e-4


def test_rigid_noise_magnitude_and_step_size_scaling():
    """The perturbation at the last input frame has the configured standard deviations (step size 1), scaled by
    step_size ** step_size_exponent."""
    torch.manual_seed(4)
    pos_std, rot_deg = 0.002, 0.3
    aug = GPUAugmentation(make_config(rigid_noise={'enable': True, 'pos_std': pos_std, 'rot_std_deg': rot_deg,
                                                   'step_size_exponent': 1.0}), CPU)
    B, T, Nobj, Nv = 256, 8, 4, 6
    for s in (1.0, 5.0):
        batch = make_batch(B=B, T=T, Nobj=Nobj, Nv=Nv, seed=5, step_sizes=[s] * B)
        batch['valid_vtx_mask'][:] = True
        batch['valid_obj_mask'][:] = True
        out = aug(clone(batch), split='train')
        clean, noisy = batch['obj_positions'][:, T - 2].double(), out['obj_positions_input'][:, T - 2].double()
        shifts, angles = [], []
        for b in range(B):
            for o in range(Nobj):
                R, _ = kabsch(clean[b, o].numpy(), noisy[b, o].numpy())
                angles.append(math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
                shifts.append((noisy[b, o].mean(0) - clean[b, o].mean(0)).numpy())
        shifts = np.array(shifts)
        assert abs(shifts.std() / (pos_std * s) - 1) < 0.1
        # rotation angle of an isotropic Gaussian rotation vector: RMS = sqrt(3) * per-axis std
        rms_angle = math.sqrt(np.mean(np.square(angles)))
        assert abs(rms_angle / (math.sqrt(3) * rot_deg * s) - 1) < 0.1


def test_rigid_noise_off_by_default_and_outside_training():
    batch = make_batch()
    aug = GPUAugmentation(make_config(), CPU)
    assert 'obj_positions_input' not in aug(clone(batch), split='train')
    aug1 = GPUAugmentation(make_config(rigid_noise={'enable': True, 'pos_std': 0.01, 'rot_std_deg': 1.0}), CPU)
    assert 'obj_positions_input' not in aug1(clone(batch), split='test')


def test_rigid_noise_rejects_scene_centring():
    cfg = make_config(rigid_noise={'enable': True, 'pos_std': 0.01, 'rot_std_deg': 1.0})
    cfg['data']['center_scene_by_t0'] = True
    with pytest.raises(ValueError):
        GPUAugmentation(cfg, CPU)


def test_centered_anchor_error_measures_rotation_independent_of_size():
    theta = math.radians(1.0)
    Rz = torch.tensor([[math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]],
                      dtype=torch.float64)
    anchors = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [-1.0, 0, 0], [0, -1.0, 0]], dtype=torch.float64)
    for size in (0.05, 1.0, 3.0):
        gt = anchors * size + torch.tensor([2.0, -1.0, 0.5], dtype=torch.float64)
        # pure translation error: zero
        assert centered_anchor_error(gt + 0.3, gt).abs().max() < 1e-12
        # rotation error about the centroid: the RMS norm is 2 sin(theta / 2) for anchors perpendicular to the axis
        c = gt.mean(0)
        pred = (gt - c) @ Rz.T + c + 0.1
        err = centered_anchor_error(pred, gt)
        rms = err.pow(2).sum(-1).mean().sqrt().item()
        assert abs(rms - 2 * math.sin(theta / 2)) < 1e-9
