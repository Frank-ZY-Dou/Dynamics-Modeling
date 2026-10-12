"""grouped_masked_batch_norm == G sequential calls of the packed masked BatchNorm (RigidBodyEncoder._bn):
outputs, input and parameter gradients, running statistics and num_batches_tracked. CPU, fp64 reference
tolerance (the two paths reduce in different orders)."""
import copy

import torch

from src.models.encoder import grouped_masked_batch_norm


def _packed_bn(bn, x, mask):
    # the packed path of RigidBodyEncoder._bn (training mode)
    xt = x.transpose(1, 2)
    idx = mask.nonzero(as_tuple=True)
    y = bn(xt[idx].t().unsqueeze(0)).squeeze(0).t()
    out = xt.new_zeros(xt.shape)
    out[idx] = y.to(out.dtype)
    return out.transpose(1, 2)


def _case(G=6, M=7, C=5, N=11, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(G * M, C, N, generator=g) * 3 + 1.5
    mask = torch.rand(G * M, N, generator=g) > 0.35
    mask[0, :] = False                                   # a fully padded row
    return x, mask


def test_matches_sequential_packed_bn():
    torch.manual_seed(0)
    G, M, C, N = 6, 7, 5, 11
    x, mask = _case(G, M, C, N)
    bn_ref = torch.nn.BatchNorm1d(C)
    with torch.no_grad():
        bn_ref.weight.uniform_(0.5, 1.5)
        bn_ref.bias.uniform_(-0.5, 0.5)
        bn_ref.running_mean.uniform_(-1, 1)
        bn_ref.running_var.uniform_(0.5, 2.0)
    bn_new = copy.deepcopy(bn_ref)
    bn_ref.train()
    bn_new.train()
    gout = torch.randn(G * M, C, N)

    x1 = x.clone().requires_grad_(True)
    y_ref = torch.cat([_packed_bn(bn_ref, x1[g * M:(g + 1) * M], mask[g * M:(g + 1) * M]) for g in range(G)])
    (y_ref * gout).sum().backward()

    x2 = x.clone().requires_grad_(True)
    y_new = grouped_masked_batch_norm(bn_new, x2, mask, G)
    (y_new * gout).sum().backward()

    tol = dict(rtol=1e-5, atol=1e-5)
    assert torch.allclose(y_new, y_ref, **tol), (y_new - y_ref).abs().max()
    assert torch.allclose(x2.grad, x1.grad, **tol), (x2.grad - x1.grad).abs().max()
    assert torch.allclose(bn_new.weight.grad, bn_ref.weight.grad, **tol)
    assert torch.allclose(bn_new.bias.grad, bn_ref.bias.grad, **tol)
    assert torch.allclose(bn_new.running_mean, bn_ref.running_mean, rtol=1e-6, atol=1e-6)
    assert torch.allclose(bn_new.running_var, bn_ref.running_var, rtol=1e-6, atol=1e-6)
    assert int(bn_new.num_batches_tracked) == int(bn_ref.num_batches_tracked) == G
    # padded positions are exactly zero
    assert torch.all(y_new.transpose(1, 2)[~mask] == 0)


def test_single_group_equals_packed():
    x, mask = _case(G=1, M=9, seed=1)
    bn_ref = torch.nn.BatchNorm1d(5)
    bn_new = copy.deepcopy(bn_ref)
    y_ref = _packed_bn(bn_ref, x, mask)
    y_new = grouped_masked_batch_norm(bn_new, x, mask, 1)
    assert torch.allclose(y_new, y_ref, rtol=1e-5, atol=1e-5)
    assert torch.allclose(bn_new.running_var, bn_ref.running_var, rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    test_matches_sequential_packed_bn()
    test_single_group_equals_packed()
    print('grouped_masked_batch_norm tests: 2 passed')
