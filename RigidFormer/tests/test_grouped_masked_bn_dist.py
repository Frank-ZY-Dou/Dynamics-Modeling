"""Distributed semantics of grouped_masked_batch_norm (CPU, gloo, 2 processes): two ranks holding half of
every group's rows must produce the same outputs, input gradients and summed parameter gradients as one
process holding all rows (the statistics and their gradients are synchronized across ranks)."""
import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from src.models.encoder import grouped_masked_batch_norm

G, M, C, N = 3, 8, 4, 9          # M rows per group in total; each rank holds M // 2 of them


def _data():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(G * M, C, N, generator=g) * 2 + 0.7
    mask = torch.rand(G * M, N, generator=g) > 0.3
    gout = torch.randn(G * M, C, N, generator=g)
    return x, mask, gout


def _rank_rows(rank):
    # rows of every group that this rank holds, in group-major order
    h = M // 2
    return torch.cat([torch.arange(g * M + rank * h, g * M + (rank + 1) * h) for g in range(G)])


def _worker(rank, world, init_file, out_file):
    dist.init_process_group('gloo', init_method=f'file://{init_file}', rank=rank, world_size=world)
    torch.manual_seed(0)
    bn = torch.nn.BatchNorm1d(C)
    with torch.no_grad():
        bn.weight.uniform_(0.5, 1.5)
        bn.bias.uniform_(-0.5, 0.5)
    x, mask, gout = _data()
    rows = _rank_rows(rank)
    xl = x[rows].clone().requires_grad_(True)
    y = grouped_masked_batch_norm(bn, xl, mask[rows], G)
    (y * gout[rows]).sum().backward()
    wg, bg = bn.weight.grad.clone(), bn.bias.grad.clone()
    dist.all_reduce(wg)
    dist.all_reduce(bg)
    torch.save({'rows': rows, 'y': y.detach(), 'xg': xl.grad, 'wg': wg, 'bg': bg,
                'rm': bn.running_mean.clone(), 'rv': bn.running_var.clone()}, f'{out_file}.{rank}')
    dist.destroy_process_group()


def test_two_ranks_equal_one_process():
    x, mask, gout = _data()
    torch.manual_seed(0)
    bn = torch.nn.BatchNorm1d(C)
    with torch.no_grad():
        bn.weight.uniform_(0.5, 1.5)
        bn.bias.uniform_(-0.5, 0.5)
    xr = x.clone().requires_grad_(True)
    # the single-process reference sees each group's rows in rank order: rank 0 half, then rank 1 half,
    # which is the original group-major order, so plain row indexing lines up
    y_ref = grouped_masked_batch_norm(bn, xr, mask, G)
    (y_ref * gout).sum().backward()
    with tempfile.TemporaryDirectory() as d:
        init_file, out_file = os.path.join(d, 'init'), os.path.join(d, 'out')
        mp.spawn(_worker, args=(2, init_file, out_file), nprocs=2, join=True)
        res = [torch.load(f'{out_file}.{r}') for r in range(2)]
    tol = dict(rtol=1e-5, atol=1e-5)
    for r in res:
        assert torch.allclose(r['y'], y_ref[r['rows']].detach(), **tol)
        assert torch.allclose(r['xg'], xr.grad[r['rows']], **tol), (r['xg'] - xr.grad[r['rows']]).abs().max()
        assert torch.allclose(r['rm'], bn.running_mean, rtol=1e-6, atol=1e-6)
        assert torch.allclose(r['rv'], bn.running_var, rtol=1e-6, atol=1e-6)
    assert torch.allclose(res[0]['wg'], bn.weight.grad, **tol)
    assert torch.allclose(res[0]['bg'], bn.bias.grad, **tol)


if __name__ == '__main__':
    test_two_ranks_equal_one_process()
    print('grouped_masked_batch_norm distributed test: passed')
