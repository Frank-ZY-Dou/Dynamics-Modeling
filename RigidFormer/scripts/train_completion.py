"""Train the completion network of data.partial_view.completion (src/data/shape_completion.py) on the observations of
scripts/make_completion_data.py: L1 Chamfer distance, in the view frame, between the predicted points and points
spread uniformly on the posed collision surface of the observed object. The checkpoint with the lowest Chamfer distance
on the validation observations is kept.

    python scripts/train_completion.py --train completion/train.npz --val completion/val.npz \\
        --out checkpoints/completion/completion.pth [--epochs 40 --batch 256 --lr 1e-3]
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.data.shape_completion import CompletionNet, chamfer, surface_points  # noqa: E402

OBJECTS = str(REPO / 'data_generation' / 'objects')


def load(path, device):
    d = np.load(path)
    n = torch.from_numpy(d['n'].astype(np.int64))
    pts = torch.from_numpy(d['pts'].astype(np.float32))
    return {'pts': pts.to(device), 'mask': (torch.arange(pts.shape[1])[None] < n[:, None]).to(device),
            'extra': torch.from_numpy(d['extra']).to(device),
            'shape': torch.from_numpy(d['shape'].astype(np.int64)).to(device),
            'size': torch.from_numpy(d['size']).to(device), 'rot': torch.from_numpy(d['rot']).to(device),
            'trans': torch.from_numpy(d['trans']).to(device), 'shapes': [str(s) for s in d['shapes']],
            'split': str(d['split']), 'm': int(len(n))}


def targets(data, idx, surface, k, gen):
    """Points on the posed collision surfaces (len(idx), k, 3) in the view frame."""
    u = surface[data['shape'][idx]]                                       # (b, S, 3)
    sel = torch.randint(u.shape[1], (k,), generator=gen, device=u.device)
    u = u[:, sel]
    return data['trans'][idx, None] + data['size'][idx, None, None] * u @ data['rot'][idx].transpose(1, 2)


def evaluate(net, data, surface, batch, gen):
    net.eval()
    tot = torch.zeros(2, device=data['pts'].device)
    per = torch.zeros(len(data['shapes']), 3, device=data['pts'].device)
    with torch.no_grad():
        for b0 in range(0, data['m'], batch):
            idx = torch.arange(b0, min(b0 + batch, data['m']), device=data['pts'].device)
            pred = net(data['pts'][idx], data['mask'][idx], data['extra'][idx])
            a, b = chamfer(pred, targets(data, idx, surface, 2048, gen))
            tot += torch.stack([a.sum(), b.sum()])
            per.index_add_(0, data['shape'][idx], torch.stack([a, b, torch.ones_like(a)], dim=1))
    net.train()
    return (tot / data['m']).tolist(), {s: (per[i, :2] / per[i, 2].clamp(min=1)).tolist()
                                        for i, s in enumerate(data['shapes']) if per[i, 2] > 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train', required=True)
    ap.add_argument('--val', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--n_out', type=int, default=512)
    ap.add_argument('--latent', type=int, default=512)
    ap.add_argument('--target_points', type=int, default=1024)
    ap.add_argument('--limit', type=int, default=0, help='use only the first N training records (quick checks)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr, va = load(args.train, device), load(args.val, device)
    if tr['split'] != 'train' or va['split'] != 'val':
        raise SystemExit(f'--train must hold train records and --val val records; got {tr["split"]} and {va["split"]}')
    if tr['shapes'] != va['shapes']:
        raise SystemExit('the two files list different shapes')
    if args.limit:
        tr = {k: (v[:args.limit] if torch.is_tensor(v) else v) for k, v in tr.items()}
        tr['m'] = min(tr['m'], args.limit)
    surface = torch.from_numpy(np.stack([surface_points(OBJECTS, s, 8192, seed=i) for i, s in enumerate(tr['shapes'])])
                               ).float().to(device)
    net = CompletionNet(n_out=args.n_out, latent=args.latent).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * math.ceil(tr['m'] / args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(i / steps, 1))))
    gen = torch.Generator(device=device).manual_seed(args.seed)
    vgen = torch.Generator(device=device)
    print(f'{tr["m"]} train / {va["m"]} val records, {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M parameters, '
          f'{steps} steps on {device}', flush=True)
    best, t0 = math.inf, time.time()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        perm = torch.randperm(tr['m'], generator=gen, device=device)
        run = torch.zeros(2, device=device)
        for b0 in range(0, tr['m'], args.batch):
            idx = perm[b0:b0 + args.batch]
            pred = net(tr['pts'][idx], tr['mask'][idx], tr['extra'][idx])
            a, b = chamfer(pred, targets(tr, idx, surface, args.target_points, gen))
            loss = (a + b).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += torch.stack([a.sum(), b.sum()]).detach()
        vgen.manual_seed(1)
        (va_a, va_b), per = evaluate(net, va, surface, args.batch, vgen)
        tr_a, tr_b = (run / tr['m']).tolist()
        mark = ''
        if va_a + va_b < best:
            best = va_a + va_b
            torch.save({'net': {'n_out': args.n_out, 'latent': args.latent}, 'state_dict': net.state_dict(),
                        'epoch': epoch, 'val_chamfer': [va_a, va_b], 'val_per_shape': per, 'args': vars(args),
                        'input_points': int(tr['pts'].shape[1])}, args.out)
            mark = ' *'
        print(f'epoch {epoch:3d} ({time.time() - t0:5.0f} s): train {tr_a:.4f} / {tr_b:.4f}, val {va_a:.4f} / '
              f'{va_b:.4f} m (pred->surface / surface->pred){mark}', flush=True)
    ck = torch.load(args.out, map_location='cpu', weights_only=False)
    print(f'kept epoch {ck["epoch"]}: val {ck["val_chamfer"][0]:.4f} / {ck["val_chamfer"][1]:.4f} m; per shape '
          + ', '.join(f'{s} {v[0]:.3f}/{v[1]:.3f}' for s, v in ck['val_per_shape'].items()))


if __name__ == '__main__':
    main()
