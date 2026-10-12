"""Recalibrate the BatchNorm running statistics of a checkpoint under its own weights. An EMA checkpoint averages the
running statistics that were collected under the raw weights of the preceding iterations; this recomputes them under
the checkpoint's weights from training batches, the way training accumulates them (masked per-step batch statistics of
the encoder under bf16 autocast, the training augmentation, dropout off), as a cumulative average over the given
number of batches instead of the momentum average. Nothing but the BatchNorm buffers changes; the output is a copy of
the checkpoint for the standard evaluation driver.

Batches have the size of the training run's SyncBatchNorm batch (--global_batch: GPUs x per-GPU batch, e.g. 4 x 32 =
128 for S/A, 6 x 21 = 126 for B), and the rotation augmentation is drawn per per-GPU slice as in training, so every
BatchNorm call sees the batch composition of training. Single process only.

  CUDA_VISIBLE_DEVICES=0 python scripts/recalibrate_bn.py --config configs/movi_sphere.yaml \
      --use_multi_dataset --checkpoint=epoch_0005_ema.pth --global_batch 128 --batches 200 --out epoch_0005_ema_bnrecal.pth
"""
import argparse
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--batches', type=int, required=True)
    ap.add_argument('--global_batch', type=int, required=True, help="the training run's SyncBatchNorm batch")
    ap.add_argument('--out', type=str, required=True)
    own, rest = ap.parse_known_args()
    ck_args = [a for a in rest if a.startswith('--checkpoint')]
    if len(ck_args) != 1:
        raise ValueError('pass the checkpoint as --checkpoint=<path> (one argument)')
    ckpt_path = ck_args[0].split('=', 1)[1] if '=' in ck_args[0] else rest[rest.index(ck_args[0]) + 1]
    out_path = os.path.abspath(own.out)
    if os.path.abspath(ckpt_path) == out_path:
        raise ValueError('--out must differ from the checkpoint')
    sys.path.insert(0, REPO)
    os.chdir(REPO)
    import src.train as T  # noqa: E402
    from torch.nn.modules.batchnorm import _BatchNorm  # noqa: E402

    def patched_train(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            raise RuntimeError('recalibrate_bn.py runs in one process (use --global_batch for the training batch)')
        per_gpu = int(self.config['training']['train_batch_size'])
        if own.global_batch % per_gpu != 0:
            raise ValueError(f'--global_batch {own.global_batch} is not a multiple of the per-GPU batch {per_gpu}')
        src = self.train_loader
        loader = torch.utils.data.DataLoader(src.dataset, batch_size=own.global_batch, shuffle=True, drop_last=True,
                                             num_workers=src.num_workers, collate_fn=src.collate_fn, pin_memory=True)

        def augment(batch):
            # rotation drawn per per-GPU slice, as each training rank draws its own
            if self.gpu_augmentation is None:
                return batch
            op, rp = batch['obj_positions'], batch['ref_obj_positions']
            if op.shape[0] != own.global_batch or rp.shape[0] != own.global_batch:
                raise ValueError('unexpected batch layout for the per-slice augmentation')
            outs = [self.gpu_augmentation({'obj_positions': op[i:i + per_gpu], 'ref_obj_positions': rp[i:i + per_gpu]},
                                          split='train') for i in range(0, own.global_batch, per_gpu)]
            batch['obj_positions'] = torch.cat([o['obj_positions'] for o in outs], 0)
            batch['ref_obj_positions'] = torch.cat([o['ref_obj_positions'] for o in outs], 0)
            return batch
        model = self.model.module if hasattr(self.model, 'module') else self.model
        enc = model.shared_obj_encoder
        bns = {n: m for n, m in model.named_modules() if isinstance(m, _BatchNorm)}
        enc_bns = {id(m) for m in enc.modules() if isinstance(m, _BatchNorm)}
        if not bns or set(id(m) for m in bns.values()) != enc_bns:
            raise ValueError('expected every BatchNorm module inside the encoder')
        # Encoder in training mode (its masked BatchNorm path and running-statistics updates), dropout off,
        # everything else in evaluation mode (step-by-step forward)
        model.eval()
        enc.train()
        for m in enc.modules():
            if isinstance(m, torch.nn.Dropout):
                m.eval()
        momenta = {n: m.momentum for n, m in bns.items()}
        for m in bns.values():
            m.reset_running_stats()
            m.momentum = None  # cumulative average over all calls
        n_done = 0
        with torch.no_grad():
            while n_done < own.batches:
                for batch in loader:
                    batch = {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
                             for k, v in batch.items()}
                    batch = augment(batch)
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        model(batch, teacher_forcing_ratio=1.0)
                    n_done += 1
                    if n_done >= own.batches:
                        break
        for n, m in bns.items():
            m.momentum = momenta[n]
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd = ck['model_state_dict']
        prefix = 'module.' if any(k.startswith('module.') for k in sd) else ''
        msd = model.state_dict()
        changed = []
        for n in bns:
            for b in ('running_mean', 'running_var', 'num_batches_tracked'):
                key = f'{n}.{b}'
                if prefix + key not in sd:
                    raise KeyError(f'{prefix + key} missing in the checkpoint')
                old = sd[prefix + key]
                sd[prefix + key] = msd[key].detach().to(device='cpu', dtype=old.dtype).clone()
                if b != 'num_batches_tracked':
                    changed.append(f'  {key}: relative change '
                                   f'{float((sd[prefix + key].double() - old.double()).norm() / old.double().norm().clamp_min(1e-30)):.3e}')
        ck['bn_recalibration'] = {'batches': n_done, 'global_batch': own.global_batch, 'source': os.path.abspath(ckpt_path),
                                  'calls_per_module': int(msd[f'{next(iter(bns))}.num_batches_tracked'])}
        torch.save(ck, out_path)
        text = (f'BatchNorm statistics of {ckpt_path} recalibrated over {n_done} training batches of {own.global_batch} '
                f'({ck["bn_recalibration"]["calls_per_module"]} calls per module) -> {out_path}\n' + '\n'.join(changed))
        print(text, flush=True)
        open(out_path + '.txt', 'w').write(text + '\n')
        sys.exit(0)

    T.Trainer.train = patched_train
    sys.argv = [os.path.join(REPO, 'src', 'train.py')] + rest
    T.main()


if __name__ == '__main__':
    main()
