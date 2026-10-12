"""Stage trained RigidFormer models as one Hugging Face model repository (a local folder; uploading is a separate step).

Per dataset, two steps, so that the card reports an evaluation of the very file that is released:

  weights   writes rigidformer_<dataset>.pth (the model weights, the configuration from --config and the training
            epoch; every tensor checked equal to the checkpoint's) and config.yaml. The configuration must equal the one
            the checkpoint was trained with in every setting except the run name and the data file names.
  card      checks that an eval.sh record evaluates exactly these files (its checkpoint and config SHA-256 lines),
            then writes eval.txt (without local paths) and a model card README.md for this model, whose table is read
            from it.

Then, once for all datasets:

  repo      checks the staged folders again (weights, configuration and record) and writes the repository: the weights
            files at the top level, their configurations under configs/, their evaluation records under eval/ and one
            model card for all of them.

    python scripts/package_hf_model.py weights --checkpoint <run>/epoch_0100_ema.pth --config configs/movi_b.yaml \\
        --dataset movi_b --out hf_models/RigidFormer-MOVi-B
    CHECKPOINT=hf_models/RigidFormer-MOVi-B/rigidformer_movib.pth BASE_CONFIG=hf_models/RigidFormer-MOVi-B/config.yaml \\
        TARGETS=B OUTPUT_FILE=eval_B.txt bash scripts/eval.sh
    python scripts/package_hf_model.py card --dir hf_models/RigidFormer-MOVi-B --eval eval_B.txt --dataset movi_b \\
        --repo_id frankzydou/RigidFormer
    # the same for movi_sphere and movi_a, then
    python scripts/package_hf_model.py repo --dirs hf_models/RigidFormer-MOVi-Sphere hf_models/RigidFormer-MOVi-A \\
        hf_models/RigidFormer-MOVi-B --repo_id frankzydou/RigidFormer --out hf_models/RigidFormer
"""
import argparse
import hashlib
import math
import os
import re
import shutil
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
DATASETS = {'movi_sphere': ('MOVi-Sphere', 'S'), 'movi_a': ('MOVi-A', 'A'), 'movi_b': ('MOVi-B', 'B')}
DATASET_REPO = 'frankzydou/RigidFormer-MOVi'
CODE = 'https://github.com/Frank-ZY-Dou/Dynamics-Modeling/tree/main/RigidFormer'
# settings that may differ between the training configuration and --config (names), and defaults the training run adds
RENAMED = {'experiment_tag', 'data.dataset_path', 'data.split_file', 'data.object_scale_file'}
RUNTIME_DEFAULTS = {'data.augmentation.dt': 1.0}
CITATION = """@inproceedings{dou2026rigidformer,
  title     = {RigidFormer: Learning Rigid Dynamics using Transformers},
  author    = {Dou, Zhiyang and Guo, Minghao and Wu, Haixu and Roble, Doug and Stuyck, Tuur and Matusik, Wojciech},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}"""


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def weights_name(dataset):
    return f"rigidformer_{dataset.replace('_', '')}.pth"


def flat(d, prefix=''):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out.update(flat(v, f'{prefix}{k}.'))
        else:
            out[f'{prefix}{k}'] = v
    return out


def config_differences(trained, given):
    """Settings in which the configuration given differs from the training configuration (names excepted)."""
    a, b = flat(trained), flat(given)
    diff = []
    for k in sorted(set(a) | set(b)):
        if k in RENAMED:
            continue
        if k not in b and k in RUNTIME_DEFAULTS and a[k] == RUNTIME_DEFAULTS[k]:
            continue
        if a.get(k, '<absent>') != b.get(k, '<absent>'):
            diff.append(f"{k}: trained {a.get(k, '<absent>')!r}, given {b.get(k, '<absent>')!r}")
    return diff


def public_record(text, fname, cfg_name):
    """The eval.sh record as shipped: the checkpoint named by its released file, the configuration by its path in the
    code (the caller checks that they are the files evaluated), no local log folders."""
    out = []
    for line in text.splitlines():
        if line.startswith('Checkpoint:'):
            line = re.sub(r'^(Checkpoint:\s*)\S+$', lambda m: m.group(1) + f'checkpoints/{fname}', line)
        elif line.startswith('Base config:'):
            line = re.sub(r'^(Base config:\s*)\S+$', lambda m: m.group(1) + f'configs/{cfg_name}', line)
        elif line.strip().startswith('"log_dir":'):
            continue
        out.append(line)
    text = '\n'.join(out) + '\n'
    if re.search(r'(?<![\w.])/(data|home|tmp|mnt|scratch)/', text):
        raise ValueError('the evaluation record still names a local path')
    return text


def read_eval(path, key):
    """Header fields and the step-size-1 rows (position RMSE, rotation error) of one target in an eval.sh record."""
    text = Path(path).read_text()
    head = dict(re.findall(r'^(Base config|Checkpoint|Checkpoint sha256|Config sha256|Split|Estimator):\s*(.+)$',
                           text.split('Target dataset:')[0], re.M))
    block = re.search(rf'Target dataset: \S+\s+\(key={key}\)(.*?)(?=Target dataset:|Machine-readable|\Z)', text, re.S)
    if block is None:
        raise ValueError(f'{path}: no results for target {key}')
    s1 = re.search(r'--- step_size = 1 ---\n(.*?)(?=\n--- |\Z)', block.group(1), re.S)
    if s1 is None:
        raise ValueError(f'{path}: no step size 1 results for target {key}')
    lines = s1.group(1).splitlines()
    rows = {}
    for i, line in enumerate(lines):
        for name in ('Pos RMSE (m)', 'Rot Err (deg)'):
            if line.strip().startswith(name):
                cols = [c.strip() for c in line.split('|')[1:]]
                vals = [v.strip() for v in lines[i + 1].split('|')[1:]]
                rows[name] = dict(zip(cols, vals))
    for name, row in rows.items():
        for col, v in row.items():
            try:
                ok = math.isfinite(float(v))
            except ValueError:
                ok = False
            if not ok:
                raise ValueError(f'{path}: {name} {col} is {v!r}, not a finite number')
    protocol = re.search(r'protocol: (.+)', s1.group(1))
    return head, rows, protocol.group(1).strip() if protocol else ''


def cmd_weights(args):
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f'{out} exists: stage into a new folder')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = yaml.safe_load(open(args.config))
    if not isinstance(ckpt.get('config'), dict):
        raise ValueError(f'{args.checkpoint} holds no training configuration to check {args.config} against')
    diff = config_differences(ckpt['config'], config)
    if diff:
        raise ValueError(f'{args.config} differs from the training configuration: ' + '; '.join(diff[:10]))
    out.mkdir(parents=True)
    fname = weights_name(args.dataset)
    weights = {'model_state_dict': ckpt['model_state_dict'], 'config': config}
    if 'epoch' in ckpt:
        weights['epoch'] = ckpt['epoch']
    torch.save(weights, out / fname)
    back = torch.load(out / fname, map_location='cpu', weights_only=True)
    src = ckpt['model_state_dict']
    if back['model_state_dict'].keys() != src.keys() or not all(torch.equal(back['model_state_dict'][k], src[k]) for k in src):
        raise RuntimeError('the weights file differs from the checkpoint')
    if back['config'] != config:
        raise RuntimeError('the configuration in the weights file differs from --config')
    shutil.copyfile(args.config, out / 'config.yaml')
    print(f'{out / fname}  sha256 {sha256(out / fname)}\n{out / "config.yaml"}  sha256 {sha256(out / "config.yaml")}')
    print(f'next: CHECKPOINT={out / fname} BASE_CONFIG={out / "config.yaml"} TARGETS={DATASETS[args.dataset][1]} '
          f'OUTPUT_FILE=<record> bash scripts/eval.sh, then the card step')


def cmd_card(args):
    d = Path(args.dir)
    name, key = DATASETS[args.dataset]
    fname = weights_name(args.dataset)
    head, rows, protocol = read_eval(args.eval, key)
    if head.get('Split') != 'test':
        raise ValueError(f"{args.eval} is on the {head.get('Split')} split")
    if head.get('Checkpoint sha256') != sha256(d / fname):
        raise ValueError(f'{args.eval} evaluates another checkpoint than {d / fname}')
    if head.get('Config sha256') != sha256(d / 'config.yaml'):
        raise ValueError(f'{args.eval} was run with another configuration than {d / "config.yaml"}')
    cfg_name = {'movi_sphere': 'movi_sphere.yaml', 'movi_a': 'movi_a.yaml', 'movi_b': 'movi_b.yaml'}[args.dataset]
    if sha256(d / 'config.yaml') != sha256(REPO / 'configs' / cfg_name):
        raise ValueError(f'{d / "config.yaml"} differs from configs/{cfg_name}: the card would name the wrong configuration')
    (d / 'eval.txt').write_text(public_record(Path(args.eval).read_text(), fname, cfg_name))
    src = torch.load(d / fname, map_location='cpu', weights_only=True)['model_state_dict']
    n_params = sum(v.numel() for v in src.values() if torch.is_tensor(v) and v.is_floating_point())
    cols = ['@ 25', '@ 50', '@100', '@200', '@300', '@400']
    pos, rot = rows['Pos RMSE (m)'], rows['Rot Err (deg)']
    table = ('| | ' + ' | '.join(c.replace(' ', '') for c in cols) + ' |\n|---|' + '---|' * len(cols) + '\n'
             + '| position RMSE (m) | ' + ' | '.join(pos[c] for c in cols) + ' |\n'
             + '| rotation error (deg) | ' + ' | '.join(rot[c] for c in cols) + ' |')
    card = f"""---
license: mit
library_name: pytorch
tags:
- physics
- rigid-body-dynamics
- simulation
- learned-simulator
datasets:
- {DATASET_REPO}
---

# RigidFormer, {name}

RigidFormer ([paper](https://arxiv.org/abs/2605.09196), [code]({CODE})) learns multi-object rigid-body dynamics with an
object-centric Transformer. This model was trained on {name} ([dataset](https://huggingface.co/datasets/{DATASET_REPO}),
config `{args.dataset}`) with `configs/{cfg_name}`; {n_params / 1e6:.1f} M parameters.

## Results

Test split, step size 1, rolled out from four ground-truth frames; error at physical frames 25 to 400:

{table}

Protocol: {protocol}. The full record, with step sizes 5 and 10 and the rotation MAE, is `eval.txt`.

## Usage

From the RigidFormer code folder, with the data prepared as in the dataset card:

```bash
hf download {args.repo_id} {fname} --local-dir checkpoints
CHECKPOINT=checkpoints/{fname} BASE_CONFIG=configs/{cfg_name} TARGETS={key} bash scripts/eval.sh
```

`{fname}`: SHA-256 `{sha256(d / fname)}`.

## Citation

```bibtex
{CITATION}
```
"""
    (d / 'README.md').write_text(card)
    print('card written', d, sorted(os.listdir(d)))


def staged_model(d):
    """(dataset, weights file name, configuration name) of a folder written by the weights and card steps, after checking
    that its eval.txt evaluates exactly its weights file and config.yaml on the test split, and that config.yaml is the
    code's configuration of that dataset."""
    d = Path(d)
    head = dict(re.findall(r'^(Base config|Checkpoint|Checkpoint sha256|Config sha256|Split|Estimator):\s*(.+)$',
                           (d / 'eval.txt').read_text().split('Target dataset:')[0], re.M))
    cfg_name = Path(head.get('Base config', '')).name
    dataset = cfg_name[:-len('.yaml')]
    if dataset not in DATASETS or head['Base config'] != f'configs/{cfg_name}':
        raise ValueError(f"{d / 'eval.txt'}: unexpected configuration {head.get('Base config')!r}")
    fname = weights_name(dataset)
    if head.get('Checkpoint') != f'checkpoints/{fname}':
        raise ValueError(f"{d / 'eval.txt'} names the checkpoint {head.get('Checkpoint')!r}, not checkpoints/{fname}")
    if head.get('Split') != 'test':
        raise ValueError(f"{d / 'eval.txt'} is on the {head.get('Split')} split")
    if head.get('Checkpoint sha256') != sha256(d / fname):
        raise ValueError(f"{d / 'eval.txt'} evaluates another checkpoint than {d / fname}")
    if head.get('Config sha256') != sha256(d / 'config.yaml'):
        raise ValueError(f"{d / 'eval.txt'} was run with another configuration than {d / 'config.yaml'}")
    if sha256(d / 'config.yaml') != sha256(REPO / 'configs' / cfg_name):
        raise ValueError(f"{d / 'config.yaml'} differs from configs/{cfg_name}")
    return dataset, fname, cfg_name


def cmd_repo(args):
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f'{out} exists: write the repository into a new folder')
    models = {}
    for d in args.dirs:
        dataset, fname, cfg_name = staged_model(d)
        if dataset in models:
            raise ValueError(f'two folders for {dataset}: {models[dataset][0]} and {d}')
        models[dataset] = (Path(d), fname, cfg_name)
    if set(models) != set(DATASETS):
        raise ValueError(f'one folder per dataset expected ({sorted(DATASETS)}), got {sorted(models)}')
    order = ['movi_sphere', 'movi_a', 'movi_b']
    (out / 'configs').mkdir(parents=True)
    (out / 'eval').mkdir()
    cols = ['@ 25', '@ 50', '@ 75', '@100', '@200', '@300', '@400']
    results, protocols, sizes, digests = [], set(), set(), {}
    for dataset in order:
        d, fname, cfg_name = models[dataset]
        name, key = DATASETS[dataset]
        for src, dst in ((d / fname, out / fname), (d / 'config.yaml', out / 'configs' / cfg_name),
                         (d / 'eval.txt', out / 'eval' / f'{dataset}.txt')):
            shutil.copyfile(src, dst)
            if sha256(dst) != sha256(src):
                raise RuntimeError(f'the copy of {src} differs from it')
        digests[fname] = sha256(out / fname)
        _, rows, protocol = read_eval(out / 'eval' / f'{dataset}.txt', key)
        pos, rot = rows['Pos RMSE (m)'], rows['Rot Err (deg)']
        results.append(f'| {name} | position RMSE (m) | ' + ' | '.join(pos[c] for c in cols) + ' |\n'
                       + '| | rotation error (deg) | ' + ' | '.join(rot[c] for c in cols) + ' |')
        protocols.add(re.sub(r'\s*\(dataset_split_\w+_1200\.json\)', '', protocol))
        src = torch.load(out / fname, map_location='cpu', weights_only=True)['model_state_dict']
        sizes.add(sum(v.numel() for v in src.values() if torch.is_tensor(v) and v.is_floating_point()))
    if len(protocols) != 1:
        raise ValueError(f'the evaluation protocols differ beyond the split file: {sorted(protocols)}')
    if len(sizes) != 1:
        raise ValueError(f'the models differ in size: {sorted(sizes)}')
    protocol, n_params = protocols.pop(), sizes.pop()
    models_table = '| Weights | Trained on | Configuration |\n|---|---|---|\n' + '\n'.join(
        f'| `{models[k][1]}` | {DATASETS[k][0]} | `configs/{models[k][2]}` |' for k in order)
    table = ('| Model | | ' + ' | '.join(c.replace(' ', '') for c in cols) + ' |\n|---|---|' + '---|' * len(cols) + '\n'
             + '\n'.join(results))
    width = max(len(models[k][1]) for k in order)
    usage = '\n'.join(f'CHECKPOINT=checkpoints/{models[k][1]:<{width}} BASE_CONFIG=configs/{models[k][2]:<16} '
                      f'TARGETS={DATASETS[k][1]} bash scripts/eval.sh' for k in order)
    files = ' '.join(models[k][1] for k in order)
    sums = '\n'.join(f'- `{models[k][1]}`: `{digests[models[k][1]]}`' for k in order)
    card = f"""---
license: mit
library_name: pytorch
tags:
- physics
- rigid-body-dynamics
- simulation
- learned-simulator
datasets:
- {DATASET_REPO}
---

# RigidFormer

RigidFormer ([paper](https://arxiv.org/abs/2605.09196), [code]({CODE})) learns multi-object rigid-body dynamics with an
object-centric Transformer. This repository holds one pretrained model per dataset of
[RigidFormer-MOVi](https://huggingface.co/datasets/{DATASET_REPO}), {n_params / 1e6:.1f} M parameters each:

{models_table}

`configs/` holds the configurations the models were trained and evaluated with; they are the files of the same name in
the code's `configs/` folder.

## Results

Test split, step size 1, rolled out from four ground-truth frames; error at physical frames 25 to 400:

{table}

Protocol: {protocol}. The full records, with step sizes 5 and 10 and the rotation MAE, are in `eval/`.

## Usage

From the RigidFormer code folder, with the data prepared as in the dataset card:

```bash
hf download {args.repo_id} {files} --local-dir checkpoints
{usage}
```

SHA-256 of the weights files:

{sums}

## Citation

```bibtex
{CITATION}
```
"""
    (out / 'README.md').write_text(card)
    print('repository written', out, sorted(os.listdir(out)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='step', required=True)
    w = sub.add_parser('weights', help='write the weights file and config.yaml')
    w.add_argument('--checkpoint', required=True, help='trained checkpoint (with its training configuration)')
    w.add_argument('--config', required=True, help='the configuration to release and evaluate the model with')
    w.add_argument('--dataset', required=True, choices=sorted(DATASETS))
    w.add_argument('--out', required=True, help='new folder')
    c = sub.add_parser('card', help='check an evaluation of the staged files and write eval.txt and the card')
    c.add_argument('--dir', required=True, help='folder written by the weights step')
    c.add_argument('--eval', required=True, help='scripts/eval.sh record of the staged weights file and config.yaml')
    c.add_argument('--dataset', required=True, choices=sorted(DATASETS))
    c.add_argument('--repo_id', required=True, help='model repository, e.g. frankzydou/RigidFormer')
    r = sub.add_parser('repo', help='write the model repository of all datasets from the staged folders')
    r.add_argument('--dirs', required=True, nargs='+', help='folders written by the weights and card steps, one per dataset')
    r.add_argument('--repo_id', required=True, help='model repository, e.g. frankzydou/RigidFormer')
    r.add_argument('--out', required=True, help='new folder')
    args = ap.parse_args()
    {'weights': cmd_weights, 'card': cmd_card, 'repo': cmd_repo}[args.step](args)


if __name__ == '__main__':
    main()
