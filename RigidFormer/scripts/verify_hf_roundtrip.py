"""Check that the Hugging Face dataset reproduces the conversion of the raw scenes exactly.

materialize_hf_dataset.py writes the training layout from the dataset's Parquet files; movi_to_sim_data.py writes it
from the complete raw scenes (data_generation/README.md). For every scene of the three datasets, each file the
converter writes must be equal in the two folders, byte for byte: per-frame vertices (frame_*/obj_*_pos.npy), faces,
the FPS anchor caches anchors/obj_*_anchors_fps<K>.npy, the pose trajectory vis_traj.npy and physics_params.json; the
two folders must hold exactly the files the scene's object and frame counts call for, and conversion.json must agree
except for its provenance fields (source, metadata_sha256). Caches written after conversion (the *_cf anchor and FPS-order caches, anchors of other
counts) are not compared.

    python scripts/verify_hf_roundtrip.py --materialized <--out of materialize_hf_dataset.py> \\
        --reference <folder with MOVi-Sphere, MOVi-A, MOVi-B converted from the raw scenes> --out <report.json>
"""
import argparse
import json
import os
import re
import sys
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'data_generation'))
from materialize_hf_dataset import FOLDERS  # noqa: E402
from movi_to_sim_data import ANCHOR_COUNTS  # noqa: E402

SPLITS = {'movi_sphere': 'dataset_split_MoviSphere_1200.json', 'movi_a': 'dataset_split_MoviA_1200.json',
          'movi_b': 'dataset_split_MoviB_1200.json'}
# the files the converter writes (other anchor counts are caches of the loader)
CONVERTER_FILE = re.compile(r'^(frame_\d{5}/obj_\d+_pos\.npy|meshes/obj_\d+_faces\.npy'
                            r'|anchors/obj_\d+_anchors_fps(' + '|'.join(map(str, ANCHOR_COUNTS)) + r')\.npy'
                            r'|physics_params\.json|vis_traj\.npy)$')
PROVENANCE = ('source', 'metadata_sha256')


def converter_files(root):
    out = set()
    for d, _, files in os.walk(root):
        for f in files:
            rel = os.path.relpath(os.path.join(d, f), root)
            if CONVERTER_FILE.match(rel):
                out.add(rel)
    return out


def expected_files(root):
    """The converter files a scene must have, from its conversion.json (object and frame counts)."""
    man = json.load(open(os.path.join(root, 'conversion.json')))
    n, T = man['exported_objects'], man['num_frames']
    files = {f'frame_{t:05d}/obj_{i}_pos.npy' for t in range(T) for i in range(n)}
    files |= {f'meshes/obj_{i}_faces.npy' for i in range(n)}
    files |= {f'anchors/obj_{i}_anchors_fps{K}.npy' for i in range(n) for K in ANCHOR_COUNTS}
    return files | {'physics_params.json', 'vis_traj.npy'}


def same_bytes(a, b):
    if os.path.getsize(a) != os.path.getsize(b):
        return False
    with open(a, 'rb') as fa, open(b, 'rb') as fb:
        return fa.read() == fb.read()


def check_scene(job):
    mat, ref, scene = job
    m, r = os.path.join(mat, scene), os.path.realpath(os.path.join(ref, scene))
    res = {'scene': scene, 'files': 0, 'differ': [], 'only_materialized': [], 'only_reference': [], 'manifest': None,
           'not_expected': []}
    fm, fr = converter_files(m), converter_files(r)
    res['only_materialized'] = sorted(fm - fr)
    res['only_reference'] = sorted(fr - fm)
    if os.path.isfile(os.path.join(m, 'conversion.json')):
        want = expected_files(m)
        res['not_expected'] = sorted((fm ^ want) | (fr ^ want))[:20]   # missing or extra on either side
    for rel in sorted(fm & fr):
        res['files'] += 1
        if not same_bytes(os.path.join(m, rel), os.path.join(r, rel)):
            res['differ'].append(rel)
    if not all(os.path.isfile(os.path.join(x, 'conversion.json')) for x in (m, r)):
        res['manifest'] = 'missing'
    else:
        a, b = (json.load(open(os.path.join(x, 'conversion.json'))) for x in (m, r))
        for k in PROVENANCE:
            a.pop(k, None)
            b.pop(k, None)
        res['manifest'] = 'equal' if a == b else 'differs'
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--materialized', required=True)
    ap.add_argument('--reference', required=True, help='folder with MOVi-Sphere, MOVi-A, MOVi-B')
    ap.add_argument('--out', required=True, help='report (JSON)')
    ap.add_argument('--configs', default='movi_sphere,movi_a,movi_b')
    ap.add_argument('--workers', type=int, default=32)
    args = ap.parse_args()
    report, ok = {}, True
    for cfg in [c.strip() for c in args.configs.split(',') if c.strip()]:
        folder = FOLDERS[cfg]
        split = json.load(open(os.path.join(REPO, SPLITS[cfg])))
        scenes = split['train_datasets'] + split['val_datasets'] + split['test_datasets']
        mat, ref = os.path.join(args.materialized, folder), os.path.join(args.reference, folder)
        extra = sorted(set(os.listdir(mat)) - set(scenes))
        with Pool(args.workers) as pool:
            res = pool.map(check_scene, [(mat, ref, s) for s in scenes], chunksize=4)
        bad = [x for x in res if x['differ'] or x['only_materialized'] or x['only_reference'] or x['not_expected']
               or x['manifest'] != 'equal']
        report[folder] = {'scenes': len(res), 'files_compared': sum(x['files'] for x in res), 'scenes_differing': len(bad),
                          'extra_materialized_scenes': extra, 'differing': bad[:50]}
        ok &= not bad and not extra
        print(folder, {k: v for k, v in report[folder].items() if k != 'differing'}, flush=True)
    with open(args.out, 'w') as fh:
        json.dump(report, fh, indent=1)
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
