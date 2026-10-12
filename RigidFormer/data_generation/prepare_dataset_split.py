#!/usr/bin/env python3
"""
Create a reproducible train/validation/test split over converted scene directories.

The splits of MOVi-Sphere, MOVi-A and MOVi-B ship at the repository root (dataset_split_*_1200.json); this tool is for
building splits over newly generated data (complete scenes only).

Usage:
  python data_generation/prepare_dataset_split.py \
    --data_path ./datasets/MOVi-B --name_pattern "movi_*" \
    --output dataset_split_custom.json
"""

import os
import json
import glob
import random
import argparse
from pathlib import Path


def _count_frames(dataset_dir: Path) -> int:
    """Count number of frame_* subdirectories inside a dataset directory."""
    if not dataset_dir.exists():
        return 0
    # Only count immediate subdirectories named frame_*
    return len([p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith('frame_')])


def create_dataset_split(
    data_path,
    train_ratio=0.9,
    val_ratio=0.1,
    seed=42,
    output_file='dataset_split.json',
    min_frames: int = 0,
    name_pattern: str = 'movi_*'
):
    """
    Create a reproducible train/validation/test split over the complete converted scenes.

    Only scenes whose conversion.json (written by movi_to_sim_data.py) marks them complete are used: a scene whose
    metadata omits simulated bodies has exported trajectories driven by bodies the model never sees. The split is
    n_train = round(N * train_ratio), n_val = round(N * val_ratio), n_test = the rest, after a seeded shuffle; the
    three lists are disjoint. The output declares scene_set 'complete' (data.scene_set in the configs).

    Args:
        data_path: parent directory containing converted scene folders
        train_ratio: fraction of the complete scenes for training (e.g., 0.8)
        val_ratio: fraction for validation (e.g., 0.1); the remainder is the test split
        seed: Random seed for reproducibility
        output_file: Output JSON file to save the split
        min_frames: scenes with fewer frames are left out (listed)
        name_pattern: glob pattern of the scene directories

    Returns:
        The split dictionary
    """
    if not (0.0 < train_ratio and 0.0 <= val_ratio and train_ratio + val_ratio <= 1.0):
        raise ValueError(f"need 0 < train_ratio, 0 <= val_ratio, train_ratio + val_ratio <= 1; got {train_ratio}, {val_ratio}")
    dataset_dirs = [
        d for d in sorted(glob.glob(str(Path(data_path) / name_pattern)))
        if os.path.isdir(d)
    ]
    if not dataset_dirs:
        raise FileNotFoundError(f"no scene directories matching '{name_pattern}' in {data_path}")

    kept, filtered_out, incomplete = [], [], []
    for d in dataset_dirs:
        manifest = Path(d) / 'conversion.json'
        if not manifest.exists():
            raise FileNotFoundError(f"{manifest} missing: convert with data_generation/movi_to_sim_data.py")
        man = json.loads(manifest.read_text())
        n_frames = _count_frames(Path(d))
        if n_frames != man['num_frames']:
            raise RuntimeError(f"{d}: {n_frames} frame directories, the manifest records {man['num_frames']}")
        if not man['complete']:
            incomplete.append((os.path.basename(d), man['simulated_objects'], man['exported_objects']))
        elif min_frames and n_frames < min_frames:
            filtered_out.append((d, n_frames))
        else:
            kept.append(d)

    dataset_names = [os.path.basename(d) for d in kept]
    num_datasets = len(dataset_names)
    print(f"Found {len(dataset_dirs)} scenes in {data_path} (pattern='{name_pattern}'); {num_datasets} complete and used")
    if incomplete:
        print(f"Left out {len(incomplete)} incomplete scenes (simulated / exported objects): "
              + ', '.join(f"{n} ({s}/{e})" for n, s, e in incomplete[:10]))
    if filtered_out:
        print(f"Left out {len(filtered_out)} scenes with < {min_frames} frames:")
        for d, n in filtered_out[:10]:  # show first 10
            print(f"  - {os.path.basename(d)} (frames={n})")
    if num_datasets == 0:
        raise RuntimeError("no complete scenes to split")

    # Seeded shuffle, then train / val / test blocks
    indices = list(range(num_datasets))
    random.seed(seed)
    random.shuffle(indices)
    n_train = int(round(num_datasets * train_ratio))
    n_val = int(round(num_datasets * val_ratio))
    n_val = min(n_val, num_datasets - n_train)
    train_indices = sorted(indices[:n_train])
    val_indices = sorted(indices[n_train:n_train + n_val])
    test_indices = sorted(indices[n_train + n_val:])
    train_datasets = [dataset_names[i] for i in train_indices]
    val_datasets = [dataset_names[i] for i in val_indices]
    test_datasets = [dataset_names[i] for i in test_indices]
    if set(train_datasets) & set(val_datasets) or set(train_datasets) & set(test_datasets) or set(val_datasets) & set(test_datasets):
        raise AssertionError("train/val/test splits overlap")

    split_info = {
        'scene_set': 'complete',
        'metadata': {
            'total_datasets': num_datasets,
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': 1.0 - train_ratio - val_ratio,
            'seed': seed,
            'n_train': len(train_datasets),
            'n_val': len(val_datasets),
            'n_test': len(test_datasets),
            'min_frames': min_frames,
            'name_pattern': name_pattern,
            'incomplete_left_out': [n for n, _, _ in incomplete],
        },
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices,
        'train_datasets': train_datasets,
        'val_datasets': val_datasets,
        'test_datasets': test_datasets
    }

    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(split_info, f, indent=2)

    print(f"\nDataset split saved to: {output_path}")
    print(f"Train: {len(train_datasets)}  Val: {len(val_datasets)}  Test: {len(test_datasets)}")
    print(f"\nFirst 5 training datasets: {train_datasets[:5]}")
    print(f"First 5 validation datasets: {val_datasets[:5]}")
    print(f"First 5 test datasets: {test_datasets[:5]}")

    return split_info


def load_dataset_split(split_file='dataset_split.json'):
    """
    Load a previously saved dataset split.
    
    Args:
        split_file: Path to the JSON file containing the split
    
    Returns:
        Dictionary with train and validation dataset information
    """
    with open(split_file, 'r') as f:
        split_info = json.load(f)
    
    print(f"Loaded dataset split from: {split_file}")
    print(f"Total datasets: {split_info['metadata']['total_datasets']}")
    print(f"Train: {split_info['metadata']['n_train']} datasets")
    print(f"Val: {split_info['metadata']['n_val']} datasets")
    print(f"Test: {split_info['metadata']['n_test']} datasets")
    print(f"Random seed used: {split_info['metadata']['seed']}")
    
    return split_info


def verify_split(data_path, split_file='dataset_split.json', min_frames: int = 0):
    """
    Verify that all datasets in the split file exist in the data directory.
    
    Args:
        data_path: parent directory containing converted scene folders
        split_file: Path to the JSON file containing the split
    
    Returns:
        Boolean indicating if all datasets exist
    """
    split_info = load_dataset_split(split_file)
    
    all_datasets = (split_info['train_datasets'] + 
                   split_info['val_datasets'] + 
                   split_info['test_datasets'])
    
    missing_datasets = []
    for dataset_name in all_datasets:
        dataset_path = Path(data_path) / dataset_name
        if not dataset_path.exists():
            missing_datasets.append(dataset_name)
            continue
        if min_frames and min_frames > 0:
            n_frames = _count_frames(dataset_path)
            if n_frames < min_frames:
                missing_datasets.append(f"{dataset_name} (frames={n_frames} < {min_frames})")
    
    if missing_datasets:
        print(f"\nWarning: {len(missing_datasets)} datasets not found:")
        for name in missing_datasets[:10]:  # Show first 10
            print(f"  - {name}")
        return False
    else:
        print(f"\nAll {len(all_datasets)} datasets exist in {data_path}")
        return True


def main():
    parser = argparse.ArgumentParser(description='Create or load a dataset train/validation/test split')
    parser.add_argument('--data_path', type=str, 
                       default='./datasets/MOVi-B',
                       help='Parent directory containing converted scene folders')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Fraction of the complete scenes for training (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Fraction for validation (default: 0.1); the remainder is the test split')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output', type=str, default='dataset_split.json',
                       help='Output JSON file for split (default: dataset_split.json)')
    parser.add_argument('--load', action='store_true',
                       help='Load existing split file instead of creating new one')
    parser.add_argument('--verify', action='store_true',
                       help='Verify that all datasets in split file exist')
    parser.add_argument('--min_frames', type=int, default=400,
                       help='Minimum number of frames required per dataset (0 to disable, default: 400)')
    parser.add_argument('--name_pattern', type=str, default='movi_*',
                       help="Glob pattern for scene directories (default: 'movi_*')")
    
    args = parser.parse_args()
    
    if args.load:
        # Load existing split
        split_info = load_dataset_split(args.output)
    elif args.verify:
        # Verify existing split
        verify_split(args.data_path, args.output, min_frames=args.min_frames)
    else:
        # Create new split
        split_info = create_dataset_split(
            args.data_path,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            output_file=args.output,
            min_frames=args.min_frames,
            name_pattern=args.name_pattern
        )


if __name__ == '__main__':
    main()