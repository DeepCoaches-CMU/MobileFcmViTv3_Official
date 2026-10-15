"""
scripts/build_kfold_splits.py

Builds K stratified folds from the existing train+validation pool (the test
split is never touched here — it stays held out for a single final check,
not used for model selection or CV). Motivation: the current single
train/validation split has only 19 'normal'-class images in validation, so
per-class metrics for that class swing by ~5 points per single misclassified
sample. Averaging over K folds gives a materially less noisy estimate of
true performance than any one arbitrary split.

Builds datasets/BUSI_split_kfold/fold{i}/{train,validation}/{class}/ as
symlinks into the real files already in datasets/BUSI_split/{train,validation}/
— no image data is duplicated on disk.

Usage:
    python3 scripts/build_kfold_splits.py --k 5 --seed 42
"""

import argparse
import shutil
from pathlib import Path

from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = REPO_ROOT / 'datasets' / 'BUSI_split'
OUTPUT_DIR = REPO_ROOT / 'datasets' / 'BUSI_split_kfold'


def collect_trainval_files():
    """Returns {class_name: [Path, ...]} combining train + validation (not test)."""
    classes = sorted(d.name for d in (SOURCE_DIR / 'train').iterdir() if d.is_dir())
    files_by_class = {}
    for cls in classes:
        files = []
        for split in ('train', 'validation'):
            split_dir = SOURCE_DIR / split / cls
            if split_dir.exists():
                files.extend(sorted(split_dir.glob('*')))
        files_by_class[cls] = files
    return files_by_class


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    files_by_class = collect_trainval_files()
    for cls, files in files_by_class.items():
        print(f"{cls}: {len(files)} images (train+validation pool)")

    if OUTPUT_DIR.exists():
        print(f"Removing existing {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    # Per-class StratifiedKFold, so each fold gets a proportional share of
    # every class regardless of class imbalance.
    fold_assignment = {cls: {} for cls in files_by_class}
    for cls, files in files_by_class.items():
        skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
        dummy_y = [0] * len(files)  # single-class split, labels don't matter here
        for fold_idx, (_, val_idx) in enumerate(skf.split(files, dummy_y)):
            for i in val_idx:
                fold_assignment[cls][files[i]] = fold_idx

    for fold_idx in range(args.k):
        for cls, files in files_by_class.items():
            train_dir = OUTPUT_DIR / f'fold{fold_idx}' / 'train' / cls
            val_dir = OUTPUT_DIR / f'fold{fold_idx}' / 'validation' / cls
            train_dir.mkdir(parents=True, exist_ok=True)
            val_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                target = val_dir if fold_assignment[cls][f] == fold_idx else train_dir
                (target / f.name).symlink_to(f.resolve())

        n_train = sum(len(list((OUTPUT_DIR / f'fold{fold_idx}' / 'train' / cls).iterdir()))
                     for cls in files_by_class)
        n_val = sum(len(list((OUTPUT_DIR / f'fold{fold_idx}' / 'validation' / cls).iterdir()))
                   for cls in files_by_class)
        print(f"fold{fold_idx}: train={n_train} val={n_val}")

    print(f"\nBuilt {args.k} folds under {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
