#!/usr/bin/env python3
"""
build_busi_split.py

Rebuilds the BUSI train/validation/test split from raw data, fixing two problems
in the original ad hoc data/BUSI_split.json:

1. No test split existed — test_results/ and evaluation_results/ were both
   scoring the same validation set used for checkpoint selection.
2. One raw image is byte-identical across two different class folders
   (benign (433).png == malignant (145).png) — a mislabeled duplicate present
   in the raw Kaggle dataset itself. Since we can't know which label is
   correct, both copies are excluded rather than arbitrarily picking one.

Produces data/BUSI_split_v2.json with 'train'/'val'/'test' keys (same
per-class filename-list schema as the original data/BUSI_split.json, so
setup_dataset.py's existing organize_split()/verify_split() logic can be
reused with a 3-way split_key_map).

Usage:
    python3 scripts/build_busi_split.py
"""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "datasets" / "BUSI_raw" / "Dataset_BUSI_with_GT"
OUT_JSON = REPO_ROOT / "data" / "BUSI_split_v2.json"
CLASSES = ["benign", "malignant", "normal"]

SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remainder (0.15) goes to test


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_images():
    """Return {class: [filename, ...]} for non-mask images, and a hash->[(cls, fname)] map."""
    by_class = defaultdict(list)
    hash_to_files = defaultdict(list)
    for cls in CLASSES:
        cls_dir = RAW_DIR / cls
        for path in sorted(cls_dir.glob("*.png")):
            if "_mask" in path.name:
                continue
            by_class[cls].append(path.name)
            hash_to_files[md5_of(path)].append((cls, path.name))
    return by_class, hash_to_files


def find_exclusions(hash_to_files):
    """
    Return (cross_class_exclude, same_class_dupe_keep_drop) where:
    - cross_class_exclude: set of (cls, fname) to drop entirely (ambiguous label)
    - same_class_dupe_keep_drop: set of (cls, fname) to drop (keep first occurrence only)
    """
    cross_class_exclude = set()
    same_class_drop = set()
    for file_hash, occurrences in hash_to_files.items():
        if len(occurrences) < 2:
            continue
        classes_seen = {cls for cls, _ in occurrences}
        if len(classes_seen) > 1:
            print(f"  ⚠️  Cross-class duplicate (excluding all copies, label ambiguous): {occurrences}")
            cross_class_exclude.update(occurrences)
        else:
            # same-class exact duplicate: keep first, drop the rest
            for cls, fname in occurrences[1:]:
                print(f"  ℹ️  Same-class duplicate (dropping extra copy): {cls}/{fname}")
                same_class_drop.add((cls, fname))
    return cross_class_exclude, same_class_drop


def stratified_split(by_class, excluded):
    rng = random.Random(SEED)
    split = {"train": {}, "val": {}, "test": {}}
    for cls in CLASSES:
        files = [f for f in by_class[cls] if (cls, f) not in excluded]
        rng.shuffle(files)
        n = len(files)
        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)
        split["train"][cls] = sorted(files[:n_train])
        split["val"][cls] = sorted(files[n_train:n_train + n_val])
        split["test"][cls] = sorted(files[n_train + n_val:])
    return split


def main():
    print("Scanning raw BUSI images and hashing …")
    by_class, hash_to_files = collect_images()
    total_raw = sum(len(v) for v in by_class.values())
    print(f"Found {total_raw} raw images across {len(CLASSES)} classes.")

    cross_class_exclude, same_class_drop = find_exclusions(hash_to_files)
    excluded = cross_class_exclude | same_class_drop
    print(f"Excluding {len(excluded)} file(s) total "
          f"({len(cross_class_exclude)} cross-class ambiguous, {len(same_class_drop)} same-class dupes).")

    split = stratified_split(by_class, excluded)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(split, f, indent=2)

    print(f"\n✅ Wrote {OUT_JSON}")
    for name in ("train", "val", "test"):
        counts = {cls: len(split[name][cls]) for cls in CLASSES}
        print(f"  {name}: {counts}  (total {sum(counts.values())})")


if __name__ == "__main__":
    main()
