import os
from collections import defaultdict

# Paths to your BUSI split directories
SPLIT_DIRS = [
    'datasets/BUSI_split/train/benign',
    'datasets/BUSI_split/train/malignant',
    'datasets/BUSI_split/train/normal',
    'datasets/BUSI_split/validation/benign',
    'datasets/BUSI_split/validation/malignant',
    'datasets/BUSI_split/validation/normal',
]

results = defaultdict(lambda: {'images': 0, 'masks': 0, 'total': 0})

for split_dir in SPLIT_DIRS:
    class_name = split_dir.split('/')[-1]
    split_type = split_dir.split('/')[-3]
    for fname in os.listdir(split_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        results[(split_type, class_name)]['total'] += 1
        if '_mask' in fname:
            results[(split_type, class_name)]['masks'] += 1
        else:
            results[(split_type, class_name)]['images'] += 1

print(f"{'Split':<10} {'Class':<10} {'Images':>8} {'Masks':>8} {'Total':>8}")
print('-' * 44)
for (split, cls), counts in sorted(results.items()):
    print(f"{split:<10} {cls:<10} {counts['images']:>8} {counts['masks']:>8} {counts['total']:>8}")
