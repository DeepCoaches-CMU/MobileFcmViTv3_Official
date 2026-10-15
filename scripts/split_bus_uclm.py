import os
import shutil
import random
from PIL import Image
import numpy as np

# Set random seed for reproducibility
random.seed(42)

# Paths
src_img_dir = 'BUS-UCLM/images'
src_mask_dir = 'BUS-UCLM/masks'
dst_root = 'datasets2/BUSI_split'
split_ratio = 0.8  # 80% train, 20% val

# Helper to get class from mask
# Green: benign, Red: malignant, Black: normal
COLOR_BENIGN = [0, 255, 0]
COLOR_MALIGNANT = [255, 0, 0]


def get_class_from_mask(mask_path):
    mask = np.array(Image.open(mask_path))
    if np.any(np.all(mask == COLOR_BENIGN, axis=-1)):
        return 'benign'
    elif np.any(np.all(mask == COLOR_MALIGNANT, axis=-1)):
        return 'malignant'
    else:
        return 'normal'

# Gather all images and their classes
samples = []
for fname in os.listdir(src_img_dir):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    img_path = os.path.join(src_img_dir, fname)
    mask_path = os.path.join(src_mask_dir, fname)
    if os.path.exists(mask_path):
        cls = get_class_from_mask(mask_path)
        samples.append((img_path, mask_path, cls))

# Split per class
by_class = {'benign': [], 'malignant': [], 'normal': []}
for img, mask, cls in samples:
    by_class[cls].append((img, mask))


moved_files = set()
for cls, items in by_class.items():
    random.shuffle(items)
    split = int(len(items) * split_ratio)
    train_items = items[:split]
    val_items = items[split:]
    for split_name, split_items in [('train', train_items), ('validation', val_items)]:
        out_img_dir = os.path.join(dst_root, split_name, cls)
        out_mask_dir = os.path.join('datasets2/BUSI_split_masks', split_name, cls)
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_mask_dir, exist_ok=True)
        for img, mask in split_items:
            shutil.copy(img, out_img_dir)
            shutil.copy(mask, out_mask_dir)
            moved_files.add(img)
            moved_files.add(mask)


# (No deletion: Only copy images/masks, do not remove originals)

print('Done!')
