import os

# List of known BUS-UCLM image/mask filenames (from BUS-UCLM/images and BUS-UCLM/masks)
bus_uclm_img_dir = 'BUS-UCLM/images'
bus_uclm_mask_dir = 'BUS-UCLM/masks'

def get_filenames(folder):
    return set(f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg')))

img_names = get_filenames(bus_uclm_img_dir)
mask_names = get_filenames(bus_uclm_mask_dir)

# Target folders to clean
target_dirs = [
    'datasets/BUSI_split/train/benign',
    'datasets/BUSI_split/train/malignant',
    'datasets/BUSI_split/train/normal',
    'datasets/BUSI_split/validation/benign',
    'datasets/BUSI_split/validation/malignant',
    'datasets/BUSI_split/validation/normal',
    'datasets/BUSI_split_masks/train/benign',
    'datasets/BUSI_split_masks/train/malignant',
    'datasets/BUSI_split_masks/train/normal',
    'datasets/BUSI_split_masks/validation/benign',
    'datasets/BUSI_split_masks/validation/malignant',
    'datasets/BUSI_split_masks/validation/normal',
]

removed = 0
for d in target_dirs:
    if not os.path.exists(d):
        continue
    for f in os.listdir(d):
        if f in img_names or f in mask_names:
            try:
                os.remove(os.path.join(d, f))
                removed += 1
            except Exception as e:
                print(f"Failed to remove {f} from {d}: {e}")

print(f"Removed {removed} BUS-UCLM files from datasets and BUSI_split_masks.")
