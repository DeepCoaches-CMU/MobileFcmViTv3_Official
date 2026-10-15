"""
Quantize MobileFCMViTv3 for mobile deployment (static quantization).
Reads configuration from config.yaml.

Usage:
    python export_and_quantization/quantize_model.py
"""

import torch
import os
import yaml
from torch.utils.data import DataLoader
from mobilefcmvitv3.models.mobilefcmvitv3_net import MobileFCMViTv3
from mobilefcmvitv3.utils.dataset import BUSIDataset
from mobilefcmvitv3.utils.augmentation import build_val_transform

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

CHECKPOINT_PATH = config.get('checkpoint_path', 'checkpoints/mobilefcmvitv3_s_BUSI/model_best.pth.tar')
EXPORT_DIR = config.get('export_dir', 'export_and_quantization/')
IMG_SIZE = config.get('img_size', 224)
N_CLASSES = config.get('num_classes', 3)
CALIBRATION_BATCH_SIZE = config.get('calibration_batch_size', 8)
USE_DUMMY_CALIBRATION = config.get('use_dummy_calibration', False)
DATASET_DIR = config.get('dataset_dir', 'datasets/BUSI_split')


def load_model():
    model = MobileFCMViTv3(num_classes=N_CLASSES)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def quantize_model(model):
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    if USE_DUMMY_CALIBRATION:
        sample = torch.randn(CALIBRATION_BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
        model(sample)
        print("Calibrated with dummy data.")
    else:
        print(f"Calibrating with real data from {DATASET_DIR} ...")
        val_transform = build_val_transform(img_size=IMG_SIZE)
        dataset = BUSIDataset(DATASET_DIR, transform=val_transform)
        loader = DataLoader(dataset, batch_size=CALIBRATION_BATCH_SIZE, shuffle=True, num_workers=0)
        model.eval()
        with torch.no_grad():
            for i, (images, _) in enumerate(loader):
                model(images)
                if i >= 4:
                    break
        print("Calibration with real data complete.")
    torch.quantization.convert(model, inplace=True)
    quant_path = os.path.join(EXPORT_DIR, 'mobilefcmvitv3_quantized.pt')
    scripted = torch.jit.trace(model, torch.randn(1, 3, IMG_SIZE, IMG_SIZE))
    scripted.save(quant_path)
    print(f"Quantized TorchScript model saved to {quant_path}")
    return model


if __name__ == "__main__":
    os.makedirs(EXPORT_DIR, exist_ok=True)
    model = load_model()
    quantize_model(model)