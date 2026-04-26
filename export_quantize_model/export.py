"""
Export MobileFCMViTv3 for mobile deployment (TorchScript, ONNX, CoreML).
Reads configuration from config.yaml.

Usage:
    python export_and_quantization/export_model.py
"""

import torch
import os
import yaml
from pathlib import Path

try:
    import coremltools as ct
except ImportError:
    ct = None
try:
    import onnx
except ImportError:
    onnx = None

from mobilefcmvitv3.models.mobilefcmvitv3_net import MobileFCMViTv3

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

def resolve_checkpoint_path(path):
    # Try relative to script directory
    script_dir = os.path.dirname(__file__)
    candidate = os.path.join(script_dir, path)
    if os.path.exists(candidate):
        return candidate
    # Try relative to project root (one level up)
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    candidate = os.path.join(project_root, path)
    if os.path.exists(candidate):
        return candidate
    # Fallback to original path
    return path

CHECKPOINT_PATH = resolve_checkpoint_path(config.get('checkpoint_path', 'checkpoints/mobilefcmvitv3_s_BUSI/model_best.pth.tar'))
EXPORT_DIR = config.get('export_dir', 'output/')
IMG_SIZE = config.get('img_size', 224)
N_CLASSES = config.get('num_classes', 3)
EXPORT_COREML = config.get('export_coreml', True)
EXPORT_ONNX = config.get('export_onnx', True)
EXPORT_TORCHSCRIPT = config.get('export_torchscript', True)


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


def export_torchscript(model):
    sample = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    traced = torch.jit.trace(model, sample)
    torchscript_path = os.path.join(EXPORT_DIR, 'mobilefcmvitv3_scripted.pt')
    traced.save(torchscript_path)
    print(f"TorchScript model saved to {torchscript_path}")


def export_onnx(model):
    sample = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    onnx_path = os.path.join(EXPORT_DIR, 'mobilefcmvitv3.onnx')
    torch.onnx.export(
        model, sample, onnx_path,
        input_names=['input'], output_names=['output'],
        opset_version=12, dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
    )
    print(f"ONNX model saved to {onnx_path}")


def export_coreml(model):
    if ct is None:
        print("coremltools not installed. Skipping CoreML export.")
        return
    sample = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    traced = torch.jit.trace(model, sample)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="input", shape=sample.shape, scale=1/255.0, bias=[0,0,0])],
        classifier_config=None
    )
    coreml_path = os.path.join(EXPORT_DIR, 'mobilefcmvitv3.mlmodel')
    mlmodel.save(coreml_path)
    print(f"CoreML model saved to {coreml_path}")


if __name__ == "__main__":
    os.makedirs(EXPORT_DIR, exist_ok=True)
    model = load_model()
    if EXPORT_TORCHSCRIPT:
        export_torchscript(model)
    if EXPORT_ONNX:
        export_onnx(model)
    if EXPORT_COREML:
        export_coreml(model)