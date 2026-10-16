#!/usr/bin/env python3
"""
Export a trained MobileFCMViTv3 checkpoint to ONNX for mobile deployment.

Usage:
    python export_onnx.py --checkpoint checkpoints/<run>/model_best.pth.tar
    python export_onnx.py --checkpoint checkpoints/<run>/model_best.pth.tar --no-quantize
    python export_onnx.py --checkpoint checkpoints/<run>/model_best.pth.tar \
                          --output mamovision-ai-mobile/assets/models/mobilefcmvitv3.onnx

Model input  : "image"         float32  [1, 3, 224, 224]
Model output : "probabilities" float32  [1, 3]
Class order  : [benign, malignant, normal]   (sorted BUSI folder names)

The mobile app reorders to [normal, benign, malignant] internally.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from mobilefcmvitv3.models.mobilefcmvitv3_net import MobileFCMViTv3


class _InferenceWrapper(nn.Module):
    """Strip auxiliary clustering losses; apply softmax for mobile deployment."""

    def __init__(self, model: MobileFCMViTv3):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.model(image)
        return torch.softmax(logits, dim=-1)


def _build_model() -> MobileFCMViTv3:
    return MobileFCMViTv3(
        num_classes=3,
        in_chans=3,
        fcm_k=3,
        fcm_proj_dim=32,
        tau=1.0,
        fcm_m=2.0,
        membership='softmax',
        normalize_feat=True,
        fusion_type='attention',
        drop_path_rate=0.0,
        dropout=0.0,
        attn_dropout=0.0,
        ffn_dropout=0.0,
    )


def _load_checkpoint(model: MobileFCMViTv3, ckpt_path: str) -> MobileFCMViTv3:
    raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # Prefer EMA weights when available (typically 0.5-1% better than last epoch)
    state = (
        raw.get('ema_state_dict')
        or raw.get('state_dict')
        or raw.get('model')
        or raw
    )
    state = {k.replace('module.', ''): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [WARN] {len(missing)} missing keys — first 5: {missing[:5]}")
    if unexpected:
        print(f"  [WARN] {len(unexpected)} unexpected keys — first 5: {unexpected[:5]}")

    backbone_loaded = sum(
        1 for k in state
        if not k.startswith(('fcm_proj', 'clustering', 'fcm_fusion', 'classifier'))
    )
    head_loaded = sum(
        1 for k in state
        if k.startswith(('fcm_proj', 'clustering', 'fcm_fusion', 'classifier'))
    )
    print(f"  Loaded {backbone_loaded} backbone keys + {head_loaded} head keys")
    return model


def export_onnx(checkpoint_path: str, output_path: str, quantize: bool = True) -> None:
    output_path = str(Path(output_path).with_suffix('.onnx'))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print("Building model …")
    model = _build_model()
    print(f"Loading checkpoint: {checkpoint_path}")
    _load_checkpoint(model, checkpoint_path)
    model.eval()

    wrapper = _InferenceWrapper(model)
    wrapper.eval()

    dummy = torch.zeros(1, 3, 224, 224)

    # Verify forward pass before export
    with torch.no_grad():
        probs = wrapper(dummy)
    assert probs.shape == (1, 3), f"Unexpected output shape {probs.shape}"
    assert abs(probs.sum().item() - 1.0) < 1e-4, "Output does not sum to 1 — softmax issue"
    print(f"  Forward pass OK — dummy probs: {probs.detach().numpy().round(4)}")

    print(f"\nExporting ONNX (opset 17) → {output_path} …")
    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        input_names=['image'],
        output_names=['probabilities'],
        dynamic_axes={'image': {0: 'batch'}, 'probabilities': {0: 'batch'}},
        opset_version=17,
        do_constant_folding=True,
    )
    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"  Saved {output_path}  ({size_mb:.1f} MB)")

    if quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            out_int8 = output_path.replace('.onnx', '_int8.onnx')
            print(f"\nQuantizing to INT8 → {out_int8} …")
            quantize_dynamic(output_path, out_int8, weight_type=QuantType.QInt8)
            size_int8 = Path(out_int8).stat().st_size / 1e6
            print(f"  Saved {out_int8}  ({size_int8:.1f} MB, "
                  f"{size_int8/size_mb*100:.0f}% of FP32)")
            output_path = out_int8
        except ImportError:
            print("[WARN] onnxruntime not installed — skipping quantization.")
            print("       pip install onnxruntime")

    mobile_dest = Path('mamovision-ai-mobile/assets/models/mobilefcmvitv3.onnx')
    print(f"\n{'─'*60}")
    print("Next steps:")
    print(f"  1. Copy model to app assets:")
    print(f"       cp \"{output_path}\" \"{mobile_dest}\"")
    print(f"  2. Install ONNX runtime (if not done):")
    print(f"       cd mamovision-ai-mobile && npm install onnxruntime-react-native")
    print(f"  3. Rebuild the app:")
    print(f"       npx expo run:android")
    print(f"{'─'*60}")
    print("\nClass order in ONNX output: [benign=0, malignant=1, normal=2]")
    print("The app reorders to [normal, benign, malignant] automatically.")


def main() -> None:
    p = argparse.ArgumentParser(
        description='Export MobileFCMViTv3 checkpoint to ONNX for mobile deployment.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--checkpoint', required=True,
                   help='Path to model_best.pth.tar (or last.pth.tar)')
    p.add_argument('--output', default='mobilefcmvitv3.onnx',
                   help='Output ONNX path (default: mobilefcmvitv3.onnx)')
    p.add_argument('--no-quantize', action='store_true',
                   help='Skip INT8 dynamic quantization')
    args = p.parse_args()

    export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        quantize=not args.no_quantize,
    )


if __name__ == '__main__':
    main()
