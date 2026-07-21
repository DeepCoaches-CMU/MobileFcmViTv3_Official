"""
mobilefcmvitv3/scripts/eval_external.py

Out-of-distribution generalization check: evaluate a BUSI-trained checkpoint
against BUS-UCLM (a different clinic/scanner source), which the model never
saw during training. This is NOT the in-distribution held-out test set (see
test.py --split test for that) — results here measure cross-dataset
generalization and should always be reported separately, clearly labeled as
OOD/external.

IMPORTANT: do not run this against the tune_06_regdrop checkpoint — that run
trained on datasets2/BUSI_split, which IS the BUS-UCLM data (see
scripts/split_bus_uclm.py), so it would not be a held-out evaluation for it.

Usage:
    cd /path/to/ultrascannet_reproduced
    PYTHONPATH=. python mobilefcmvitv3/scripts/eval_external.py \\
        -c mobilefcmvitv3/configs/mobilefcmvitv3_s_busi.yaml \\
        --checkpoint tuning/tune_05_augmix/checkpoints/model_best.pth.tar \\
        --external_dir datasets2/BUSI_split \\
        --output_dir test_results/external_bus_uclm
"""

import argparse
import logging
import sys
from contextlib import suppress
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from mobilefcmvitv3.models import MobileFCMViTv3
from mobilefcmvitv3.utils.dataset import BUSIDataset
from mobilefcmvitv3.utils.augmentation import build_val_transform
from mobilefcmvitv3.utils.metrics import (
    compute_extended_metrics, save_metrics_json, compute_efficiency
)
from mobilefcmvitv3.scripts.validate import evaluate

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)s  %(message)s')
_logger = logging.getLogger('eval_external')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config', required=True, type=str)
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--external_dir', type=str, default='datasets2/BUSI_split',
                   help='Root of the external dataset (must contain train/ and '
                        'validation/ class subfolders); both are combined since '
                        'none of it was used to train the checkpoint being evaluated')
    p.add_argument('--output_dir', type=str, default='test_results/external_bus_uclm')
    p.add_argument('--no_ema', action='store_true', default=False,
                   help='Use live model weights instead of EMA weights')
    return p.parse_args()


def main():
    args = parse_args()
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model = MobileFCMViTv3(
        num_classes=cfg.get('num_classes', 3),
        in_chans=3,
        fcm_k=cfg.get('fcm_k', 3),
        fcm_proj_dim=cfg.get('fcm_proj_dim', 32),
        tau=cfg.get('tau', 1.0),
        fcm_m=cfg.get('fcm_m', 2.0),
        membership=cfg.get('membership', 'softmax'),
        normalize_feat=cfg.get('normalize_feat', True),
        fusion_type=cfg.get('fusion_type', 'attention'),
        drop_path_rate=cfg.get('drop_path_rate', 0.1),
        dropout=cfg.get('dropout', 0.1),
        attn_dropout=cfg.get('attn_dropout', 0.0),
        ffn_dropout=cfg.get('ffn_dropout', 0.0),
    ).cuda()

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if not args.no_ema and ckpt.get('ema_state_dict') is not None:
        state = ckpt['ema_state_dict']
        _logger.info("Using EMA weights")
    else:
        state = ckpt.get('state_dict', ckpt)
        _logger.info("Using live model weights")
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    _logger.info(f"Loaded: {args.checkpoint}")

    params_M, flops_G, inf_ms = compute_efficiency(model, in_chans=3)

    val_tf = build_val_transform(img_size=224)
    external_root = Path(args.external_dir)
    sub_datasets = [
        BUSIDataset(str(external_root / sub), transform=val_tf)
        for sub in ('train', 'validation')
        if (external_root / sub).exists()
    ]
    if not sub_datasets:
        raise RuntimeError(f'No train/ or validation/ subfolders found under {external_root}')
    dataset = ConcatDataset(sub_datasets)
    class_names = getattr(sub_datasets[0], 'classes', None)
    _logger.info(f"External OOD eval on {external_root}: {len(dataset)} samples | classes: {class_names}")

    loader = DataLoader(dataset, batch_size=cfg.get('validation_batch_size', 16),
                        shuffle=False, num_workers=cfg.get('workers', 8), pin_memory=True)
    amp_autocast = partial(torch.amp.autocast, 'cuda') if cfg.get('amp', True) else suppress
    results = evaluate(model, loader, amp_autocast)
    _logger.info(f"Top-1: {results['top1']:.2f}% | Loss: {results['loss']:.4f}")

    ext = compute_extended_metrics(
        results['targets'], results['preds'], results['probs'],
        class_names=class_names, n_boot=500
    )
    _logger.info(f"Precision: {ext.get('precision_macro', 0):.4f} | "
                 f"Recall: {ext.get('recall_macro', 0):.4f} | "
                 f"F1: {ext.get('f1_macro', 0):.4f}")

    cm = ext.get('confusion_matrix')
    if cm is not None:
        _logger.info(f"Confusion matrix:\n{np.array(cm)}")

    record = {
        'checkpoint': args.checkpoint,
        'split': 'external_bus_uclm_ood',
        'external_dir': str(external_root),
        'loss': results['loss'],
        'top1': results['top1'],
        'params_M': params_M,
        'flops_G': flops_G,
        'inference_ms': inf_ms,
        **{k: v for k, v in ext.items() if k != 'confusion_matrix'},
        'confusion_matrix': cm,
    }
    out_path = Path(args.output_dir) / 'external_bus_uclm_metrics.json'
    save_metrics_json(record, str(out_path))
    _logger.info(f"Metrics saved to: {out_path}")


if __name__ == '__main__':
    main()
