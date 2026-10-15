"""
mobilefcmvitv3/scripts/ablation.py

Ablation study launcher for MobileFCMViTv3 on BUSI.
Sweeps over:
  - FCM channel on/off
  - Dropout values
  - Mixup on/off
  - Pretrained vs. scratch

All paths are read from paths.yaml at the repo root.

Usage:
    cd /path/to/ultrascannet_reproduced
    PYTHONPATH=. python mobilefcmvitv3/scripts/ablation.py
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent.parent
PATHS_YAML = REPO_ROOT / 'paths.yaml'
TRAIN_SCRIPT = Path(__file__).parent / 'train.py'


def resolve_repo_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def load_paths():
    with open(PATHS_YAML) as f:
        paths = yaml.safe_load(f)
    for key in ('datasets_root', 'busi_split_dir', 'output_dir'):
        if key in paths:
            paths[key] = resolve_repo_path(paths[key])
    return paths


# ── Ablation grid ─────────────────────────────────────────────────────────────
# Each entry is (experiment_suffix, config_overrides_as_cli_args)
# NOTE: "_pretrained" variants all load ablation_base.yaml, which fixes
# pretrained_weights to the common untrained-on-BUSI VOC checkpoint. They
# used to load mobilefcmvitv3_s_busi.yaml, whose pretrained_weights points at
# tune_05_augmix's own converged checkpoint — every ablation then warm-started
# already-converged and produced bit-identical metrics regardless of the
# ablated setting. See ablation_base.yaml's header comment for the full
# explanation.
ABLATIONS = [
    # 1. Full model (FCM + pretrained) — baseline for comparison
    (
        'full_fcm_pretrained',
        ['-c', 'mobilefcmvitv3/configs/ablation_base.yaml'],
    ),
    # 2. No FCM channel (3-ch, pretrained)
    (
        'no_fcm_pretrained',
        ['-c', 'mobilefcmvitv3/configs/ablation_no_fcm.yaml'],
    ),
    # 3. Full FCM, no pretrained weights (train from scratch)
    (
        'full_fcm_scratch',
        [
            '-c', 'mobilefcmvitv3/configs/ablation_base.yaml',
            '--pretrained_weights=',
        ],
    ),
    # 4. No FCM, no pretrained (pure scratch baseline)
    (
        'no_fcm_scratch',
        [
            '-c', 'mobilefcmvitv3/configs/ablation_no_fcm.yaml',
            '--pretrained_weights=',
        ],
    ),
    # 5. Full FCM, no mixup/cutmix
    (
        'full_fcm_no_mixup',
        [
            '-c', 'mobilefcmvitv3/configs/ablation_base.yaml',
            '--mixup=0',
            '--cutmix=0',
        ],
    ),
    # 6. Full FCM, no class weights
    (
        'full_fcm_no_classweights',
        [
            '-c', 'mobilefcmvitv3/configs/ablation_base.yaml',
            '--use_class_weights=false',
        ],
    ),
    # 7. Full FCM, higher dropout
    (
        'full_fcm_dropout_0.3',
        [
            '-c', 'mobilefcmvitv3/configs/ablation_base.yaml',
            '--dropout=0.3',
        ],
    ),
    # 8. Full FCM, no label smoothing
    (
        'full_fcm_no_smoothing',
        [
            '-c', 'mobilefcmvitv3/configs/ablation_base.yaml',
            '--smoothing=0.0',
        ],
    ),
    # 9. FCM-only (no backbone at all) — previously configured but never run
    (
        'fcm_only',
        ['-c', 'mobilefcmvitv3/configs/ablation_fcm_only.yaml'],
    ),
    # 10. MobileViTv3-only (no FCM module at all) — previously configured but
    # never run (and would have crashed: its model class called a timm model
    # name that doesn't exist — see mobilevitv3_only.py)
    (
        'mobilevitv3_only',
        ['-c', 'mobilefcmvitv3/configs/ablation_mobilevitv3_only.yaml'],
    ),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['force', 'resume'], default='force',
                   help="'force' (default): each ablation variant always starts fresh, "
                        "overwriting any previous run in ablation_results/ (this is the "
                        "normal way to use this script — ablations are meant to be "
                        "regenerated, e.g. after a dataset rebuild, not incrementally "
                        "resumed). 'resume': continue each variant from its own "
                        "last checkpoint instead.")
    return p.parse_args()


def main():
    args = parse_args()
    paths = load_paths()
    busi_dir = paths['busi_split_dir']

    for suffix, extra_args in ABLATIONS:
        experiment_name = f'mobilefcmvitv3_{suffix}'
        # Always set output dirs to ablation_results for ablation runs
        ablation_ckpt_dir = f'ablation_results/{experiment_name}/checkpoints'
        ablation_res_dir = f'ablation_results/{experiment_name}/results'
        cmd = [
            sys.executable, str(TRAIN_SCRIPT),
            '--data_dir', busi_dir,
            '--experiment', experiment_name,
            f'--checkpoint_dir={ablation_ckpt_dir}',
            f'--results_dir={ablation_res_dir}',
            f'--{args.mode}',
        ] + extra_args

        print(f'\n🔬 Ablation: {experiment_name}')
        print('   ' + ' '.join(cmd))
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


if __name__ == '__main__':
    main()
