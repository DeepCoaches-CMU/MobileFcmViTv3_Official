"""
P0.1 — per-file artifact inventory: path, size, mtime, SHA-256, artifact type,
and whether git tracks the file. Run against the frozen snapshot so the hashes
describe the archived copy.

Types are assigned from path/extension only, to make the inventory browsable;
a type label says what kind of file it is, never which run produced it.

Usage: python build_inventory.py <snapshot_root> > inventory.csv
"""
import csv
import hashlib
import os
import subprocess
import sys
from datetime import datetime, timezone

ROOT = sys.argv[1]
SKIP = {'.git', '.venv', '__pycache__'}


def artifact_type(rel):
    p = rel.lower()
    name = os.path.basename(p)
    if p.startswith(('datasets/', 'datasets2/', 'bus-uclm/', '__macosx/')):
        return 'dataset_image' if name.endswith(('.png', '.jpg', '.jpeg', '.bmp')) else 'dataset_other'
    if name.endswith(('.pth', '.pth.tar', '.pt', '.ckpt')):
        return 'checkpoint'
    if name.endswith(('.onnx', '.onnx.data', '.tflite')):
        return 'exported_model'
    if name.startswith('predictions_') and name.endswith('.csv'):
        return 'prediction_csv'
    if 'confusion' in name or p.startswith('baseline/cm/'):
        return 'confusion_matrix_figure'
    if p.startswith('data/') or 'split' in name and name.endswith('.json') or 'kfold' in name and name.endswith('.json'):
        return 'split_manifest'
    if name.endswith(('.yaml', '.yml')) and not p.startswith('wandb/'):
        return 'config'
    if p.startswith(('slurm_logs/', 'log_dir/')) or name.endswith(('.log', '.out', '.err')):
        return 'log'
    if p.startswith('wandb/'):
        return 'wandb_record'
    if name in ('best_metrics.json', 'summary.csv', 'evaluation_metrics.json') or name.startswith(('metrics_epoch_', 'test_metrics')) \
            or name.endswith(('_summary.json', '_metrics.json')) or p.startswith('baseline/mambavision_fcm/metrics/'):
        return 'metrics'
    if name.endswith(('.png', '.jpg', '.pdf', '.svg')):
        return 'figure'
    if name.endswith(('.py', '.slurm', '.sh', '.ipynb')):
        return 'script'
    if name.endswith(('.tex', '.md', '.rst', '.txt')):
        return 'document'
    return 'other'


def git_tracked(root):
    try:
        out = subprocess.run(['git', '-C', root, 'ls-files', '-z'], capture_output=True, check=True).stdout
        return set(out.decode().split('\0'))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    tracked = git_tracked(ROOT)
    w = csv.writer(sys.stdout)
    w.writerow(['path', 'type', 'size_bytes', 'mtime_utc', 'sha256', 'git_tracked'])
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP)
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, ROOT)
            if os.path.islink(full):
                w.writerow([rel, 'symlink->' + os.readlink(full), '', '', '', rel in tracked])
                continue
            st = os.stat(full)
            mtime = datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec='seconds')
            w.writerow([rel, artifact_type(rel), st.st_size, mtime, sha256(full), rel in tracked])


if __name__ == '__main__':
    main()
