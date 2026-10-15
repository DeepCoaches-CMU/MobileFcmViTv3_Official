"""
P0.1 — build a run register from every W&B run directory plus the on-disk
results directories.

One row per W&B run. Columns are taken only from what the run itself recorded
(wandb-metadata.json, config.yaml, wandb-summary.json, output.log). Nothing is
inferred from directory or file names; where a link cannot be read from a
recorded artifact the cell says MISSING.

  eval_n         sum of val/<class>/support in the final summary — the number of
                 images the logged metrics were computed on (115 = v2 split
                 validation, 156 = old non-deduped validation, etc.)
  commit_in_repo whether the recorded commit exists in this clone (after
                 `git fetch --all`)
  results_dir    --results_dir from the command line, else results_dir from the
                 config file the run loaded (config.yaml as recorded by W&B)

Usage: python audit/P0.1/scripts/build_run_register.py [repo_root] > run_register.csv
"""
import csv
import glob
import json
import os
import re
import subprocess
import sys

import yaml

ROOT = sys.argv[1] if len(sys.argv) > 1 else '.'


def load_json(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def wandb_config(p):
    """W&B config.yaml stores {key: {value: ...}}; flatten it."""
    try:
        with open(p) as f:
            raw = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return {k: (v.get('value') if isinstance(v, dict) and 'value' in v else v)
            for k, v in raw.items()}


def arg(args, name):
    """Value of --name / --name=value / -c style flag in a recorded argv list."""
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(name + '='):
            return a.split('=', 1)[1]
    return ''


def commit_exists(c):
    if not c:
        return 'no-commit-recorded'
    r = subprocess.run(['git', '-C', ROOT, 'cat-file', '-e', c + '^{commit}'],
                       capture_output=True)
    return 'yes' if r.returncode == 0 else 'NO'


def first_log_line(path, pattern):
    try:
        with open(path, errors='replace') as f:
            for line in f:
                m = re.search(pattern, line)
                if m:
                    return m.group(1).strip()
    except OSError:
        pass
    return ''


FIELDS = ['wandb_run', 'started_utc', 'experiment', 'config_file', 'cli_overrides',
          'git_commit', 'commit_in_repo', 'host', 'gpu', 'data_dir', 'pretrained_weights',
          'seed', 'checkpoint_dir', 'results_dir', 'final_epoch', 'eval_n',
          'eval_class_support', 'val_top1', 'val_precision', 'val_recall', 'val_f1',
          'val_loss', 'val_roc_auc', 'params_flops_infer', 'ckpt_loaded_line']


def main():
    w = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
    w.writeheader()
    for d in sorted(glob.glob(os.path.join(ROOT, 'wandb', 'run-*'))):
        files = os.path.join(d, 'files')
        meta = load_json(os.path.join(files, 'wandb-metadata.json')) or {}
        summ = load_json(os.path.join(files, 'wandb-summary.json')) or {}
        cfg = wandb_config(os.path.join(files, 'config.yaml'))
        args = meta.get('args') or []
        commit = (meta.get('git') or {}).get('commit', '')
        supports = {k.split('/')[1]: v for k, v in summ.items()
                    if re.fullmatch(r'val/[^/]+/support', k)}
        log = os.path.join(files, 'output.log')
        overrides = [a for a in args if a.startswith('--') and a.split('=')[0] not in
                     ('--experiment', '--data_dir', '-c')]
        w.writerow({
            'wandb_run': os.path.basename(d),
            'started_utc': meta.get('startedAt', 'MISSING'),
            'experiment': arg(args, '--experiment') or cfg.get('experiment', '') or 'MISSING',
            'config_file': arg(args, '-c') or 'MISSING',
            'cli_overrides': ' '.join(overrides),
            'git_commit': commit or 'MISSING',
            'commit_in_repo': commit_exists(commit),
            'host': meta.get('host', ''),
            'gpu': meta.get('gpu', ''),
            'data_dir': arg(args, '--data_dir') or cfg.get('data_dirs', '') or cfg.get('data_dir', ''),
            'pretrained_weights': cfg.get('pretrained_weights', 'MISSING'),
            'seed': arg(args, '--seed') or cfg.get('seed', 'MISSING'),
            'checkpoint_dir': arg(args, '--checkpoint_dir') or cfg.get('checkpoint_dir', 'MISSING'),
            'results_dir': arg(args, '--results_dir') or cfg.get('results_dir', 'MISSING'),
            'final_epoch': summ.get('epoch', ''),
            'eval_n': sum(supports.values()) if supports else 'MISSING',
            'eval_class_support': ';'.join(f'{k}={v}' for k, v in sorted(supports.items())),
            'val_top1': summ.get('val/top1', ''),
            'val_precision': summ.get('val/precision', ''),
            'val_recall': summ.get('val/recall', ''),
            'val_f1': summ.get('val/f1', ''),
            'val_loss': summ.get('val/loss', ''),
            'val_roc_auc': summ.get('val/roc_auc', ''),
            'params_flops_infer': first_log_line(log, r'(Params:.*)'),
            'ckpt_loaded_line': first_log_line(log, r'((?:Loaded|Loading|Resum|pretrained).*)')[:200],
        })


if __name__ == '__main__':
    main()
