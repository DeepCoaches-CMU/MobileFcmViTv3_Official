"""
mobilefcmvitv3/scripts/aggregate_seeds.py

Aggregates best_metrics.json across multiple seed runs of the same
hyperparameter config into mean +/- std, so the paper reports a distribution
instead of a single cherry-picked run. (Original report.tex reported only the
single best-of-six tuning run, at/below the MambaVision baseline for 3 of the
6 configs — this makes that variance visible instead of hidden.)

Usage:
    PYTHONPATH=. python mobilefcmvitv3/scripts/aggregate_seeds.py \\
        --inputs tuning/tune_05_seed42/results/best_metrics.json \\
                 tuning/tune_05_seed123/results/best_metrics.json \\
                 tuning/tune_05_seed7/results/best_metrics.json \\
        --output tuning/tune_05_augmix_seed_summary.json
"""

import argparse
import json
import statistics
from pathlib import Path

# Scalar metrics worth reporting mean +/- std for. All are top-level float/int
# fields in best_metrics.json (see mobilefcmvitv3/scripts/train.py's `record`).
METRIC_KEYS = [
    'val_top1', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
    'roc_auc_macro', 'pr_auc_macro',
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', nargs='+', required=True,
                   help='Paths to best_metrics.json, one per seed run (need >= 2 to compute std)')
    p.add_argument('--output', type=str, default='',
                   help='Optional path to write the aggregated summary JSON')
    return p.parse_args()


def main():
    args = parse_args()
    records = []
    for path in args.inputs:
        with open(path) as f:
            records.append(json.load(f))

    n = len(records)
    if n < 2:
        raise SystemExit(f"Need at least 2 seed runs to compute mean/std, got {n}.")

    summary = {'n_seeds': n, 'sources': args.inputs, 'metrics': {}}
    print(f"Aggregating {n} seed run(s):")
    for path in args.inputs:
        print(f"  - {path}")
    print()

    for key in METRIC_KEYS:
        values = [r[key] for r in records if key in r]
        if len(values) != n:
            continue
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        summary['metrics'][key] = {
            'mean': round(mean, 4),
            'std': round(std, 4),
            'values': [round(v, 4) for v in values],
        }
        print(f"  {key:20s}: {mean:.4f} +/- {std:.4f}  (values: {[round(v, 2) for v in values]})")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary to {args.output}")


if __name__ == '__main__':
    main()
