"""
P0.1 — search every machine-readable result file for each quantitative claim in
the submitted paper (43_MobileFCMViTv3_A_Lightweigh.pdf).

A value "matches" if some number in a file equals the paper value after rounding
to the paper's printed precision, either as-is or scaled by 100 (fraction vs
percent). Matches are leads, not proof: a common value (e.g. 0.315) can match
by coincidence, so every hit must be checked by hand before it is written into
the provenance table.

Usage: python audit/P0.1/scripts/trace_paper_numbers.py [repo_root] > out.tsv
"""
import os
import re
import sys
from decimal import Decimal

ROOT = sys.argv[1] if len(sys.argv) > 1 else '.'
SKIP_DIRS = {'.venv', '.git', 'datasets', 'datasets2', 'BUS-UCLM', '__MACOSX',
             '__pycache__', 'MobileViTv3-ref', 'audit', '_build'}
EXTS = ('.json', '.csv', '.txt', '.log', '.out', '.err', '.yaml', '.yml',
        '.tex', '.md', '.py', '.ipynb')

# (claim id, where in paper, printed value)
CLAIMS = [
    ('T2.ours.acc', 'Abstract/Table 2', '92.95'), ('T2.ours.prec', 'Table 2', '92.81'),
    ('T2.ours.rec', 'Table 2', '92.47'), ('T2.ours.f1', 'Table 2', '92.58'),
    ('T2.ours.loss', 'Table 2', '0.315'),
    ('T2.usn.acc', 'Table 2', '91.67'), ('T2.usn.prec', 'Table 2', '90.72'),
    ('T2.usn.rec', 'Table 2', '91.74'), ('T2.usn.f1', 'Table 2', '90.96'),
    ('T2.usn.loss', 'Table 2', '0.337'),
    ('T2.mamba.acc', 'Table 2', '91.02'), ('T2.mamba.prec', 'Table 2', '92.41'),
    ('T2.mamba.rec', 'Table 2', '88.32'), ('T2.mamba.f1', 'Table 2', '90.03'),
    ('T2.mamba.loss', 'Table 2', '0.351'),
    ('T2.fcmonly.acc', 'Table 2', '85.26'), ('T2.fcmonly.loss', 'Table 2', '0.503'),
    ('T2.mv3only.acc', 'Table 2', '73.71'), ('T2.mv3only.prec', 'Table 2', '71.12'),
    ('T2.mv3only.rec', 'Table 2', '77.63'), ('T2.mv3only.f1', 'Table 2', '72.86'),
    ('T2.mv3only.loss', 'Table 2', '0.620'),
    ('T3.benign.prec', 'Table 3', '0.933'), ('T3.benign.f1', 'Table 3', '0.943'),
    ('T3.benign.auc', 'Table 3', '0.951'), ('T3.malig.prec', 'Table 3', '0.923'),
    ('T3.malig.f1', 'Table 3', '0.889'), ('T3.malig.auc', 'Table 3', '0.962'),
    ('T3.normal.prec', 'Table 3', '0.929'), ('T3.normal.f1', 'Table 3', '0.946'),
    ('T3.normal.auc', 'Table 3', '0.994'),
    ('T4.macro_auc', 'Table 4', '0.969'), ('T4.inference_ms', 'Table 4/Abstract', '7.222'),
    ('T4.params_M', 'Table 4/Abstract', '3.712'), ('T4.flops_G', 'Table 4/Abstract', '1.176'),
    ('S3.2.preproc_s', 'Sec 3.2', '1.59'),
    ('S4.1.benign_area', 'Sec 4.1', '6.70'), ('S4.1.malig_area', 'Sec 4.1', '14.72'),
]

NUM = re.compile(r'(?<![\w.])-?\d+\.\d+(?:[eE]-?\d+)?')


def matches(found: str, printed: str) -> bool:
    places = Decimal(printed).as_tuple().exponent  # e.g. -2
    q = Decimal(1).scaleb(places)
    try:
        x = Decimal(found)
    except Exception:
        return False
    target = Decimal(printed)
    for cand in (x, x * 100, x / 100):
        if cand.quantize(q) == target:
            # require at least the printed precision in the source, so that
            # e.g. "0.9" never matches "0.933"
            return True
    return False


def main():
    print('claim_id\tpaper_location\tprinted\tfile\tline\tcontext')
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if not fn.endswith(EXTS) or fn.startswith('metrics_epoch_'):
                continue
            path = os.path.join(dirpath, fn)
            try:
                if os.path.getsize(path) > 20_000_000:
                    continue
                with open(path, errors='replace') as fh:
                    lines = fh.readlines()
            except OSError:
                continue
            for i, line in enumerate(lines, 1):
                nums = NUM.findall(line)
                if not nums:
                    continue
                for cid, loc, printed in CLAIMS:
                    p_places = -Decimal(printed).as_tuple().exponent
                    for n in nums:
                        n_places = len(n.split('.')[1].split('e')[0].split('E')[0])
                        if n_places + 2 < p_places:  # source too coarse
                            continue
                        if matches(n, printed):
                            ctx = line.strip().replace('\t', ' ')[:160]
                            print(f'{cid}\t{loc}\t{printed}\t{os.path.relpath(path, ROOT)}\t{i}\t{ctx}')
                            break


if __name__ == '__main__':
    main()
