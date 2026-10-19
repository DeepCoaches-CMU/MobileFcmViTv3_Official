# P0.1 — Freeze and inventory artifacts

Owner: John Dalton Gibson · Date: 2026-10-15

## Archive (cluster copy, not read-only — see below)

`/data/user_data/jgibson2/mobilefcmvitv3_P0.1_freeze_20261015/`

| Item | What it is |
|---|---|
| `repo_snapshot/` | `rsync -a` copy of the working tree, `.venv/` excluded (6.1 GB, 18,493 files). Original mtimes preserved. |
| `repo_all_refs.bundle` | `git bundle --all`: every branch and tag, including `p0.1-freeze-20261015`. |
| `paper/` | Empty on the cluster (quota). The submitted PDF `43_MobileFCMViTv3_A_Lightweigh.pdf` is kept beside the tar on the owner's PC; its SHA-256 is in `paper.sha256`. |
| `FREEZE_STARTED` / `FREEZE_FINISHED` | Copy timestamps. |

**Not done on the cluster copy.** The filesystem quota filled up right after the snapshot
was written (2026-10-15), so:
- the audit files were not copied in,
- no `SHA256SUMS` was written, and
- `chmod -R a-w` failed, so the cluster copy is **not** read-only.

**Protected copy.** The archive was streamed as a single tar file:
`tar --sort=name -cf - mobilefcmvitv3_P0.1_freeze_20261015`. Its SHA-256 and size are in
`archive_tar.sha256`. The copy kept as the read-only original is that tar file on the
owner's Windows PC: `C:\Users\rabia\Dalton\mobilefcmvitv3_P0.1_freeze_20261015.tar`.
Keep it unextracted, because the archive contains 3,419 symlinks, 113 of them already broken.

To verify:
- **Snapshot files:** check them against `inventory.csv`.
- **Tar file:** compare its hash with `archive_tar.sha256`, using `certutil -hashfile <tar> SHA256`
  on Windows.

## Deliverables in this folder

- `inventory.csv`: one row per archived file, with path, type, size, mtime, SHA-256 and
  whether git tracks it.
- `run_register_wandb.csv`: one row per W&B run (117). Columns: config, CLI overrides,
  git commit and whether it exists here, warm-start weights, checkpoint and results
  directories, evaluation N and class support, final metrics, and params/FLOPs/latency.
  Every cell comes from the run's own W&B record. Anything unavailable is marked
  `MISSING`.
- `headline_trace.csv`: every quantitative claim in the paper mapped to its source
  artifact, or marked UNTRACEABLE or CONTRADICTED.
- `gaps.md`: missing provenance and uncertainties.
- `environment.txt`: Python/torch/CUDA, `pip freeze`, the host and GPU recorded for the
  headline run, and the git state.
- `scripts/`: regenerates everything above.
  ```
  python audit/P0.1/scripts/build_inventory.py <archive>/repo_snapshot > audit/P0.1/inventory.csv
  .venv/bin/python audit/P0.1/scripts/build_run_register.py . > audit/P0.1/run_register_wandb.csv
  python audit/P0.1/scripts/trace_paper_numbers.py . > trace.tsv   # raw leads; hand-check each
  ```

## Acceptance status

- **Originals preserved, hashes recorded: met.**
- **Every headline number linked to a run or marked untraceable: met.** See
  `headline_trace.csv`.
- **Missing items listed: met.** See `gaps.md`.
- **Every reported result mapped to run ID, config, checkpoint, split manifest, prediction
  file and code commit: not met, and it can't be met from existing evidence.**
  - There are no per-sample prediction files for any paper number.
  - Code commits for the April runs are not in this repo. *Correction 2026-10-19 (P0.3):*
    they are recoverable from the old repo `DeepCoaches-CMU/ultrascannet_reproduced`.
  - The ablation rows have no run at all.

  That's the honest outcome of this task. Per the handbook, those results should be
  treated as untraceable, and the old headline result should be retired or re-derived in
  P0.2.
