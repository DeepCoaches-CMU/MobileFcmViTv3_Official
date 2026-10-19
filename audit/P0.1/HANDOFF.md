# P0.1 → P0.2 and onward: team handoff

From: John Dalton Gibson (P0.1 owner) · 2026-10-15

P0.1 (freeze and inventory) is **Done with documented gaps**. This note tells each owner
what P0.1 produced for their task, what is already known, and where to start. Task IDs
match the Research Plan workbook and the Task Implementation Handbook.

## What P0.1 produced

All files are in `audit/P0.1/`:

| File | Use it for |
|---|---|
| `inventory.csv` | Every frozen file, with its SHA-256 hash, size, mtime and type, and whether git tracks it. |
| `run_register_wandb.csv` | All 117 W&B runs, with config, CLI overrides, recorded commit, warm-start checkpoint, evaluation N and class support, final metrics, and params/FLOPs/latency. |
| `headline_trace.csv` | Every number in the submitted paper, mapped to its source or marked UNTRACEABLE / CONTRADICTED. |
| `gaps.md` | 18 provenance gaps. Read this first. |
| `environment.txt` | Software versions, plus the host and GPU of the headline run. |
| `scripts/` | Regenerates the inventory, the run register and the paper-number search. |

**Frozen original:** `mobilefcmvitv3_P0.1_freeze_20261015.tar` (read-only, on the P0.1
owner's PC). Its SHA-256 is in `archive_tar.sha256`, and the submitted paper's hash is in
`paper.sha256`. The cluster copy is at
`/data/user_data/jgibson2/mobilefcmvitv3_P0.1_freeze_20261015/`. It is **not** read-only,
so don't write to it. Git tag: `p0.1-freeze-20261015`.

## Rules for everyone

- **Check hashes.** Before relying on a result file, checkpoint or manifest, confirm its
  SHA-256 matches `inventory.csv`. A mismatch means it changed after the freeze.
- **Cite W&B run IDs** (e.g. `run-20260425_071258-2z1ie560`) from the run register, not
  folder names. Folders were reused and overwritten.
- **Write new outputs to your own task folder** (`audit/P0.2/`, `audit/P0.3/`, …), never
  over existing results.
- **Label regenerated evidence as such.** A prediction file produced today from an old
  checkpoint is new evidence, not the original.
- **Treat the submitted paper's numbers as unusable evidence until P0.2 decides
  otherwise.** In particular, its "held-out test" results are validation results.

## What P0.1 established

1. Table 2 (our model), Table 3, Table 4's macro-AUC and Fig. 2 are tune_05_augmix's
   checkpoint scored **at epoch 0 on the old 156-image validation set**. The split is
   `data/BUSI_split.json`: 624 train / 156 val, no test partition. The same 156 images
   selected that checkpoint.
2. The Mamba Vision and "UltraScanNet" rows are on the same 156 validation images
   (`baseline/mambavision_fcm/metrics/extended_metrics.txt`). The source file labels the
   "UltraScanNet" row `MobileFCMVit2`.
3. The MobileViTv3-Only row, and FCM-Only's precision/recall/F1/loss, appear in no result
   file. No April run of either model exists, so the 19.24-point claim has no source.
4. The paper's 545/115/120 split matches neither manifest: old = 624/156,
   v2 = 544/115/119 (built July 2026).
5. 7.222 ms is a single desktop-GPU timing; the same model measured 3.7–11.9 ms elsewhere.
6. No code in the repo implements the paper's preprocessing (anisotropic diffusion, CLAHE),
   the 1.59 s timing, the lesion-area statistics or the KS test.
7. *Corrected 2026-10-19:* the April commits are recoverable from the old repo
   `DeepCoaches-CMU/ultrascannet_reproduced` (gaps.md item 9). Recorded commits still
   don't always match the code that ran. Only 3 per-sample prediction CSVs exist, none of them for a paper number.

## Gate G0: next tasks

P0.2–P0.5 depend only on P0.1, so they can run in parallel.

### P0.2: 120-vs-156 discrepancy (Evaluation lead)

- **Start with:** the `Fig2` and `T2.ours.*` rows of `headline_trace.csv`, and gaps.md items 1–2.
- **Known:** Fig. 2 is `results/mobilefcmvitv3_s_BUSI/best_confusion_matrix.png`. Its 156
  images are the old validation set (87 benign / 42 malignant / 27 normal).
- **Steps:**
  1. Rebuild the 156-image list from `data/BUSI_split.json` against the raw BUSI images.
     `datasets/BUSI_split/` was rebuilt in July and no longer matches it.
  2. Run the **headline checkpoint** `checkpoints/mobilefcmvitv3_s_BUSI/model_best.pth.tar`
     (and the April tune_05 weights it started from,
     `tuning/tune_05_augmix_oldsplit/checkpoints/model_best.pth.tar`) on those images. Try
     both the raw `state_dict` and `ema_state_dict`, since training kept whichever scored
     higher. Save a per-sample CSV with true label, prediction and class probabilities.
     *Correction 2026-10-19:* do **not** use `tuning/tune_05_augmix/checkpoints/model_best.pth.tar`;
     that file is the July retrain (epoch 23, 89.57 %), not the April model.
  3. Confirm the CSV reproduces `[[83,2,2],[6,36,0],[0,1,26]]` and 92.95%. Recompute every
     metric from that one CSV with a script.
  4. Write the discrepancy report and record the retain/retire decision.
- **Expected decision:** retire the result. It is validation on the checkpoint-selection
  set, not a held-out test.

### P0.3: architecture versus paper (Modeling lead)

- **Start with:** gaps.md items 6 and 10, the model code in `mobilefcmvitv3/models/`,
  and the paper's Sections 3.1–3.3.
- **Known conflict:** the paper describes:
  - a 4-channel input with a pixel-space FCM map,
  - a Dense(1024) head,
  - TFLite export,
  - CLAHE and anisotropic-diffusion preprocessing.

  The code is a 3-channel model with feature-space soft clustering and torchvision
  augmentation.
- **Steps:**
  1. Establish what was trained from the headline checkpoint's own weights (e.g. the stem
     convolution's input-channel count and the head layer shapes), not from configs.
     Recorded commits don't match the code that ran.
  2. Instantiate the model from a clean checkout and run a forward pass.
  3. Draw the architecture diagram from that forward pass.
  4. List each manuscript claim that must change.
- **Also include:** the learned FCM cluster centers in the tune_05 and ablation
  checkpoints are identical in direction (pairwise cosine 1.0). Memberships are therefore
  uniform (1/3), and the clustering carries no per-image information. This feeds P2.2.

### P0.4: checkpoint lineage (Modeling lead)

- **Start with:** the `pretrained_weights`, `git_commit` and `eval_n` columns of
  `run_register_wandb.csv`.
- **Known chain:** tune_01 → tune_05 (logged no git commit) → reloaded at epoch 0 for the
  headline evaluation.
- **To trace:** tune_01's own starting checkpoint (expected to be the VOC checkpoint; verify
  this from its W&B record).
- **Record:**
  - The April ablation outputs were overwritten on 2026-07-21; only their W&B summaries
    remain.
  - tune_11 (cost-sensitive loss) was motivated by test-set error analysis. Log that
    test-set access.
- **Deliverable:** a lineage table that sorts runs into *safe to compare* versus *needs a
  rerun*.

### P0.5: dataset manifest and labels (Data lead)

- **Start with:** the `dataset_image` and `split_manifest` rows of `inventory.csv`, plus
  `scripts/build_busi_split.py`.
- **Reconcile:**
  - raw 780 images versus 778 in v2 — which two were removed, and why;
  - the paper's Table 1 counts (437/210/133);
  - the paper's 545/115/120 split;
  - exclusion of mask files.
- **Note:** class weights `[0.595, 1.238, 1.955]` are hard-coded from all 780 images, not
  the training split (gaps.md item 7; code-audit item C7).

## After G0: gate G1

- **P1.1, patient-level split (Data lead).** Check early whether patient IDs can be
  recovered at all. The paper says 780 images from 600 patients, so some patients have
  several images, but we have not found a per-image patient mapping in the public release.
  If none exists, document the limitation and don't claim patient-independent results.
- **P1.2, single-source evaluation (Evaluation lead).** Build on the per-sample CSV that
  `mobilefcmvitv3/scripts/test.py` already writes and on the P0.2 metric script. Every
  table and figure is generated from one prediction file.
- **P1.3, locked protocol (Research lead).** Name the split manifest version and its
  SHA-256 hash.

## Who to ask

- **P0.1 evidence and audit files:** John Dalton Gibson.
- **Run history:** check the run register first; it holds more than folder names suggest.
