# P0.1 — Provenance gaps and uncertainties

Owner: John Dalton Gibson · Compiled 2026-10-15 · Freeze tag `p0.1-freeze-20261015` (HEAD 3d4f630)

Each item says what is missing, how it was established, and what it blocks. Items are
facts about the evidence. None of them has been fixed here; fixes belong to later tasks.

## A. Headline results (paper ↔ runs)

1. **The "held-out test set" results are validation-set results.** Table 2 (our model),
   Table 3, Table 4 macro-AUC and Fig. 2 all come from tune_05_augmix's checkpoint
   (the run loaded `tuning/tune_05_augmix/checkpoints/model_best.pth.tar`; *correction
   2026-10-19, P0.3:* that path now holds a July retrain, and the April file survives as
   `tuning/tune_05_augmix_oldsplit/checkpoints/model_best.pth.tar`). It was reloaded in full
   (318/318 params), scored at epoch 0 on the old 156-image validation set, and logged
   as `val/*` (W&B `run-20260425_065747-149rh2m3` and `run-20260425_071258-2z1ie560`).
   The run log says `Train: 624 | Val: 156`, and the split manifest in use
   (`data/BUSI_split.json`) has no test partition. The same validation set chose
   tune_05's best checkpoint. → P0.2, P0.4, P4.4.
2. **The 120-vs-156 discrepancy has an identified cause.** Fig. 2 is
   `results/mobilefcmvitv3_s_BUSI/best_confusion_matrix.png`. Its 156 is the old validation
   set (87/42/27). The paper's 545/115/120 split matches neither manifest:
   old = 624/156, v2 = 544/115/119 (v2 built 2026-07, after submission). → P0.2.
3. **Both baseline rows of Table 2 are on the same 156-image validation set.** They come from
   `baseline/mambavision_fcm/metrics/extended_metrics.txt`, which stores summaries only:
   no per-sample predictions, and no checkpoint stated per row. The "UltraScanNet" row
   is labelled **`MobileFCMVit2`** in the source file. Someone needs to confirm which
   model it is.
4. **The ablation rows can't be traced.** MobileViTv3-Only
   (73.71/71.12/77.63/72.86/0.620) has no source anywhere. FCM-Only's 85.26 equals
   133/156, a value logged by runs of the full model (`run-20260425_062354-r77il796`,
   `run-20260425_064106-wpv1gtvz`) and by efficientnet_b0 in the baseline file. Its
   82.00/89.00/85.00/0.503 have no source. No April run of either standalone model exists
   (`mobilevitv3_only.py` could not run before 2026-07). The 19.24 pp claim is derived
   from these values.
5. **7.222 ms is one desktop-GPU forward-pass timing.** It comes from
   `run-20260425_071258-2z1ie560`. The same model measured 3.68–11.9 ms in other runs
   (see `run_register_wandb.csv`, column `params_flops_infer`). There's no phone or
   end-to-end measurement. → P3.1/P3.2.
6. **No code or output exists for parts of Sections 3.2 and 4.1.** Nothing in the repo
   implements the six-step preprocessing (anisotropic diffusion, CLAHE, …), the 1.59 s
   timing, the lesion-area statistics or the KS test. The training code uses torchvision
   crop/flip/colour-jitter with ImageNet normalisation on RGB.
7. **Class weights come from the full 780-image dataset**, not the training split. They are
   hard-coded in `mobilefcmvitv3/utils/class_imbalance.py` and include the val/test
   images and the duplicates. → C7, P2.5.
8. **Fig. 3 is a W&B dashboard screenshot of a tune_01_lr_aug run**, not a figure generated
   by a script. Two 199-epoch tune_01 runs are candidates (`run-20260424_134733-su0jyg0h`,
   `run-20260425_055534-lra75jab`). tune_01 is an upstream warm-start, not the reported
   model.

## B. Code version

9. **Every April W&B run logged a commit from an earlier git history** (0df6e8c, 170f399,
   1dbf997, 81ea701, e19e9fa). *Correction 2026-10-19 (P0.3):* these commits are not in
   this repo, but **all five exist in the old repo
   `git@github.com:DeepCoaches-CMU/ultrascannet_reproduced.git`** (the remote those runs
   recorded), so the April code is recoverable. The model code there is byte-identical
   to this repo's `5158ed1` (see `audit/P0.3/ARCHITECTURE_SPEC.md` §6). The tune_05/tune_06
   runs that produced the headline checkpoint still logged **no commit**; the nearest
   earlier commit is `e19e9fa` (2026-04-24 14:14 EDT).
10. **A recorded commit isn't the code that ran.** The headline runs logged 5158ed1, but
    that commit's `mobilefcmvitv3_s_busi.yaml` points `pretrained_weights` at the VOC
    checkpoint, while the runs loaded tune_05's checkpoint. tune_11 logged b78bbef,
    whose `train.py` has no `cost_sensitive_beta`. In both cases the runs used uncommitted
    changes, and W&B saved no diff of them.

## C. Evidence not protected by version control

11. Results, checkpoints, `data/` split manifests and `*.tex` are gitignored, so before
    this freeze nothing made accidental replacement detectable. They are now hashed in
    `inventory.csv`.
12. **Most runs have no per-sample predictions.** Only 3 prediction CSVs exist:
    `test_results/{tune_05_augmix,tune_07_balanced_sampler}/predictions_test.csv` and
    `test_results/tuning_05_augmix/predictions_validation.csv`. None of them is from the
    April headline evaluation. Predictions regenerated from a checkpoint would be new
    evidence and must be labelled that way (P0.2/P1.2).
13. **Old-split runs can't be fully reconstructed.** `datasets/BUSI_split/` was rebuilt
    on 2026-07-21, so it no longer holds the 624/156 layout the April runs read.
    `data/BUSI_split.json` records which files the split contained, but not their
    hashes.
14. **Some links in the snapshot point outside the archive:**
    - The 3,295 files in `datasets/BUSI_split_kfold/` are absolute symlinks into the live
      repo's `datasets/BUSI_split/`. The frozen snapshot of that folder holds the same
      content as of the freeze, but the links themselves point to the live copy.
    - Two W&B debug-core logs are symlinks into `~/.cache/wandb`, which is not archived.
15. **Some April checkpoints were overwritten.** `ablation_results/*` was re-run on
    2026-07-21, so the April ablation outputs now exist only as W&B summaries.
    `checkpoints/mobilefcmvitv3_s_BUSI/` and `results/mobilefcmvitv3_s_BUSI/` are still
    the April files.

## D. Outside the repo

16. The original ZIP isn't in the workspace. If a ZIP was submitted, it should be added
    to `<archive>/paper/` and hashed.
17. The submitted paper PDF (`43_MobileFCMViTv3_A_Lightweigh.pdf`) and the plan
    documents are not on disk. They need to be copied into `<archive>/paper/`.
18. The archive filesystem (`/data/user_data/jgibson2`) was 98% full at freeze time
    (22 GB free).
