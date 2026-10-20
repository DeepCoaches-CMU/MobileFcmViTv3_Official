# P0.4: Checkpoint and training provenance audit

Owner: Obasi. Draft prepared from the repo (`DeepCoaches-CMU/MobileFcmViTv3_Official`, HEAD `ef81b68`) and the P0.1/P0.3 audit files. Table: `P0.4_checkpoint_lineage.csv`.

**Limits of this audit.** No checkpoints, W&B logs or prediction CSVs were available to me (they are gitignored). Run facts come from Dalton's `run_register_wandb.csv`, whose `val_top1` column looks like W&B last-epoch summaries, not best-epoch values. Items marked "verify" need a check on the cluster.

## 1. Answers to the P0.4 question

| Question | Answer |
|---|---|
| Where did the headline checkpoint come from? | VOC segmentation encoder, then `tune_01` (April), then `tune_05` (April), then reloaded for the "headline" run. No training happened in the headline run: it scored the loaded checkpoint at epoch 0 and no later epoch beat it. |
| What was it scored on? | The old 156-image validation split (87/42/27), the same set that selected the checkpoint. 145/156 = 92.95%. |
| Did test data influence selection? | The paper's "test set" was the selection set, so yes for the headline. In the v2 era, selection used the 115-image validation set. The 119-image test set was used for error analysis that designed the tune_11 loss, so it is no longer pristine. |
| How was "best" chosen? | Highest validation top-1 each epoch, taking the better of live and EMA weights (not recorded in the checkpoint). Patience 80 in the configs; the paper says 20. |
| Are the ablations comparable to the headline? | No. The April ablations that warm-started from tune_05 are bit-identical copies of the headline (19 runs, all 92.9487%). The ablation_base.yaml fix (July) produced valid runs, but with one seed and N=115. |

## 2. Findings

1. **Headline result is a validation score, not a test score.** Compromised (L1).
2. **The two ablation rows the paper's main claim depends on have no source run** (L3). The numbers coincide with other runs: 85.26 = 133/156 (full model, VOC init) and 73.72 = 115/156 (tune_01's "baseline run"). Treat these as leads.
3. **19 April ablation runs measured nothing** (L4): they loaded tune_05 and reproduced its number.
4. **Warm-starting from a checkpoint trained on overlapping data inflates results.** The team's own k-fold comparison shows it: tune_01-initialised folds scored 95.5-99.2% (mean 97.3), VOC-initialised folds 77-87% (L9). The v2 runs that chain from tune_01 are clean only if the tune_01 checkpoint on disk was the July retrain on v2 train. **Verify by checkpoint hash and mtime** (L8).
5. **The honest estimate is about 81-89%, not 93%** (L7, L8, L9), on 115-133-image evaluation sets, one image about 0.8 pp.
6. **Best-epoch selection on small validation sets is itself optimistic:** the gap between last-epoch and best-epoch values in the register is 2-4 points.
7. **v2 test results are the key unknown** (L10). A code comment implies 32 test errors; on 119 images that is about 73% accuracy. Compute it from `test_results/tune_05_augmix/predictions_test.csv` before anything else.
8. **Code-vs-paper mismatches that affect lineage:** ran with uncommitted changes (headline and tune_11), W&B saved no diff; vflip augmentation was never applied before 2026-09-05; class weights were computed from all 780 images; FCM clusters in the trained checkpoint collapsed (P0.3), so the ablation gaps may not come from clustering.

## 3. Proposed Decision Log entries

| Question | Proposed decision | Rationale |
|---|---|---|
| Was the headline checkpoint trained or selected using test data? | Selected on the 156-image validation set that was then reported as test. Retire the result. | L1, gaps.md 1-2 |
| Which architecture generated the submitted results? | The 3-channel feature-space soft-clustering model (P0.3). The 4-channel pixel-FCM description is not what ran. | P0.3 spec |
| Can old results be trusted? | No for L1-L6. Usable as development evidence only: L7-L9(b). | This table |
| New rule | No run may warm-start from any checkpoint trained on BUSI. Use the common VOC (or ImageNet) checkpoint. | L9 leak |
| New rule | Report seeds, last-epoch and best-epoch metrics, and which weights (live/EMA) were evaluated. | Finding 6 |

## 4. Verify on the cluster (in order)

1. Metrics from `test_results/tune_05_augmix/predictions_test.csv` (N, accuracy, malignant recall).
2. Hash and mtime of `tuning/tune_01_lr_aug/checkpoints/model_best.pth.tar` against the July runs.
3. `best_metrics.json` versus W&B summary for the L7-L9 runs (explains the seed-summary mismatch).
4. Which weights (live or EMA) `best_metrics.json` came from.
5. Identity of the "UltraScanNet" baseline row (labelled MobileFCMVit2).

## 5. Status

P0.4's acceptance criterion (every result mapped, no test-driven selection) is **not met for the headline**: the finding is that it cannot be met. Close the task as "Done with findings" once items 1-3 above are verified and the Decision Log is updated.
