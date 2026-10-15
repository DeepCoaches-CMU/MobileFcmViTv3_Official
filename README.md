# UltraScanNet — MobileFCMViTv3

**UltraScanNet** is a research codebase for MobileFCMViTv3, a lightweight, high-performance deep learning model for breast ultrasound image classification. The model combines the MobileViTv3 backbone with a differentiable feature-space Fuzzy C-Means (FCM) clustering module, designed to improve classification of challenging malignant cases in the BUSI dataset.

## Key Features
- **MobileViTv3 Backbone:** Efficient vision transformer architecture for mobile and edge deployment.
- **Differentiable FCM Clustering:** Integrates feature-space fuzzy clustering for improved representation and decision boundaries, especially for hard malignant/benign cases.
- **Standalone Metrics Suite:** Computes accuracy, F1, ROC-AUC, PR-AUC, and confidence intervals without external dependencies.
- **SLURM Integration:** Ready-to-use scripts for training, evaluation, ablation, and tuning on SLURM clusters.
- **WandB Logging:** Integrated experiment tracking and visualization.

## Project Structure
- `mobilefcmvitv3/` — Model, clustering, training, validation, and utility scripts
- `tuning/` — Hyperparameter tuning configs and results
- `slurm_scripts/` — SLURM job scripts for all experiments
- `datasets/` — BUSI dataset splits (train/validation/test — see below)
- `fcm/` — Pixel-space FCM utilities (not used in main model)
- `paths.yaml` — Centralized path management
- `scripts/` — Dataset-split rebuilding, seed aggregation, report-table generation

## How to Run
1. **Set up your Python environment:**
	- Python 3.10+
	- Install dependencies from `requirements.txt`
	- Set `PYTHONPATH` to the repo root
2. **Prepare the dataset (train/validation/test, deduped):**
	- `python3 setup_dataset.py` downloads raw BUSI and materializes
	  `datasets/BUSI_split/{train,validation,test}/` from `data/BUSI_split_v2.json`.
	- That manifest is built by `scripts/build_busi_split.py`, which hashes every
	  raw image to drop duplicates (including one cross-class mislabeled
	  duplicate found in the raw Kaggle dataset itself) and does a stratified
	  70/15/15 train/val/test split. Only re-run it if you need to regenerate
	  the manifest from scratch — the checked-in `data/BUSI_split_v2.json` is
	  the manifest actually used for the reported results.
	- `data/BUSI_split.json` (train/validation only, no dedup) is kept for
	  provenance of earlier results but is no longer the default.
3. **Train the model:**
	- Submit a job, e.g.:
	  ```bash
	  sbatch slurm_scripts/mobilefcmvitv3_training.slurm tuning/tune_01_lr_aug.yaml tune_01_lr_aug
	  ```
	- If `checkpoint_dir` already has a run in it, `train.py` refuses to start
	  and asks for an explicit `resume` or `force` third argument to the
	  SLURM script (`sbatch ... <config> <experiment> resume`) — this is what
	  previously allowed repeated resubmissions to silently clobber a run.
4. **Run ablations:**
	- `python3 mobilefcmvitv3/scripts/ablation.py` runs all 10 ablation
	  variants (see `mobilefcmvitv3/scripts/ablation.py`'s `ABLATIONS` list),
	  including the FCM-only and MobileViTv3-only standalone architectures.
	  All "_pretrained" variants warm-start from `ablation_base.yaml`'s common
	  untrained-on-BUSI checkpoint, not from any already fine-tuned model —
	  see that file's header comment for why this matters.
5. **Evaluate:**
	- `mobilefcmvitv3/scripts/validate.py` (checkpoint-selection metrics) and
	  `mobilefcmvitv3/scripts/test.py` (`--split test` for the held-out test
	  set) both accept `train`/`validation`/`test`.
	- `mobilefcmvitv3/scripts/eval_external.py` runs a checkpoint against
	  BUS-UCLM (`datasets2/BUSI_split`) as an out-of-distribution
	  generalization check — never run this against `tune_06_regdrop`, which
	  trained on that same data.
6. **Regenerate report tables:**
	- `python3 scripts/generate_report_tables.py` reads the real result JSON
	  files and writes `report_tables_generated.tex` (`\input`-ed from
	  `report.tex`), so table rows can't drift from or be mistranscribed from
	  the underlying runs.
	- `mobilefcmvitv3/scripts/aggregate_seeds.py` computes mean ± std across
	  multiple seed runs of the same config for `--seed_summary`.

## Citation
If you use this codebase, please cite the original MobileViTv3 and FCM papers, and reference this repository.

## License
This project is for academic research and benchmarking. See LICENSE for details.
