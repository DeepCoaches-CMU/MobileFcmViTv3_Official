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
- `datasets/` — BUSI dataset splits
- `fcm/` — Pixel-space FCM utilities (not used in main model)
- `paths.yaml` — Centralized path management

## How to Run
1. **Set up your Python environment:**
	- Python 3.10+
	- Install dependencies from `requirements.txt`
	- Set `PYTHONPATH` to the repo root
2. **Prepare the dataset:**
	- Place BUSI data in `datasets/BUSI_split/`
3. **Train the model:**
	- Submit a job, e.g.:
	  ```bash
	  sbatch slurm_scripts/mobilefcmvitv3_training.slurm tuning/tune_01_lr_aug.yaml tune_01_lr_aug
	  ```
4. **Evaluate or tune:**
	- Use the corresponding SLURM scripts and config files in `tuning/`

## Citation
If you use this codebase, please cite the original MobileViTv3 and FCM papers, and reference this repository.

## License
This project is for academic research and benchmarking. See LICENSE for details.
