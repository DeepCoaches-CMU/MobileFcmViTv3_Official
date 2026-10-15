import os
import subprocess

# Root of the external MambaVision repo this baseline drives (not part of this
# repo). Set MAMBAVISION_ROOT in your environment, e.g.:
#   export MAMBAVISION_ROOT=/path/to/MambaVision/mambavision
MAMBAVISION_ROOT = os.environ.get("MAMBAVISION_ROOT")
if not MAMBAVISION_ROOT:
    raise SystemExit("Set MAMBAVISION_ROOT to the external MambaVision repo root before running this script.")

# BUSI_SPLIT_DIR defaults to this repo's own datasets/BUSI_split (see paths.yaml)
BUSI_SPLIT_DIR = os.environ.get("BUSI_SPLIT_DIR", "datasets/BUSI_split")

model_list = [
    "mambavision_fcm_net",
]

dataset_list = [
    BUSI_SPLIT_DIR,
]

dataset_names = [
    'BUSI',
]

data_len = [
    3,
]

# ✅ Launch validation runs
for model in model_list:
    for i, dataset_path in enumerate(dataset_list):
        experiment_name = f"{model}_{dataset_names[0]}"
        args_path = f"{MAMBAVISION_ROOT}/output_busi/{experiment_name}/args.yaml"
        checkpoint_path = f"{MAMBAVISION_ROOT}/output_busi/{experiment_name}/checkpoint-113.pth.tar"

        cmd = [
            "python3", "val_simple.py",
            "-c", args_path,
            "--loadcheckpoint", checkpoint_path,
            "--infer_only"
        ]

        print(f"\n✅ Validating: {experiment_name}\n")
        subprocess.run(cmd)

# python3 val_simple.py -c $MAMBAVISION_ROOT/best-model-article-1/args.yaml --loadcheckpoint $MAMBAVISION_ROOT/best-model-article-1/model_best.pth.tar --validate_only