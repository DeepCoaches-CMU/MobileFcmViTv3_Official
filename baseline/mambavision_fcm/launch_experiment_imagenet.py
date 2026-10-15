import os
import subprocess

# Root of the external MambaVision repo this baseline drives (not part of this
# repo). Set MAMBAVISION_ROOT in your environment, e.g.:
#   export MAMBAVISION_ROOT=/path/to/MambaVision/mambavision
MAMBAVISION_ROOT = os.environ.get("MAMBAVISION_ROOT")
if not MAMBAVISION_ROOT:
    raise SystemExit("Set MAMBAVISION_ROOT to the external MambaVision repo root before running this script.")

model_list = [
    # "mambavision_fcm_net"
    "mamba_vision_T3_pyramid_fusion",
    # "mamba_vision_T4_pyramid_fusion_refinement"
    # 'mamba_vision_T6_refinement_attn',
    # "mamba_vision_T5_refinement"
    # "resnet50",
    # "mobilenetv2_100",
    # "densenet121",
    # "vit_small_patch16_224",
    # "efficientnet_b0",
    # "convnext_tiny",
    # "swin_tiny_patch4_window7_224",
    # "deit_tiny_patch16_224",
    # "maxvit_tiny_rw_224",
]



for model in model_list:
        experiment_name = f"{model}"

        cmd = ["python3", "train.py",
            "-c", f"{MAMBAVISION_ROOT}/configs/imagenet/mambavision_tiny2_1k.yaml",
            f"--group={model}",
            f"--model={model}",
            f"--experiment=imagenet_{experiment_name}",
            f"--resume={MAMBAVISION_ROOT}/output_imagenet/imagenet_mamba_vision_T3_pyramid_fusion/checkpoint-190.pth.tar",
            f"--log-wandb"
        ]

        print(f"\n🚀 Running: {experiment_name}\n")
        subprocess.run(cmd)