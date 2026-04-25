import torch
import torch.nn as nn
from timm.models import create_model

class MobileViTv3OnlyNet(nn.Module):
    """
    MobileViTv3-only model for ablation: standard MobileViTv3 backbone with classification head, no FCM.
    """
    def __init__(self, num_classes=3, pretrained=False, **kwargs):
        super().__init__()
        # Use timm to create MobileViTv3 backbone
        self.backbone = create_model(
            'mobilevitv3_small_100', pretrained=pretrained, num_classes=num_classes, **kwargs
        )

    def forward(self, x):
        logits = self.backbone(x)
        return logits, None, None  # For compatibility
