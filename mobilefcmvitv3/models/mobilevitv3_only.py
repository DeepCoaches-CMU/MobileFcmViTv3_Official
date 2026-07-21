import torch
import torch.nn as nn

from .mobilevitv3_block import build_mobilevitv3_s_backbone


class MobileViTv3OnlyNet(nn.Module):
    """
    MobileViTv3-only model for ablation: the same custom MobileViTv3-S
    backbone used by MobileFCMViTv3, with a classification head and no FCM
    module at all (not just a zeroed-out fusion — the clustering/fusion
    parameters don't exist in this model).

    Note: previously this used timm's create_model('mobilevitv3_small_100', ...),
    which does not exist in timm's model registry (only mobilevit_s/xs/xxs and
    mobilevitv2_* are registered) — that call would raise at construction time.
    This repo implements MobileViTv3 itself (see mobilevitv3_block.py), so the
    backbone-only ablation now reuses that implementation instead.
    """
    def __init__(self, num_classes=3, dropout=0.1, attn_dropout=0.0, ffn_dropout=0.0, **kwargs):
        super().__init__()
        self.backbone, exp_ch = build_mobilevitv3_s_backbone(
            in_chans=3, dropout=dropout, attn_dropout=attn_dropout, ffn_dropout=ffn_dropout,
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(exp_ch, num_classes),
        )

    def forward(self, x):
        x = self.backbone['conv_1'](x)
        x = self.backbone['layer_1'](x)
        x = self.backbone['layer_2'](x)
        x = self.backbone['layer_3'](x)
        x = self.backbone['layer_4'](x)
        x = self.backbone['layer_5'](x)
        x = self.backbone['conv_1x1_exp'](x)
        logits = self.classifier(x)
        return logits, None, None  # no auxiliary FCM losses — for compatibility with train.py
