import torch
import torch.nn as nn

class FCMOnlyNet(nn.Module):
    """
    FCM-only model for ablation: no backbone, just FCM clustering and classification.
    Expects input features (e.g., global average pooled image features) as input.
    """
    def __init__(self, num_classes=3, fcm_k=3, fcm_proj_dim=32, tau=1.0, fcm_m=2.0, normalize_feat=True):
        super().__init__()
        self.fcm_proj = nn.Linear(224*224*3, fcm_proj_dim)  # Example: flatten image, project
        self.fcm_k = fcm_k
        self.fcm_m = fcm_m
        self.tau = tau
        self.normalize_feat = normalize_feat
        self.prototypes = nn.Parameter(torch.randn(fcm_k, fcm_proj_dim))
        self.classifier = nn.Linear(fcm_k, num_classes)

    def forward(self, x):
        # x: (B, 3, 224, 224) -> flatten
        x = x.view(x.size(0), -1)
        feat = self.fcm_proj(x)
        if self.normalize_feat:
            feat = nn.functional.normalize(feat, dim=1)
        # Compute distances to prototypes
        dists = torch.cdist(feat.unsqueeze(1), self.prototypes.unsqueeze(0))  # (B, 1, K)
        dists = dists.squeeze(1)  # (B, K)
        # FCM membership (softmax or fcm-style)
        if self.fcm_m == 1.0:
            membership = torch.softmax(-dists / self.tau, dim=1)
        else:
            # FCM membership formula
            inv_dist = 1.0 / (dists + 1e-6)
            membership = inv_dist ** (1.0 / (self.fcm_m - 1))
            membership = membership / membership.sum(dim=1, keepdim=True)
        logits = self.classifier(membership)
        return logits, None, None  # For compatibility
