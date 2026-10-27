"""Independent-encoder preservation distance (paper Eq. 10, RD-cos).

Extracted from the publication notebook. RAD-DINO appears in no
training term and no selection rule of the selected configuration, so this
distance is the post-hoc check reported in paper Section 5.5.
"""

from typing import Dict

import torch
import torch.nn.functional as F

from .config import Config
from .losses import RadDINOPerceptualLoss


class IndependentPreservationMetric:
    """Preservation distance in a frozen encoder that no term of the selected
    configuration optimises."""

    def __init__(self, device, config: Config = None):
        self.device = device
        self.backbone = RadDINOPerceptualLoss(config or Config()).to(device)
        self.backbone.eval()

    @torch.no_grad()
    def compute(self, denoised: torch.Tensor, clean: torch.Tensor) -> Dict[str, float]:
        feats_d = self.backbone._patch_features(
            self.backbone._preprocess(denoised.clamp(0, 1))
        )
        feats_c = self.backbone._patch_features(
            self.backbone._preprocess(clean.clamp(0, 1))
        )
        pooled_d = F.adaptive_avg_pool2d(F.relu(feats_d), 1).flatten(1)
        pooled_c = F.adaptive_avg_pool2d(F.relu(feats_c), 1).flatten(1)
        dimension = pooled_d.shape[1]
        return {
            "indep_perc_l2": float(
                ((pooled_d - pooled_c).norm(dim=1) / dimension).mean()
            ),
            "indep_perc_cos": float(
                (1.0 - F.cosine_similarity(pooled_d, pooled_c, dim=1)).mean()
            ),
        }
