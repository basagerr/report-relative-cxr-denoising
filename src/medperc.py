"""Original MedPercMetrics extracted from the publication notebook.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from .config import Config

class MedPercMetrics:
    """MF-L2 / MF-cos (paper Eq. 9): feature-space distance to the reference in
    frozen XRV DenseNet-121 features. A representation-level measure, not a
    validated clinical measure.

    Two variants:
      - L2:  ‖GAP(ReLU(ψ_d)) − GAP(ReLU(ψ_c))‖₂ / D     (magnitude)
      - cos: 1 − cos(GAP(ReLU(ψ_d)), GAP(ReLU(ψ_c)))    (direction)

    Both lower-is-better.
    """

    def __init__(self, device: torch.device):
        import torchxrayvision as xrv_lib
        self.device = device
        self.xrv = xrv_lib.models.DenseNet(weights="densenet121-res224-all")
        self.xrv.to(device).eval()
        for p in self.xrv.parameters():
            p.requires_grad = False

        # Hook the penultimate feature map (before global pool + classifier)
        self._features: Dict[str, torch.Tensor] = {}
        def hook(_mod, _inp, out):
            self._features["feat"] = out.detach()
        self.xrv.features.register_forward_hook(hook)

    def _preprocess(self, img: torch.Tensor) -> torch.Tensor:
        """[B, 1|3, H, W] in [0, 1] -> [B, 1, 224, 224] in XRV's [-1024, 1024] range."""
        x = img * 2048.0 - 1024.0
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        if x.shape[-2] != 224 or x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return x

    def _pool(self, spatial_feat: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(F.relu(spatial_feat), 1).flatten(1)

    @torch.no_grad()
    def compute(self, denoised: torch.Tensor, clean: torch.Tensor) -> Dict[str, float]:
        self.xrv(self._preprocess(denoised))
        feats_d = self._pool(self._features["feat"])
        self.xrv(self._preprocess(clean))
        feats_c = self._pool(self._features["feat"])

        D = feats_d.shape[1]
        l2  = ((feats_d - feats_c).norm(dim=1) / D).mean().item()
        cos = (1.0 - F.cosine_similarity(feats_d, feats_c, dim=1)).mean().item()
        return {"med_perc_l2": l2, "med_perc_cos": cos}
