from typing import Dict
import torch
import lpips
from pytorch_msssim import ssim as _ssim
from .config import Config

"""Original Metrics from the publication notebook.
"""

class Metrics:
    """Per-image PSNR, SSIM, and LPIPS with scalar convenience wrappers."""
    DATA_RANGE = 1.0

    def __init__(self, config: Config, lpips_net: str = "alex", device=None):
        self.config = config
        self.lpips_model = lpips.LPIPS(net=lpips_net).to(torch.device(device) if device is not None else torch.device("cpu")).eval()
        for p in self.lpips_model.parameters():
            p.requires_grad = False

    @staticmethod
    def psnr_per_image(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
        mse = (pred - target).pow(2).flatten(1).mean(1).clamp_min(1e-12)
        return 10.0 * torch.log10((data_range ** 2) / mse)

    @staticmethod
    def ssim_per_image(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
        return _ssim(pred, target, data_range=data_range, size_average=False, nonnegative_ssim=True)

    def lpips_per_image(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape[1] == 1:
            pred, target = pred.repeat(1,3,1,1), target.repeat(1,3,1,1)
        with torch.no_grad():
            out = self.lpips_model(pred * 2 - 1, target * 2 - 1)
        return out.flatten(1).mean(1)

    @classmethod
    def psnr(cls, pred, target, data_range: float = 1.0) -> float:
        return cls.psnr_per_image(pred, target, data_range).mean().item()

    @classmethod
    def ssim(cls, pred, target, data_range: float = 1.0) -> float:
        return cls.ssim_per_image(pred, target, data_range).mean().item()

    def lpips_score(self, pred, target) -> float:
        return self.lpips_per_image(pred, target).mean().item()

    def compute_per_image(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        p, t = pred.clamp(0,1), target.clamp(0,1)
        return {
            "psnr": self.psnr_per_image(p,t),
            "ssim": self.ssim_per_image(p,t),
            "lpips": self.lpips_per_image(p,t),
        }

    def compute_all(self, pred, target) -> Dict[str, float]:
        d = self.compute_per_image(pred, target)
        return {k: v.mean().item() for k,v in d.items()}
