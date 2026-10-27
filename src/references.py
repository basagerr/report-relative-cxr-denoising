"""Reference-model components (paper Eq. 12).

The Noise2Noise and supervised references share the residual U-Net and the
optimisation schedule of the proposed method and differ only in the target:

* Noise2Noise: input y1 and target y2 are two independent corruptions of the
  same archived image, with one noise family and severity drawn per pair;
* supervised: input is the corrupted image and target is the archived image.

Both use a mean-squared error over all pixels. ``N2NDataset`` is reproduced
verbatim from the publication notebook. The notebook's reference
training driver depends on its run-manifest infrastructure and is
not included in this repository; see README, "Reproducibility scope".
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .noise import add_iid_noise


class N2NDataset(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        clean = item["clean_image"]
        noise_type = str(np.random.choice(self.base.config.noise_types))
        sigma = float(np.random.uniform(self.base.config.noise_sigma_min, self.base.config.noise_sigma_max))
        y1 = add_iid_noise(clean, noise_type, sigma, clip=self.base.config.clip_noisy_before_model)
        y2 = add_iid_noise(clean, noise_type, sigma, clip=self.base.config.clip_noisy_before_model)
        return {"input": y1, "target": y2, "clean_image": clean}


def reference_loss(model, batch, kind: str) -> torch.Tensor:
    """Unscaled Eq. 12 objective (mean-squared error over all pixels)."""
    if kind == "noise2noise":
        input_image, target = batch["input"], batch["target"]
    elif kind == "supervised":
        input_image, target = batch["noisy_image"], batch["clean_image"]
    else:
        raise ValueError(f"Unsupported reference kind: {kind!r}")
    device = next(model.parameters()).device
    return F.mse_loss(model(input_image.to(device)), target.to(device))
