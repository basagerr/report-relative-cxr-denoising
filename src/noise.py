"""Original publication noise and masking utilities."""

from typing import Optional, Tuple
import hashlib
import numpy as np
import torch

def stable_int_seed(s: str, mod: int = 2**31 - 1) -> int:
    """Map one string to one stable integer seed."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16) % mod


def add_iid_noise(image: torch.Tensor, noise_type: str, sigma: float,
                  generator: Optional[torch.Generator] = None,
                  deterministic_seed: Optional[int] = None,
                  clip: bool = False) -> torch.Tensor:
    """Add conditionally unbiased noise. Clipping is OFF for publication training.

    All three families may exceed 1, and Gaussian and speckle may also go below 0.
    The U-Net receives those values; only
    encoder inputs, visualisations, and metric predictions are clamped.
    """
    if generator is None and deterministic_seed is not None:
        generator = torch.Generator(device=image.device); generator.manual_seed(int(deterministic_seed))
    if noise_type == "gaussian":
        noisy = image + torch.randn(image.shape, generator=generator, dtype=image.dtype, device=image.device) * sigma
    elif noise_type == "speckle":
        noisy = image + image * torch.randn(image.shape, generator=generator, dtype=image.dtype, device=image.device) * sigma
    elif noise_type == "poisson":
        scale = float(np.clip((0.1 / sigma) ** 2 * 100, 10, 10000))
        scaled = image.clamp_min(0) * scale
        if deterministic_seed is not None:
            rng = np.random.RandomState(int(deterministic_seed))
            noisy = torch.from_numpy(rng.poisson(scaled.cpu().numpy()).astype(np.float32)).to(image.device) / scale
        else:
            noisy = torch.poisson(scaled) / scale
    else:
        raise ValueError(f"Unknown noise_type={noise_type!r}")
    return noisy.clamp(0,1) if clip else noisy


def make_blind_spot_mask(
    image_shape: Tuple[int, int, int],
    ratio: float,
    deterministic_seed: Optional[int] = None,
    exclusion_radius: int = 1,
) -> torch.Tensor:
    """Select targets while keeping every selected pixel away from other targets."""
    _, height, width = image_shape
    number_of_targets = int(height * width * ratio)

    # Validation uses an explicit per-image seed. Training uses NumPy's seeded
    # worker RNG, so a resumed run continues reproducibly.
    if deterministic_seed is None:
        order = np.random.permutation(height * width)
    else:
        order = np.random.default_rng(deterministic_seed).permutation(height * width)

    available = np.ones((height, width), dtype=bool)
    chosen = []

    for flat_index in order:
        row, column = divmod(int(flat_index), width)
        if not available[row, column]:
            continue

        chosen.append(flat_index)
        row_start = max(0, row - exclusion_radius)
        row_end = min(height, row + exclusion_radius + 1)
        column_start = max(0, column - exclusion_radius)
        column_end = min(width, column + exclusion_radius + 1)
        available[row_start:row_end, column_start:column_end] = False

        if len(chosen) == number_of_targets:
            break

    if len(chosen) != number_of_targets:
        raise RuntimeError(
            f"Could select only {len(chosen)}/{number_of_targets} leakage-safe pixels"
        )

    mask = torch.zeros(1, height, width)
    mask.view(-1)[torch.tensor(chosen, dtype=torch.long)] = 1
    return mask
