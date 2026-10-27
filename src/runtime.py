"""Shared runtime helpers for the command-line scripts."""

import os

# Same cuBLAS setting as the publication notebook; it must be in place before
# the first CUDA matrix multiplication.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


GLOBAL_SEED = 42


def set_seed(seed: int = GLOBAL_SEED) -> None:
    """Seed Python / NumPy / PyTorch (CPU+CUDA) and enable deterministic cuDNN.

    Identical to ``set_seed`` in the publication notebook, which ran
    before every training and evaluation run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print(f"  (note) torch.use_deterministic_algorithms unavailable: {e}")


def worker_init_fn(worker_id: int) -> None:
    """Seed one worker from PyTorch's per-iterator worker seed.

    Identical to the publication notebook. Final evaluation does not depend on
    it, because holdout noise and masks are seeded per image.
    """
    del worker_id
    worker_seed = int(torch.initial_seed() % (2**32))
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def resolve_device(name: str) -> torch.device:
    """Return the requested device, failing loudly if CUDA is unavailable."""
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested but no CUDA GPU is available; "
            "pass --device cpu to run (much more slowly) on the CPU"
        )
    if name not in ("cuda", "cpu"):
        raise ValueError(f"Unsupported device {name!r}")
    return torch.device(name)


def assert_private_output_dir(output_dir: Path, publication_dir: Path) -> Path:
    """Refuse output locations inside the repository or the publication results."""
    output_dir = Path(output_dir).resolve()
    if output_dir.is_relative_to(REPOSITORY_ROOT):
        raise ValueError("Private outputs must be written outside the repository")
    if output_dir.is_relative_to(Path(publication_dir).resolve()):
        raise ValueError("Do not write into the original publication results")
    return output_dir
