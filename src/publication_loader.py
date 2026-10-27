"""Load a verified publication checkpoint.

Only use with checkpoints from trusted sources.
"""
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.serialization import safe_globals

from .config import Config
from .model import UNet
from . import checkpoint_utils as cu


def load_publication_model(record, freeze, roots, device="cpu"):
    """Verify the checkpoint and reconstruct its original model."""

    recorded = Path(record["checkpoint_path"]).expanduser()

    candidates = (
        [recorded] if recorded.is_absolute()
        else [Path(root) / recorded for root in roots]
    )

    verified = []

    for path in candidates:
        if not path.is_file():
            continue

        digest = hashlib.sha256()

        with path.open("rb") as stream:
            for block in iter(
                lambda: stream.read(1024 * 1024), b""
            ):
                digest.update(block)

        if digest.hexdigest() == record["checkpoint_sha256"].lower():
            verified.append(path.resolve())

    verified = list(dict.fromkeys(verified))

    if len(verified) != 1:
        raise RuntimeError(
            "Expected exactly one SHA-256-verified checkpoint"
        )

    allowed = [
        np._core.multiarray._reconstruct,
        np.dtype,
        np.ndarray,
        type(np.dtype(np.float32)),
        np.dtypes.UInt32DType,
    ]

    with safe_globals(allowed):
        checkpoint = torch.load(
            verified[0],
            map_location="cpu",
            weights_only=True,
        )

    cfg = Config(**{
        key: value
        for key, value in checkpoint["config"].items()
        if key in Config.__dataclass_fields__
    })

    # Use the split hashes recorded in the verified publication freeze record.
    train_hash_value = freeze["train_hash"]
    val_hash_value = freeze["val_hash"]

    expected = cu.training_signature(
        cfg,
        train_hash_value=train_hash_value,
        val_hash_value=val_hash_value,
    )

    if expected != checkpoint["training_signature"]:
        raise RuntimeError("Checkpoint training signature mismatch")

    if expected != record["training_signature"]:
        raise RuntimeError("Publication freeze signature mismatch")

    if not cu.checkpoint_is_training_compatible(
        checkpoint,
        cfg,
        train_hash_value=train_hash_value,
        val_hash_value=val_hash_value,
    ):
        raise RuntimeError("Original compatibility check failed")

    model = UNet(cfg)
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    model = model.to(device).eval()

    return model, cfg
