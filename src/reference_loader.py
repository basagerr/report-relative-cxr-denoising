"""Safely load previously trained reference UNet checkpoints."""

import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .model import UNet


def load_reference_model(kind, seed, references_dir, device="cpu"):
    """Load and verify a completed Noise2Noise or supervised run."""

    if kind not in ("noise2noise", "supervised"):
        raise ValueError("Unsupported reference model kind")

    run_name = f"ref_{kind}_seed{int(seed)}"
    run_dir = Path(references_dir) / run_name

    marker_path = run_dir / "completed.json"
    checkpoint_path = run_dir / "checkpoints" / "best.pt"

    marker = json.loads(marker_path.read_text())

    if marker.get("status") != "complete":
        raise ValueError("Reference training is not complete")

    if marker.get("run_name") != run_name:
        raise ValueError("Reference run-name mismatch")

    if marker.get("epochs_recorded") != marker.get("planned_epochs"):
        raise ValueError("Incomplete reference training history")

    safe_types = [
        np.ndarray,
        np.dtype,
        np._core.multiarray._reconstruct,
        *[
            type(np.dtype(dtype))
            for dtype in (
                np.float32, np.float64, np.int64, np.uint32
            )
        ],
    ]

    with torch.serialization.safe_globals(safe_types):
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

    if payload.get("training_signature") != marker["training_signature"]:
        raise ValueError("Reference training signature mismatch")

    if payload.get("experiment_manifest_hash") != marker["experiment_manifest_hash"]:
        raise ValueError("Reference manifest hash mismatch")

    if payload.get("best_epoch") != marker["best_epoch"]:
        raise ValueError("Reference best-epoch mismatch")

    config_fields = {field.name for field in fields(Config)}
    saved_config = payload["config"]

    unknown = set(saved_config) - config_fields
    if unknown:
        raise ValueError(f"Unknown configuration fields: {unknown}")

    cfg = Config(**saved_config)
    model = UNet(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(torch.device(device)).eval()

    return model, cfg
