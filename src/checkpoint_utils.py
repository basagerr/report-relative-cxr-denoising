"""Original publication checkpoint and manifest utilities."""

from pathlib import Path

import json
import hashlib
import torch
import numpy as np
from dataclasses import asdict
from typing import Any, Dict
import pandas as pd
from .config import Config

# Module-level split hashes used by the original notebook when a caller does
# not pass them explicitly. They must be set (or passed) before use.
train_hash = None
val_hash = None

_NON_TRAINING_CONFIG_KEYS = {
    "run_id",
    "device",
    "output_dir",
    "save_dir",
    "log_dir",
    "metrics_csv_path",
    "history_pt_path",
    "history_json_path",
    "final_plot_path",
    "save_interval",
    "experiment_manifest_json",
    "experiment_manifest_hash",
}

def _json_safe(value: Any) -> Any:
    """Recursively convert scientific Python objects to strict JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        # JSON NaN/Infinity are non-standard. Store missing non-finite metadata
        # as null rather than writing invalid cross-language JSON.
        return value if bool(np.isfinite(value)) else None

    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        if detached.numel() == 1:
            return _json_safe(detached.item())
        return _json_safe(detached.tolist())

    if isinstance(value, np.generic):
        return _json_safe(value.item())

    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, set):
        converted = [_json_safe(item) for item in value]
        return sorted(
            converted,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                allow_nan=False,
            ),
        )

    # Descriptive metadata objects such as torch.device and torch.dtype.
    return str(value)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(
        _json_safe(obj),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def dataframe_manifest_hash(df: pd.DataFrame) -> str:
    preferred = (
        "subject_id",
        "study_id",
        "dicom_id",
        "relative_path",
        "image_sha256",
        "report_sha256",
        "ViewPosition",
        "split",
    )
    columns = [
        column
        for column in preferred
        if column in df.columns
    ]
    if not columns:
        raise ValueError("No stable manifest columns are available")

    sort_columns = [
        column
        for column in ("subject_id", "study_id", "dicom_id")
        if column in columns
    ]
    stable = df[columns].copy().astype(str)
    if sort_columns:
        stable = stable.sort_values(sort_columns)

    return sha256_text(stable.to_csv(index=False))


def training_config_payload(
    cfg_or_dict,
) -> Dict[str, Any]:
    if isinstance(cfg_or_dict, dict):
        payload = dict(cfg_or_dict)
    else:
        payload = asdict(cfg_or_dict)

    for key in _NON_TRAINING_CONFIG_KEYS:
        payload.pop(key, None)

    return _json_safe(payload)


def training_signature(
    cfg_run: Config,
    train_hash_value=None,
    val_hash_value=None,
) -> str:
    """Stable skip/resume key, independent of audit/code hashes."""
    resolved_train = train_hash if train_hash_value is None else train_hash_value
    resolved_val = val_hash if val_hash_value is None else val_hash_value
    if not resolved_train or not resolved_val:
        raise ValueError(
            "training_signature requires the train and validation split hashes"
        )
    payload = {
        "signature_version": (
            "training_config_and_split_v1"
        ),
        "protocol": cfg_run.protocol_version,
        "experiment_name": (
            cfg_run.experiment_name
        ),
        "config": training_config_payload(
            cfg_run
        ),
        "train_hash": resolved_train,
        "val_hash": resolved_val,
    }
    return sha256_text(
        canonical_json(payload)
    )


def checkpoint_is_training_compatible(
    checkpoint: Dict[str, Any],
    cfg_run: Config,
    train_hash_value=None,
    val_hash_value=None,
) -> bool:
    """Accept new signatures and safely migrate older Config-only saves."""
    expected = training_signature(
        cfg_run,
        train_hash_value=train_hash_value,
        val_hash_value=val_hash_value,
    )
    saved_signature = checkpoint.get(
        "training_signature"
    )

    if saved_signature is not None:
        return saved_signature == expected

    saved_config = checkpoint.get("config")
    if not isinstance(saved_config, dict):
        return False

    return (
        canonical_json(
            training_config_payload(saved_config)
        )
        == canonical_json(
            training_config_payload(cfg_run)
        )
    )
