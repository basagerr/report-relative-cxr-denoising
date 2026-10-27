"""Discover completed reference runs without retraining."""

import json
from pathlib import Path


def load_reference_records(references_dir, seeds=(7, 42, 123)):
    """Reconstruct the original reference record dictionaries."""

    root = Path(references_dir)
    records = []

    for kind in ("noise2noise", "supervised"):
        for seed in seeds:
            run_name = f"ref_{kind}_seed{seed}"
            run_dir = root / run_name

            marker = json.loads(
                (run_dir / "completed.json").read_text()
            )

            checkpoint = run_dir / "checkpoints" / "best.pt"

            if marker.get("status") != "complete":
                raise ValueError(f"Incomplete run: {run_name}")

            if marker.get("run_name") != run_name:
                raise ValueError("Reference run-name mismatch")

            if (
                marker.get("epochs_recorded")
                != marker.get("planned_epochs")
            ):
                raise ValueError("Reference training incomplete")

            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)

            manifest_hash = marker.get("experiment_manifest_hash")

            if not manifest_hash:
                raise ValueError("Missing experiment manifest hash")

            records.append({
                "run_name": run_name,
                "stage": kind,
                "config_name": kind,
                "seed": seed,
                "lambda_semantic": 0.0,
                "margin": 0.0,
                "lambda_perceptual": 0.0,
                "perceptual_backbone": "none",
                "semantic_variant": "none",
                "checkpoint_path": str(checkpoint),
                "manifest_hash": manifest_hash,
            })

    return records
