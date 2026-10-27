"""Preflight verification of the locked final holdout (no model is loaded).

Checks the publication freeze record, the eligible-manifest content lock and
the SHA-256 of every holdout report and image against the private manifests,
then exits. Formerly ``scripts/run_final_evaluation.py``; use
``scripts/run_one_seed.py`` for the actual evaluation.
"""

import argparse
import hashlib
import json
from pathlib import Path

from src.final_data import prepare_locked_final_data
from src.evaluation import FULL_NOISE_LEVELS, VALIDATION_NOISE_TYPES


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--eligible-manifest", type=Path, required=True)
    parser.add_argument("--eligible-meta", type=Path, required=True)
    parser.add_argument("--content-lock", type=Path, required=True)
    parser.add_argument("--final-data-dir", type=Path, required=True)
    args = parser.parse_args()

    freeze = json.loads(args.freeze.read_text())
    meta = json.loads(args.eligible_meta.read_text())
    lock = json.loads(args.content_lock.read_text())

    if meta["candidate_manifest_hash"] != freeze["final_candidate_manifest_hash"]:
        raise ValueError("Candidate manifest reference mismatch")

    if lock["final_content_hash"] != meta["content_hash"]:
        raise ValueError("Final content lock mismatch")

    if sha256_file(args.eligible_manifest) != lock["eligible_manifest_hash"]:
        raise ValueError("Eligible manifest file SHA256 mismatch")

    for name in ("selected_runs", "baseline_runs"):
        records = freeze.get(name)
        if not isinstance(records, list) or len(records) != 3:
            raise ValueError(f"Unexpected checkpoint group: {name}")

        for record in records:
            for key in (
                "seed", "checkpoint_path",
                "checkpoint_sha256", "training_signature"
            ):
                if key not in record:
                    raise ValueError(f"Missing {key} in {name}")

    frame = prepare_locked_final_data(
        eligible_manifest=args.eligible_manifest,
        candidate_manifest=args.candidate_manifest,
        final_data_dir=args.final_data_dir,
        expected_content_hash=meta["content_hash"],
    )

    print("=== PUBLICATION FINAL EVALUATION PREFLIGHT ===")
    print("Final records:", len(frame))
    print("Unique patients:", frame["subject_id"].nunique())
    print("Content integrity: PASS")
    print("Selected runs:", len(freeze["selected_runs"]))
    print("Baseline runs:", len(freeze["baseline_runs"]))
    print("Noise types:", VALIDATION_NOISE_TYPES)
    print("Noise levels:", FULL_NOISE_LEVELS)
    print("Models loaded: NO")
    print("Results generated: NO")
    print("PUBLICATION PREFLIGHT: PASS")


if __name__ == "__main__":
    main()
