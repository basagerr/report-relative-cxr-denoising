"""Prepare the locked final holdout for evaluation.

Contains no bundled patient data.
"""

import hashlib
from pathlib import Path
import pandas as pd

from .checkpoint_utils import dataframe_manifest_hash


def _safe_path(root, relative):
    root = Path(root).resolve()
    relative = Path(str(relative))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe relative path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Path escapes data directory")
    return path


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_locked_final_data(
    eligible_manifest,
    candidate_manifest,
    final_data_dir,
    expected_content_hash,
):
    """Return the original 204 rows with path and full report text."""

    keys = ["subject_id", "study_id", "dicom_id"]

    eligible = pd.read_csv(eligible_manifest)
    candidate = pd.read_csv(candidate_manifest)

    frame = eligible.merge(
        candidate[
            keys + ["report_relative_path", "ViewPosition", "split"]
        ],
        on=keys,
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if len(frame) != 204:
        raise ValueError("Unexpected final holdout size")

    if frame["report_relative_path"].isna().any():
        raise ValueError("Missing report mapping")

    paths = []
    texts = []
    image_hashes = []

    for row in frame.itertuples(index=False):
        image_path = _safe_path(final_data_dir, row.relative_path)
        report_path = _safe_path(
            final_data_dir, row.report_relative_path
        )

        report_bytes = report_path.read_bytes()
        text = report_bytes.decode("utf-8")

        report_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if report_hash != str(row.report_sha256):
            raise ValueError("Report hash mismatch")

        paths.append(str(image_path))
        texts.append(text)
        image_hashes.append(_sha256_file(image_path))

    frame["image_sha256"] = image_hashes

    actual_hash = dataframe_manifest_hash(frame)

    if actual_hash != expected_content_hash:
        raise ValueError("Final content hash mismatch")

    frame["path"] = paths
    frame["text"] = texts

    return frame
