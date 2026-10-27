"""Checkpoint adapter for the four publication evaluation methods."""

from .evaluation_loader import make_checkpoint_loader
from .reference_loader import load_reference_model


def make_combined_loader(
    method,
    *,
    freeze,
    roots,
    references_dir,
    device,
    evaluation_config,
):
    """Return a callable accepting a record and returning a loaded model."""

    publication_methods = {"proposed", "n2v_baseline"}
    reference_methods = {"noise2noise", "supervised"}

    if method in publication_methods:
        return make_checkpoint_loader(
            freeze,
            roots,
            device,
            evaluation_config,
        )

    if method not in reference_methods:
        raise ValueError(f"Unknown evaluation method: {method}")

    def load_record(record):
        kind = record["stage"]

        if kind != method:
            raise ValueError("Reference record method mismatch")

        model, cfg = load_reference_model(
            kind=kind,
            seed=record["seed"],
            references_dir=references_dir,
            device=device,
        )

        if cfg.image_size != evaluation_config.image_size:
            raise ValueError("Reference image-size mismatch")

        if cfg.in_channels != evaluation_config.in_channels:
            raise ValueError("Reference channel-count mismatch")

        return model

    return load_record
