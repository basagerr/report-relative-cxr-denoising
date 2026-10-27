"""Adapter between publication checkpoints and final evaluation."""

from .publication_loader import load_publication_model


def make_checkpoint_loader(freeze, roots, device, evaluation_config):
    """Return a model-only loader with checkpoint compatibility checks."""

    def checkpoint_loader(record):
        model, checkpoint_config = load_publication_model(
            record=record,
            freeze=freeze,
            roots=roots,
            device=device,
        )

        for field in ("image_size", "in_channels"):
            if hasattr(evaluation_config, field) and hasattr(
                checkpoint_config, field
            ):
                expected = getattr(evaluation_config, field)
                actual = getattr(checkpoint_config, field)

                if expected != actual:
                    raise ValueError(
                        f"Checkpoint configuration mismatch: {field}: "
                        f"evaluation={expected}, checkpoint={actual}"
                    )

        return model

    return checkpoint_loader
