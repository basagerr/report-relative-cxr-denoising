"""Configuration overrides of the published models (paper Sections 4.2-4.3).

The published checkpoints store their complete configuration, and
``publication_loader`` rebuilds each model from that stored configuration;
these overrides are documentation and a starting point for retraining, not a
substitute for the stored configuration.
"""

from dataclasses import replace

from .config import Config

PUBLICATION_SEEDS = (7, 42, 123)
NOISE_TYPES = ("gaussian", "poisson", "speckle")
NOISE_LEVELS = (0.05, 0.10, 0.15, 0.20)

PUBLISHED_OVERRIDES = {
    # Plain Noise2Void baseline: blind-spot loss only. BioViL-T is used for
    # monitoring the report-relative change but not optimised.
    "n2v_baseline": dict(
        model_variant="noise2void_only",
        lambda_semantic=0.0,
        lambda_perceptual=0.0,
    ),
    # Selected configuration: softplus one-sided report-relative constraint,
    # m = 0, beta = 20, lambda_report = 0.001, XRV dense feature term 0.005.
    "proposed": dict(
        model_variant="hinge",
        semantic_variant="softplus",
        semantic_margin=0.0,
        softplus_beta=20.0,
        lambda_semantic=0.001,
        semantic_warmup_epochs=10,
        lambda_perceptual=0.005,
        perceptual_backbone="xrv",
        perceptual_warmup_epochs=5,
    ),
}


def published_config(method: str, **extra) -> Config:
    """Return Config() with the published overrides for ``method`` applied."""
    if method not in PUBLISHED_OVERRIDES:
        raise ValueError(
            f"Unknown method {method!r}; expected one of {sorted(PUBLISHED_OVERRIDES)}"
        )
    return replace(Config(), **PUBLISHED_OVERRIDES[method], **extra)
