import torch
from dataclasses import dataclass, field
from typing import List

GLOBAL_SEED = 42

"""Original Config definition from the publication notebook.
"""

@dataclass
class Config:
    """Publication configuration. Important choices are stored in every run manifest."""

    protocol_version: str = "Q1-PUB-v1.7.4-local"
    experiment_name: str = "vlm_guided_denoising_publication"
    seed: int = GLOBAL_SEED
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset
    data_dir: str = "./mimic-cxr-jpg"
    train_split: float = 0.70
    val_split: float = 0.15
    publication_split_seed: int = 20260731
    image_size: int = 256
    num_images_to_download: int = 1500
    final_holdout_patients: int = 204
    final_holdout_candidates: int = 260
    final_holdout_seed: int = 20260731

    # Noise. `sigma` is a type-specific severity parameter for Poisson/speckle.
    noise_sigma_min: float = 0.05
    noise_sigma_max: float = 0.30
    noise_types: List[str] = field(default_factory=lambda: ["gaussian", "poisson", "speckle"])
    clip_noisy_before_model: bool = False

    # U-Net
    in_channels: int = 1
    out_channels: int = 1
    features: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    bilinear: bool = True
    dropout: float = 0.1

    # BioViL-T
    vlm_name: str = "biovil_t"
    vlm_frozen: bool = True
    vlm_embedding_dim: int = 128
    vlm_temperature: float = 0.07

    # Losses. Sweep values replace these placeholders.
    lambda_ssl: float = 1.0
    lambda_semantic: float = 0.0
    lambda_reg: float = 0.0
    lambda_perceptual: float = 0.0
    perceptual_backbone: str = "xrv"
    perceptual_warmup_epochs: int = 5
    semantic_margin: float = 0.0
    softplus_beta: float = 20.0
    semantic_variant: str = "softplus"
    semantic_warmup_epochs: int = 10

    # Leakage-safe blind spot
    blind_spot_ratio: float = 0.05
    mask_exclusion_radius: int = 1

    # Training
    epochs: int = 35
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    scheduler_gamma: float = 0.95
    gradient_accumulation: int = 4
    max_grad_norm: float = 1.0
    common_selection_start_epoch: int = 10
    use_early_stopping: bool = False
    patience: int = 12
    min_delta: float = 1e-4

    # Validation selection rule
    selection_psnr_tolerance_db: float = 0.10
    selection_ssim_tolerance: float = 0.002
    selection_primary_metric: str = "lpips"

    # Hardware / output
    num_workers: int = 4
    use_amp: bool = True
    output_dir: str = "./outputs_publication_q1_v1_7_4_local"
    save_dir: str = "./outputs_publication_q1_v1_7_4_local/checkpoints"
    log_dir: str = "./outputs_publication_q1_v1_7_4_local/logs"
    save_interval: int = 10

    # Runtime fields
    model_variant: str = "noise2void_only"
    run_id: str = ""
    metrics_csv_path: str = ""
    history_pt_path: str = ""
    history_json_path: str = ""
    final_plot_path: str = ""
    experiment_manifest_json: str = ""
    experiment_manifest_hash: str = ""
