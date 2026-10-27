"""Publication final evaluation pipeline."""

import torch
import pandas as pd
from torch.utils.data import DataLoader

from .dataset import MIMICCXRDataset
from .metrics import Metrics

VALIDATION_NOISE_TYPES = ("gaussian", "poisson", "speckle")
FULL_NOISE_LEVELS = (0.05, 0.10, 0.15, 0.20)


def final_per_seed(method, records, frame, *, config, device, vlm, medperc, checkpoint_loader, worker_init_fn, noise_types=VALIDATION_NOISE_TYPES, noise_levels=FULL_NOISE_LEVELS, independent_metric=None):
    """Per-image locked-holdout evaluation of one method (paper Section 4.5).

    Row schema is identical to the publication notebook. When
    ``independent_metric`` (``IndependentPreservationMetric``) is given, two
    extra columns, ``indep_perc_l2`` and ``indep_perc_cos`` (RD-cos, Eq. 10),
    are appended; the original columns are unchanged.
    """
    metrics = Metrics(config, device=device)
    rows = []

    for record in sorted(records, key=lambda item: item["seed"]):
        model = checkpoint_loader(record)

        for noise_type in noise_types:
            for severity in noise_levels:
                dataset = MIMICCXRDataset(
                    frame,
                    config,
                    split="final",
                    fixed_noise_type=noise_type,
                    fixed_sigma=severity,
                )

                loader = DataLoader(
                    dataset,
                    batch_size=config.batch_size,
                    shuffle=False,
                    num_workers=config.num_workers,
                    worker_init_fn=worker_init_fn,
                )

                with torch.no_grad():
                    for batch in loader:
                        clean = batch["clean_image"].to(device)
                        noisy = batch["noisy_image"].to(device)

                        # Model receives the raw/unclipped noisy tensor.
                        prediction = model(noisy).clamp(0, 1)

                        # Denoised-image metrics.
                        values = metrics.compute_per_image(
                            prediction,
                            clean,
                        )

                        # Measured corruption severity.
                        # Metrics are evaluated in the valid image range,
                        # matching the validation evaluator.
                        noisy_for_metrics = noisy.clamp(0, 1)

                        input_psnr = Metrics.psnr_per_image(
                            noisy_for_metrics,
                            clean,
                        )
                        input_ssim = Metrics.ssim_per_image(
                            noisy_for_metrics,
                            clean,
                        )

                        # Report/image alignment metrics.
                        text_features = vlm.encode_text(
                            batch["report_text"]
                        )
                        noisy_features = vlm.encode_image(
                            noisy_for_metrics.float()
                        )
                        prediction_features = vlm.encode_image(
                            prediction.float()
                        )

                        cos_noisy = (
                            noisy_features * text_features
                        ).sum(dim=-1)

                        cos_prediction = (
                            prediction_features * text_features
                        ).sum(dim=-1)

                        for image_index in range(clean.size(0)):
                            medperc_values = medperc.compute(
                                prediction[
                                    image_index:image_index + 1
                                ],
                                clean[
                                    image_index:image_index + 1
                                ],
                            )

                            rows.append({
                                "method": method,
                                "seed": int(record["seed"]),
                                "subject_id": int(
                                    batch["subject_id"][image_index]
                                ),
                                "dicom_id": str(
                                    batch["dicom_id"][image_index]
                                ),
                                "noise_type": noise_type,
                                "noise_sigma": float(severity),

                                # Input/noisy-image metrics.
                                "input_psnr": float(
                                    input_psnr[image_index]
                                ),
                                "input_ssim": float(
                                    input_ssim[image_index]
                                ),

                                # Denoised-image metrics.
                                "psnr": float(
                                    values["psnr"][image_index]
                                ),
                                "ssim": float(
                                    values["ssim"][image_index]
                                ),
                                "lpips": float(
                                    values["lpips"][image_index]
                                ),

                                # Medical perceptual metrics.
                                "medperc_l2": float(
                                    medperc_values["med_perc_l2"]
                                ),
                                "medperc_cos": float(
                                    medperc_values["med_perc_cos"]
                                ),

                                # Image-report alignment.
                                "cos_noisy": float(
                                    cos_noisy[image_index]
                                ),
                                "cos_den": float(
                                    cos_prediction[image_index]
                                ),
                                "cos_change": float(
                                    cos_prediction[image_index]
                                    - cos_noisy[image_index]
                                ),
                            })

                            if independent_metric is not None:
                                rows[-1].update(
                                    independent_metric.compute(
                                        prediction[image_index:image_index + 1],
                                        clean[image_index:image_index + 1],
                                    )
                                )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)
