"""Original publication training implementation.

Extracted from the research notebook.
"""

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config import Config
from src.losses import CombinedLoss
from src.metrics import Metrics

if TYPE_CHECKING:
    from src.vlm import VLMWrapper

from src.checkpoint_utils import (
    training_signature,
    checkpoint_is_training_compatible,
)
from src.noise import stable_int_seed
from src.training_utils import (
    atomic_json_write,
    atomic_torch_save,
    capture_rng_state,
    restore_rng_state,
    save_history,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Resuming from latest.pt is disabled by default: an existing latest.pt raises
# an error instead of being resumed. Set to True to resume interrupted runs.
RESUME_INTERRUPTED_RUNS = False


class Trainer:
    """Train one publication model and keep strict resume information."""

    HISTORY_KEYS = (
        "train_loss",
        "val_loss",
        "train_psnr",
        "val_psnr",
        "train_ssim",
        "val_ssim",
        "vlm_similarity",
        "semantic_metrics",
    )

    def __init__(
        self,
        model: nn.Module,
        vlm: Optional["VLMWrapper"],
        config: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str,
        train_hash: str,
        val_hash: str,
    ):
        self.model = model.to(device)
        self.is_data_parallel = isinstance(model, nn.DataParallel)
        self.vlm = vlm
        self.config = config
        if not isinstance(train_hash, str) or not train_hash.strip():
            raise ValueError('A non-empty train_hash is required')
        if not isinstance(val_hash, str) or not val_hash.strip():
            raise ValueError('A non-empty val_hash is required')
        self.train_hash = train_hash
        self.val_hash = val_hash
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.criterion = CombinedLoss(vlm, config).to(device)

        trainable_model = self.model.module if self.is_data_parallel else self.model
        self.optimizer = torch.optim.AdamW(
            trainable_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=config.scheduler_gamma,
        )

        amp_is_on = config.use_amp and torch.cuda.is_available()
        self.scaler = GradScaler("cuda") if amp_is_on else None
        self.metrics = Metrics

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_psnr = -float("inf")
        self.patience_counter = 0
        self.best_epoch = -1
        self.best_val_metrics = {}
        self.grad_accum = config.gradient_accumulation
        self.history = {key: [] for key in self.HISTORY_KEYS}

        # If latest.pt exists and matches this exact run, we continue from it.
        self._maybe_resume()



def _trainer_maybe_resume(self) -> None:
    """Resume from latest.pt using a stable training signature.

    The full scientific manifest remains stored for audit, but it is not used
    as the restart key because implementation/audit metadata can change without
    changing the already-trained optimisation problem.
    """
    latest_path = self.checkpoint_dir / "latest.pt"
    if not latest_path.exists():
        return

    if not RESUME_INTERRUPTED_RUNS:
        raise RuntimeError(
            f"Interrupted run found at {latest_path}. "
            "Enable resume or force retraining."
        )

    checkpoint = torch.load(
        latest_path,
        map_location=device,
        weights_only=False,
    )

    if not checkpoint_is_training_compatible(
        checkpoint,
        self.config,
        train_hash_value=self.train_hash,
        val_hash_value=self.val_hash,
    ):
        raise RuntimeError(
            "The saved latest.pt was produced by a genuinely different "
            f"training configuration: {latest_path}"
        )

    model = (
        self.model.module
        if self.is_data_parallel
        else self.model
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    self.optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )
    self.scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    if (
        self.scaler is not None
        and "scaler_state_dict" in checkpoint
    ):
        self.scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

    self.current_epoch = (
        int(checkpoint["epoch"]) + 1
    )
    self.global_step = int(
        checkpoint.get("global_step", 0)
    )
    self.best_val_psnr = float(
        checkpoint.get(
            "best_val_psnr",
            -float("inf"),
        )
    )
    self.best_epoch = int(
        checkpoint.get("best_epoch", -1)
    )
    self.best_val_metrics = dict(
        checkpoint.get(
            "best_val_metrics",
            {},
        )
    )
    self.patience_counter = int(
        checkpoint.get(
            "patience_counter",
            0,
        )
    )
    self.history = checkpoint.get(
        "history",
        self.history,
    )

    restore_rng_state(
        checkpoint.get("rng_state"),
        self.train_loader,
    )

    print(
        f"[resume verified] next epoch="
        f"{self.current_epoch}"
    )



def _trainer_grad_norm(self) -> float:
    trainable_model = self.model.module if self.is_data_parallel else self.model

    squared_norms = []
    for parameter in trainable_model.parameters():
        if parameter.grad is not None:
            squared_norms.append(parameter.grad.data.norm(2).item() ** 2)

    return sum(squared_norms) ** 0.5



def _trainer_semantic_weight(self, epoch: int) -> float:
    warmup_epochs = getattr(self.config, "semantic_warmup_epochs", 5)
    warmup_fraction = min(1.0, epoch / max(1, warmup_epochs))
    return self.config.lambda_semantic * warmup_fraction



def _trainer_optimizer_step(self) -> float:
    # AMP gradients must be unscaled before finiteness checks and clipping.
    if self.scaler:
        self.scaler.unscale_(self.optimizer)

    trainable_model = (
        self.model.module
        if self.is_data_parallel
        else self.model
    )
    nonfinite_gradients = [
        name
        for name, parameter in trainable_model.named_parameters()
        if parameter.grad is not None
        and not torch.isfinite(parameter.grad).all()
    ]
    if nonfinite_gradients:
        raise FloatingPointError(
            "Non-finite gradients before optimizer step: "
            f"{nonfinite_gradients[:5]}"
        )

    gradient_norm = self._get_grad_norm()
    if not np.isfinite(gradient_norm):
        raise FloatingPointError(
            f"Non-finite aggregate gradient norm: {gradient_norm}"
        )

    torch.nn.utils.clip_grad_norm_(
        trainable_model.parameters(),
        self.config.max_grad_norm,
    )

    if self.scaler:
        self.scaler.step(self.optimizer)
        self.scaler.update()
    else:
        self.optimizer.step()

    self.optimizer.zero_grad()
    self.global_step += 1
    return gradient_norm



def _trainer_train_epoch(self) -> Dict[str, float]:
    self.model.train()

    metric_names = (
        "loss",
        "psnr",
        "ssim",
        "loss_ssl",
        "loss_semantic",
        "loss_perceptual",
        "preserve_loss",
        "drift_loss",
        "embedding_drift",
        "cos_noisy",
        "cos_den",
        "hinge_activation_rate",
    )
    sums = {name: 0.0 for name in metric_names}

    number_of_images = 0
    gradient_norm_sum = 0.0
    optimizer_steps = 0
    accumulated_batches = 0

    self.optimizer.zero_grad()
    semantic_weight = self._get_semantic_weight(self.current_epoch)
    amp_is_on = self.config.use_amp and device.type == "cuda"

    for batch_index, batch in enumerate(
        tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
    ):
        noisy = batch["noisy_image"].to(device)
        clean = batch["clean_image"].to(device)
        mask = batch["blind_spot_mask"].to(device)
        reports = batch["report_text"]

        blind_input = self.criterion.ssl_loss.create_blind_spot_input(
            noisy,
            mask,
        )

        # denoised_full only needs gradients when an auxiliary loss consumes
        # it. For noise2void_only runs both weights are zero and the graph
        # would be built, held, and discarded unused every batch.
        _needs_full_grad = (
            (self.criterion.semantic_loss is not None and semantic_weight > 0)
            or (
                self.criterion.perceptual_loss is not None
                and self.criterion._perceptual_weight(self.current_epoch) > 0
            )
        )

        with autocast(device_type=device.type, enabled=amp_is_on):
            denoised_blind = self.model(blind_input)
            if _needs_full_grad:
                denoised_full = self.model(noisy)
            else:
                with torch.no_grad():
                    denoised_full = self.model(noisy)

            loss, batch_metrics = self.criterion(
                denoised_blind,
                denoised_full,
                noisy,
                mask,
                reports,
                lambda_semantic_eff=semantic_weight,
                epoch=self.current_epoch,
            )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at batch {batch_index}"
                )

            nonfinite_metrics = {
                name: value
                for name, value in batch_metrics.items()
                if isinstance(value, (int, float, np.number))
                and not np.isfinite(float(value))
            }
            if nonfinite_metrics:
                raise FloatingPointError(
                    f"non-finite metrics at batch {batch_index}: "
                    f"{nonfinite_metrics}"
                )

            loss_for_backward = loss / self.grad_accum

        if self.scaler:
            self.scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        accumulated_batches += 1
        if accumulated_batches == self.grad_accum:
            gradient_norm_sum += self._optimizer_step()
            optimizer_steps += 1
            accumulated_batches = 0

        batch_size = clean.size(0)
        number_of_images += batch_size
        sums["loss"] += loss.item() * batch_size

        tracked_losses = (
            "loss_ssl",
            "loss_semantic",
            "loss_perceptual",
            "preserve_loss",
            "drift_loss",
            "embedding_drift",
            "cos_noisy",
            "cos_den",
            "hinge_activation_rate",
        )
        for name in tracked_losses:
            sums[name] += batch_metrics.get(name, 0.0) * batch_size

        with torch.no_grad():
            prediction = denoised_full.clamp(0, 1)
            sums["psnr"] += self.metrics.psnr_per_image(
                prediction,
                clean,
            ).sum().item()
            sums["ssim"] += self.metrics.ssim_per_image(
                prediction,
                clean,
            ).sum().item()

    # The last group can contain fewer batches than gradient_accumulation.
    # We correct its scale before the final optimizer step.
    if accumulated_batches > 0:
        correction = self.grad_accum / accumulated_batches
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(correction)

        gradient_norm_sum += self._optimizer_step()
        optimizer_steps += 1

    averaged = {
        name: value / max(1, number_of_images)
        for name, value in sums.items()
    }
    averaged["lambda_semantic_eff"] = semantic_weight
    averaged["avg_grad_norm"] = gradient_norm_sum / max(1, optimizer_steps)
    return averaged



@torch.no_grad()
def _trainer_validate(self) -> Dict[str, float]:
    self.model.eval()
    metric_names = (
        "loss", "psnr", "ssim", "loss_ssl", "loss_semantic",
        "loss_perceptual", "preserve_loss", "drift_loss",
        "embedding_drift", "hinge_activation_rate",
        "vlm_cos_noisy", "vlm_cos_denoised",
    )
    sums = {name: 0.0 for name in metric_names}
    number_of_images = 0
    semantic_weight = self._get_semantic_weight(self.current_epoch)

    for batch in tqdm(self.val_loader, desc="Validating"):
        noisy = batch["noisy_image"].to(device)
        clean = batch["clean_image"].to(device)
        mask = batch["blind_spot_mask"].to(device)
        reports = batch["report_text"]

        blind_parts = []
        for image_index, dicom_id in enumerate(batch["dicom_id"]):
            generator = torch.Generator(device=device)
            generator.manual_seed(stable_int_seed(f"{dicom_id}:blind_replacement"))
            blind_parts.append(
                self.criterion.ssl_loss.create_blind_spot_input(
                    noisy[image_index:image_index + 1],
                    mask[image_index:image_index + 1],
                    generator=generator,
                )
            )
        blind_input = torch.cat(blind_parts, dim=0)

        denoised_blind = self.model(blind_input)
        denoised_full = self.model(noisy)
        loss, batch_metrics = self.criterion(
            denoised_blind,
            denoised_full,
            noisy,
            mask,
            reports,
            lambda_semantic_eff=semantic_weight,
            epoch=self.current_epoch,
        )

        batch_size = clean.size(0)
        number_of_images += batch_size
        sums["loss"] += loss.item() * batch_size
        for name in (
            "loss_ssl", "loss_semantic", "loss_perceptual", "preserve_loss",
            "drift_loss", "embedding_drift", "hinge_activation_rate",
        ):
            sums[name] += batch_metrics.get(name, 0.0) * batch_size

        prediction = denoised_full.clamp(0, 1)
        sums["psnr"] += self.metrics.psnr_per_image(prediction, clean).sum().item()
        sums["ssim"] += self.metrics.ssim_per_image(prediction, clean).sum().item()

        if self.criterion.semantic_loss is not None:
            sums["vlm_cos_noisy"] += batch_metrics.get("cos_noisy", 0.0) * batch_size
            sums["vlm_cos_denoised"] += batch_metrics.get("cos_den", 0.0) * batch_size
        elif self.vlm is not None:
            noisy_features = self.vlm.encode_image(noisy.clamp(0, 1).float())
            denoised_features = self.vlm.encode_image(prediction.float())
            text_features = self.vlm.encode_text(reports)
            sums["vlm_cos_noisy"] += self.vlm.compute_similarity_raw(noisy_features, text_features).sum().item()
            sums["vlm_cos_denoised"] += self.vlm.compute_similarity_raw(denoised_features, text_features).sum().item()

    averaged = {name: value / max(1, number_of_images) for name, value in sums.items()}
    averaged["vlm_sim"] = averaged["vlm_cos_denoised"]
    return averaged



def _trainer_save_checkpoint(
    self,
    is_best: bool = False,
):
    """Save a complete, restartable state after every epoch."""
    model = (
        self.model.module
        if self.is_data_parallel
        else self.model
    )

    checkpoint = {
        "epoch": self.current_epoch,
        "global_step": self.global_step,
        "model_state_dict": model.state_dict(),
        "config": asdict(self.config),
        "optimizer_state_dict": (
            self.optimizer.state_dict()
        ),
        "scheduler_state_dict": (
            self.scheduler.state_dict()
        ),
        "best_val_psnr": self.best_val_psnr,
        "best_epoch": self.best_epoch,
        "best_val_metrics": (
            self.best_val_metrics
        ),
        "patience_counter": (
            self.patience_counter
        ),
        "history": self.history,
        "rng_state": capture_rng_state(
            self.train_loader
        ),
        # Stable restart key. Audit hashes are retained separately below.
        "training_signature": (
            training_signature(
                self.config,
                train_hash_value=self.train_hash,
                val_hash_value=self.val_hash,
            )
        ),
        "experiment_manifest_json": (
            self.config.experiment_manifest_json
        ),
        "experiment_manifest_hash": (
            self.config.experiment_manifest_hash
        ),
    }

    if self.scaler:
        checkpoint[
            "scaler_state_dict"
        ] = self.scaler.state_dict()

    # latest.pt is always overwritten atomically after every epoch.
    atomic_torch_save(
        checkpoint,
        self.checkpoint_dir / "latest.pt",
    )

    if is_best:
        atomic_torch_save(
            checkpoint,
            self.checkpoint_dir / "best.pt",
        )
        print(
            f"[ok] best saved: "
            f"{self.best_val_psnr:.3f} dB "
            f"at epoch {self.best_epoch}"
        )

    if (
        self.current_epoch
        % self.config.save_interval
        == 0
    ):
        atomic_torch_save(
            checkpoint,
            self.checkpoint_dir
            / f"epoch_{self.current_epoch}.pt",
        )



def _trainer_train(self):
    selection_start = int(self.config.common_selection_start_epoch)
    if selection_start >= self.config.epochs:
        raise ValueError("common_selection_start_epoch must be smaller than epochs")

    print("Training:", self.config.experiment_name)
    print("Common checkpoint-selection start:", selection_start)
    print("Early stopping:", self.config.use_early_stopping)
    stopped_early = False

    for epoch in range(self.current_epoch, self.config.epochs):
        self.current_epoch = epoch
        train_metrics = self.train_epoch()
        val_metrics = self.validate()
        self.scheduler.step()

        self.history["train_loss"].append(train_metrics["loss"])
        self.history["val_loss"].append(val_metrics["loss"])
        self.history["train_psnr"].append(train_metrics["psnr"])
        self.history["val_psnr"].append(val_metrics["psnr"])
        self.history["train_ssim"].append(train_metrics["ssim"])
        self.history["val_ssim"].append(val_metrics["ssim"])
        self.history["vlm_similarity"].append(val_metrics.get("vlm_sim", 0.0))
        self.history["semantic_metrics"].append({
            "vlm_cos_noisy": val_metrics.get("vlm_cos_noisy", 0.0),
            "vlm_cos_denoised": val_metrics.get("vlm_cos_denoised", 0.0),
            "lambda_semantic_eff": train_metrics.get("lambda_semantic_eff", 0.0),
            "hinge_activation_rate": train_metrics.get("hinge_activation_rate", 0.0),
        })
        save_history(self.history, self.config.history_pt_path, self.config.history_json_path)

        selection_is_active = epoch >= selection_start
        improved = selection_is_active and (
            val_metrics["psnr"] > self.best_val_psnr + self.config.min_delta
        )
        if improved:
            self.best_val_psnr = float(val_metrics["psnr"])
            self.best_epoch = int(epoch)
            self.best_val_metrics = {
                "psnr": float(val_metrics["psnr"]),
                "ssim": float(val_metrics["ssim"]),
                "loss": float(val_metrics["loss"]),
                "vlm_cos_noisy": float(val_metrics.get("vlm_cos_noisy", 0.0)),
                "vlm_cos_denoised": float(val_metrics.get("vlm_cos_denoised", 0.0)),
            }
            self.patience_counter = 0
        elif selection_is_active:
            self.patience_counter += 1

        self.save_checkpoint(is_best=improved)
        print(
            f"epoch={epoch:02d} train_psnr={train_metrics['psnr']:.3f} "
            f"val_psnr={val_metrics['psnr']:.3f} val_ssim={val_metrics['ssim']:.5f}"
        )

        if (
            self.config.use_early_stopping
            and selection_is_active
            and self.patience_counter >= self.config.patience
        ):
            print("Early stop at epoch", epoch)
            stopped_early = True
            break

    best_path = self.checkpoint_dir / "best.pt"
    if not best_path.exists():
        raise RuntimeError("Training ended without creating best.pt")
    if not stopped_early and len(self.history["val_psnr"]) != self.config.epochs:
        raise RuntimeError("A fixed-epoch publication run ended before all epochs were recorded")

    completion = {
        "status": "complete",
        "protocol": self.config.protocol_version,
        "run_name": self.config.experiment_name,
        "experiment_manifest_hash": self.config.experiment_manifest_hash,
        "training_signature": training_signature(
                self.config,
                train_hash_value=self.train_hash,
                val_hash_value=self.val_hash,
            ),
        "epochs_recorded": len(self.history["val_psnr"]),
        "last_epoch": int(self.current_epoch),
        "planned_epochs": int(self.config.epochs),
        "stopped_early": bool(stopped_early),
        "best_epoch": int(self.best_epoch),
        "best_val_metrics": self.best_val_metrics,
        "completed_utc": pd.Timestamp.utcnow().isoformat(),
    }
    atomic_json_write(self.output_dir / "completed.json", completion)
    print(f"[complete] {self.config.experiment_name}: {completion['epochs_recorded']} epochs")
    return self.history



Trainer._maybe_resume = _trainer_maybe_resume
Trainer._get_grad_norm = _trainer_grad_norm
Trainer._get_semantic_weight = _trainer_semantic_weight
Trainer._optimizer_step = _trainer_optimizer_step
Trainer.train_epoch = _trainer_train_epoch
Trainer.validate = _trainer_validate
Trainer.save_checkpoint = _trainer_save_checkpoint
Trainer.train = _trainer_train
