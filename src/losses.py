"""Original publication training loss functions."""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Config

if TYPE_CHECKING:
    from src.vlm import VLMWrapper

class Noise2VoidLoss(nn.Module):
    NEIGHBORHOOD_SHIFTS = ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1))
    def __init__(self,config): super().__init__(); self.config=config
    def create_blind_spot_input(self,noisy,mask,generator=None):
        B,C,H,W=noisy.shape; mask_bool=mask.bool()
        # The exclusion mask guarantees all eight donors are non-target pixels.
        shifts=torch.randint(0,len(self.NEIGHBORHOOD_SHIFTS),(B,1,H,W),device=noisy.device,generator=generator)
        padded=F.pad(noisy,(1,1,1,1),mode="reflect"); out=noisy.clone()
        for i,(dy,dx) in enumerate(self.NEIGHBORHOOD_SHIFTS):
            donor=padded[:,:,1+dy:1+dy+H,1+dx:1+dx+W]
            out=torch.where(((shifts==i)&mask_bool).expand_as(out),donor,out)
        return out
    def forward(self,denoised,noisy,mask):
        target_mask=mask.bool().expand_as(denoised)
        return F.mse_loss(denoised[target_mask],noisy[target_mask])



class HingeSemanticLoss(nn.Module):
    """One-sided softplus hinge on semantic degradation vs the original noisy input.

    Loss is UNWEIGHTED here. CombinedLoss applies `lambda_semantic`.

    Paper Section 3.3, Eq. 5. Returns (loss, metrics_dict) where
    metrics_dict contains diagnostics: cos values, activation rate, effective
    gradient, embedding drift.
    """

    def __init__(self, vlm: "VLMWrapper", config: Config):
        super().__init__()
        self.vlm    = vlm
        self.config = config
        self.margin = config.semantic_margin
        self.beta   = getattr(config, "softplus_beta", 5.0)

    def forward(self, denoised: torch.Tensor, noisy: torch.Tensor,
                report_texts: List[str]) -> Tuple[torch.Tensor, Dict[str, float]]:
        denoised_clamped = denoised.clamp(0, 1).float()
        noisy_f32        = noisy.float()

        # Anchor branch: noisy + text encoded WITHOUT gradient (frozen references)
        with torch.no_grad():
            noisy_features = self.vlm.encode_image(noisy_f32.clamp(0, 1))
            text_features  = self.vlm.encode_text(report_texts)
            cos_noisy      = torch.sum(noisy_features * text_features, dim=-1)

        # Live branch: denoised gets gradient through the frozen VLM into the U-Net
        denoised_features = self.vlm.encode_image(denoised_clamped)
        cos_denoised      = torch.sum(denoised_features * text_features, dim=-1)

        # One-sided softplus hinge on semantic degradation
        degradation     = cos_noisy - cos_denoised - self.margin
        per_sample_loss = F.softplus(degradation, beta=self.beta)
        semantic_loss   = per_sample_loss.mean()

        # Diagnostics (no gradient)
        with torch.no_grad():
            activation_rate = (degradation > 0).float().mean().item()
            eff_grad        = torch.sigmoid(self.beta * degradation).mean().item()
            drift = (1.0 - F.cosine_similarity(denoised_features, noisy_features, dim=-1)).mean().item()

        return semantic_loss, {
            "variant":               "softplus",
            "cos_noisy":             cos_noisy.mean().item(),
            "cos_den":               cos_denoised.mean().item(),
            "cos_change":            (cos_denoised - cos_noisy).mean().item(),
            "semantic_loss_raw":     semantic_loss.item(),
            "hinge_activation_rate": activation_rate,
            "hinge_eff_gradient":    eff_grad,
            "hinge_margin":          self.margin,
            "softplus_beta":         self.beta,
            "drift_loss_raw":        drift,
            "embedding_drift":       drift,
        }


class SemanticEnergyFamily(HingeSemanticLoss):
    """The five semantic losses use the same inputs and return the same outputs."""

    VARIANTS = (
        "neg_cos",
        "abs_pres",
        "mse_pres",
        "hard_hinge",
        "softplus",
    )

    def __init__(self, vlm, config, variant: str = "softplus"):
        super().__init__(vlm, config)
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}")
        self.variant = variant

    def _energy(self, cos_noisy, cos_denoised):
        # This value is positive when the semantic requirement is not met.
        violation = cos_noisy - cos_denoised - self.margin

        if self.variant == "neg_cos":
            return -cos_denoised

        if self.variant == "abs_pres":
            return torch.abs(cos_denoised - cos_noisy)

        if self.variant == "mse_pres":
            # One score per image, then the forward method takes the batch mean.
            # So this is MSE preservation and also scalar squared L2 preservation.
            return (cos_denoised - cos_noisy).pow(2)

        if self.variant == "hard_hinge":
            return torch.clamp(violation, min=0.0)

        return F.softplus(violation, beta=self.beta)



# Original forward of the five-loss family from the publication notebook.
# Without it the family inherited the softplus forward of HingeSemanticLoss.
def _semantic_energy_forward(self, denoised, noisy, report_texts):
    # The noisy score is the fixed anchor. It does not receive a gradient.
    with torch.no_grad():
        noisy_features = self.vlm.encode_image(noisy.float().clamp(0, 1))
        text_features = self.vlm.encode_text(report_texts)
        cos_noisy = torch.sum(noisy_features * text_features, dim=-1)

    # The denoised branch stays connected to the U-Net.
    denoised_features = self.vlm.encode_image(denoised.clamp(0, 1).float())
    cos_denoised = torch.sum(denoised_features * text_features, dim=-1)

    per_image_energy = self._energy(cos_noisy, cos_denoised)
    loss = per_image_energy.mean()

    with torch.no_grad():
        violation = cos_noisy - cos_denoised - self.margin
        difference = cos_denoised - cos_noisy
        drift = 1.0 - F.cosine_similarity(denoised_features, noisy_features, dim=-1)

        if self.variant == "neg_cos":
            effective_gradient = torch.ones_like(difference)
        elif self.variant == "abs_pres":
            effective_gradient = (difference != 0).float()
        elif self.variant == "mse_pres":
            effective_gradient = 2.0 * difference.abs()
        elif self.variant == "hard_hinge":
            effective_gradient = (violation > 0).float()
        else:
            effective_gradient = torch.sigmoid(self.beta * violation)

    metrics = {
        "variant": self.variant,
        "cos_noisy": cos_noisy.mean().item(),
        "cos_den": cos_denoised.mean().item(),
        "cos_change": difference.mean().item(),
        "semantic_loss_raw": loss.item(),
        "hinge_activation_rate": (violation > 0).float().mean().item(),
        "hinge_eff_gradient": effective_gradient.mean().item(),
        "hinge_margin": self.margin,
        "softplus_beta": self.beta,
        "embedding_drift": drift.mean().item(),
        "drift_loss_raw": drift.mean().item(),
    }
    return loss, metrics


SemanticEnergyFamily.forward = _semantic_energy_forward


class PatchFeatureLoss(nn.Module):
    """Base class for perceptual losses on hooked intermediate features.

    Subclasses must:
      1. Load `self.backbone` and freeze it
      2. Register forward hooks (filling `self._features: Dict[str, Tensor]`)
      3. Override `_preprocess(imgs)` to map [B, 1, H, W] in [0,1] -> backbone input
    """

    def __init__(self):
        super().__init__()
        self._features: Dict[str, torch.Tensor] = {}
        self._target_layers: List[str] = []

    def _make_hook(self, name: str):
        def hook(_module, _input, output):
            self._features[name] = output
        return hook

    def _preprocess(self, imgs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _normalize_features(self, feats: torch.Tensor) -> torch.Tensor:
        """L2-normalize along the channel dimension (stabilizes the L1 distance)."""
        return feats / (feats.norm(dim=1, keepdim=True) + 1e-8)

    def forward(self, noisy_imgs: torch.Tensor, denoised_imgs: torch.Tensor) -> torch.Tensor:
        """L1 on L2-normalized features at each hooked layer, averaged across layers."""
        noisy_pp    = self._preprocess(noisy_imgs.detach().clamp(0, 1))
        denoised_pp = self._preprocess(denoised_imgs.clamp(0, 1))

        # Reference branch (no gradient)
        with torch.no_grad():
            self.backbone(noisy_pp)
            noisy_feats = {k: v.clone() for k, v in self._features.items()}

        # Live branch (gradient flows back to U-Net)
        self.backbone(denoised_pp)
        den_feats = self._features

        loss = torch.tensor(0.0, device=denoised_imgs.device)
        for layer in self._target_layers:
            nf = self._normalize_features(noisy_feats[layer])
            df = self._normalize_features(den_feats[layer])
            loss = loss + F.l1_loss(df, nf)
        return loss / max(1, len(self._target_layers))


class XRVPerceptualLoss(PatchFeatureLoss):
    """Perceptual loss using TorchXRayVision DenseNet-121 features (denseblock2 + 3)."""

    TARGET_LAYERS = ("features.denseblock2", "features.denseblock3")
    INPUT_SIZE    = 224

    def __init__(self, config: Config):
        super().__init__()
        import torchxrayvision as xrv
        print("Loading TorchXRayVision DenseNet-121 for perceptual loss…")

        self.backbone = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        for name, module in self.backbone.named_modules():
            if name in self.TARGET_LAYERS:
                module.register_forward_hook(self._make_hook(name))
        self._target_layers = list(self.TARGET_LAYERS)
        print(f"  [ok] XRV ready (layers: {self._target_layers})")

    def _preprocess(self, imgs: torch.Tensor) -> torch.Tensor:
        """[B, 1, H, W] in [0,1] -> [B, 1, 224, 224] in [-1024, 1024]."""
        x = F.interpolate(imgs, size=(self.INPUT_SIZE, self.INPUT_SIZE),
                          mode="bilinear", align_corners=False)
        return x * 2048.0 - 1024.0


# Hugging Face revision of microsoft/rad-dino. The publication notebook
# resolved this at run time and recorded it in each private run manifest.
# Set it to that recorded revision to reproduce the RAD-DINO experiments
# exactly; None loads the current default revision.
RAD_DINO_REVISION = None


class RadDINOPerceptualLoss(PatchFeatureLoss):
    """Perceptual loss using RAD-DINO spatial patch embeddings."""

    MODEL_ID = "microsoft/rad-dino"
    INPUT_SIZE = 518
    PATCH_GRID = 37
    MEAN = (0.5307, 0.5307, 0.5307)
    STD = (0.2583, 0.2583, 0.2583)
    RESAMPLE_ID = 3  # PIL.Image.Resampling.BICUBIC
    PREPROCESS_VERSION = "official_bit_processor_square_tensor_v1"

    def __init__(self, config: Config):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel

        revision = RAD_DINO_REVISION
        kwargs = {"revision": revision} if revision else {}
        print(
            f"Loading RAD-DINO ({self.MODEL_ID}, "
            f"revision={revision or 'default'})…"
        )

        processor = AutoImageProcessor.from_pretrained(
            self.MODEL_ID,
            **kwargs,
        )
        self._verify_official_processor(processor)

        self.backbone = AutoModel.from_pretrained(
            self.MODEL_ID,
            **kwargs,
        )
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        self._target_layers = ["patch_tokens"]
        print(
            f"  [ok] RAD-DINO ready "
            f"({self.PATCH_GRID}×{self.PATCH_GRID} patches; "
            f"official processor verified)"
        )

    @classmethod
    def _verify_official_processor(cls, processor) -> None:
        mean = tuple(float(value) for value in processor.image_mean)
        std = tuple(float(value) for value in processor.image_std)
        crop_size = getattr(processor, "crop_size", {})
        size = getattr(processor, "size", {})

        assert bool(getattr(processor, "do_resize", False))
        assert bool(getattr(processor, "do_center_crop", False))
        assert bool(getattr(processor, "do_rescale", False))
        assert bool(getattr(processor, "do_normalize", False))
        assert bool(getattr(processor, "do_convert_rgb", False))
        assert int(crop_size["height"]) == cls.INPUT_SIZE
        assert int(crop_size["width"]) == cls.INPUT_SIZE
        assert int(size["shortest_edge"]) == cls.INPUT_SIZE
        assert int(getattr(processor, "resample")) == cls.RESAMPLE_ID
        assert np.allclose(mean, cls.MEAN, atol=1e-7)
        assert np.allclose(std, cls.STD, atol=1e-7)

    @classmethod
    def _preprocess_tensor(
        cls,
        imgs: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable equivalent of the official processor for square [0,1] tensors."""
        if imgs.ndim != 4:
            raise ValueError(
                f"RAD-DINO expects [B,C,H,W], got {tuple(imgs.shape)}"
            )
        if imgs.shape[-2] != imgs.shape[-1]:
            raise ValueError(
                "RAD-DINO manual preprocessing is locked to square images; "
                "the official resize+center-crop would differ for non-square input"
            )
        if imgs.shape[1] == 1:
            x = imgs.repeat(1, 3, 1, 1)
        elif imgs.shape[1] == 3:
            x = imgs
        else:
            raise ValueError(
                f"RAD-DINO expects 1 or 3 channels, got {imgs.shape[1]}"
            )

        # Inputs are already floats in [0,1], so the processor's 1/255 rescale
        # has already been applied by ToTensor.
        x = F.interpolate(
            x,
            size=(cls.INPUT_SIZE, cls.INPUT_SIZE),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        mean = x.new_tensor(cls.MEAN).view(1, 3, 1, 1)
        std = x.new_tensor(cls.STD).view(1, 3, 1, 1)
        return (x - mean) / std

    def _preprocess(self, imgs: torch.Tensor) -> torch.Tensor:
        return self._preprocess_tensor(imgs)

    def _patch_features(self, imgs: torch.Tensor) -> torch.Tensor:
        output = self.backbone(pixel_values=imgs)
        patches = output.last_hidden_state[:, 1:, :]
        batch_size, number_of_patches, dimension = patches.shape
        expected = self.PATCH_GRID * self.PATCH_GRID
        if number_of_patches != expected:
            raise RuntimeError(
                f"RAD-DINO returned {number_of_patches} patches; "
                f"expected {expected}"
            )
        return patches.transpose(1, 2).reshape(
            batch_size,
            dimension,
            self.PATCH_GRID,
            self.PATCH_GRID,
        )

    def forward(
        self,
        noisy_imgs: torch.Tensor,
        denoised_imgs: torch.Tensor,
    ) -> torch.Tensor:
        noisy_preprocessed = self._preprocess(
            noisy_imgs.detach().clamp(0, 1)
        )
        denoised_preprocessed = self._preprocess(
            denoised_imgs.clamp(0, 1)
        )
        with torch.no_grad():
            noisy_patches = self._patch_features(
                noisy_preprocessed
            )
        denoised_patches = self._patch_features(
            denoised_preprocessed
        )
        return F.l1_loss(
            self._normalize_features(denoised_patches),
            self._normalize_features(noisy_patches),
        )


class CombinedLoss(nn.Module):
    """Linearly combines SSL, hinge, and perceptual losses; dispatches on `config.model_variant`."""

    PERCEPTUAL_BACKBONES = {
        "xrv":      lambda vlm, cfg: XRVPerceptualLoss(cfg).to(torch.device(cfg.device)),
        "rad_dino": lambda vlm, cfg: RadDINOPerceptualLoss(cfg).to(torch.device(cfg.device)),
    }

    def __init__(self, vlm: Optional["VLMWrapper"], config: Config):
        super().__init__()
        self.config    = config
        self.vlm       = vlm
        self.loss_mode = config.model_variant
        self.ssl_loss  = Noise2VoidLoss(config)

        # Hinge — only built if variant requires it
        if self.loss_mode == "hinge":
            assert vlm is not None, "VLM required for hinge mode"
            # Default path is byte-identical to before: variant 'softplus' == the
            # reported model. Only when config.semantic_variant is set to one of the
            # alternatives (neg_cos / abs_pres / mse_pres / hard_hinge) does the family class
            # take over, so the published configuration cannot be perturbed by this.
            _variant = getattr(config, "semantic_variant", "softplus")
            if _variant == "softplus":
                self.semantic_loss = HingeSemanticLoss(vlm, config)
            else:
                self.semantic_loss = SemanticEnergyFamily(vlm, config, variant=_variant)
        else:
            self.semantic_loss = None

        # Perceptual — only built if λ_perceptual > 0
        self.perceptual_loss = None
        if config.lambda_perceptual > 0:
            backbone = getattr(config, "perceptual_backbone", "xrv")
            if backbone not in self.PERCEPTUAL_BACKBONES:
                raise ValueError(f"Unknown perceptual backbone: {backbone!r} "
                                 f"(supported: {list(self.PERCEPTUAL_BACKBONES)})")
            self.perceptual_loss = self.PERCEPTUAL_BACKBONES[backbone](vlm, config)
            print(f"  [ok] Perceptual loss enabled (backbone={backbone}, λ={config.lambda_perceptual})")

        print(f"[ok] CombinedLoss initialized in {self.loss_mode!r} mode"
              f"{' + perceptual' if self.perceptual_loss else ''}")

    def _perceptual_weight(self, epoch: Optional[int]) -> float:
        """Linear warmup: λ_perc(t) = λ_perc · min(1, t / t_warmup)."""
        if epoch is None:
            return self.config.lambda_perceptual          # full weight if epoch unspecified
        warmup = getattr(self.config, "perceptual_warmup_epochs", 5)
        return self.config.lambda_perceptual * min(1.0, epoch / max(1, warmup))


# Original forward function from the publication notebook.
def _combined_forward(self, den_blind, den_full, noisy, mask, report_texts,
                      lambda_semantic_eff=None, epoch=None):
    """Total loss + metrics dict. Schema is stable across all variants."""
    metrics: Dict[str, float] = {}
    lam_sem = lambda_semantic_eff if lambda_semantic_eff is not None else self.config.lambda_semantic

    # ---------- 1. SSL (always active) ----------
    l_ssl = self.ssl_loss(den_blind, noisy, mask)
    metrics["loss_ssl"] = l_ssl.item()
    total_loss = self.config.lambda_ssl * l_ssl

    # ---------- 2. Semantic (hinge or just monitoring) ----------
    if self.semantic_loss is not None and lam_sem > 0:
        sem_loss, sem_m = self.semantic_loss(den_full, noisy, report_texts)
        metrics.update(sem_m)
        metrics.setdefault("variant", getattr(self.config, "semantic_variant", "softplus"))
        metrics.setdefault("hinge_eff_gradient", 0.0)
        metrics.setdefault("drift_loss_raw", metrics.get("embedding_drift", 0.0))
        w_sem = lam_sem * sem_loss
        metrics["loss_semantic"] = w_sem.item()
        metrics["preserve_loss"] = sem_loss.item()
        metrics["drift_loss"] = metrics.get("drift_loss_raw", 0.0)
        total_loss = total_loss + w_sem
    else:
        # Monitoring-only — still report cos values for the baseline
        if self.vlm is not None:
            with torch.no_grad():
                img_n = self.vlm.encode_image(noisy.float().clamp(0, 1))
                img_d = self.vlm.encode_image(den_full.clamp(0, 1).float())
                txt   = self.vlm.encode_text(report_texts)
                metrics["cos_noisy"]  = self.vlm.compute_similarity_raw(img_n, txt).mean().item()
                metrics["cos_den"]    = self.vlm.compute_similarity_raw(img_d, txt).mean().item()
                metrics["cos_change"] = metrics["cos_den"] - metrics["cos_noisy"]
        else:
            metrics.update({"cos_noisy": 0.0, "cos_den": 0.0, "cos_change": 0.0})
        metrics.update({
            # zero-fill for the hinge-active branch's metrics
            "loss_semantic":         0.0,
            "preserve_loss":         0.0,
            "drift_loss":            0.0,
            "embedding_drift":       0.0,
            "semantic_loss_raw":     0.0,
            "drift_loss_raw":        0.0,
            # ALSO zero-fill the four hinge-diagnostic keys to keep schema stable
            "hinge_activation_rate": 0.0,
            "hinge_eff_gradient":    0.0,
            "hinge_margin":          getattr(self.config, "semantic_margin", 0.0),
            "softplus_beta":         getattr(self.config, "softplus_beta", 20.0),
        })

    # ---------- 3. Perceptual (with warmup) ----------
    if self.perceptual_loss is not None:
        lam_perc_eff = self._perceptual_weight(epoch)
        if lam_perc_eff > 0:
            l_perc = self.perceptual_loss(noisy, den_full)
            w_perc = lam_perc_eff * l_perc
            total_loss = total_loss + w_perc
            metrics["loss_perceptual"]       = w_perc.item()
            metrics["loss_perceptual_raw"]   = l_perc.item()
            metrics["lambda_perceptual_eff"] = lam_perc_eff
        else:
            metrics.update({"loss_perceptual": 0.0, "loss_perceptual_raw": 0.0,
                            "lambda_perceptual_eff": 0.0})
    else:
        metrics.update({"loss_perceptual": 0.0, "loss_perceptual_raw": 0.0,
                        "lambda_perceptual_eff": 0.0})

    metrics["loss_total"]            = total_loss.item()
    metrics["lambda_semantic_eff"]   = lam_sem
    return total_loss, metrics

CombinedLoss.forward = _combined_forward
