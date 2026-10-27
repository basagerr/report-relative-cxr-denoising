"""GPU smoke test of the proposed training objective on synthetic tensors.

Requires a CUDA GPU and the BioViL-T and TorchXRayVision weights, which are
downloaded on first use. Skipped automatically when CUDA, ``hi-ml-multimodal``
or ``torchxrayvision`` is unavailable.
Run directly with ``python -m tests.test_gpu_training_smoke`` or via pytest.
"""

from dataclasses import replace
import hashlib
import json
import math
import tempfile

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import Config
from src.model import UNet
from src.trainer import Trainer
from src.vlm import VLMWrapper


class SyntheticDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        g = torch.Generator().manual_seed(100 + index)

        clean = torch.rand(1, 256, 256, generator=g)
        noisy = (
            clean + 0.05 * torch.randn(clean.shape, generator=g)
        ).clamp(0, 1)

        mask = torch.zeros_like(clean)
        mask[:, ::8, ::8] = 1.0

        return {
            "clean_image": clean,
            "noisy_image": noisy,
            "blind_spot_mask": mask,
            "report_text": "Synthetic test report.",
            "dicom_id": f"synthetic_{index:03d}",
        }


def split_hash(name):
    specification = {
        "dataset": "proposed-synthetic-smoke-v1",
        "split": name,
        "indices": [0, 1],
        "seed_rule": "100+index",
        "shape": [1, 256, 256],
        "noise_std": 0.05,
        "mask_stride": 8,
    }
    return hashlib.sha256(
        json.dumps(specification, sort_keys=True).encode()
    ).hexdigest()


def main():
    print("=== PROPOSED UNET TRAINING SMOKE TEST ===")

    assert torch.cuda.is_available(), "A CUDA GPU is required"

    torch.manual_seed(42)
    device = torch.device("cuda")

    cfg = replace(
        Config(),
        model_variant="hinge",
        semantic_variant="softplus",
        lambda_semantic=0.001,
        lambda_perceptual=0.005,
        perceptual_backbone="xrv",
        semantic_margin=0.0,
        semantic_warmup_epochs=0,
        perceptual_warmup_epochs=0,
        use_amp=False,
        gradient_accumulation=1,
    )

    loader = DataLoader(
        SyntheticDataset(),
        batch_size=2,
        shuffle=False,
    )

    print("Loading BioViL-T...")
    vlm = VLMWrapper(cfg).to(device).eval()

    for parameter in vlm.parameters():
        parameter.requires_grad_(False)

    with tempfile.TemporaryDirectory(
        prefix="cxr_proposed_smoke_"
    ) as output_dir:

        trainer = Trainer(
            model=UNet(cfg),
            vlm=vlm,
            config=cfg,
            train_loader=loader,
            val_loader=loader,
            output_dir=output_dir,
            train_hash=split_hash("train"),
            val_hash=split_hash("validation"),
        )

        for parameter in trainer.criterion.perceptual_loss.parameters():
            parameter.requires_grad_(False)

        before = {
            name: p.detach().clone()
            for name, p in trainer.model.named_parameters()
            if p.requires_grad
        }

        trainer.current_epoch = 1
        assert abs(trainer._get_semantic_weight(1) - 0.001) < 1e-12
        assert abs(trainer.criterion._perceptual_weight(1) - 0.005) < 1e-12
        metrics = trainer.train_epoch()
        assert metrics['loss_semantic'] > 0
        print("First epoch perceptual:", metrics["loss_perceptual"])

        # After the first optimizer update, the UNet is no longer
        # an exact identity map. Test the next training epoch.
        trainer.current_epoch = 2
        metrics = trainer.train_epoch()

        print("Second epoch perceptual:", metrics["loss_perceptual"])
        assert metrics["loss_semantic"] > 0
        assert metrics["loss_perceptual"] > 0, (
            "Perceptual loss remained zero after model update"
        )

        assert metrics
        assert all(
            isinstance(value, (int, float))
            and math.isfinite(value)
            for value in metrics.values()
        )

        changed = [
            name
            for name, p in trainer.model.named_parameters()
            if name in before
            and not torch.equal(before[name], p.detach())
        ]

        assert changed, "No model weights were updated"

        print("\n=== RESULTS ===")
        for name in (
            "loss",
            "loss_ssl",
            "loss_semantic",
            "loss_perceptual",
        ):
            print(f"{name}: {metrics.get(name)}")

        print("Updated parameter tensors:", len(changed))

        print("\n=== PROPOSED VALIDATION ===")

        before_validation = {
            name: p.detach().clone()
            for name, p in trainer.model.named_parameters()
        }

        validation = trainer.validate()

        print("Validation metrics:", validation)

        assert validation, "Validation returned no metrics"
        assert all(
            isinstance(value, (int, float))
            and math.isfinite(value)
            for value in validation.values()
        )

        for name, p in trainer.model.named_parameters():
            assert torch.equal(
                before_validation[name], p.detach()
            ), f"Validation modified model parameter: {name}"

        print("PROPOSED VALIDATION: PASS")
        print("MODEL WEIGHTS UNCHANGED: PASS")
        print("\nPROPOSED UNET TRAIN EPOCH: PASS")
        print("MODEL WEIGHTS UPDATED: PASS")
        print("Synthetic data only: YES")


@pytest.mark.gpu
def test_proposed_training_smoke_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU not available")
    pytest.importorskip("health_multimodal")
    pytest.importorskip("torchxrayvision")
    main()


if __name__ == "__main__":
    main()
