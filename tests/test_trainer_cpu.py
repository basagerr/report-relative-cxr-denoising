"""CPU smoke test of the Noise2Void training loop on synthetic tensors.

Exercises one training epoch, validation without gradient tracking, the
checkpoint files and the completion marker. No BioViL-T or XRV weights are
needed because the N2V baseline uses neither for optimisation.
"""

import json
from dataclasses import replace

import torch
from torch.utils.data import DataLoader, Dataset

from src import trainer as trainer_module
from src.config import Config
from src.model import UNet
from src.noise import make_blind_spot_mask
from src.published import published_config


class SyntheticDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        g = torch.Generator().manual_seed(100 + index)
        clean = torch.rand(1, 32, 32, generator=g)
        noisy = clean + 0.1 * torch.randn(clean.shape, generator=g)
        return {
            "clean_image": clean,
            "noisy_image": noisy,
            "blind_spot_mask": make_blind_spot_mask((1, 32, 32), 0.05, deterministic_seed=index),
            "report_text": "synthetic",
            "dicom_id": f"synthetic_{index:03d}",
        }


def test_n2v_training_loop_on_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer_module, "device", torch.device("cpu"))
    cfg = replace(
        published_config("n2v_baseline"),
        experiment_name="synthetic_n2v",
        epochs=2,
        common_selection_start_epoch=0,
        gradient_accumulation=2,
        use_amp=False,
        features=[8, 16, 32, 64],
        history_pt_path=str(tmp_path / "history.pt"),
        history_json_path=str(tmp_path / "history.json"),
    )
    loader = DataLoader(SyntheticDataset(), batch_size=2, shuffle=False)
    trainer = trainer_module.Trainer(
        model=UNet(cfg), vlm=None, config=cfg,
        train_loader=loader, val_loader=loader,
        output_dir=str(tmp_path / "run"), train_hash="train", val_hash="val",
    )

    grad_modes = []
    original_forward = trainer.model.forward

    def recording_forward(x):
        grad_modes.append(torch.is_grad_enabled())
        return original_forward(x)

    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    trainer.train_epoch()
    assert any(not torch.equal(before[k], v) for k, v in trainer.model.state_dict().items())

    trainer.model.forward = recording_forward
    validation = trainer.validate()
    trainer.model.forward = original_forward
    assert grad_modes and not any(grad_modes), "validation must run without gradients"
    assert torch.isfinite(torch.tensor(validation["psnr"]))

    trainer.current_epoch = 0
    trainer.history = {key: [] for key in trainer.HISTORY_KEYS}
    trainer.train()
    run = tmp_path / "run"
    completion = json.loads((run / "completed.json").read_text())
    assert completion["status"] == "complete" and completion["epochs_recorded"] == 2
    assert (run / "checkpoints" / "best.pt").is_file()
    assert (run / "checkpoints" / "latest.pt").is_file()
    saved = torch.load(run / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    assert saved["training_signature"] == completion["training_signature"]
    restored = UNet(Config(**{k: v for k, v in saved["config"].items()}))
    restored.load_state_dict(saved["model_state_dict"], strict=True)
