"""CPU unit tests of the method components on synthetic tensors.

No MIMIC-CXR data, checkpoint, GPU or pretrained weight is needed.
"""

import json
import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import checkpoint_utils as cu
from src.config import Config
from src.losses import CombinedLoss, HingeSemanticLoss, Noise2VoidLoss, SemanticEnergyFamily
from src.model import UNet
from src.noise import add_iid_noise, make_blind_spot_mask, stable_int_seed
from src.paper_tables import noisy_input_row, table4_paired, table5_absolute
from src.published import published_config
from src.report_utils import REPORT_FALLBACK_TEXT, extract_impression
from src.training_utils import atomic_json_write


# --------------------------------------------------------------------- noise
def test_noise_is_conditionally_unbiased_and_unclipped():
    clean = torch.full((1, 128, 128), 0.5)
    for noise_type in ("gaussian", "speckle", "poisson"):
        noisy = add_iid_noise(clean, noise_type, 0.2, deterministic_seed=123)
        assert abs(float(noisy.mean()) - 0.5) < 0.01, noise_type
    gaussian = add_iid_noise(clean, "gaussian", 0.3, deterministic_seed=1)
    assert float(gaussian.max()) > 1.0 and float(gaussian.min()) < 0.0
    clipped = add_iid_noise(clean, "gaussian", 0.3, deterministic_seed=1, clip=True)
    assert float(clipped.max()) <= 1.0 and float(clipped.min()) >= 0.0


def test_noise_is_deterministic_per_seed():
    clean = torch.rand(1, 64, 64, generator=torch.Generator().manual_seed(0))
    for noise_type in ("gaussian", "speckle", "poisson"):
        a = add_iid_noise(clean, noise_type, 0.1, deterministic_seed=7)
        b = add_iid_noise(clean, noise_type, 0.1, deterministic_seed=7)
        c = add_iid_noise(clean, noise_type, 0.1, deterministic_seed=8)
        assert torch.equal(a, b) and not torch.equal(a, c)
    with pytest.raises(ValueError):
        add_iid_noise(clean, "salt", 0.1)


def test_stable_int_seed_is_stable():
    assert stable_int_seed("abc") == stable_int_seed("abc")
    assert stable_int_seed("abc") != stable_int_seed("abd")


# --------------------------------------------------------------------- mask
def test_blind_spot_mask_count_and_exclusion_radius():
    mask = make_blind_spot_mask((1, 256, 256), 0.05, deterministic_seed=11, exclusion_radius=1)
    assert int(mask.sum()) == int(256 * 256 * 0.05)
    m = mask[0].bool()
    padded = F.pad(m[None, None].float(), (1, 1, 1, 1))[0, 0]
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbour = padded[1 + dy:1 + dy + 256, 1 + dx:1 + dx + 256].bool()
            assert not (m & neighbour).any(), "two targets are neighbours"
    again = make_blind_spot_mask((1, 256, 256), 0.05, deterministic_seed=11)
    assert torch.equal(mask, again)


def test_blind_spot_input_replaces_only_targets_with_a_neighbour():
    torch.manual_seed(0)
    noisy = torch.rand(2, 1, 32, 32)
    mask = torch.stack([make_blind_spot_mask((1, 32, 32), 0.05, deterministic_seed=s) for s in (1, 2)])
    loss = Noise2VoidLoss(Config())
    blind = loss.create_blind_spot_input(noisy, mask, generator=torch.Generator().manual_seed(3))
    unchanged = ~mask.bool()
    assert torch.equal(blind[unchanged], noisy[unchanged])
    padded = F.pad(noisy, (1, 1, 1, 1), mode="reflect")
    neighbours = torch.stack([
        padded[:, :, 1 + dy:1 + dy + 32, 1 + dx:1 + dx + 32]
        for dy, dx in Noise2VoidLoss.NEIGHBORHOOD_SHIFTS
    ])
    targets = mask.bool()
    matches = (neighbours == blind.unsqueeze(0)).any(dim=0)
    assert matches[targets].all()
    assert not torch.equal(blind[targets], noisy[targets])


def test_n2v_loss_uses_only_masked_pixels():
    noisy = torch.zeros(1, 1, 8, 8)
    mask = torch.zeros(1, 1, 8, 8)
    mask[0, 0, 2, 3] = 1
    denoised = torch.ones(1, 1, 8, 8)
    denoised[0, 0, 2, 3] = 0.5
    assert math.isclose(float(Noise2VoidLoss(Config())(denoised, noisy, mask)), 0.25)


# --------------------------------------------------------------------- model
def test_unet_is_identity_at_initialisation_and_keeps_checkpoint_keys():
    model = UNet(Config()).eval()
    x = torch.rand(1, 1, 64, 64)
    with torch.no_grad():
        assert torch.equal(model(x), x)
    keys = set(model.state_dict())
    for key in ("inc.double_conv.0.weight", "down1.maxpool_conv.1.double_conv.0.weight",
                "up4.conv.double_conv.4.weight", "outc.weight", "res_scale"):
        assert key in keys
    assert math.isclose(float(model.res_scale.detach()), 0.1, rel_tol=1e-6)


# ------------------------------------------------------- report-relative loss
class FakeVLM(nn.Module):
    """Tiny differentiable stand-in for BioViL-T with unit-norm embeddings."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 8)
        for p in self.parameters():
            p.requires_grad_(False)

    def encode_image(self, image):
        pooled = F.adaptive_avg_pool2d(image, 2).flatten(1)
        return F.normalize(self.proj(pooled), dim=-1)

    def encode_text(self, texts):
        g = torch.Generator().manual_seed(len(texts))
        return F.normalize(torch.randn(len(texts), 8, generator=g), dim=-1)

    def compute_similarity_raw(self, a, b):
        return (a * b).sum(-1)


def _report_inputs():
    torch.manual_seed(0)
    noisy = torch.rand(3, 1, 16, 16)
    denoised = (noisy + 0.2 * torch.randn_like(noisy)).requires_grad_(True)
    return noisy, denoised, ["a", "b", "c"]


def test_softplus_default_matches_family_and_eq5():
    cfg = replace(Config(), semantic_margin=0.0, softplus_beta=20.0)
    vlm = FakeVLM()
    noisy, denoised, reports = _report_inputs()
    a, metrics = HingeSemanticLoss(vlm, cfg)(denoised, noisy, reports)
    b, _ = SemanticEnergyFamily(vlm, cfg, "softplus")(denoised, noisy, reports)
    assert torch.allclose(a, b)
    s_noisy = (vlm.encode_image(noisy.clamp(0, 1)) * vlm.encode_text(reports)).sum(-1)
    s_den = (vlm.encode_image(denoised.clamp(0, 1)) * vlm.encode_text(reports)).sum(-1)
    d = s_noisy - s_den
    expected = (torch.log1p(torch.exp(20.0 * d)) / 20.0).mean()
    assert torch.allclose(a, expected, atol=1e-6)
    assert math.isclose(metrics["cos_change"], float((s_den - s_noisy).mean()), abs_tol=1e-6)


def test_report_loss_gradient_reaches_denoised_only():
    cfg = Config()
    noisy, denoised, reports = _report_inputs()
    noisy.requires_grad_(True)
    loss, _ = HingeSemanticLoss(FakeVLM(), cfg)(denoised, noisy, reports)
    loss.backward()
    assert denoised.grad is not None and denoised.grad.abs().sum() > 0
    assert noisy.grad is None


def test_energy_family_variants():
    cfg = replace(Config(), semantic_margin=0.0)
    vlm = FakeVLM()
    noisy, _, reports = _report_inputs()
    same = noisy.clone()
    hard, _ = SemanticEnergyFamily(vlm, cfg, "hard_hinge")(same, noisy, reports)
    absolute, _ = SemanticEnergyFamily(vlm, cfg, "abs_pres")(same, noisy, reports)
    squared, _ = SemanticEnergyFamily(vlm, cfg, "mse_pres")(same, noisy, reports)
    assert float(hard) == 0.0 and float(absolute) == 0.0 and float(squared) == 0.0
    neg, metrics = SemanticEnergyFamily(vlm, cfg, "neg_cos")(same, noisy, reports)
    assert metrics["variant"] == "neg_cos"
    assert math.isclose(float(neg), -metrics["cos_den"], abs_tol=1e-6)
    _, denoised, _ = _report_inputs()
    for variant in SemanticEnergyFamily.VARIANTS:
        loss, _ = SemanticEnergyFamily(vlm, cfg, variant)(denoised, noisy, reports)
        loss.backward()
        assert denoised.grad is not None
        denoised.grad = None
    with pytest.raises(ValueError):
        SemanticEnergyFamily(vlm, cfg, "unknown")


def test_combined_loss_warmup_and_n2v_only():
    cfg = published_config("proposed")
    combined = CombinedLoss.__new__(CombinedLoss)
    nn.Module.__init__(combined)
    combined.config = cfg
    assert combined._perceptual_weight(0) == 0.0
    assert math.isclose(combined._perceptual_weight(2), 0.005 * 2 / 5)
    assert math.isclose(combined._perceptual_weight(9), 0.005)

    base = CombinedLoss(None, published_config("n2v_baseline"))
    noisy = torch.rand(2, 1, 16, 16)
    mask = torch.zeros_like(noisy)
    mask[:, :, ::4, ::4] = 1
    total, metrics = base(noisy + 0.1, noisy, noisy, mask, ["r", "r"], lambda_semantic_eff=0.0, epoch=3)
    assert math.isclose(float(total), 0.01, rel_tol=1e-5)
    assert metrics["loss_semantic"] == 0.0 and metrics["loss_perceptual"] == 0.0


def test_unknown_perceptual_backbone_is_rejected():
    with pytest.raises(ValueError):
        CombinedLoss(None, replace(Config(), lambda_perceptual=0.1, perceptual_backbone="vgg"))


# ------------------------------------------------------------ report parsing
def test_report_section_priority_and_word_cap():
    report = (
        "INDICATION: cough.\n\nFINDINGS: The lungs are clear without focal consolidation.\n\n"
        "IMPRESSION: No acute cardiopulmonary process identified today."
    )
    assert extract_impression(report).startswith("No acute cardiopulmonary")
    findings_only = "FINDINGS: The lungs are clear without focal consolidation or effusion."
    assert extract_impression(findings_only).startswith("The lungs are clear")
    long = "IMPRESSION: " + " ".join(["word"] * 150)
    assert len(extract_impression(long).split()) == 100
    assert extract_impression("") == REPORT_FALLBACK_TEXT


# ------------------------------------------------------- checkpoint signature
def test_training_signature_rules():
    cfg = published_config("proposed")
    sig = cu.training_signature(cfg, train_hash_value="t", val_hash_value="v")
    assert sig == cu.training_signature(cfg, train_hash_value="t", val_hash_value="v")
    moved = replace(cfg, output_dir="/elsewhere", device="cpu", save_interval=3)
    assert sig == cu.training_signature(moved, train_hash_value="t", val_hash_value="v")
    assert sig != cu.training_signature(replace(cfg, lambda_semantic=0.01),
                                        train_hash_value="t", val_hash_value="v")
    assert sig != cu.training_signature(cfg, train_hash_value="t2", val_hash_value="v")
    with pytest.raises(ValueError):
        cu.training_signature(cfg)
    assert cu.checkpoint_is_training_compatible({"training_signature": sig}, cfg, "t", "v")


def test_atomic_json_write_handles_dataclasses_and_non_finite(tmp_path):
    path = tmp_path / "x.json"
    atomic_json_write(path, {"config": Config(), "nan": float("nan")})
    payload = json.loads(path.read_text())
    assert payload["config"]["image_size"] == 256 and payload["nan"] is None


# ---------------------------------------------------------------- paper tables
def _synthetic_results():
    rng = np.random.default_rng(0)
    rows = []
    for seed_index, seed in enumerate((7, 42, 123)):
        for patient in range(10):
            for noise_type in ("gaussian", "poisson", "speckle"):
                for sigma in (0.05, 0.10, 0.15, 0.20):
                    base = 30 + patient * 0.1 - sigma * 10
                    input_psnr = 20 - sigma * 20
                    for method, shift in (("n2v_baseline", 0.0), ("proposed", -0.2 - 0.01 * seed_index)):
                        rows.append({
                            "method": method, "seed": seed, "subject_id": f"p{patient}",
                            "dicom_id": f"d{patient}", "noise_type": noise_type, "noise_sigma": sigma,
                            "input_psnr": input_psnr, "input_ssim": 0.4,
                            "psnr": base + shift, "ssim": 0.9, "lpips": 0.15 + shift / 100,
                            "medperc_l2": 0.001, "medperc_cos": 0.1,
                            "cos_change": (0.3 if method == "proposed" else 0.0) + rng.normal(0, 1e-3),
                        })
    return pd.DataFrame(rows)


def test_table4_matches_hand_computation():
    frame = _synthetic_results()
    table = table4_paired(frame).set_index("metric")
    expected = np.array([-0.20, -0.21, -0.22])
    assert math.isclose(table.loc["psnr", "mean_difference"], expected.mean(), abs_tol=1e-9)
    assert math.isclose(table.loc["psnr", "sd_across_seeds"], expected.std(ddof=1), abs_tol=1e-9)
    assert table.loc["psnr", "patients_favourable_percent"] == 0.0
    assert table.loc["cos_change", "patients_favourable_percent"] == 100.0
    assert table.loc["lpips", "patients_favourable_percent"] == 100.0
    assert table.loc["psnr", "n_patients"] == 10


def test_table5_and_noisy_input_row():
    frame = _synthetic_results()
    table = table5_absolute(frame)
    assert list(table.index) == ["n2v_baseline", "proposed"]
    assert (table["n_seeds"] == 3).all()
    noisy = noisy_input_row(frame)
    assert math.isclose(noisy["psnr"], 20 - 0.125 * 20)
    broken = frame.copy()
    broken.loc[broken.index[0], "input_psnr"] += 1
    with pytest.raises(ValueError):
        noisy_input_row(broken)
