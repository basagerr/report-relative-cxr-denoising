from .report_utils import extract_impression
import re
from typing import Any, Tuple

"""Original VLMWrapper extracted from the publication notebook.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from .config import Config

class VLMWrapper(nn.Module):
    """Small interface around the frozen BioViL-T image and text encoders."""

    def __init__(self, config: Config):
        super().__init__()
        if config.vlm_name != "biovil_t":
            raise ValueError(f"This notebook supports BioViL-T only, not {config.vlm_name!r}")

        self.config = config
        self.temperature = config.vlm_temperature
        self.image_model, self.biovil_transform = _load_biovil_image_encoder()
        self.text_inference = _load_biovil_text_encoder()
        self._text_embedding_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

        if config.vlm_frozen:
            self._freeze()

        print("[ok] BioViL-T image and text encoders loaded")

    def _freeze(self):
        for parameter in self.parameters():
            parameter.requires_grad = False
        if hasattr(self.text_inference, "model"):
            self.text_inference.model.eval()
            for parameter in self.text_inference.model.parameters():
                parameter.requires_grad = False
        self.eval()

    def train(self, mode=True):
        # The encoder stays frozen even when the U-Net enters train mode.
        return super().train(False)

    def to(self, target_device):
        super().to(target_device)
        self._device = target_device
        self.image_model = self.image_model.to(target_device)
        if hasattr(self.text_inference, "model"):
            self.text_inference.model = self.text_inference.model.to(target_device)
        return self

    @property
    def device(self):
        if hasattr(self, "_device"):
            return self._device
        return next(self.image_model.parameters()).device



def _load_biovil_image_encoder() -> Tuple[Any, Any]:
    """Load BioViL-T's image encoder. Returns (model, transform).

    `hi-ml-multimodal` exposes two APIs across versions; we try both.
    Raises RuntimeError if neither succeeds.
    """
    errors = []
    # API 1 (newer): get_image_inference
    try:
        from health_multimodal.image.utils import get_image_inference, ImageModelType
        eng = get_image_inference(image_model_type=ImageModelType.BIOVIL_T)
        return eng.model, eng.transform
    except Exception as e:
        errors.append(("get_image_inference", repr(e)))

    # API 2 (older): ImageInferenceEngine.from_biovil_t
    try:
        from health_multimodal.image import ImageInferenceEngine
        eng = ImageInferenceEngine.from_biovil_t()
        return eng.model, eng.transform
    except Exception as e:
        errors.append(("ImageInferenceEngine", repr(e)))

    msg = "Could not load BioViL-T image encoder.\n"
    for name, err in errors:
        msg += f"  via {name}: {err}\n"
    msg += "Install: pip install hi-ml-multimodal"
    raise RuntimeError(msg)


def _load_biovil_text_encoder() -> Any:
    """Load CXR-BERT (BioViL-T's text encoder). Returns the inference engine."""
    try:
        from health_multimodal.text import get_bert_inference
        return get_bert_inference()
    except Exception:
        from health_multimodal.text import TextInferenceEngine
        return TextInferenceEngine.from_biovil_t()


def _vlm_encode_image(self, image):
    image = self._preprocess_image(image.clamp(0, 1))
    output = self.image_model(image)

    if hasattr(output, "projected_global_embedding"):
        features = output.projected_global_embedding
    elif hasattr(output, "img_embedding"):
        features = output.img_embedding
    else:
        features = output if isinstance(output, torch.Tensor) else output[0]

    return F.normalize(features, dim=-1)


@torch.no_grad()
def _vlm_encode_text(self, texts):
    cleaned_reports = [self._clean_report(text) for text in texts]
    embeddings = []

    for text in cleaned_reports:
        cached = self._text_embedding_cache.get(text)
        if cached is not None:
            self._cache_hits += 1
            embeddings.append(cached.to(self.device))
            continue

        self._cache_misses += 1
        try:
            tokens = self.text_inference.tokenizer(
                [text],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
                add_special_tokens=True,
            ).to(self.device)
            embedding = self.text_inference.model.get_projected_text_embeddings(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            )
        except Exception as exc:
            raise RuntimeError(f"BioViL-T text encoding failed: {type(exc).__name__}") from exc

        self._text_embedding_cache[text] = embedding.cpu()
        embeddings.append(embedding)

    return F.normalize(torch.cat(embeddings, dim=0), dim=-1)


VLMWrapper.encode_image = _vlm_encode_image
VLMWrapper.encode_text = _vlm_encode_text


def _vlm_preprocess_image(self, image):
    if image.shape[1] == 1:
        image = image.repeat(1, 3, 1, 1)
    image = F.interpolate(
        image,
        size=(512, 512),
        mode="bilinear",
        align_corners=False,
    )
    crop_start = (512 - 448) // 2
    return image[:, :, crop_start:crop_start + 448, crop_start:crop_start + 448]


VLMWrapper._preprocess_image = _vlm_preprocess_image


def _vlm_clean_report(self, report):
    if not report or not isinstance(report, str) or len(report.strip()) < 5:
        return "chest radiograph"

    report = report.strip()
    placeholder_patterns = (
        r"^no report",
        r"^not available",
        r"^n/a$",
        r"^none$",
        r"^missing",
    )
    if any(re.match(pattern, report, re.IGNORECASE) for pattern in placeholder_patterns):
        return "chest radiograph"

    extracted = extract_impression(report, max_words=100)
    return extracted if len(extracted) >= 5 else "chest radiograph"


VLMWrapper._clean_report = _vlm_clean_report


# Original VLM methods from the publication notebook.
def _vlm_similarity_raw(self, image_features, text_features):
    return torch.sum(image_features * text_features, dim=-1)


def _vlm_similarity(self, image_features, text_features):
    return self.compute_similarity_raw(image_features, text_features) / self.temperature


def _vlm_cache_stats(self):
    total = self._cache_hits + self._cache_misses
    return {
        "cache_size": len(self._text_embedding_cache),
        "hit_rate_percent": (self._cache_hits / total * 100) if total else 0.0,
        "hits": self._cache_hits,
        "misses": self._cache_misses,
    }


VLMWrapper.compute_similarity_raw = _vlm_similarity_raw
VLMWrapper.compute_similarity = _vlm_similarity
VLMWrapper.get_cache_stats = _vlm_cache_stats
