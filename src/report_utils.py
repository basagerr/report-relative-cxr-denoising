"""Report extraction utilities from the original publication notebook."""

import re
from typing import List, Optional

REPORT_FALLBACK_TEXT = "chest radiograph"


_FINDINGS_PATTERNS = [
    r"FINDINGS[:\s]*(.+?)(?=\n\n|\nIMPRESSION|\nHISTORY|\nINDICATION|\Z)",
    r"FINDINGS[:\s]*(.+)",
]


_IMPRESSION_PATTERNS = [
    r"IMPRESSION[:\s]*(.+?)(?=\n\n|\nFINDINGS|\nHISTORY|\nINDICATION|\nTECHNIQUE|\Z)",
    r"IMPRESSION[:\s]*(.+)",
]


_METADATA_LINE_RE = re.compile(
    r"\b(HISTORY|TECHNIQUE|COMPARISON|INDICATION)[:\s]*[^\n]*\n?",
    re.I,
)


def _try_section(text: str, patterns: List[str], min_chars: int = 20) -> Optional[str]:
    for pattern in patterns:
        match = re.search(pattern, text, re.I | re.S)
        if match and len(match.group(1).strip()) > min_chars:
            return match.group(1).strip()
    return None


def extract_impression(text: str, max_words: int = 100) -> str:
    if not text or not isinstance(text, str) or len(text.strip()) < 5:
        return REPORT_FALLBACK_TEXT
    text = text.strip()
    for patterns in (_IMPRESSION_PATTERNS, _FINDINGS_PATTERNS):
        body = _try_section(text, patterns)
        if body is not None:
            return " ".join(body.split()[:max_words])
    cleaned = " ".join(_METADATA_LINE_RE.sub("", text).split())
    return " ".join(cleaned.split()[:max_words]) if len(cleaned) >= 10 else REPORT_FALLBACK_TEXT
