"""Tiền xử lý và chuẩn hoá văn bản tiếng Việt + Từ điển phát âm trước khi tổng hợp giọng nói."""

from __future__ import annotations

import re
from typing import List, Tuple

from vieneu_utils.phonemize_text import normalize_to_chunks_v3_with_gaps

from .lexicon_manager import lexicon_manager

_RE_TP_HCM = re.compile(r"\bTP\.?\s*H[ồo]\s*Ch[íi]\s*Minh\b", re.IGNORECASE)
_RE_TP_GENERIC = re.compile(r"\bTP\.\s*", re.IGNORECASE)
_RE_MULTI_SPACES = re.compile(r"[ \t]+")
_RE_MULTI_NEWLINES = re.compile(r"\n{3,}")


def preprocess_text(text: str, *, apply_lexicon: bool = True) -> str:
    """Làm sạch nhẹ văn bản đầu vào, áp dụng từ điển phát âm, giữ nguyên dấu câu và emotion tag."""
    if not text:
        return ""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    # Tránh lỗi sea-g2p để lại dấu chấm sau "TP." làm ngắt câu giữa "thành phố Hồ Chí Minh"
    cleaned = _RE_TP_HCM.sub("thành phố Hồ Chí Minh", cleaned)
    cleaned = _RE_TP_GENERIC.sub("thành phố ", cleaned)
    if apply_lexicon:
        cleaned = lexicon_manager.apply(cleaned)
    cleaned = _RE_MULTI_SPACES.sub(" ", cleaned)
    cleaned = _RE_MULTI_NEWLINES.sub("\n\n", cleaned)
    return cleaned.strip()


def split_and_normalize(text: str, max_chars: int = 220) -> Tuple[List[str], List[str]]:
    """Chuẩn hoá và tách văn bản dài thành các chunk an toàn kèm loại khoảng nghỉ."""
    cleaned = preprocess_text(text)
    if not cleaned:
        return [], []
    return normalize_to_chunks_v3_with_gaps(cleaned, max_chars=max_chars)
