"""Tiền xử lý và chuẩn hoá văn bản tiếng Việt + Từ điển phát âm trước khi tổng hợp giọng nói."""

from __future__ import annotations

import re
from typing import List, Tuple

from vieneu_utils.core_utils import gaps_to_silence
from vieneu_utils.phonemize_text import normalize_to_chunks_v3_with_gaps

from .lexicon_manager import lexicon_manager

_RE_TP_HCM = re.compile(r"\bTP\.?\s*H[ồo]\s*Ch[íi]\s*Minh\b", re.IGNORECASE)
_RE_TP_GENERIC = re.compile(r"\bTP\.\s*", re.IGNORECASE)
_RE_MULTI_SPACES = re.compile(r"[ \t]+")
_RE_MULTI_NEWLINES = re.compile(r"\n{3,}")
_RE_PAUSE_TAG = re.compile(r"\[\s*ngh[ỉi]\s*([0-9]+(?:[.,][0-9]+)?)\s*s?\s*\]", re.IGNORECASE)
_RE_EMOTION_TAG_STRIP = re.compile(
    r"\[(?:cười|chuckle|thở\s*dài|sigh|hắng\s*giọng|clear\s*throat)\]|<\|emotion_\d+\|>",
    re.IGNORECASE,
)


def strip_emotion_cues(text: str) -> str:
    """Loại bỏ thẻ biểu cảm dành cho v3 khi tổng hợp bằng VieNeu v2 GGUF."""
    return _RE_MULTI_SPACES.sub(" ", _RE_EMOTION_TAG_STRIP.sub(" ", text)).strip()


def gaps_to_silence_durations(gaps: List[str]) -> List[float]:
    """Chuyển danh sách gap chuẩn + gap tùy chỉnh `[nghỉ X]` thành thời lượng im lặng (giây)."""
    out: List[float] = []
    for g in gaps:
        if isinstance(g, str) and g.startswith("pause:"):
            try:
                out.append(float(g.split(":", 1)[1]))
                continue
            except ValueError:
                pass
        out.extend(gaps_to_silence([g]))
    return out


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
    """Chuẩn hoá và tách văn bản dài thành các chunk an toàn, đồng thời giữ chính xác khoảng nghỉ `[nghỉ X]`."""
    cleaned = preprocess_text(text)
    if not cleaned:
        return [], []

    # Nếu không có thẻ [nghỉ X], đi thẳng nhánh chuẩn để giữ tốc độ tối ưu
    if not _RE_PAUSE_TAG.search(cleaned):
        return normalize_to_chunks_v3_with_gaps(cleaned, max_chars=max_chars)

    parts = _RE_PAUSE_TAG.split(cleaned)
    all_chunks: List[str] = []
    all_gaps: List[str] = []
    pending_pause: float | None = None
    carried_emotions: List[str] = []

    for idx in range(0, len(parts), 2):
        seg = parts[idx].strip()
        if seg:
            # Không bao giờ để thẻ biểu cảm đứng lẻ không có chữ trước [nghỉ X]
            if not _RE_EMOTION_TAG_STRIP.sub("", seg).strip():
                carried_emotions.append(seg)
                seg = ""
        if seg:
            if carried_emotions:
                seg = f"{' '.join(carried_emotions)} {seg}"
                carried_emotions.clear()
            seg_chunks, seg_gaps = normalize_to_chunks_v3_with_gaps(seg, max_chars=max_chars)
            if seg_chunks:
                if all_chunks:
                    if pending_pause is not None:
                        all_gaps.append(f"pause:{pending_pause:.2f}")
                    else:
                        all_gaps.append("sentence")
                all_chunks.extend(seg_chunks)
                all_gaps.extend(seg_gaps)
                pending_pause = None

        if idx + 1 < len(parts):
            try:
                sec = float(parts[idx + 1].replace(",", "."))
                sec = max(0.15, min(10.0, sec))
            except ValueError:
                sec = 0.6
            pending_pause = sec if pending_pause is None else min(10.0, pending_pause + sec)

    return all_chunks, all_gaps
