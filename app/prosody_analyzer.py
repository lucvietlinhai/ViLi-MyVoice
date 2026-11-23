"""Bộ phân tích ngữ điệu tiếng Việt tự động (1-click) chèn ngắt hơi, dấu phẩy và thẻ biểu cảm tự nhiên."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

# Các liên từ / cụm chuyển ý tiếng Việt thường lấy hơi nhẹ phía trước nếu mệnh đề trước đủ dài
_BREATH_PIVOTS = (
    "nhưng mà",
    "tuy nhiên",
    "thế nhưng",
    "tuy vậy",
    "vì vậy",
    "cho nên",
    "bởi vì",
    "tại vì",
    "nếu như",
    "mặc dù",
    "trong khi",
    "sau khi",
    "trước khi",
    "ngoài ra",
    "đặc biệt là",
    "thực ra",
    "thậm chí",
    "để rồi",
    "nhưng",
    "nên",
    "rồi",
)

_RE_OPENING_HOST = re.compile(
    r"^\s*(?:xin\s+chào|chào\s+mừng|chào\s+các\s+bạn|thưa\s+quý\s+vị|các\s+bạn\s+thân\s+mến|"
    r"mọi\s+người\s+ơi|hôm\s+nay|sau\s+đây|trước\s+hết|nào\s+chúng\s+ta)\b",
    re.IGNORECASE,
)

_RE_TRANSITION_OPENERS = re.compile(
    r"(^|[.!?…]\s+|(?:\[ngh[ỉi]\s+[0-9.]+\]\s*))"
    r"(Tuy nhiên|Thế nhưng|Cuối cùng|Thực ra|Thật vậy|Bất ngờ thay|May mắn thay|"
    r"Đáng tiếc là|Nói cách khác|Tóm lại|Chợt|Bỗng nhiên)(\s*,)",
    re.IGNORECASE,
)

_RE_LAUGH_TRIGGER = re.compile(
    r"\b(haha|hihi|hì\s*hì|buồn\s+cười|hài\s+hước|thú\s+vị\s+thật|tuyệt\s+vời|"
    r"dễ\s+thương|đáng\s+yêu|vui\s+quá|vui\s+thật|mừng\s+quá|bật\s+cười|tủm\s+tỉm|"
    r"khúc\s+khích|đùa\s+thôi|đùa\s+chút|chúc\s+mừng|thật\s+bất\s+ngờ|nghe\s+cũng\s+hay)\b",
    re.IGNORECASE,
)

_RE_SIGH_TRIGGER = re.compile(
    r"\b(thở\s+dài|mệt\s+mỏi|kiệt\s+sức|chán\s+nản|tiếc\s+nuối|đáng\s+tiếc|giá\s+mà|"
    r"ước\s+gì|thôi\s+đành|cũng\s+đành|cô\s+đơn|lặng\s+lẽ|trống\s+trải|buồn\s+bã|"
    r"khuya\s+rồi|đêm\s+đã\s+về\s+khuya|ngày\s+xưa|đã\s+qua\s+rồi|thật\s+lòng\s+mà\s+nói|"
    r"nghĩ\s+lại|thở\s+phào|cuối\s+cùng\s+cũng\s+xong)\b",
    re.IGNORECASE,
)

_RE_EXISTING_TAG = re.compile(r"\[[^\]]+\]")
_RE_SENTENCE_END = re.compile(r"([.!?…]+)(\s+|$)")
_RE_ELLIPSIS_NEXT = re.compile(r"(\.\.\.|…)(?!\s*\[ngh[ỉi])\s+(?=\S)")
_RE_MISSING_STARTER_COMMA = re.compile(
    r"(^|[.!?…]\s+)(Tuy nhiên|Thế nhưng|Cuối cùng|Thực ra|Thật vậy|Nói chung|Tóm lại)\s+(?=[a-zà-ỹ0-9])",
    re.IGNORECASE,
)


def _has_nearby_tag(text: str, pos: int, window: int = 32) -> bool:
    lo = max(0, pos - window)
    hi = min(len(text), pos + window)
    return bool(_RE_EXISTING_TAG.search(text[lo:hi]))


def _insert_breath_commas(paragraph: str) -> Tuple[str, int]:
    """Chèn dấu phẩy lấy hơi tự nhiên trước liên từ ở những câu văn dài thiếu dấu phẩy."""
    commas_added = 0
    # Tách theo câu để xử lý từng câu dài
    parts = re.split(r"([.!?…]+)", paragraph)
    rebuilt: List[str] = []
    for idx in range(0, len(parts), 2):
        seg = parts[idx]
        punct = parts[idx + 1] if idx + 1 < len(parts) else ""
        # Nếu câu ngắn hoặc đã có dấu ngắt vừa đủ thì giữ nguyên
        if len(seg.strip()) < 72:
            rebuilt.append(seg + punct)
            continue

        # Xét từng vế chưa có dấu phẩy dài > 70 ký tự
        sub_clauses = seg.split(",")
        new_clauses: List[str] = []
        for clause in sub_clauses:
            if len(clause.strip()) < 72 or "[" in clause:
                new_clauses.append(clause)
                continue
            words = clause.split()
            if len(words) < 14:
                new_clauses.append(clause)
                continue

            inserted = False
            lower_clause = clause.lower()
            # Tìm liên từ nằm ở khoảng giữa câu (tối thiểu 5 từ phía trước và 4 từ phía sau)
            for pivot in _BREATH_PIVOTS:
                for m in re.finditer(rf"(?<=\s){re.escape(pivot)}(?=\s|$)", lower_clause):
                    pos = m.start()
                    before = clause[:pos].rstrip()
                    after = clause[pos:].lstrip()
                    if (
                        len(before.split()) >= 5
                        and len(after.split()) >= 4
                        and not before.endswith((",", ";", ":", "-", "—", "–"))
                    ):
                        clause = f"{before}, {after}"
                        commas_added += 1
                        inserted = True
                        break
                if inserted:
                    break
            new_clauses.append(clause)

        rebuilt.append(",".join(new_clauses) + punct)
    return "".join(rebuilt), commas_added


def _annotate_paragraph_cadence(
    para: str,
    *,
    style: str,
    stats: Dict[str, object],
    last_emo_global: List[int],
    char_offset: int,
) -> str:
    """Chèn ngắt hơi [nghỉ ...] và biểu cảm [cười]/[thở dài] tự nhiên trong một đoạn văn."""
    # 1. Ngắt hơi sau dấu ba chấm (...) nếu phía sau còn câu tiếp
    def _ellipsis_sub(m: re.Match) -> str:
        stats["pauses_added"] = int(stats["pauses_added"]) + 1
        return f"{m.group(1)} [nghỉ 0.7] "

    out = _RE_ELLIPSIS_NEXT.sub(_ellipsis_sub, para)

    # 2. Chuyển ý sau trạng ngữ mở đầu ("Tuy nhiên, [nghỉ 0.4] ..." — tự thêm phẩy nếu thiếu)
    out = _RE_MISSING_STARTER_COMMA.sub(lambda m: f"{m.group(1)}{m.group(2)}, ", out)

    def _trans_sub(m: re.Match) -> str:
        prefix, word, comma = m.group(1), m.group(2), m.group(3)
        if not prefix.strip():
            return m.group(0)
        stats["pauses_added"] = int(stats["pauses_added"]) + 1
        return f"{prefix}{word}{comma} [nghỉ 0.4]"

    if not _has_nearby_tag(out, 0, 20):
        out = _RE_TRANSITION_OPENERS.sub(_trans_sub, out)

    # 3. Chia câu trong đoạn để chèn nhịp thở đều cho câu liên tiếp (tránh đọc liền tù tì)
    tokens = _RE_SENTENCE_END.split(out)
    # _RE_SENTENCE_END.split trả [seg0, punct0, space0, seg1, punct1, space1, ...]
    rebuilt: List[str] = []
    chars_since_pause = 0
    sentence_idx = 0
    min_pause_stride = 95 if style == "story" else 130
    emo_cooldown = 190 if style == "story" else 260
    emotion_tags_used: List[str] = stats["emotion_tags"]  # type: ignore[assignment]

    n_triples = (len(tokens) + 2) // 3
    for i in range(0, len(tokens), 3):
        seg = tokens[i]
        punct = tokens[i + 1] if i + 1 < len(tokens) else ""
        space = tokens[i + 2] if i + 2 < len(tokens) else ""
        if not seg and not punct:
            continue

        cur_global_pos = char_offset + sum(len(x) for x in rebuilt)
        has_next = any(tokens[k].strip() for k in range(i + 3, len(tokens), 3))

        # Kiểm tra chèn biểu cảm đúng ngữ cảnh (chặt cooldown để không lạm dụng)
        if (
            seg.strip()
            and "[" not in seg
            and (cur_global_pos - last_emo_global[0]) >= emo_cooldown
        ):
            m_sigh = _RE_SIGH_TRIGGER.search(seg)
            m_laugh = _RE_LAUGH_TRIGGER.search(seg)
            if m_sigh:
                # Đặt [thở dài] ngay trước mệnh đề hoặc đầu câu cảm xúc
                seg = f"[thở dài] {seg.lstrip()}" if not seg.startswith(" ") else f" [thở dài] {seg.lstrip()}"
                stats["emotions_added"] = int(stats["emotions_added"]) + 1
                emotion_tags_used.append("[thở dài]")
                last_emo_global[0] = cur_global_pos
            elif m_laugh:
                seg = f"[cười] {seg.lstrip()}" if not seg.startswith(" ") else f" [cười] {seg.lstrip()}"
                stats["emotions_added"] = int(stats["emotions_added"]) + 1
                emotion_tags_used.append("[cười]")
                last_emo_global[0] = cur_global_pos

        chunk_str = seg + punct
        chars_since_pause += len(seg)
        sentence_idx += 1

        if has_next and punct:
            already_has_pause = bool(re.search(r"\[ngh[ỉi]\s+[0-9.]+\]\s*$", seg)) or bool(
                i + 3 < len(tokens) and re.match(r"^\s*\[ngh[ỉi]\s+[0-9.]+\]", tokens[i + 3])
            )
            if not already_has_pause:
                if "?" in punct and chars_since_pause >= 45:
                    chunk_str += " [nghỉ 0.6]"
                    stats["pauses_added"] = int(stats["pauses_added"]) + 1
                    chars_since_pause = 0
                elif "!" in punct and chars_since_pause >= 55:
                    chunk_str += " [nghỉ 0.5]"
                    stats["pauses_added"] = int(stats["pauses_added"]) + 1
                    chars_since_pause = 0
                elif chars_since_pause >= min_pause_stride or (style == "story" and sentence_idx % 2 == 0 and n_triples >= 3):
                    dur = "0.6" if style == "story" else "0.5"
                    chunk_str += f" [nghỉ {dur}]"
                    stats["pauses_added"] = int(stats["pauses_added"]) + 1
                    chars_since_pause = 0

        if space:
            chunk_str += " " if has_next else ""
        rebuilt.append(chunk_str)

    return "".join(rebuilt).strip()


def analyze_and_annotate_prosody(text: str, style: str = "natural") -> dict:
    """Tự động phân tích văn bản tiếng Việt để chèn ngắt hơi, dấu phẩy và thẻ biểu cảm tự nhiên."""
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        raise ValueError("Vui lòng nhập nội dung văn bản tiếng Việt trước khi phân tích.")

    norm_style = (style or "natural").strip().lower()
    if norm_style not in ("natural", "story", "podcast"):
        norm_style = "natural"

    stats: Dict[str, object] = {
        "pauses_added": 0,
        "commas_added": 0,
        "emotions_added": 0,
        "emotion_tags": [],
        "changed": False,
    }

    # Chuẩn hoá ngắt đoạn giữ cấu trúc đoạn văn người dùng
    raw_paras = [p.strip() for p in re.split(r"\n+", raw) if p.strip()]
    processed_paras: List[str] = []
    last_emo_global = [-999]
    running_offset = 0

    for idx, p in enumerate(raw_paras):
        # Bước 1: Chèn dấu phẩy lấy hơi ở các vế quá dài chưa ngắt phẩy
        with_commas, c_added = _insert_breath_commas(p)
        stats["commas_added"] = int(stats["commas_added"]) + c_added

        # Bước 2: Mở bài kiểu MC / giới thiệu -> hắng giọng nhẹ gắn liền vào câu chào (không tách riêng thành chunk rỗng)
        if (
            idx == 0
            and len(raw) >= 110
            and "[hắng giọng]" not in with_commas.lower()
            and not _has_nearby_tag(with_commas, 0, 25)
            and _RE_OPENING_HOST.search(with_commas)
        ):
            with_commas = f"[hắng giọng] {with_commas}"
            stats["emotions_added"] = int(stats["emotions_added"]) + 1
            stats["emotion_tags"].append("[hắng giọng]")  # type: ignore[union-attr]
            last_emo_global[0] = 0

        # Bước 3: Phân tích ngắt hơi câu & biểu cảm từng đoạn
        annotated = _annotate_paragraph_cadence(
            with_commas,
            style=norm_style,
            stats=stats,
            last_emo_global=last_emo_global,
            char_offset=running_offset,
        )
        processed_paras.append(annotated)
        running_offset += len(annotated)

    # Bước 4: Ngắt hơi chuyển đoạn rõ ràng giữa các đoạn văn
    if len(processed_paras) > 1:
        joined_parts: List[str] = []
        for idx, para in enumerate(processed_paras):
            if idx > 0:
                prev = joined_parts[-1]
                has_pause_end = bool(re.search(r"\[ngh[ỉi]\s+[0-9.]+\]\s*$", prev))
                has_pause_start = bool(re.match(r"^\s*\[ngh[ỉi]\s+[0-9.]+\]", para))
                if not has_pause_end and not has_pause_start:
                    gap_sec = "0.9" if len(prev) > 35 else "0.6"
                    joined_parts[-1] = f"{prev} [nghỉ {gap_sec}]"
                    stats["pauses_added"] = int(stats["pauses_added"]) + 1
            joined_parts.append(para)
        final_text = "\n\n".join(joined_parts)
    else:
        final_text = processed_paras[0]

    # Nếu văn bản chỉ có 1 đoạn hơi dài (~120+ ký tự) mà chưa thêm được gì, chèn 1 nhịp nghỉ giữa hợp lý
    if (
        int(stats["pauses_added"]) == 0
        and int(stats["commas_added"]) == 0
        and int(stats["emotions_added"]) == 0
        and len(final_text) >= 90
    ):
        mid_m = re.search(r"([.!?…])(\s+)(?=\S)", final_text[max(25, len(final_text) // 4):])
        if mid_m:
            cut_pos = max(25, len(final_text) // 4) + mid_m.start()
            final_text = f"{final_text[:cut_pos + 1]} [nghỉ 0.5] {final_text[cut_pos + 1:].lstrip()}"
            stats["pauses_added"] = 1

    stats["changed"] = final_text.strip() != raw.strip()
    summary_parts = []
    if stats["pauses_added"]:
        summary_parts.append(f"{stats['pauses_added']} ngắt hơi")
    if stats["commas_added"]:
        summary_parts.append(f"{stats['commas_added']} dấu phẩy lấy hơi")
    if stats["emotions_added"]:
        uniq = list(dict.fromkeys(stats["emotion_tags"]))  # type: ignore[arg-type]
        summary_parts.append(f"{stats['emotions_added']} biểu cảm ({', '.join(uniq)})")

    summary = (
        f"Đã tối ưu ngữ điệu: {', '.join(summary_parts)}"
        if summary_parts
        else "Văn bản đã có nhịp ngắt nghỉ cân đối sẵn."
    )
    return {
        "annotated_text": final_text,
        "stats": stats,
        "summary": summary,
    }
