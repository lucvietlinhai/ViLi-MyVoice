"""Quản lý từ điển phát âm tùy chỉnh (Lexicon) lưu bằng Rust orjson."""

from __future__ import annotations

import re
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import orjson

from .config import settings

STARTER_TEMPLATES: List[Dict[str, Any]] = [
    {"pattern": "AI", "replacement": "ây ai", "whole_word": True, "case_sensitive": True, "note": "Trí tuệ nhân tạo"},
    {"pattern": "UI", "replacement": "iu ai", "whole_word": True, "case_sensitive": True, "note": "Giao diện người dùng"},
    {"pattern": "API", "replacement": "ây pi ai", "whole_word": True, "case_sensitive": True, "note": "Cổng kết nối lập trình"},
    {"pattern": "TTS", "replacement": "ti ti ét", "whole_word": True, "case_sensitive": True, "note": "Chuyển văn bản thành giọng nói"},
    {"pattern": "KOC", "replacement": "cây ô xi", "whole_word": True, "case_sensitive": True, "note": "Người tiêu dùng có ảnh hưởng"},
    {"pattern": "TikTok", "replacement": "Tích Tóc", "whole_word": True, "case_sensitive": False, "note": "Mạng xã hội TikTok"},
    {"pattern": "YouTube", "replacement": "Diu Túp", "whole_word": True, "case_sensitive": False, "note": "Nền tảng video YouTube"},
    {"pattern": "VNĐ", "replacement": "Việt Nam đồng", "whole_word": False, "case_sensitive": False, "note": "Đơn vị tiền tệ"},
]


class LexiconManager:
    """Lưu trữ, quản lý và áp dụng quy tắc phát âm tiếng Việt/từ mượn."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._compiled_cache: List[tuple[re.Pattern, str]] = []
        self._rules: List[Dict[str, Any]] = []
        self._load_unlocked()

    def _load_unlocked(self) -> None:
        if not self.path.is_file():
            self._rules = []
            self._rebuild_compiled_unlocked()
            return
        try:
            data = orjson.loads(self.path.read_bytes())
            if isinstance(data, list):
                self._rules = data
        except Exception:
            self._rules = []
        self._rebuild_compiled_unlocked()

    def _save_unlocked(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_bytes(orjson.dumps(self._rules, option=orjson.OPT_INDENT_2))
        tmp.replace(self.path)

    def _rebuild_compiled_unlocked(self) -> None:
        compiled: List[tuple[re.Pattern, str]] = []
        # Ưu tiên cụm dài trước để tránh rule ngắn cắt ngang rule dài
        enabled_rules = [r for r in self._rules if r.get("enabled", True) and r.get("pattern")]
        enabled_rules.sort(key=lambda r: len(str(r.get("pattern", ""))), reverse=True)
        for r in enabled_rules:
            pat = str(r["pattern"])
            rep = str(r.get("replacement", ""))
            whole = bool(r.get("whole_word", True))
            case_sens = bool(r.get("case_sensitive", False))
            escaped = re.escape(pat)
            if whole:
                expr = rf"(?<!\w){escaped}(?!\w)"
            else:
                expr = escaped
            flags = 0 if case_sens else re.IGNORECASE
            try:
                compiled.append((re.compile(expr, flags), rep))
            except re.error:
                continue
        self._compiled_cache = compiled

    def list_rules(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._rules]

    def add_rule(
        self,
        *,
        pattern: str,
        replacement: str,
        whole_word: bool = True,
        case_sensitive: bool = False,
        enabled: bool = True,
        note: str = "",
    ) -> Dict[str, Any]:
        pat = (pattern or "").strip()
        rep = (replacement or "").strip()
        if not pat or not rep:
            raise ValueError("Từ gốc và cách đọc thay thế không được để trống.")
        with self._lock:
            rule = {
                "id": uuid.uuid4().hex[:10],
                "pattern": pat,
                "replacement": rep,
                "whole_word": bool(whole_word),
                "case_sensitive": bool(case_sensitive),
                "enabled": bool(enabled),
                "note": (note or "").strip(),
            }
            self._rules.insert(0, rule)
            self._save_unlocked()
            self._rebuild_compiled_unlocked()
            return dict(rule)

    def update_rule(self, rule_id: str, **fields: Any) -> Dict[str, Any]:
        with self._lock:
            for r in self._rules:
                if r.get("id") == rule_id:
                    for k in ("pattern", "replacement", "whole_word", "case_sensitive", "enabled", "note"):
                        if k in fields and fields[k] is not None:
                            val = fields[k]
                            if isinstance(val, str):
                                val = val.strip()
                            r[k] = val
                    if not r.get("pattern") or not r.get("replacement"):
                        raise ValueError("Từ gốc và cách đọc không được để trống.")
                    self._save_unlocked()
                    self._rebuild_compiled_unlocked()
                    return dict(r)
            raise KeyError(f"Không tìm thấy quy tắc phát âm '{rule_id}'.")

    def delete_rule(self, rule_id: str) -> bool:
        with self._lock:
            before = len(self._rules)
            self._rules = [r for r in self._rules if r.get("id") != rule_id]
            if len(self._rules) == before:
                return False
            self._save_unlocked()
            self._rebuild_compiled_unlocked()
            return True

    def install_starter_presets(self) -> List[Dict[str, Any]]:
        with self._lock:
            existing_patterns = {str(r.get("pattern", "")).lower() for r in self._rules}
            added = []
            for tpl in STARTER_TEMPLATES:
                if tpl["pattern"].lower() in existing_patterns:
                    continue
                rule = {
                    "id": uuid.uuid4().hex[:10],
                    "pattern": tpl["pattern"],
                    "replacement": tpl["replacement"],
                    "whole_word": tpl["whole_word"],
                    "case_sensitive": tpl["case_sensitive"],
                    "enabled": True,
                    "note": tpl["note"],
                }
                self._rules.append(rule)
                added.append(rule)
            if added:
                self._save_unlocked()
                self._rebuild_compiled_unlocked()
            return added

    def apply(self, text: str) -> str:
        if not text:
            return text
        with self._lock:
            compiled = list(self._compiled_cache)
        out = text
        for regex, rep in compiled:
            out = regex.sub(lambda _m, r=rep: r, out)
        return out


lexicon_manager = LexiconManager(settings.lexicon_path)
