"""Quản lý lưu trữ lâu dài các file audio kết quả (WAV/MP3/M4A) trong thư mục history/ dùng Rust orjson."""

from __future__ import annotations

import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

import orjson

from .audio_postprocess import FORMAT_META

MAX_HISTORY_ITEMS = 250
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{4,80}$")


class HistoryManager:
    """Lưu file audio kết quả và metadata index.json vào thư mục history/."""

    def __init__(self, history_dir: Path) -> None:
        self.history_dir = history_dir
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.history_dir / "index.json"
        self._lock = threading.Lock()

    def _read_index_unlocked(self) -> list[dict[str, Any]]:
        if not self.index_path.is_file():
            return []
        try:
            data = orjson.loads(self.index_path.read_bytes())
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return []

    def _write_index_unlocked(self, items: list[dict[str, Any]]) -> None:
        tmp_path = self.index_path.with_suffix(".json.tmp")
        tmp_path.write_bytes(orjson.dumps(items, option=orjson.OPT_INDENT_2))
        tmp_path.replace(self.index_path)

    def save_item(
        self,
        audio_bytes: bytes,
        *,
        text: str,
        voice: str,
        model: str,
        duration_sec: float,
        sample_rate: int,
        inference_sec: float = 0.0,
        rtf: float = 0.0,
        speed: float = 1.0,
        temperature: float = 0.45,
        fmt: str = "wav",
        fx_summary: str = "",
        label: str = "",
    ) -> dict[str, Any]:
        now = datetime.now()
        short_uid = uuid.uuid4().hex[:6]
        safe_voice = re.sub(r"[^a-zA-Z0-9_-]+", "_", (voice or "default").strip())[:24].strip("_") or "voice"
        _, ext = FORMAT_META.get((fmt or "wav").lower(), ("audio/wav", "wav"))
        item_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{safe_voice}_{short_uid}"
        filename = f"{item_id}.{ext}"
        file_path = self.history_dir / filename

        with self._lock:
            file_path.write_bytes(audio_bytes)
            items = self._read_index_unlocked()

            entry: dict[str, Any] = {
                "id": item_id,
                "filename": filename,
                "format": ext,
                "label": (label or "").strip(),
                "fx_summary": fx_summary,
                "audio_url": f"/v1/history/{item_id}/audio",
                "text": text,
                "voice": voice or "default",
                "engine": model or "vieneu",
                "duration": round(float(duration_sec or 0.0), 2),
                "sample_rate": int(sample_rate or 48000),
                "inference_sec": round(float(inference_sec or 0.0), 2),
                "rtf": round(float(rtf or 0.0), 3),
                "speed": round(float(speed or 1.0), 2),
                "temperature": round(float(temperature or 0.45), 2),
                "size_bytes": len(audio_bytes),
                "created_at": now.isoformat(timespec="seconds"),
                "timestamp": now.strftime("%H:%M"),
                "dateStr": now.strftime("%d/%m"),
            }
            items.insert(0, entry)

            if len(items) > MAX_HISTORY_ITEMS:
                overflow = items[MAX_HISTORY_ITEMS:]
                items = items[:MAX_HISTORY_ITEMS]
                for old in overflow:
                    old_file = old.get("filename")
                    if old_file:
                        (self.history_dir / old_file).unlink(missing_ok=True)

            self._write_index_unlocked(items)
            return entry

    def list_items(self) -> list[dict[str, Any]]:
        with self._lock:
            items = self._read_index_unlocked()
            valid_items: list[dict[str, Any]] = []
            changed = False
            for item in items:
                filename = item.get("filename")
                item_id = item.get("id")
                if not filename or not item_id:
                    changed = True
                    continue
                if not (self.history_dir / filename).is_file():
                    changed = True
                    continue
                item.setdefault("audio_url", f"/v1/history/{item_id}/audio")
                item.setdefault("format", Path(filename).suffix.lstrip(".") or "wav")
                item.setdefault("label", "")
                item.setdefault("fx_summary", "")
                valid_items.append(item)

            if changed:
                self._write_index_unlocked(valid_items)
            return valid_items

    def update_label(self, item_id: str, label: str) -> dict[str, Any]:
        if not item_id or not _SAFE_ID_RE.match(item_id):
            raise KeyError("ID bản ghi không hợp lệ.")
        with self._lock:
            items = self._read_index_unlocked()
            for item in items:
                if item.get("id") == item_id:
                    item["label"] = (label or "").strip()[:120]
                    self._write_index_unlocked(items)
                    return dict(item)
            raise KeyError(f"Không tìm thấy bản ghi '{item_id}'.")

    def get_audio_info(self, item_id: str) -> Optional[Tuple[Path, str]]:
        if not item_id or not _SAFE_ID_RE.match(item_id):
            return None
        for ext, (mime, _) in [("wav", ("audio/wav", "wav")), ("mp3", ("audio/mpeg", "mp3")), ("m4a", ("audio/mp4", "m4a"))]:
            candidate = self.history_dir / f"{item_id}.{ext}"
            if candidate.is_file():
                return candidate, mime
        return None

    def get_audio_path(self, item_id: str) -> Optional[Path]:
        info = self.get_audio_info(item_id)
        return info[0] if info else None

    def delete_item(self, item_id: str) -> bool:
        if not item_id or not _SAFE_ID_RE.match(item_id):
            return False
        with self._lock:
            items = self._read_index_unlocked()
            kept: list[dict[str, Any]] = []
            removed = False
            for item in items:
                if item.get("id") == item_id:
                    removed = True
                    filename = item.get("filename") or f"{item_id}.wav"
                    (self.history_dir / filename).unlink(missing_ok=True)
                else:
                    kept.append(item)
            for ext in ("wav", "mp3", "m4a"):
                p = self.history_dir / f"{item_id}.{ext}"
                if p.is_file():
                    p.unlink(missing_ok=True)
                    removed = True
            if removed:
                self._write_index_unlocked(kept)
            return removed

    def clear_all(self) -> int:
        with self._lock:
            items = self._read_index_unlocked()
            count = len(items)
            for ext in ("*.wav", "*.mp3", "*.m4a"):
                for audio_file in self.history_dir.glob(ext):
                    audio_file.unlink(missing_ok=True)
            self._write_index_unlocked([])
            return count
