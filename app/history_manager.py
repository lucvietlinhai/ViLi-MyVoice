"""Quản lý lưu trữ lâu dài các file audio kết quả trong thư mục history/."""

from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

MAX_HISTORY_ITEMS = 200
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{4,80}$")


class HistoryManager:
    """Lưu file WAV kết quả và metadata index.json vào thư mục history/."""

    def __init__(self, history_dir: Path) -> None:
        self.history_dir = history_dir
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.history_dir / "index.json"
        self._lock = threading.Lock()

    def _read_index_unlocked(self) -> list[dict[str, Any]]:
        if not self.index_path.is_file():
            return []
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return []

    def _write_index_unlocked(self, items: list[dict[str, Any]]) -> None:
        tmp_path = self.index_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.index_path)

    def save_item(
        self,
        wav_bytes: bytes,
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
    ) -> dict[str, Any]:
        now = datetime.now()
        short_uid = uuid.uuid4().hex[:6]
        safe_voice = re.sub(r"[^a-zA-Z0-9_-]+", "_", (voice or "default").strip())[:24].strip("_") or "voice"
        item_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{safe_voice}_{short_uid}"
        filename = f"{item_id}.wav"
        wav_path = self.history_dir / filename

        with self._lock:
            wav_path.write_bytes(wav_bytes)
            items = self._read_index_unlocked()

            entry: dict[str, Any] = {
                "id": item_id,
                "filename": filename,
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
                "size_bytes": len(wav_bytes),
                "created_at": now.isoformat(timespec="seconds"),
                "timestamp": now.strftime("%H:%M"),
                "dateStr": now.strftime("%d/%m"),
            }
            items.insert(0, entry)

            # Giữ tối đa MAX_HISTORY_ITEMS bản ghi, tự dọn file cũ vượt ngưỡng
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
                valid_items.append(item)

            if changed:
                self._write_index_unlocked(valid_items)
            return valid_items

    def get_audio_path(self, item_id: str) -> Optional[Path]:
        if not item_id or not _SAFE_ID_RE.match(item_id):
            return None
        candidate = self.history_dir / f"{item_id}.wav"
        if candidate.is_file():
            return candidate
        return None

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
            if not removed:
                # Xóa file lẻ nếu còn tồn tại
                wav_path = self.history_dir / f"{item_id}.wav"
                if wav_path.is_file():
                    wav_path.unlink(missing_ok=True)
                    removed = True
            else:
                self._write_index_unlocked(kept)
            return removed

    def clear_all(self) -> int:
        with self._lock:
            items = self._read_index_unlocked()
            count = len(items)
            for wav_file in self.history_dir.glob("*.wav"):
                wav_file.unlink(missing_ok=True)
            self._write_index_unlocked([])
            return count
