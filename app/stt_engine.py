"""Module Speech-to-Text (STT) Tiếng Việt siêu nhẹ sử dụng Whisper Tiny.
Chạy hoàn toàn local, offline và bảo mật trên CPU.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class STTEngine:
    """Singleton quản lý mô hình STT Whisper Tiny."""

    _instance: Optional[STTEngine] = None

    def __init__(self, model_id: str = "openai/whisper-tiny") -> None:
        self.model_id = model_id
        self._pipe = None
        self._load_seconds: Optional[float] = None

    @classmethod
    def get_instance(cls) -> STTEngine:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_loaded(self) -> None:
        if self._pipe is not None:
            return

        t0 = time.time()
        logger.info(f"Đang nạp mô hình STT {self.model_id} trên CPU...")
        from transformers import pipeline

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            device="cpu",
        )
        self._load_seconds = time.time() - t0
        logger.info(f"Đã nạp xong STT {self.model_id} trong {self._load_seconds:.2f}s")

    def transcribe(self, audio_path: Path | str) -> str:
        """Nhận diện lời thoại tiếng Việt từ file audio (WAV, MP3, M4A...)."""
        self._ensure_loaded()
        path_str = str(audio_path)
        result = self._pipe(
            path_str,
            generate_kwargs={"language": "vietnamese", "task": "transcribe"},
        )
        text = result.get("text", "").strip()
        return text
