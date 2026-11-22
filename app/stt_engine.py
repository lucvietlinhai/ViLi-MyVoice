"""Module Speech-to-Text (STT) chuyên sâu tiếng Việt sử dụng VinAI PhoWhisper-base trên transformers."""

from __future__ import annotations

import gc
import logging
import threading
import time
from pathlib import Path
from typing import Optional

from .config import settings
from .log_stream import push_log

logger = logging.getLogger(__name__)


class STTEngine:
    """Singleton quản lý mô hình STT tiếng Việt (PhoWhisper-base trên CPU)."""

    _instance: Optional[STTEngine] = None
    _init_lock = threading.Lock()

    def __init__(self, model_id: Optional[str] = None) -> None:
        self.model_id = (model_id or settings.stt_model_id).strip()
        self._pipe = None
        self._infer_lock = threading.Lock()
        self._load_seconds: Optional[float] = None

    @classmethod
    def get_instance(cls) -> STTEngine:
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _ensure_loaded(self) -> None:
        if self._pipe is not None:
            return

        t0 = time.perf_counter()
        push_log(
            "INFO",
            "STT",
            f"Đang nạp mô hình nhận diện tiếng Việt '{self.model_id}' (cache tại {settings.hf_home_dir})...",
        )
        logger.info(f"Đang nạp mô hình STT {self.model_id} trên CPU...")
        from huggingface_hub import snapshot_download
        from transformers import pipeline

        # Ưu tiên lấy đường dẫn snapshot offline trong .hf_cache để không gửi request thừa lên mạng
        model_source = self.model_id
        try:
            model_source = snapshot_download(self.model_id, local_files_only=True)
        except Exception:
            model_source = self.model_id

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model_source,
            device="cpu",
        )
        self._load_seconds = round(time.perf_counter() - t0, 2)
        push_log("OK", "STT", f"Đã nạp xong '{self.model_id}' trong {self._load_seconds}s.")
        logger.info(f"Đã nạp xong STT {self.model_id} trong {self._load_seconds:.2f}s")

    def transcribe(self, audio_path: Path | str) -> str:
        """Nhận diện lời thoại tiếng Việt chính xác từ file audio (WAV, MP3, M4A...)."""
        with self._infer_lock:
            self._ensure_loaded()
            path_str = str(audio_path)
            try:
                result = self._pipe(
                    path_str,
                    generate_kwargs={"language": "vietnamese", "task": "transcribe"},
                )
            except Exception as exc:
                # PhoWhisper đã fine-tune sẵn tiếng Việt; fallback nếu phiên bản transformers từ chối forced kwargs
                logger.warning(f"STT fallback không truyền generate_kwargs ({exc})")
                result = self._pipe(path_str)
            text = (result.get("text", "") if isinstance(result, dict) else "").strip()
            return text

    def unload(self) -> bool:
        """Giải phóng RAM nếu không sử dụng STT nữa."""
        with self._infer_lock:
            if self._pipe is None:
                return False
            self._pipe = None
            gc.collect()
            push_log("INFO", "STT", f"Đã giải phóng bộ nhớ mô hình '{self.model_id}'.")
            return True
