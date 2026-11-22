"""Cấu hình ứng dụng TTS Clone Voice tiếng Việt (song song v3 Turbo & v2 GGUF Q8)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(env_path: Path) -> None:
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def _resolve_dir(raw_path: str, default_name: str) -> Path:
    val = (raw_path or default_name).strip()
    p = Path(val)
    return p if p.is_absolute() else (BASE_DIR / p)


_load_dotenv(BASE_DIR / ".env")

# Ép toàn bộ bộ nhớ đệm HuggingFace & PyTorch nằm trong dự án (không ghi vào ổ C:)
_HF_HOME_DIR = _resolve_dir(os.getenv("HF_HOME", ".hf_cache"), ".hf_cache")
_HF_HOME_DIR.mkdir(parents=True, exist_ok=True)
_TORCH_HOME_DIR = _resolve_dir(os.getenv("TORCH_HOME", str(_HF_HOME_DIR / "torch")), ".hf_cache/torch")
_TORCH_HOME_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(_HF_HOME_DIR)
os.environ["HF_HUB_CACHE"] = str(_HF_HOME_DIR / "hub")
os.environ["TORCH_HOME"] = str(_TORCH_HOME_DIR)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


@dataclass(frozen=True)
class Settings:
    # Engine 1: VieNeu v3 Turbo (48kHz, ONNX CPU)
    vieneu_mode: str = os.getenv("VIENEU_MODE", "v3turbo")
    vieneu_precision: str = os.getenv("VIENEU_PRECISION", "fp32")
    vieneu_backend: str = os.getenv("VIENEU_BACKEND", "onnx")
    vieneu_threads: int = int(os.getenv("VIENEU_THREADS", "4"))
    sample_rate: int = int(os.getenv("SAMPLE_RATE", "48000"))

    # Engine 2: VieNeu v2 GGUF Q8 (24kHz, bám giọng vùng miền qua ref_text)
    v2_backbone_repo: str = os.getenv("V2_BACKBONE_REPO", "pnnbao-ump/VieNeu-TTS-q8-gguf")
    v2_gguf_filename: str = os.getenv("V2_GGUF_FILENAME", "VieNeu-TTS-q8_0.gguf")
    v2_codec_repo: str = os.getenv("V2_CODEC_REPO", "nguyensu27/neucodec")
    v2_sample_rate: int = int(os.getenv("V2_SAMPLE_RATE", "24000"))

    # STT tiếng Việt (nhận diện ref_text từ file âm thanh mẫu)
    stt_model_id: str = os.getenv("STT_MODEL_ID", "vinai/PhoWhisper-base")

    max_chars_per_chunk: int = int(os.getenv("MAX_CHARS_PER_CHUNK", "220"))
    hf_home_dir: Path = _HF_HOME_DIR
    voices_dir: Path = _resolve_dir(os.getenv("VOICES_DIR", "voices"), "voices")
    outputs_dir: Path = _resolve_dir(os.getenv("OUTPUTS_DIR", "outputs"), "outputs")
    history_dir: Path = _resolve_dir(os.getenv("HISTORY_DIR", "history"), "history")
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "8000"))


settings = Settings()
settings.voices_dir.mkdir(parents=True, exist_ok=True)
settings.outputs_dir.mkdir(parents=True, exist_ok=True)
settings.history_dir.mkdir(parents=True, exist_ok=True)
