"""Wrapper quản lý song song VieNeu v3 Turbo (48kHz) và VieNeu v2 GGUF Q8 (24kHz)."""

from __future__ import annotations

import io
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple

import numpy as np
import soundfile as sf
from vieneu import Vieneu

from .config import settings
from .normalizer import preprocess_text


def _to_pcm16_bytes(audio_f32: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(audio_f32, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def encode_wav_bytes(audio_f32: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    clipped = np.clip(np.asarray(audio_f32, dtype=np.float32), -1.0, 1.0)
    sf.write(buf, clipped, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _create_public_v2_tts() -> Any:
    """Khởi tạo VieNeu v2 GGUF Q8 sử dụng mirror công khai nguyensu27/neucodec."""
    import torch
    from huggingface_hub import hf_hub_download
    from neucodec import NeuCodec
    from vieneu.standard import VieNeuTTS

    class PublicV2VieNeuTTS(VieNeuTTS):
        def _load_codec(self, codec_repo: str, codec_device: str) -> None:
            ckpt = hf_hub_download(codec_repo, "pytorch_model.bin")
            state = torch.load(ckpt, map_location="cpu")
            ignore_keys = ("fc_post_s", "SemanticDecoder")
            state = {k: v for k, v in state.items() if not any(ig in k for ig in ignore_keys)}
            codec = NeuCodec(24_000, 480)
            codec.load_state_dict(state, strict=False)
            codec.eval().to(codec_device)
            self.codec = codec
            self._is_onnx_codec = False

        def _warmup_model(self) -> None:
            # GGUF VieNeu-TTS-q8-gguf dùng chat format; bỏ qua warmup mặc định trước khi set flag
            return

    tts2 = PublicV2VieNeuTTS(
        backbone_repo=settings.v2_backbone_repo,
        gguf_filename=settings.v2_gguf_filename,
        backbone_device="cpu",
        codec_repo=settings.v2_codec_repo,
        codec_device="cpu",
    )
    tts2.use_chat_format = True
    return tts2


class TTSEngine:
    """Quản lý v3 Turbo (nạp sẵn) và v2 GGUF Q8 (nạp lười khi dùng) kèm cache LRU."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._v2_load_lock = threading.Lock()
        self._cache: "OrderedDict[tuple, Tuple[bytes, float, float, int]]" = OrderedDict()
        self._cache_max = 64
        self.voice_revision = 0

        t0 = time.perf_counter()
        self.tts = Vieneu(
            mode=settings.vieneu_mode,
            backend=settings.vieneu_backend,
            precision=settings.vieneu_precision,
            threads=settings.vieneu_threads,
        )
        self.load_seconds = round(time.perf_counter() - t0, 2)
        self.sample_rate = int(getattr(self.tts, "sample_rate", settings.sample_rate))

        self.tts_v2: Optional[Any] = None
        self.v2_load_seconds: Optional[float] = None
        self.v2_sample_rate = settings.v2_sample_rate

    def ensure_v2(self) -> Any:
        if self.tts_v2 is not None:
            return self.tts_v2
        with self._v2_load_lock:
            if self.tts_v2 is None:
                t0 = time.perf_counter()
                self.tts_v2 = _create_public_v2_tts()
                self.v2_load_seconds = round(time.perf_counter() - t0, 2)
                self.v2_sample_rate = int(getattr(self.tts_v2, "sample_rate", settings.v2_sample_rate))
        return self.tts_v2

    def bump_voice_revision(self) -> None:
        self.voice_revision += 1

    def _resolve_voice_arg(self, voice: Optional[str]) -> Optional[str]:
        if not voice or voice == "default":
            return None
        return voice

    def _encode_v2_wav(self, wav_path: Path) -> np.ndarray:
        tts2 = self.ensure_v2()
        codes = tts2.encode_reference(str(wav_path))
        if hasattr(codes, "cpu"):
            codes = codes.cpu().numpy()
        return np.asarray(codes, dtype=np.int64)

    def synthesize_wav(
        self,
        text: str,
        *,
        model: str = "vieneu",
        voice: Optional[str] = None,
        speed: float = 1.0,
        temperature: float = 0.45,
        top_k: int = 15,
        top_p: float = 0.85,
        repetition_penalty: float = 1.2,
        use_ref_codes: bool = True,
        ref_text_override: Optional[str] = None,
        voice_manager: Any = None,
    ) -> Tuple[bytes, dict]:
        cleaned = preprocess_text(text)
        if not cleaned:
            raise ValueError("Văn bản đầu vào đang trống.")
        if not (0.5 <= speed <= 2.0):
            raise ValueError("Tham số speed phải nằm trong khoảng 0.5 đến 2.0.")

        engine_key = model.lower()
        if engine_key not in ("vieneu", "vieneu-v2"):
            raise ValueError(f"Model '{model}' không được hỗ trợ. Chọn 'vieneu' hoặc 'vieneu-v2'.")

        resolved_voice = self._resolve_voice_arg(voice)
        if (
            engine_key == "vieneu-v2"
            and resolved_voice
            and voice_manager is not None
            and voice_manager.is_custom_voice(resolved_voice)
            and not (ref_text_override and ref_text_override.strip())
        ):
            meta = voice_manager._custom_meta.get(resolved_voice, {})
            if not meta.get("has_ref_text"):
                raise ValueError(
                    f"Giọng '{resolved_voice}' chưa có lời thoại mẫu (ref_text). "
                    "Chế độ VieNeu v2 yêu cầu nhập đúng câu nói trong file audio mẫu."
                )
        cache_key = (
            engine_key,
            cleaned,
            resolved_voice or "__default__",
            (ref_text_override or "").strip(),
            round(speed, 2),
            settings.vieneu_precision if engine_key == "vieneu" else "gguf-q8",
            self.voice_revision,
            round(temperature, 2),
            top_k,
            round(top_p, 2),
            round(repetition_penalty, 2),
            bool(use_ref_codes),
        )

        with self._lock:
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                wav_bytes, audio_dur, elapsed, sr = self._cache[cache_key]
                return wav_bytes, {
                    "cached": True,
                    "audio_duration_sec": round(audio_dur, 3),
                    "inference_sec": 0.0,
                    "original_inference_sec": round(elapsed, 3),
                    "rtf": 0.0,
                    "sample_rate": sr,
                    "precision": settings.vieneu_precision if engine_key == "vieneu" else "gguf-q8",
                }

            t0 = time.perf_counter()
            if engine_key == "vieneu":
                sr = self.sample_rate
                audio = self.tts.infer(
                    cleaned,
                    voice=resolved_voice,
                    max_chars=settings.max_chars_per_chunk,
                    use_ref_codes=use_ref_codes,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
            else:
                tts2 = self.ensure_v2()
                sr = self.v2_sample_rate
                if resolved_voice and voice_manager is not None and voice_manager.is_custom_voice(resolved_voice):
                    ref_codes, ref_text = voice_manager.get_v2_reference(
                        resolved_voice,
                        self._encode_v2_wav,
                        ref_text_override=ref_text_override,
                    )
                    audio = tts2.infer(
                        cleaned,
                        ref_codes=ref_codes,
                        ref_text=ref_text,
                        max_chars=settings.max_chars_per_chunk,
                        temperature=temperature,
                        top_k=top_k,
                    )
                else:
                    v2_voice = tts2.get_preset_voice(resolved_voice)
                    audio = tts2.infer(
                        cleaned,
                        voice=v2_voice,
                        max_chars=settings.max_chars_per_chunk,
                        temperature=temperature,
                        top_k=top_k,
                    )

            if abs(speed - 1.0) > 0.02 and len(audio) > 0:
                import librosa

                audio = librosa.effects.time_stretch(
                    np.asarray(audio, dtype=np.float32), rate=float(speed)
                )

            elapsed = time.perf_counter() - t0
            audio_dur = float(len(audio)) / float(sr) if len(audio) else 0.0
            wav_bytes = encode_wav_bytes(audio, sr)

            self._cache[cache_key] = (wav_bytes, audio_dur, elapsed, sr)
            while len(self._cache) > self._cache_max:
                self._cache.popitem(last=False)

        rtf = elapsed / audio_dur if audio_dur > 0 else 0.0
        return wav_bytes, {
            "cached": False,
            "audio_duration_sec": round(audio_dur, 3),
            "inference_sec": round(elapsed, 3),
            "rtf": round(rtf, 3),
            "sample_rate": sr,
            "precision": settings.vieneu_precision if engine_key == "vieneu" else "gguf-q8",
        }

    def stream_pcm(
        self,
        text: str,
        *,
        model: str = "vieneu",
        voice: Optional[str] = None,
        temperature: float = 0.45,
        top_k: int = 15,
        top_p: float = 0.85,
        repetition_penalty: float = 1.2,
        use_ref_codes: bool = True,
        ref_text_override: Optional[str] = None,
        voice_manager: Any = None,
    ) -> Tuple[int, Generator[bytes, None, None]]:
        cleaned = preprocess_text(text)
        if not cleaned:
            raise ValueError("Văn bản đầu vào đang trống.")
        engine_key = model.lower()
        if engine_key not in ("vieneu", "vieneu-v2"):
            raise ValueError(f"Model '{model}' không được hỗ trợ.")
        resolved_voice = self._resolve_voice_arg(voice)
        if (
            engine_key == "vieneu-v2"
            and resolved_voice
            and voice_manager is not None
            and voice_manager.is_custom_voice(resolved_voice)
            and not (ref_text_override and ref_text_override.strip())
        ):
            meta = voice_manager._custom_meta.get(resolved_voice, {})
            if not meta.get("has_ref_text"):
                raise ValueError(
                    f"Giọng '{resolved_voice}' chưa có lời thoại mẫu (ref_text). "
                    "Chế độ VieNeu v2 yêu cầu nhập đúng câu nói trong file audio mẫu."
                )

        if engine_key == "vieneu":
            sr = self.sample_rate

            def _gen_v3() -> Generator[bytes, None, None]:
                with self._lock:
                    for chunk in self.tts.infer_stream(
                        cleaned,
                        voice=resolved_voice,
                        max_chars=settings.max_chars_per_chunk,
                        use_ref_codes=use_ref_codes,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                    ):
                        if chunk is not None and len(chunk) > 0:
                            yield _to_pcm16_bytes(chunk)

            return sr, _gen_v3()

        tts2 = self.ensure_v2()
        sr = self.v2_sample_rate

        def _gen_v2() -> Generator[bytes, None, None]:
            with self._lock:
                if resolved_voice and voice_manager is not None and voice_manager.is_custom_voice(resolved_voice):
                    ref_codes, ref_text = voice_manager.get_v2_reference(
                        resolved_voice,
                        self._encode_v2_wav,
                        ref_text_override=ref_text_override,
                    )
                    gen = tts2.infer_stream(
                        cleaned,
                        ref_codes=ref_codes,
                        ref_text=ref_text,
                        max_chars=settings.max_chars_per_chunk,
                        temperature=temperature,
                        top_k=top_k,
                    )
                else:
                    v2_voice = tts2.get_preset_voice(resolved_voice)
                    gen = tts2.infer_stream(
                        cleaned,
                        voice=v2_voice,
                        max_chars=settings.max_chars_per_chunk,
                        temperature=temperature,
                        top_k=top_k,
                    )
                for chunk in gen:
                    if chunk is not None and len(chunk) > 0:
                        yield _to_pcm16_bytes(chunk)

        return sr, _gen_v2()
