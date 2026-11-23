"""Wrapper quản lý song song VieNeu v3 Turbo (48kHz) và VieNeu v2 GGUF Q8 (24kHz)."""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple

import numpy as np
import soundfile as sf
from vieneu import Vieneu

from .audio_postprocess import FORMAT_META, apply_studio_fx, encode_audio_bytes
from .config import settings
from .log_stream import push_log
from .normalizer import gaps_to_silence_durations, preprocess_text, split_and_normalize, strip_emotion_cues
from .progress_store import ProgressCallback


def _to_pcm16_bytes(audio_f32: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(audio_f32, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def encode_wav_bytes(audio_f32: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    clipped = np.clip(np.asarray(audio_f32, dtype=np.float32), -1.0, 1.0)
    sf.write(buf, clipped, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _time_stretch_ffmpeg(audio_f32: np.ndarray, sample_rate: int, speed: float) -> np.ndarray:
    """Chỉnh tốc độ đọc bằng FFmpeg atempo để giữ cao độ và thanh điệu tiếng Việt tự nhiên."""
    arr = np.asarray(audio_f32, dtype=np.float32).reshape(-1)
    if len(arr) == 0 or abs(speed - 1.0) <= 0.02:
        return arr

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        # Chuỗi atempo chuẩn [0.5..2.0] an toàn với mọi phiên bản FFmpeg
        factors = []
        rem = float(speed)
        while rem > 2.0:
            factors.append("atempo=2.0")
            rem /= 2.0
        while rem < 0.5:
            factors.append("atempo=0.5")
            rem /= 0.5
        factors.append(f"atempo={rem:.4f}")
        filter_str = ",".join(factors)

        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "f32le",
            "-ar",
            str(int(sample_rate)),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-filter:a",
            filter_str,
            "-f",
            "f32le",
            "pipe:1",
        ]
        try:
            proc = subprocess.run(cmd, input=arr.tobytes(), capture_output=True, check=False)
            if proc.returncode == 0 and len(proc.stdout) >= 4:
                out = np.frombuffer(proc.stdout, dtype=np.float32).copy()
                if len(out) > 0:
                    return out
        except Exception:
            pass

    import librosa

    return np.asarray(librosa.effects.time_stretch(arr, rate=float(speed)), dtype=np.float32)


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

        def _build_ggml_prompt(self, ref_codes: Any, ref_phonemes: str, chunk_phonemes: str, emotion_tag: Optional[str] = None) -> tuple[str, list]:
            ref_codes_list = self.to_list(ref_codes)
            codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes_list])
            # Chat-format GGUF không dùng <|emotion_0|>; tránh chèn token lạ gây lệch alignment
            emotion_prefix = "" if self.use_chat_format else (emotion_tag or "")
            if self.use_chat_format:
                prompt = (
                    f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{emotion_prefix}{ref_phonemes} {chunk_phonemes}"
                    f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
                )
            else:
                prompt = (
                    f"<|TEXT_PROMPT_START|>{emotion_prefix}{ref_phonemes} {chunk_phonemes}"
                    f"<|TEXT_PROMPT_END|><|SPEECH_GENERATION_START|>{codes_str}"
                )
            return prompt, ref_codes_list

        def _infer_ggml(self, ref_codes: Any, ref_phonemes: str, chunk_phonemes: str, temperature: float = 0.85, top_k: int = 50, emotion_tag: Optional[str] = None) -> str:
            prompt, _ = self._build_ggml_prompt(ref_codes, ref_phonemes, chunk_phonemes, emotion_tag=emotion_tag)
            rep_pen = float(getattr(self, "_repeat_penalty", 1.15))
            top_p = float(getattr(self, "_top_p", 0.90))
            max_gen_tokens = min(self.max_context - 512, max(260, int(len(chunk_phonemes) * 7.5)))
            output = self.backbone(
                prompt,
                max_tokens=max_gen_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repeat_penalty=rep_pen,
                stop=["<|SPEECH_GENERATION_END|>"],
            )
            return output["choices"][0]["text"]

    tts2 = PublicV2VieNeuTTS(
        backbone_repo=settings.v2_backbone_repo,
        gguf_filename=settings.v2_gguf_filename,
        backbone_device="cpu",
        codec_repo=settings.v2_codec_repo,
        codec_device="cpu",
    )
    tts2.use_chat_format = True
    tts2.default_emotion = None
    tts2._repeat_penalty = 1.15
    tts2._top_p = 0.90
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

    def has_v2_local_cache(self) -> bool:
        """Kiểm tra nhanh model VieNeu v2 (~2.3GB w2v-bert-2.0 + GGUF + codec) đã nằm sẵn trong .hf_cache chưa."""
        if self.tts_v2 is not None:
            return True
        try:
            from huggingface_hub import hf_hub_download

            hf_hub_download(settings.v2_backbone_repo, settings.v2_gguf_filename, local_files_only=True)
            hf_hub_download(settings.v2_codec_repo, "pytorch_model.bin", local_files_only=True)
            try:
                hf_hub_download("facebook/w2v-bert-2.0", "model.safetensors", local_files_only=True)
            except Exception:
                hf_hub_download("facebook/w2v-bert-2.0", "pytorch_model.bin", local_files_only=True)
            return True
        except Exception:
            return False

    def bump_voice_revision(self) -> None:
        self.voice_revision += 1

    def _resolve_v2_params(
        self,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
    ) -> Tuple[float, int, float, float, int]:
        """Chuẩn hóa tham số sampling phù hợp riêng cho GGUF v2 để tránh vòng lặp rè/è."""
        clamped_t = max(0.15, min(1.20, float(temperature)))
        v2_temp = round(0.75 + (clamped_t - 0.15) * (0.30 / 1.05), 2)
        v2_top_k = max(40, int(top_k))
        v2_top_p = round(max(0.85, min(0.95, float(top_p))), 2)
        v2_rep = round(max(1.08, min(1.25, float(repetition_penalty))), 2)
        v2_max_chars = min(settings.max_chars_per_chunk, 150)
        return v2_temp, v2_top_k, v2_top_p, v2_rep, v2_max_chars

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
        response_format: str = "wav",
        normalize_lufs: bool = False,
        voice_enhance: bool = False,
        bgm_path: Optional[Path] = None,
        bgm_volume: float = 0.14,
        progress_cb: Optional[Any] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Tuple[bytes, dict]:
        cleaned = preprocess_text(text)
        if not cleaned:
            raise ValueError("Văn bản đầu vào đang trống.")
        if not (0.5 <= speed <= 2.0):
            raise ValueError("Tham số speed phải nằm trong khoảng 0.5 đến 2.0.")

        fmt_key = (response_format or "wav").lower().strip()
        if fmt_key not in FORMAT_META:
            raise ValueError(f"Định dạng '{response_format}' không được hỗ trợ.")

        engine_key = model.lower()
        if engine_key not in ("vieneu", "vieneu-v2"):
            raise ValueError(f"Model '{model}' không được hỗ trợ. Chọn 'vieneu' hoặc 'vieneu-v2'.")

        resolved_voice = self._resolve_voice_arg(voice)
        if (
            engine_key == "vieneu-v2"
            and resolved_voice
            and voice_manager is not None
            and voice_manager.is_custom_voice(resolved_voice)
        ):
            meta = voice_manager._custom_meta.get(resolved_voice, {})
            if meta.get("created_model") == "v3" or not meta.get("v2_cached"):
                raise ValueError(
                    f"Giọng '{resolved_voice}' được tạo từ mô hình VieNeu v3 nên không thể tạo âm thanh với VieNeu v2. "
                    "Vui lòng chuyển sang mô hình VieNeu v3 Turbo hoặc chọn giọng clone được tạo từ VieNeu v2."
                )
            if not (ref_text_override and ref_text_override.strip()) and not meta.get("has_ref_text"):
                raise ValueError(
                    f"Giọng '{resolved_voice}' chưa có lời thoại mẫu (ref_text). "
                    "Chế độ VieNeu v2 yêu cầu nhập đúng câu nói trong file audio mẫu."
                )

        bgm_sig = f"{bgm_path.name}:{round(float(bgm_volume), 2)}" if bgm_path else ""
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
            fmt_key,
            bool(normalize_lufs),
            bool(voice_enhance),
            bgm_sig,
        )

        with self._lock:
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                out_bytes, audio_dur, elapsed, sr = self._cache[cache_key]
                if progress_cb:
                    progress_cb(98, "Trích xuất tức thì từ bộ nhớ đệm (Cache HIT)...", "", 0.2)
                return out_bytes, {
                    "cached": True,
                    "audio_duration_sec": round(audio_dur, 3),
                    "inference_sec": 0.0,
                    "original_inference_sec": round(elapsed, 3),
                    "rtf": 0.0,
                    "sample_rate": sr,
                    "precision": settings.vieneu_precision if engine_key == "vieneu" else "gguf-q8",
                    "format": FORMAT_META[fmt_key][1],
                    "mime": FORMAT_META[fmt_key][0],
                }

            from vieneu_utils.core_utils import join_audio_chunks

            max_c = min(settings.max_chars_per_chunk, 150) if engine_key == "vieneu-v2" else settings.max_chars_per_chunk
            chunks, gaps = split_and_normalize(cleaned, max_chars=max_c)
            if not chunks:
                chunks = [cleaned]
                gaps = []
            if engine_key == "vieneu-v2":
                chunks = [strip_emotion_cues(c) or c for c in chunks]
            chunk_count = len(chunks)
            if engine_key == "vieneu":
                est_sec = max(2.2, len(cleaned) * 0.055)
            else:
                v2_boot = 22.0 if self.tts_v2 is None else 0.0
                est_sec = v2_boot + max(10.0, len(cleaned) * 0.65)

            if progress_cb:
                progress_cb(
                    10,
                    f"Đã tách {chunk_count} đoạn văn bản · Đang chuẩn bị tổng hợp...",
                    f"{len(cleaned)} ký tự",
                    est_sec,
                )

            t0 = time.perf_counter()
            if engine_key == "vieneu":
                sr = self.sample_rate
                speaker_emb, ref_codes = self.tts._resolve_ref(
                    resolved_voice, None, True, use_ref_codes
                )
                sampling = dict(
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    max_new_frames=300,
                    repetition_penalty=repetition_penalty,
                )
                chunk_wavs = []
                for ci, ch_text in enumerate(chunks, start=1):
                    if cancel_event and cancel_event.is_set():
                        raise RuntimeError("Đã hủy tác vụ theo yêu cầu.")
                    pct = 14 + int(((ci - 1) / chunk_count) * 72)
                    if progress_cb:
                        progress_cb(
                            pct,
                            f"Đang tổng hợp đoạn {ci}/{chunk_count} (VieNeu v3 Turbo)...",
                            ch_text[:45],
                            est_sec,
                        )
                    w = self.tts._infer_chunks(
                        [ch_text], speaker_emb, ref_codes, use_ref_codes, 1, sampling
                    )[0]
                    chunk_wavs.append(w)

                if cancel_event and cancel_event.is_set():
                    raise RuntimeError("Đã hủy tác vụ theo yêu cầu.")
                final_wav = join_audio_chunks(
                    chunk_wavs, sr, silence_ps=gaps_to_silence_durations(gaps)
                )
                audio = (
                    self.tts._apply_watermark(final_wav)
                    if hasattr(self.tts, "_apply_watermark")
                    else final_wav
                )
            else:
                if self.tts_v2 is None and progress_cb:
                    progress_cb(
                        14,
                        "Đang nạp mô hình VieNeu v2 GGUF Q8 & NeuCodec lần đầu...",
                        "Quá trình nạp lần đầu mất ~15–25s",
                        est_sec,
                    )
                tts2 = self.ensure_v2()
                sr = self.v2_sample_rate
                v2_temp, v2_top_k, v2_top_p, v2_rep, v2_max_chars = self._resolve_v2_params(
                    temperature, top_k, top_p, repetition_penalty
                )
                tts2._repeat_penalty = v2_rep
                tts2._top_p = v2_top_p
                push_log(
                    "INFO",
                    "TTS-V2",
                    f"Áp dụng tham số tối ưu GGUF v2: temp={v2_temp}, top_k={v2_top_k}, top_p={v2_top_p}, repeat_penalty={v2_rep}, max_chars={v2_max_chars}.",
                )

                is_custom_v2 = bool(
                    resolved_voice and voice_manager is not None and voice_manager.is_custom_voice(resolved_voice)
                )
                if is_custom_v2:
                    ref_codes, ref_text = voice_manager.get_v2_reference(
                        resolved_voice,
                        self._encode_v2_wav,
                        ref_text_override=ref_text_override,
                    )
                else:
                    v2_voice = tts2.get_preset_voice(resolved_voice)

                chunk_wavs = []
                for ci, ch_text in enumerate(chunks, start=1):
                    if cancel_event and cancel_event.is_set():
                        raise RuntimeError("Đã hủy tác vụ theo yêu cầu.")
                    pct = 20 + int(((ci - 1) / chunk_count) * 66)
                    if progress_cb:
                        progress_cb(
                            pct,
                            f"Đang tổng hợp đoạn {ci}/{chunk_count} (VieNeu v2 GGUF)...",
                            ch_text[:45],
                            est_sec,
                        )
                    if is_custom_v2:
                        w = tts2.infer(
                            ch_text,
                            ref_codes=ref_codes,
                            ref_text=ref_text,
                            max_chars=v2_max_chars,
                            temperature=v2_temp,
                            top_k=v2_top_k,
                        )
                    else:
                        w = tts2.infer(
                            ch_text,
                            voice=v2_voice,
                            max_chars=v2_max_chars,
                            temperature=v2_temp,
                            top_k=v2_top_k,
                        )
                    chunk_wavs.append(np.asarray(w, dtype=np.float32))

                if cancel_event and cancel_event.is_set():
                    raise RuntimeError("Đã hủy tác vụ theo yêu cầu.")
                audio = join_audio_chunks(chunk_wavs, sr, silence_ps=gaps_to_silence_durations(gaps))

            if abs(speed - 1.0) > 0.02 and len(audio) > 0:
                if progress_cb:
                    progress_cb(88, f"Đang chỉnh tốc độ {speed:.2f}x bằng FFmpeg atempo...", "", est_sec)
                audio = _time_stretch_ffmpeg(np.asarray(audio, dtype=np.float32), sr, float(speed))

            if normalize_lufs or voice_enhance or bgm_path is not None:
                if progress_cb:
                    progress_cb(92, "Đang xử lý hậu kỳ Studio FFmpeg (-14 LUFS / BGM)...", "", est_sec)
                audio = apply_studio_fx(
                    np.asarray(audio, dtype=np.float32),
                    sr,
                    normalize_lufs=normalize_lufs,
                    voice_enhance=voice_enhance,
                    bgm_path=bgm_path,
                    bgm_volume=bgm_volume,
                )

            if progress_cb:
                progress_cb(96, f"Đang đóng gói tệp âm thanh {FORMAT_META[fmt_key][1].upper()}...", "", est_sec)
            elapsed = time.perf_counter() - t0
            audio_dur = float(len(audio)) / float(sr) if len(audio) else 0.0
            out_bytes, mime, ext = encode_audio_bytes(audio, sr, fmt=fmt_key)

            self._cache[cache_key] = (out_bytes, audio_dur, elapsed, sr)
            while len(self._cache) > self._cache_max:
                self._cache.popitem(last=False)

        rtf = elapsed / audio_dur if audio_dur > 0 else 0.0
        return out_bytes, {
            "cached": False,
            "audio_duration_sec": round(audio_dur, 3),
            "inference_sec": round(elapsed, 3),
            "rtf": round(rtf, 3),
            "sample_rate": sr,
            "precision": settings.vieneu_precision if engine_key == "vieneu" else "gguf-q8",
            "format": ext,
            "mime": mime,
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
        ):
            meta = voice_manager._custom_meta.get(resolved_voice, {})
            if meta.get("created_model") == "v3" or not meta.get("v2_cached"):
                raise ValueError(
                    f"Giọng '{resolved_voice}' được tạo từ mô hình VieNeu v3 nên không thể tạo âm thanh với VieNeu v2. "
                    "Vui lòng chuyển sang mô hình VieNeu v3 Turbo hoặc chọn giọng clone được tạo từ VieNeu v2."
                )
            if not (ref_text_override and ref_text_override.strip()) and not meta.get("has_ref_text"):
                raise ValueError(
                    f"Giọng '{resolved_voice}' chưa có lời thoại mẫu (ref_text). "
                    "Chế độ VieNeu v2 yêu cầu nhập đúng câu nói trong file audio mẫu."
                )

        if engine_key == "vieneu":
            sr = self.sample_rate

            def _gen_v3() -> Generator[bytes, None, None]:
                import re as _re
                v3_text = _re.sub(r"\[\s*ngh[ỉi]\s*[0-9.,]+\s*s?\s*\]", "...", cleaned, flags=_re.IGNORECASE)
                with self._lock:
                    for chunk in self.tts.infer_stream(
                        v3_text,
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
        v2_temp, v2_top_k, v2_top_p, v2_rep, v2_max_chars = self._resolve_v2_params(
            temperature, top_k, top_p, repetition_penalty
        )

        def _gen_v2() -> Generator[bytes, None, None]:
            import re as _re
            v2_text = strip_emotion_cues(
                _re.sub(r"\[\s*ngh[ỉi]\s*[0-9.,]+\s*s?\s*\]", ",", cleaned, flags=_re.IGNORECASE)
            ) or cleaned
            with self._lock:
                tts2._repeat_penalty = v2_rep
                tts2._top_p = v2_top_p
                if resolved_voice and voice_manager is not None and voice_manager.is_custom_voice(resolved_voice):
                    ref_codes, ref_text = voice_manager.get_v2_reference(
                        resolved_voice,
                        self._encode_v2_wav,
                        ref_text_override=ref_text_override,
                    )
                    gen = tts2.infer_stream(
                        v2_text,
                        ref_codes=ref_codes,
                        ref_text=ref_text,
                        max_chars=v2_max_chars,
                        temperature=v2_temp,
                        top_k=v2_top_k,
                    )
                else:
                    v2_voice = tts2.get_preset_voice(resolved_voice)
                    gen = tts2.infer_stream(
                        v2_text,
                        voice=v2_voice,
                        max_chars=v2_max_chars,
                        temperature=v2_temp,
                        top_k=v2_top_k,
                    )
                for chunk in gen:
                    if chunk is not None and len(chunk) > 0:
                        yield _to_pcm16_bytes(chunk)

        return sr, _gen_v2()
