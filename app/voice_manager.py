"""Quản lý upload, làm sạch và lưu trữ bền vững các giọng clone (v3 & v2)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from .progress_store import ProgressCallback

_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_\-\u00C0-\u1EF9 ]+")


def _slugify_voice_name(name: str) -> str:
    clean = _SAFE_ID_RE.sub("", name).strip()
    clean = re.sub(r"\s+", "_", clean)
    if not clean:
        raise ValueError("Tên giọng không hợp lệ hoặc chỉ chứa ký tự đặc biệt.")
    return clean


def _convert_to_wav_if_needed(src_path: Path, dst_wav_path: Path) -> float:
    """Đưa audio upload về WAV mono 48kHz PCM16 và trả về thời lượng (giây)."""
    try:
        data, sr = sf.read(str(src_path), dtype="float32", always_2d=False)
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)
        data = np.asarray(data, dtype=np.float32)
        if sr != 48000:
            raise RuntimeError("Need resample via ffmpeg")
        sf.write(str(dst_wav_path), data, 48000, subtype="PCM_16")
        return float(len(data)) / 48000.0
    except Exception:
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise RuntimeError(
                "Không đọc được định dạng audio này và không tìm thấy ffmpeg trên hệ thống."
            )
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(src_path),
            "-ac",
            "1",
            "-ar",
            "48000",
            "-c:a",
            "pcm_s16le",
            str(dst_wav_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg chuyển đổi audio thất bại: {proc.stderr.strip()}")
        info = sf.info(str(dst_wav_path))
        return float(info.duration)


def _analyze_and_trim_sample(wav_path: Path) -> Tuple[float, dict]:
    """Đo chất lượng file mẫu và cắt gọn khoảng lặng đầu/cuối an toàn (giữ đệm 180ms bảo vệ thanh điệu tiếng Việt)."""
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    data = np.asarray(data, dtype=np.float32).reshape(-1)
    if len(data) == 0:
        raise ValueError("File âm thanh mẫu trống.")

    win = max(1, int(sr * 0.02))
    trimmed_ms = 0

    def _frame_dbs(samples: np.ndarray) -> np.ndarray:
        n_frames = len(samples) // win
        if n_frames <= 0:
            rms = float(np.sqrt(np.mean(samples * samples) + 1e-12))
            return np.asarray([20.0 * np.log10(max(rms, 1e-7))], dtype=np.float32)
        blocks = samples[: n_frames * win].reshape(n_frames, win)
        rms = np.sqrt(np.mean(blocks * blocks, axis=1) + 1e-12)
        return (20.0 * np.log10(np.maximum(rms, 1e-7))).astype(np.float32)

    fdb = _frame_dbs(data)
    active = np.where(fdb > -45.0)[0]
    if len(active) > 0:
        pad = int(0.18 * sr)
        start_s = max(0, int(active[0] * win) - pad)
        end_s = min(len(data), int((active[-1] + 1) * win) + pad)
        removed_ms = int(round(((len(data) - (end_s - start_s)) / float(sr)) * 1000))
        if removed_ms >= 250 and (end_s - start_s) / float(sr) >= 1.5:
            data = data[start_s:end_s]
            sf.write(str(wav_path), data, sr, subtype="PCM_16")
            trimmed_ms = removed_ms
            fdb = _frame_dbs(data)

    duration = float(len(data)) / float(sr)
    clipping_pct = round(float(np.mean(np.abs(data) >= 0.985) * 100.0), 2)
    speech_rms_db = round(float(np.percentile(fdb, 85)), 1)
    noise_floor_db = round(float(np.percentile(fdb, 15)), 1)
    snr_db = round(max(0.0, speech_rms_db - noise_floor_db), 1)

    warnings: List[str] = []
    if clipping_pct >= 0.5:
        warnings.append(f"Âm thanh chạm đỉnh/vỡ tiếng ({clipping_pct}% mẫu)")
    if snr_db < 14.0:
        warnings.append(f"Nhiễu nền hơi cao (SNR {snr_db} dB)")
    if speech_rms_db < -34.0:
        warnings.append(f"Giọng thu hơi nhỏ ({speech_rms_db} dBFS)")

    if clipping_pct >= 2.0 or snr_db < 9.0:
        status, label = "poor", "Cần lưu ý"
    elif warnings:
        status, label = "warn", "Đạt chuẩn"
    else:
        status, label = "good", "Âm sạch"

    return duration, {
        "status": status,
        "label": label,
        "snr_db": snr_db,
        "speech_rms_db": speech_rms_db,
        "clipping_pct": clipping_pct,
        "trimmed_silence_ms": trimmed_ms,
        "warnings": warnings,
    }


class VoiceManager:
    """Lưu giữ các giọng clone đã đăng ký cho cả VieNeu v3 Turbo và v2 GGUF."""

    def __init__(self, tts: Any, voices_dir: Path) -> None:
        self.tts = tts
        self.voices_dir = Path(voices_dir)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.voices_dir / "custom_voices.json"
        self._custom_meta: Dict[str, dict] = {}
        self.load_saved_voices()

    def _read_registry(self) -> Dict[str, dict]:
        if not self.registry_path.is_file():
            return {}
        try:
            return json.loads(self.registry_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_registry(self, data: Dict[str, dict]) -> None:
        self.registry_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _build_meta(name: str, entry: dict) -> dict:
        ref_text = str(entry.get("ref_text") or "").strip()
        return {
            "name": name,
            "description": entry.get("description", ""),
            "ref_text": ref_text,
            "has_ref_text": bool(ref_text),
            "v2_cached": bool(entry.get("v2_codes")),
            "duration_sec": entry.get("duration_sec", 0.0),
            "denoise": entry.get("denoise", False),
            "consent_confirmed": entry.get("consent_confirmed", False),
            "created_at": entry.get("created_at", ""),
            "audio_file": entry.get("audio_file", ""),
            "quality": entry.get("quality"),
        }

    def load_saved_voices(self) -> int:
        """Nạp lại các giọng clone đã lưu từ ổ đĩa vào bộ nhớ VieNeu v3."""
        data = self._read_registry()
        loaded = 0
        for name, entry in data.items():
            emb = entry.get("speaker_emb")
            codes = entry.get("codes")
            if emb is None:
                continue
            self.tts._preset_voices[name] = {
                "description": entry.get("description", ""),
                "gender": entry.get("gender", ""),
                "style": "tu_nhien",
                "speaker_emb": np.asarray(emb, dtype=np.float32),
                "codes": None if codes is None else np.asarray(codes, dtype=np.int64),
            }
            self._custom_meta[name] = self._build_meta(name, entry)
            loaded += 1
        return loaded

    def list_voices(self, tts_v2: Any = None) -> Dict[str, List[dict]]:
        """Trả về danh sách giọng clone tùy chỉnh và giọng preset mặc định."""
        custom_list = list(self._custom_meta.values())
        preset_list = []
        for _, voice_id in self.tts.list_preset_voices():
            if voice_id in self._custom_meta:
                continue
            info = self.tts._preset_voices.get(voice_id, {})
            preset_list.append(
                {
                    "name": voice_id,
                    "description": info.get("description", ""),
                    "gender": info.get("gender", ""),
                }
            )
        v2_preset_list = []
        if tts_v2 is not None:
            for desc, vid in tts_v2.list_preset_voices():
                v2_preset_list.append({"name": vid, "description": desc})
        return {
            "custom_voices": custom_list,
            "preset_voices": preset_list,
            "v2_preset_voices": v2_preset_list,
        }

    def register_voice(
        self,
        name: str,
        uploaded_file_path: Path,
        *,
        description: str = "",
        ref_text: str = "",
        denoise: bool = False,
        consent_confirmed: bool = False,
        progress_cb: ProgressCallback = None,
    ) -> dict:
        """Chuẩn hoá file audio upload, trích xuất đặc trưng v3 và lưu metadata/ref_text."""
        if not consent_confirmed:
            raise ValueError("Cần xác nhận quyền sử dụng / sự đồng ý của chủ giọng mẫu.")

        def _report(pct: int, stage: str, detail: str = "") -> None:
            if progress_cb:
                progress_cb(pct, stage, detail)

        _report(12, "Đang chuyển đổi & chuẩn hóa WAV 48kHz...", "")
        safe_name = _slugify_voice_name(name)
        target_wav = self.voices_dir / f"{safe_name}.wav"
        duration = _convert_to_wav_if_needed(uploaded_file_path, target_wav)

        if duration < 1.5:
            target_wav.unlink(missing_ok=True)
            raise ValueError(
                f"File giọng mẫu quá ngắn ({duration:.2f}s). Cần tối thiểu 3–8 giây để clone ổn định."
            )

        _report(38, "Đang đo chất lượng âm & cắt khoảng lặng an toàn...", f"Thời lượng: {duration:.1f}s")
        duration, quality = _analyze_and_trim_sample(target_wav)

        _report(62, "Đang trích xuất đặc trưng giọng nói (Speaker Embedding)...", "VieNeu Voice Encoder")
        self.tts.add_voice(
            safe_name,
            str(target_wav),
            denoise=denoise,
            use_ref_codes=True,
            description=description,
            save=False,
        )

        _report(90, "Đang lưu giọng clone vào thư viện...", safe_name)
        voice_data = self.tts._preset_voices[safe_name]
        emb = voice_data.get("speaker_emb")
        codes = voice_data.get("codes")

        registry = self._read_registry()
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
        entry = {
            "name": safe_name,
            "description": description,
            "ref_text": ref_text.strip(),
            "duration_sec": round(duration, 2),
            "denoise": bool(denoise),
            "consent_confirmed": True,
            "created_at": now_iso,
            "audio_file": target_wav.name,
            "quality": quality,
            "speaker_emb": [round(float(x), 6) for x in np.asarray(emb).reshape(-1)],
            "codes": None if codes is None else np.asarray(codes, dtype=int).tolist(),
            "v2_codes": None,
        }
        registry[safe_name] = entry
        self._write_registry(registry)
        meta = self._build_meta(safe_name, entry)
        self._custom_meta[safe_name] = meta
        _report(100, "Đã lưu giọng clone vào thư viện!", safe_name)
        return meta

    def update_ref_text(self, name: str, ref_text: str) -> dict:
        """Cập nhật lời thoại mẫu (ref_text) cho giọng clone đã có để dùng trên VieNeu v2."""
        if name not in self._custom_meta:
            raise KeyError(f"Không tìm thấy giọng clone '{name}'.")
        clean_ref = ref_text.strip()
        if not clean_ref:
            raise ValueError("Lời thoại mẫu (ref_text) không được để trống.")
        registry = self._read_registry()
        if name not in registry:
            raise KeyError(f"Không tìm thấy dữ liệu lưu của giọng '{name}'.")
        registry[name]["ref_text"] = clean_ref
        self._write_registry(registry)
        meta = self._build_meta(name, registry[name])
        self._custom_meta[name] = meta
        return meta

    def is_custom_voice(self, name: Optional[str]) -> bool:
        return bool(name and name in self._custom_meta)

    def get_v2_reference(
        self,
        name: str,
        encode_fn: Callable[[Path], np.ndarray],
        ref_text_override: Optional[str] = None,
    ) -> Tuple[np.ndarray, str]:
        """Lấy (v2_codes, ref_text) cho giọng clone; tự encode & cache lần đầu nếu chưa có."""
        registry = self._read_registry()
        if name not in registry:
            raise ValueError(f"Không tìm thấy giọng clone '{name}'.")
        entry = registry[name]
        ref_text = (ref_text_override or entry.get("ref_text") or "").strip()
        if not ref_text:
            raise ValueError(
                f"Giọng '{name}' chưa có lời thoại mẫu (ref_text). "
                "Chế độ VieNeu v2 yêu cầu nhập đúng câu nói trong file audio mẫu."
            )
        audio_file = entry.get("audio_file")
        if not audio_file:
            raise ValueError(f"Không tìm thấy file audio gốc của giọng '{name}'.")
        wav_path = self.voices_dir / audio_file
        if not wav_path.is_file():
            raise ValueError(f"File audio mẫu '{wav_path.name}' không còn tồn tại.")

        cached_codes = entry.get("v2_codes")
        if cached_codes:
            codes_arr = np.asarray(cached_codes, dtype=np.int64)
        else:
            codes_arr = np.asarray(encode_fn(wav_path), dtype=np.int64).reshape(-1)
            entry["v2_codes"] = codes_arr.tolist()
            registry[name] = entry
            self._write_registry(registry)
            self._custom_meta[name] = self._build_meta(name, entry)

        return codes_arr, ref_text

    def delete_voice(self, name: str) -> bool:
        """Xoá một giọng clone tùy chỉnh (không xoá giọng preset gốc)."""
        if name not in self._custom_meta:
            return False
        self.tts.remove_voice(name, save=False)
        meta = self._custom_meta.pop(name)
        registry = self._read_registry()
        registry.pop(name, None)
        self._write_registry(registry)
        audio_file = meta.get("audio_file")
        if audio_file:
            (self.voices_dir / audio_file).unlink(missing_ok=True)
        return True
