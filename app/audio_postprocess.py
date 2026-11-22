"""Hậu kỳ âm thanh chuẩn Studio (FFmpeg 9.0): Lọc ù, -14 LUFS, BGM Auto-Ducking, xuất WAV/MP3/M4A."""

from __future__ import annotations

import io
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

_SAFE_BGM_RE = re.compile(r"[^a-zA-Z0-9_\-\u00C0-\u1EF9. ]+")

FORMAT_META: Dict[str, Tuple[str, str]] = {
    "wav": ("audio/wav", "wav"),
    "mp3": ("audio/mpeg", "mp3"),
    "m4a": ("audio/mp4", "m4a"),
    "aac": ("audio/mp4", "m4a"),
}


def _require_ffmpeg() -> str:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("Không tìm thấy FFmpeg trên hệ thống để xử lý hậu kỳ / xuất MP3-M4A.")
    return ffmpeg_bin


def apply_studio_fx(
    audio_f32: np.ndarray,
    sample_rate: int,
    *,
    normalize_lufs: bool = False,
    voice_enhance: bool = False,
    bgm_path: Optional[Path] = None,
    bgm_volume: float = 0.14,
) -> np.ndarray:
    """Áp dụng chuỗi lọc hậu kỳ Studio (highpass/EQ/compressor, -14 LUFS, BGM auto-ducking)."""
    arr = np.asarray(audio_f32, dtype=np.float32).reshape(-1)
    if len(arr) == 0:
        return arr
    if not normalize_lufs and not voice_enhance and bgm_path is None:
        return arr

    ffmpeg_bin = _require_ffmpeg()
    dur_sec = float(len(arr)) / float(sample_rate)
    fade_start = max(0.0, dur_sec - 1.2)
    bgm_vol = max(0.02, min(0.60, float(bgm_volume)))

    voice_filters: List[str] = []
    if voice_enhance:
        voice_filters.append(
            "highpass=f=80,acompressor=threshold=-18dB:ratio=2.5:attack=15:release=180,equalizer=f=220:width_type=o:width=1.2:g=1.5"
        )
    if normalize_lufs:
        voice_filters.append("loudnorm=I=-14:TP=-1.5:LRA=11")

    if bgm_path is None:
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
            "-af",
            ",".join(voice_filters),
            "-f",
            "f32le",
            "pipe:1",
        ]
        proc = subprocess.run(cmd, input=arr.tobytes(), capture_output=True, check=False)
        if proc.returncode != 0 or len(proc.stdout) < 4:
            raise RuntimeError(f"FFmpeg hậu kỳ thất bại: {proc.stderr.decode('utf-8', 'ignore').strip()}")
        return np.frombuffer(proc.stdout, dtype=np.float32).copy()

    # Mixing BGM có auto-ducking bằng sidechaincompress
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_voice:
        sf.write(tmp_voice.name, np.clip(arr, -1.0, 1.0), sample_rate, subtype="PCM_16")
        voice_wav_path = Path(tmp_voice.name)

    try:
        v_chain = ",".join(voice_filters) if voice_filters else "anull"
        filter_complex = (
            f"[0:a]{v_chain},asplit=2[v_main][v_sc];"
            f"[1:a]aformat=sample_rates={sample_rate}:channel_layouts=mono,"
            f"aloop=loop=-1:size=2000000000,atrim=0:{dur_sec:.3f},asetpts=PTS-STARTPTS,"
            f"volume={bgm_vol:.3f},afade=t=in:st=0:d=0.6,afade=t=out:st={fade_start:.3f}:d=1.2[bgm_trimmed];"
            f"[bgm_trimmed][v_sc]sidechaincompress=threshold=0.025:ratio=6:attack=40:release=350:level_in=1[bgm_ducked];"
            f"[v_main][bgm_ducked]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.96[out]"
        )
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(voice_wav_path),
            "-i",
            str(bgm_path),
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-ac",
            "1",
            "-ar",
            str(int(sample_rate)),
            "-f",
            "f32le",
            "pipe:1",
        ]
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0 or len(proc.stdout) < 4:
            raise RuntimeError(f"FFmpeg mix BGM thất bại: {proc.stderr.decode('utf-8', 'ignore').strip()}")
        return np.frombuffer(proc.stdout, dtype=np.float32).copy()
    finally:
        voice_wav_path.unlink(missing_ok=True)


def encode_audio_bytes(
    audio_f32: np.ndarray,
    sample_rate: int,
    fmt: str = "wav",
) -> Tuple[bytes, str, str]:
    """Đóng gói audio float32 thành bytes theo định dạng (wav / mp3 / m4a)."""
    fmt_key = (fmt or "wav").lower().strip()
    if fmt_key not in FORMAT_META:
        raise ValueError(f"Định dạng xuất '{fmt}' không hỗ trợ. Chọn wav, mp3 hoặc m4a.")

    mime, ext = FORMAT_META[fmt_key]
    clipped = np.clip(np.asarray(audio_f32, dtype=np.float32).reshape(-1), -1.0, 1.0)

    if ext == "wav":
        buf = io.BytesIO()
        sf.write(buf, clipped, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue(), mime, ext

    ffmpeg_bin = _require_ffmpeg()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_out:
        out_path = Path(tmp_out.name)

    try:
        codec_args = (
            ["-c:a", "libmp3lame", "-b:a", "256k"]
            if ext == "mp3"
            else ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"]
        )
        cmd = [
            ffmpeg_bin,
            "-y",
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
            *codec_args,
            str(out_path),
        ]
        proc = subprocess.run(cmd, input=clipped.tobytes(), capture_output=True, check=False)
        if proc.returncode != 0 or not out_path.is_file():
            raise RuntimeError(f"FFmpeg xuất {ext.upper()} thất bại: {proc.stderr.decode('utf-8', 'ignore').strip()}")
        return out_path.read_bytes(), mime, ext
    finally:
        out_path.unlink(missing_ok=True)


class BgmManager:
    """Quản lý thư viện nhạc nền (BGM) trong voices/bgm/."""

    ALLOWED_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}

    def __init__(self, bgm_dir: Path) -> None:
        self.bgm_dir = bgm_dir
        self.bgm_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        clean = _SAFE_BGM_RE.sub("_", Path(name).name).strip()
        if not clean or clean in (".", ".."):
            raise ValueError("Tên file nhạc nền không hợp lệ.")
        return clean

    def list_tracks(self) -> List[Dict[str, Any]]:
        tracks: List[Dict[str, Any]] = []
        for p in sorted(self.bgm_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not p.is_file() or p.suffix.lower() not in self.ALLOWED_EXTS:
                continue
            tracks.append(
                {
                    "name": p.name,
                    "size_kb": round(p.stat().st_size / 1024.0, 1),
                    "url": f"/v1/bgm/{p.name}",
                }
            )
        return tracks

    def save_track(self, filename: str, raw_bytes: bytes) -> Dict[str, Any]:
        safe_name = self._sanitize_filename(filename)
        ext = Path(safe_name).suffix.lower()
        if ext not in self.ALLOWED_EXTS:
            raise ValueError("Chỉ hỗ trợ file nhạc nền định dạng MP3, WAV, M4A, OGG hoặc FLAC.")
        dst = self.bgm_dir / safe_name
        dst.write_bytes(raw_bytes)
        return {
            "name": dst.name,
            "size_kb": round(dst.stat().st_size / 1024.0, 1),
            "url": f"/v1/bgm/{dst.name}",
        }

    def get_path(self, filename: str) -> Optional[Path]:
        if not filename:
            return None
        safe_name = Path(filename).name
        p = self.bgm_dir / safe_name
        if p.is_file() and p.suffix.lower() in self.ALLOWED_EXTS:
            return p
        return None

    def delete_track(self, filename: str) -> bool:
        p = self.get_path(filename)
        if p is None:
            return False
        p.unlink(missing_ok=True)
        return True
