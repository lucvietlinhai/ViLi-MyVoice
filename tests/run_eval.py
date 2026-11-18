"""Script nghiệm thu bộ câu test tiếng Việt và đo RTF cho VieNeu-TTS."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import soundfile as sf

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.normalizer import preprocess_text  # noqa: E402
from vieneu import Vieneu  # noqa: E402


def load_sentences(path: Path) -> list[str]:
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy bộ câu test nghiệm thu VieNeu-TTS.")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "int8"])
    parser.add_argument("--voice", default=None, help="Tên giọng preset hoặc giọng đã lưu.")
    parser.add_argument("--ref-audio", default=None, help="Đường dẫn file WAV mẫu để test clone.")
    parser.add_argument("--denoise", action="store_true", help="Khử nhiễu file mẫu trước khi clone.")
    parser.add_argument("--limit", type=int, default=0, help="Giới hạn số câu chạy (0 = toàn bộ).")
    args = parser.parse_args()

    sentences = load_sentences(BASE_DIR / "tests" / "test_sentences.txt")
    if args.limit > 0:
        sentences = sentences[: args.limit]

    out_dir = BASE_DIR / "outputs" / f"eval_{args.precision}"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    tts = Vieneu(mode="v3turbo", backend="onnx", precision=args.precision)
    print(f"Loaded VieNeu v3turbo ({args.precision}) in {time.perf_counter() - t0:.2f}s")

    voice_arg = args.voice
    if args.ref_audio:
        tts.add_voice("eval_clone", args.ref_audio, denoise=args.denoise, save=False)
        voice_arg = "eval_clone"
        print(f"Enrolled clone voice from: {args.ref_audio} (denoise={args.denoise})")

    total_audio = 0.0
    total_infer = 0.0

    for idx, raw_text in enumerate(sentences, start=1):
        cleaned = preprocess_text(raw_text)
        t1 = time.perf_counter()
        wav = tts.infer(cleaned, voice=voice_arg, max_chars=220)
        elapsed = time.perf_counter() - t1
        dur = len(wav) / tts.sample_rate
        rtf = elapsed / dur if dur > 0 else 0.0
        total_audio += dur
        total_infer += elapsed
        out_path = out_dir / f"{idx:02d}.wav"
        sf.write(str(out_path), wav, tts.sample_rate, subtype="PCM_16")
        print(f"[{idx:02d}/{len(sentences)}] dur={dur:.2f}s | infer={elapsed:.2f}s | RTF={rtf:.3f} | {raw_text}")

    avg_rtf = total_infer / total_audio if total_audio > 0 else 0.0
    print(
        f"\nSUMMARY ({args.precision}): sentences={len(sentences)} | "
        f"total_audio={total_audio:.2f}s | total_infer={total_infer:.2f}s | avg_RTF={avg_rtf:.3f}"
    )
    print(f"Saved WAVs in: {out_dir}")


if __name__ == "__main__":
    main()
