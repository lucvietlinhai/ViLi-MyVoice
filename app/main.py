"""FastAPI service + Web UI cho hệ thống TTS Clone Voice tiếng Việt (song song v3 & v2)."""

from __future__ import annotations

import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import settings
from .normalizer import split_and_normalize
from .tts_engine import TTSEngine
from .voice_manager import VoiceManager

STATIC_DIR = Path(__file__).resolve().parent / "static"


class SpeechRequest(BaseModel):
    model: Literal["vieneu", "vieneu-v2"] = Field(
        default="vieneu",
        description="'vieneu' = v3 Turbo 48kHz; 'vieneu-v2' = v2 GGUF Q8 24kHz + ref_text.",
    )
    voice: str = Field(default="default", description="Tên giọng clone hoặc giọng preset.")
    input: str = Field(..., min_length=1, description="Văn bản tiếng Việt cần đọc.")
    ref_text: Optional[str] = Field(
        default=None, description="Ghi đè lời thoại mẫu (chỉ dùng cho model='vieneu-v2')."
    )
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    response_format: Literal["wav", "pcm"] = Field(default="wav")
    temperature: float = Field(default=0.45, ge=0.1, le=1.5)
    top_k: int = Field(default=15, ge=1, le=100)
    top_p: float = Field(default=0.85, ge=0.1, le=1.0)
    repetition_penalty: float = Field(default=1.2, ge=1.0, le=1.8)
    use_ref_codes: bool = Field(default=True)


class UpdateRefTextRequest(BaseModel):
    ref_text: str = Field(..., min_length=1, description="Lời thoại chính xác của file audio mẫu.")


class PreviewTextRequest(BaseModel):
    input: str = Field(..., min_length=1)


_engine: Optional[TTSEngine] = None
_voice_mgr: Optional[VoiceManager] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _engine, _voice_mgr
    _engine = TTSEngine()
    _voice_mgr = VoiceManager(_engine.tts, settings.voices_dir)
    yield
    if _engine is not None:
        _engine.tts.close()
        if _engine.tts_v2 is not None:
            _engine.tts_v2.close()


app = FastAPI(
    title="Vietnamese TTS Voice Clone (Local v3 + v2)",
    version="1.1.0",
    lifespan=lifespan,
)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _require_ready() -> tuple[TTSEngine, VoiceManager]:
    if _engine is None or _voice_mgr is None:
        raise HTTPException(status_code=503, detail="TTS engine chưa sẵn sàng.")
    return _engine, _voice_mgr


@app.get("/")
async def index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.is_file():
        return {"status": "ok", "message": "Vietnamese TTS Clone API is running."}
    return FileResponse(str(index_file))


@app.get("/health")
async def health():
    engine, voice_mgr = _require_ready()
    voices = voice_mgr.list_voices(engine.tts_v2)
    return {
        "status": "ok",
        "models": ["vieneu", "vieneu-v2"],
        "mode": settings.vieneu_mode,
        "backend": settings.vieneu_backend,
        "precision": settings.vieneu_precision,
        "sample_rate": engine.sample_rate,
        "load_seconds": engine.load_seconds,
        "v2_loaded": engine.tts_v2 is not None,
        "v2_sample_rate": engine.v2_sample_rate,
        "v2_load_seconds": engine.v2_load_seconds,
        "custom_voice_count": len(voices["custom_voices"]),
        "preset_voice_count": len(voices["preset_voices"]),
    }


@app.get("/v1/voices")
async def list_voices():
    engine, voice_mgr = _require_ready()
    return voice_mgr.list_voices(engine.tts_v2)


@app.get("/v1/voices/{voice_id}/audio")
async def get_voice_audio(voice_id: str):
    """Trả về file audio gốc đã lưu của giọng clone để người dùng nghe thử mẫu."""
    engine, voice_mgr = _require_ready()
    from .voice_manager import _slugify_voice_name
    safe_name = _slugify_voice_name(voice_id)
    wav_path = voice_mgr.voices_dir / f"{safe_name}.wav"
    if not wav_path.is_file():
        raise HTTPException(status_code=404, detail=f"Không tìm thấy file audio cho giọng '{voice_id}'.")
    return FileResponse(str(wav_path), media_type="audio/wav")


@app.post("/v1/voices")
async def upload_voice(
    name: str = Form(..., description="Tên gợi nhớ cho giọng clone"),
    description: str = Form(default=""),
    ref_text: str = Form(default=""),
    denoise: bool = Form(default=False),
    consent_confirmed: bool = Form(default=False),
    file: UploadFile = File(...),
):
    engine, voice_mgr = _require_ready()
    suffix = Path(file.filename or "sample.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        meta = voice_mgr.register_voice(
            name=name,
            uploaded_file_path=tmp_path,
            description=description,
            ref_text=ref_text,
            denoise=denoise,
            consent_confirmed=consent_confirmed,
        )
        engine.bump_voice_revision()
        return {"status": "registered", "voice": meta}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý giọng mẫu: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)


@app.patch("/v1/voices/{voice_id}")
async def update_voice_ref_text(voice_id: str, req: UpdateRefTextRequest):
    engine, voice_mgr = _require_ready()
    try:
        meta = voice_mgr.update_ref_text(voice_id, req.ref_text)
        engine.bump_voice_revision()
        return {"status": "updated", "voice": meta}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/v1/voices/{voice_id}")
async def delete_voice(voice_id: str):
    engine, voice_mgr = _require_ready()
    removed = voice_mgr.delete_voice(voice_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy giọng clone '{voice_id}'.")
    engine.bump_voice_revision()
    return {"status": "deleted", "name": voice_id}


@app.post("/v1/text/preview-chunks")
async def preview_chunks(req: PreviewTextRequest):
    chunks, gaps = split_and_normalize(req.input, max_chars=settings.max_chars_per_chunk)
    return {"chunks": chunks, "gaps": gaps, "count": len(chunks)}


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    engine, voice_mgr = _require_ready()
    try:
        if req.response_format == "pcm":
            sr, gen = engine.stream_pcm(
                req.input,
                model=req.model,
                voice=req.voice,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
                use_ref_codes=req.use_ref_codes,
                ref_text_override=req.ref_text,
                voice_manager=voice_mgr,
            )
            return StreamingResponse(
                gen,
                media_type="audio/pcm",
                headers={
                    "X-Audio-Sample-Rate": str(sr),
                    "X-Audio-Channels": "1",
                    "X-Audio-Sample-Format": "s16le",
                    "X-TTS-Model": req.model,
                },
            )

        wav_bytes, stats = engine.synthesize_wav(
            req.input,
            model=req.model,
            voice=req.voice,
            speed=req.speed,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            use_ref_codes=req.use_ref_codes,
            ref_text_override=req.ref_text,
            voice_manager=voice_mgr,
        )
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'inline; filename="speech.wav"',
                "X-Audio-Sample-Rate": str(stats["sample_rate"]),
                "X-Audio-Duration-Sec": str(stats["audio_duration_sec"]),
                "X-Inference-Sec": str(stats["inference_sec"]),
                "X-RTF": str(stats["rtf"]),
                "X-Cached": str(stats["cached"]).lower(),
                "X-TTS-Precision": stats["precision"],
                "X-TTS-Model": req.model,
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi tổng hợp giọng nói: {exc}") from exc
