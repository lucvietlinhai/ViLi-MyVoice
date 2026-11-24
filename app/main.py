"""FastAPI service + Web UI cho hệ thống TTS Clone Voice tiếng Việt (song song v3 & v2)."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, ORJSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .audiobook_manager import AudiobookManager, extract_text_from_document
from .audio_postprocess import BgmManager
from .config import settings
from .history_manager import HistoryManager
from .lexicon_manager import STARTER_TEMPLATES, lexicon_manager
from .log_stream import attach_python_logging, log_store, push_log
from .normalizer import preprocess_text, split_and_normalize
from .progress_store import progress_store
from .prosody_analyzer import analyze_and_annotate_prosody
from .tts_engine import TTSEngine
from .voice_manager import VoiceManager

STATIC_DIR = Path(__file__).resolve().parent / "static"

_OPENAI_MODEL_ALIASES = {
    "tts-1": "vieneu",
    "tts-1-hd": "vieneu",
    "gpt-4o-mini-tts": "vieneu",
    "vieneu": "vieneu",
    "vieneu-v2": "vieneu-v2",
}

_OPENAI_VOICE_FALLBACKS = {
    "alloy": "default",
    "nova": "default",
    "shimmer": "default",
    "echo": "default",
    "onyx": "default",
    "fable": "default",
}


class SpeechRequest(BaseModel):
    model: str = Field(
        default="vieneu",
        description="'vieneu' = v3 Turbo 48kHz; 'vieneu-v2' = v2 GGUF Q8 24kHz (hoặc alias OpenAI tts-1).",
    )
    voice: str = Field(default="default", description="Tên giọng clone, preset VieNeu hoặc alias OpenAI.")
    input: str = Field(..., min_length=1, description="Văn bản tiếng Việt cần đọc.")
    ref_text: Optional[str] = Field(
        default=None, description="Ghi đè lời thoại mẫu (chỉ dùng cho model='vieneu-v2')."
    )
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    response_format: str = Field(default="wav", description="wav | mp3 | m4a | aac | pcm")
    temperature: float = Field(default=0.45, ge=0.1, le=1.5)
    top_k: int = Field(default=15, ge=1, le=100)
    top_p: float = Field(default=0.85, ge=0.1, le=1.0)
    repetition_penalty: float = Field(default=1.2, ge=1.0, le=1.8)
    use_ref_codes: bool = Field(default=True)
    normalize_lufs: bool = Field(default=False, description="Chuẩn hóa âm lượng -14 LUFS.")
    voice_enhance: bool = Field(default=False, description="Lọc ù <80Hz & làm ấm giọng Studio.")
    bgm_name: Optional[str] = Field(default=None, description="Tên file nhạc nền trong voices/bgm/ (auto-ducking).")
    bgm_volume: float = Field(default=0.14, ge=0.02, le=0.6)
    job_id: Optional[str] = Field(default=None, description="ID theo dõi % tiến trình (tùy chọn).")


class UpdateRefTextRequest(BaseModel):
    ref_text: str = Field(..., min_length=1, description="Lời thoại chính xác của file audio mẫu.")


class CommitCloneRequest(BaseModel):
    draft_id: str = Field(..., min_length=4)
    name: str = Field(..., min_length=1)
    description: str = Field(default="")
    ref_text: str = Field(default="")
    consent_confirmed: bool = Field(default=False)
    job_id: str = Field(default="")


class PreviewTextRequest(BaseModel):
    input: str = Field(..., min_length=1)


class AutoProsodyRequest(BaseModel):
    input: str = Field(..., min_length=1)
    style: str = Field(default="natural")


class LexiconRuleRequest(BaseModel):
    pattern: str = Field(..., min_length=1)
    replacement: str = Field(..., min_length=1)
    whole_word: bool = Field(default=True)
    case_sensitive: bool = Field(default=False)
    enabled: bool = Field(default=True)
    note: str = Field(default="")


class LexiconRulePatch(BaseModel):
    pattern: Optional[str] = None
    replacement: Optional[str] = None
    whole_word: Optional[bool] = None
    case_sensitive: Optional[bool] = None
    enabled: Optional[bool] = None
    note: Optional[str] = None


class HistoryPatchRequest(BaseModel):
    label: str = Field(default="")


_engine: Optional[TTSEngine] = None
_voice_mgr: Optional[VoiceManager] = None
_history_mgr = HistoryManager(settings.history_dir)
_bgm_mgr = BgmManager(settings.bgm_dir)
_audiobook_mgr = AudiobookManager(settings.audiobooks_dir, _bgm_mgr)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _engine, _voice_mgr
    attach_python_logging()
    push_log("INFO", "SERVER", f"Khởi động TTS Studio (mode={settings.vieneu_mode}, backend={settings.vieneu_backend}, precision={settings.vieneu_precision})...")
    _engine = TTSEngine()
    push_log("OK", "TTS", f"Nạp thành công VieNeu v3 Turbo ({_engine.sample_rate} Hz) trong {_engine.load_seconds}s.")
    _voice_mgr = VoiceManager(_engine.tts, settings.voices_dir)
    _audiobook_mgr.attach_runtime(_engine, _voice_mgr)
    voices = _voice_mgr.list_voices(_engine.tts_v2)
    push_log("OK", "VOICE", f"Đã nạp {len(voices['custom_voices'])} giọng clone & {len(voices['preset_voices'])} giọng mặc định VieNeu.")
    yield
    push_log("INFO", "SERVER", "Đang tắt TTS Engine...")
    if _engine is not None:
        _engine.tts.close()
        if _engine.tts_v2 is not None:
            _engine.tts_v2.close()


app = FastAPI(
    title="ViLi-MyVoice (Vietnamese TTS Voice Clone)",
    version="1.2.0",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _require_ready() -> tuple[TTSEngine, VoiceManager]:
    if _engine is None or _voice_mgr is None:
        raise HTTPException(status_code=503, detail="TTS engine chưa sẵn sàng.")
    return _engine, _voice_mgr


def _resolve_openai_compat(model: str, voice: str, voice_mgr: VoiceManager) -> tuple[str, str]:
    m_key = (model or "vieneu").strip().lower()
    mapped_model = _OPENAI_MODEL_ALIASES.get(m_key, m_key)
    v_key = (voice or "default").strip()
    if v_key.lower() in _OPENAI_VOICE_FALLBACKS and not voice_mgr.is_custom_voice(v_key):
        mapped_voice = _OPENAI_VOICE_FALLBACKS[v_key.lower()]
    else:
        mapped_voice = v_key
    return mapped_model, mapped_voice


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
        "models": ["vieneu", "vieneu-v2", "tts-1", "tts-1-hd"],
        "mode": settings.vieneu_mode,
        "backend": settings.vieneu_backend,
        "precision": settings.vieneu_precision,
        "sample_rate": engine.sample_rate,
        "load_seconds": engine.load_seconds,
        "v2_loaded": engine.tts_v2 is not None,
        "v2_sample_rate": engine.v2_sample_rate,
        "v2_load_seconds": engine.v2_load_seconds,
        "stt_model": settings.stt_model_id,
        "hf_home": str(settings.hf_home_dir),
        "threads": settings.vieneu_threads,
        "custom_voice_count": len(voices["custom_voices"]),
        "preset_voice_count": len(voices["preset_voices"]),
    }


@app.get("/v1/logs")
async def get_logs(since_id: int = 0):
    return log_store.get_since(since_id)


@app.get("/v1/progress/{job_id}")
async def get_job_progress(job_id: str):
    return progress_store.get(job_id)


@app.post("/v1/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    progress_store.cancel(job_id)
    push_log("WARN", "TTS", f"Người dùng yêu cầu hủy tác vụ '{job_id}'.")
    return {"status": "cancelling", "job_id": job_id}


@app.delete("/v1/logs")
async def clear_logs():
    log_store.clear()
    push_log("INFO", "SERVER", "Đã xóa sạch lịch sử Terminal Log.")
    return {"status": "cleared"}


@app.get("/v1/voices")
async def list_voices():
    engine, voice_mgr = _require_ready()
    return voice_mgr.list_voices(engine.tts_v2)


@app.get("/v1/voices/{voice_id}/audio")
async def get_voice_audio(voice_id: str):
    engine, voice_mgr = _require_ready()
    from .voice_manager import _slugify_voice_name
    safe_name = _slugify_voice_name(voice_id)
    wav_path = voice_mgr.voices_dir / f"{safe_name}.wav"
    if not wav_path.is_file():
        raise HTTPException(status_code=404, detail=f"Không tìm thấy file audio cho giọng '{voice_id}'.")
    return FileResponse(str(wav_path), media_type="audio/wav")


@app.get("/v1/voices/{voice_id}/sample")
def get_voice_sample(voice_id: str):
    engine, voice_mgr = _require_ready()
    voices = voice_mgr.list_voices(engine.tts_v2)
    preset_names = {v["name"] for v in voices["preset_voices"]}
    if voice_id not in preset_names:
        if voice_mgr.is_custom_voice(voice_id):
            return get_voice_audio_sync(voice_mgr, voice_id)
        raise HTTPException(status_code=404, detail=f"Không tìm thấy giọng mẫu '{voice_id}'.")

    sample_text = "Xin chào, đây là giọng đọc mẫu được tích hợp sẵn trong ViLi-MyVoice."
    wav_bytes, _stats = engine.synthesize_wav(
        sample_text,
        model="vieneu",
        voice=voice_id,
        speed=1.0,
        voice_manager=voice_mgr,
    )
    return Response(content=wav_bytes, media_type="audio/wav")


def get_voice_audio_sync(voice_mgr: VoiceManager, voice_id: str):
    safe_name = voice_mgr.resolve_audio_name(voice_id)
    wav_path = voice_mgr.voices_dir / f"{safe_name}.wav"
    if not wav_path.is_file():
        raise HTTPException(status_code=404, detail=f"Không tìm thấy file audio cho giọng '{voice_id}'.")
    return FileResponse(str(wav_path), media_type="audio/wav")


@app.post("/v1/voices/preview-clone")
async def preview_clone_voice(
    preview_text: str = Form(...),
    preview_model: str = Form(default="vieneu"),
    ref_text: str = Form(default=""),
    denoise: bool = Form(default=False),
    draft_id: str = Form(default=""),
    job_id: str = Form(default=""),
    file: Optional[UploadFile] = File(default=None),
):
    """Học giọng từ âm thanh tham chiếu và tạo bản nghe thử với câu đại diện (chưa lưu vào Thư viện)."""
    engine, voice_mgr = _require_ready()
    tmp_path: Optional[Path] = None

    if file is not None and file.filename:
        suffix = Path(file.filename).suffix or ".wav"
        progress_store.update(job_id, 6, "Đang tải tệp âm thanh tham chiếu lên server...", file.filename, estimated_sec=9.0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)
    elif not draft_id:
        raise HTTPException(status_code=400, detail="Vui lòng chọn file âm thanh hoặc thu âm tham chiếu 8.0s.")

    def _synth_v2_preview(txt: str, ref_codes, r_txt: str):
        tts2 = engine.ensure_v2()
        v2_t, v2_k, v2_p, v2_rep, v2_max = engine._resolve_v2_params(0.45, 15, 0.85, 1.2)
        tts2._repeat_penalty = v2_rep
        tts2._top_p = v2_p
        return tts2.infer(
            preprocess_text(txt),
            ref_codes=ref_codes,
            ref_text=r_txt,
            max_chars=v2_max,
            temperature=v2_t,
            top_k=v2_k,
        )

    try:
        push_log("INFO", "CLONE", f"Tạo thử giọng clone [{preview_model}] với câu đại diện ({len(preview_text)} ký tự)...")
        est = 22.0 if preview_model == "vieneu-v2" else 7.0
        result = await asyncio.to_thread(
            voice_mgr.create_clone_preview,
            uploaded_file_path=tmp_path,
            draft_id=draft_id or None,
            preview_text=preprocess_text(preview_text),
            preview_model=preview_model,
            ref_text=ref_text,
            denoise=denoise,
            encode_v2_fn=engine._encode_v2_wav,
            synth_v2_fn=_synth_v2_preview,
            progress_cb=lambda pct, stage, detail="": progress_store.update(
                job_id, pct, stage, detail, estimated_sec=est, done=(pct >= 100)
            ),
        )
        push_log("OK", "CLONE", f"Đã tạo bản nghe thử '{result['draft_id']}' ({result['preview_duration_sec']}s) — chờ xác nhận lưu.")
        return {"status": "preview_ready", "draft": result}
    except ValueError as exc:
        progress_store.update(job_id, 100, "Dữ liệu không hợp lệ", str(exc), done=True, error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        progress_store.update(job_id, 100, "Lỗi tạo thử giọng clone", str(exc), done=True, error=str(exc))
        push_log("ERROR", "CLONE", f"Lỗi tạo thử giọng clone: {exc}")
        raise HTTPException(status_code=500, detail=f"Lỗi tạo thử giọng clone: {exc}") from exc
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


@app.get("/v1/voices/drafts/{draft_id}/audio")
async def get_draft_preview_audio(draft_id: str):
    _engine_ref, voice_mgr = _require_ready()
    p = voice_mgr.get_draft_preview_path(draft_id)
    if p is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy bản nghe thử.")
    return FileResponse(str(p), media_type="audio/wav", headers={"Cache-Control": "no-store"})


@app.post("/v1/voices/commit")
def commit_clone_voice(req: CommitCloneRequest):
    """Xác nhận lưu bản giọng clone đã nghe thử ưng ý vào Thư viện (giữ âm thanh clone + thông số kỹ thuật)."""
    engine, voice_mgr = _require_ready()
    v2_encoder = engine._encode_v2_wav if engine.has_v2_local_cache() else None
    if req.ref_text.strip() and v2_encoder is None:
        push_log("WARN", "CLONE", "Model VieNeu v2 (w2v-bert-2.0) chưa tải đủ vào .hf_cache — lưu trước vector v3 để không làm treo trình duyệt.")
    try:
        meta = voice_mgr.commit_clone_draft(
            req.draft_id,
            name=req.name,
            description=req.description,
            ref_text=req.ref_text,
            consent_confirmed=req.consent_confirmed,
            encode_v2_fn=v2_encoder,
            progress_cb=lambda pct, stage, detail="": progress_store.update(
                req.job_id, pct, stage, detail, estimated_sec=6.0, done=(pct >= 100)
            ),
        )
        engine.bump_voice_revision()
        push_log("OK", "CLONE", f"Đã lưu giọng clone '{meta['name']}' vào Thư viện (v3+v2={'có' if meta.get('v2_cached') else 'v3'}).")
        return {"status": "committed", "voice": meta}
    except ValueError as exc:
        progress_store.update(req.job_id, 100, "Từ chối lưu giọng", str(exc), done=True, error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        progress_store.update(req.job_id, 100, "Lỗi lưu giọng vào thư viện", str(exc), done=True, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Lỗi lưu giọng: {exc}") from exc


@app.delete("/v1/voices/drafts/{draft_id}")
async def delete_draft_preview(draft_id: str):
    _engine_ref, voice_mgr = _require_ready()
    voice_mgr.discard_clone_draft(draft_id)
    return {"status": "discarded", "draft_id": draft_id}


@app.post("/v1/voices")
async def upload_voice(
    name: str = Form(..., description="Tên gợi nhớ cho giọng clone"),
    description: str = Form(default=""),
    ref_text: str = Form(default=""),
    denoise: bool = Form(default=False),
    consent_confirmed: bool = Form(default=False),
    job_id: str = Form(default=""),
    file: UploadFile = File(...),
):
    engine, voice_mgr = _require_ready()
    suffix = Path(file.filename or "sample.wav").suffix or ".wav"
    progress_store.update(job_id, 5, "Đang tải tệp âm thanh mẫu lên server...", file.filename or "", estimated_sec=8.0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        push_log("INFO", "CLONE", f"Bắt đầu đăng ký giọng clone '{name}' (file={file.filename}, denoise={denoise})...")
        meta = await asyncio.to_thread(
            voice_mgr.register_voice,
            name=name,
            uploaded_file_path=tmp_path,
            description=description,
            ref_text=ref_text,
            denoise=denoise,
            consent_confirmed=consent_confirmed,
            progress_cb=lambda pct, stage, detail="": progress_store.update(
                job_id, pct, stage, detail, estimated_sec=8.0, done=(pct >= 100)
            ),
        )
        engine.bump_voice_revision()
        q = meta.get("quality") or {}
        q_note = f", chất lượng={q.get('label', 'OK')} (SNR {q.get('snr_db', 0)} dB, cắt lặng {q.get('trimmed_silence_ms', 0)}ms)" if q else ""
        push_log("OK", "CLONE", f"Đã lưu giọng clone '{meta['name']}' ({meta['duration_sec']}s, ref_text={'có' if meta['has_ref_text'] else 'trống'}{q_note}).")
        return {"status": "registered", "voice": meta}
    except ValueError as exc:
        progress_store.update(job_id, 100, "Từ chối tạo giọng clone", str(exc), done=True, error=str(exc))
        push_log("WARN", "CLONE", f"Từ chối đăng ký giọng '{name}': {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        progress_store.update(job_id, 100, "Lỗi xử lý giọng mẫu", str(exc), done=True, error=str(exc))
        push_log("ERROR", "CLONE", f"Lỗi xử lý giọng mẫu '{name}': {exc}")
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
    return {"chunks": chunks, "gaps": gaps, "count": len(chunks), "normalized_text": preprocess_text(req.input)}


@app.post("/v1/text/auto-prosody")
async def auto_prosody(req: AutoProsodyRequest):
    """Phân tích văn bản tiếng Việt 1-click để tự chèn ngắt hơi, dấu phẩy và thẻ biểu cảm tự nhiên."""
    try:
        res = analyze_and_annotate_prosody(req.input, style=req.style)
        push_log("OK", "PROSODY", f"AI tự động ngắt nghỉ: {res['summary']}")
        return res
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ── Từ điển phát âm tùy chỉnh (Lexicon) ──────────────────────────────────────

@app.get("/v1/lexicon")
async def get_lexicon():
    return {"rules": lexicon_manager.list_rules(), "starter_count": len(STARTER_TEMPLATES)}


@app.post("/v1/lexicon")
async def add_lexicon_rule(req: LexiconRuleRequest):
    try:
        rule = lexicon_manager.add_rule(
            pattern=req.pattern,
            replacement=req.replacement,
            whole_word=req.whole_word,
            case_sensitive=req.case_sensitive,
            enabled=req.enabled,
            note=req.note,
        )
        push_log("OK", "LEXICON", f"Thêm quy tắc phát âm: '{rule['pattern']}' → '{rule['replacement']}'.")
        return {"status": "created", "rule": rule}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/lexicon/starter")
async def install_starter_lexicon():
    added = lexicon_manager.install_starter_presets()
    push_log("OK", "LEXICON", f"Đã nạp {len(added)} quy tắc phát âm gợi ý phổ biến.")
    return {"status": "ok", "added": len(added), "rules": lexicon_manager.list_rules()}


@app.post("/v1/lexicon/test")
async def test_lexicon(req: PreviewTextRequest):
    return {"before": req.input, "after": preprocess_text(req.input)}


@app.patch("/v1/lexicon/{rule_id}")
async def patch_lexicon_rule(rule_id: str, req: LexiconRulePatch):
    try:
        rule = lexicon_manager.update_rule(rule_id, **req.model_dump())
        return {"status": "updated", "rule": rule}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/v1/lexicon/{rule_id}")
async def delete_lexicon_rule(rule_id: str):
    if not lexicon_manager.delete_rule(rule_id):
        raise HTTPException(status_code=404, detail="Không tìm thấy quy tắc phát âm.")
    return {"status": "deleted", "id": rule_id}


# ── Thư viện nhạc nền BGM (cho Auto-Ducking) ─────────────────────────────────

@app.get("/v1/bgm")
async def list_bgm():
    return {"tracks": _bgm_mgr.list_tracks()}


@app.post("/v1/bgm")
async def upload_bgm(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        track = _bgm_mgr.save_track(file.filename or "bgm.mp3", raw)
        push_log("OK", "BGM", f"Đã tải lên nhạc nền '{track['name']}' ({track['size_kb']} KB).")
        return {"status": "uploaded", "track": track}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v1/bgm/{filename}")
async def get_bgm_file(filename: str):
    p = _bgm_mgr.get_path(filename)
    if p is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy file nhạc nền.")
    return FileResponse(str(p))


@app.delete("/v1/bgm/{filename}")
async def delete_bgm(filename: str):
    if not _bgm_mgr.delete_track(filename):
        raise HTTPException(status_code=404, detail="Không tìm thấy file nhạc nền.")
    return {"status": "deleted", "name": filename}


# ── Tổng hợp giọng nói (Tương thích OpenAI + FX + Đa định dạng) ─────────────

@app.post("/v1/audio/speech")
def create_speech(req: SpeechRequest):
    engine, voice_mgr = _require_ready()
    eff_model, eff_voice = _resolve_openai_compat(req.model, req.voice, voice_mgr)
    push_log(
        "INFO",
        "TTS",
        f"Yêu cầu tạo giọng [{eff_model}] voice='{eff_voice}' ({len(req.input)} ký tự, fmt={req.response_format}, speed={req.speed})...",
    )
    progress_store.update(req.job_id, 5, "Đang tiếp nhận yêu cầu tạo giọng...", f"{len(req.input)} ký tự")
    cancel_ev = progress_store.get_cancel_event(req.job_id)

    try:
        if req.response_format.lower() == "pcm":
            sr, gen = engine.stream_pcm(
                req.input,
                model=eff_model,
                voice=eff_voice,
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
                    "X-TTS-Model": eff_model,
                },
            )

        bgm_path = _bgm_mgr.get_path(req.bgm_name) if req.bgm_name else None
        out_bytes, stats = engine.synthesize_wav(
            req.input,
            model=eff_model,
            voice=eff_voice,
            speed=req.speed,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            use_ref_codes=req.use_ref_codes,
            ref_text_override=req.ref_text,
            voice_manager=voice_mgr,
            response_format=req.response_format,
            normalize_lufs=req.normalize_lufs,
            voice_enhance=req.voice_enhance,
            bgm_path=bgm_path,
            bgm_volume=req.bgm_volume,
            progress_cb=lambda pct, stage, detail="", est=0.0: progress_store.update(
                req.job_id, pct, stage, detail, estimated_sec=est
            ),
            cancel_event=cancel_ev,
        )

        fx_tags = []
        if req.normalize_lufs:
            fx_tags.append("-14 LUFS")
        if req.voice_enhance:
            fx_tags.append("Làm ấm giọng")
        if bgm_path:
            fx_tags.append(f"BGM: {bgm_path.name}")
        fx_summary = " · ".join(fx_tags)

        progress_store.update(req.job_id, 98, f"Đang lưu tệp {stats['format'].upper()} vào history/...", "")
        history_entry = _history_mgr.save_item(
            out_bytes,
            text=req.input,
            voice=eff_voice,
            model=eff_model,
            duration_sec=stats["audio_duration_sec"],
            sample_rate=stats["sample_rate"],
            inference_sec=stats["inference_sec"],
            rtf=stats["rtf"],
            speed=req.speed,
            temperature=req.temperature,
            fmt=stats["format"],
            fx_summary=fx_summary,
        )
        progress_store.update(req.job_id, 100, "Hoàn tất tạo giọng nói!", history_entry["filename"], done=True)
        push_log(
            "OK",
            "TTS",
            f"Hoàn tất [{eff_model}] voice='{eff_voice}' ({stats['format'].upper()}): dur={stats['audio_duration_sec']}s, infer={stats['inference_sec']}s, RTF={stats['rtf']}, cache={'HIT' if stats['cached'] else 'MISS'} · Đã lưu history/{history_entry['filename']}.",
        )
        return Response(
            content=out_bytes,
            media_type=stats["mime"],
            headers={
                "Content-Disposition": f'inline; filename="{history_entry["filename"]}"',
                "X-Audio-Sample-Rate": str(stats["sample_rate"]),
                "X-Audio-Duration-Sec": str(stats["audio_duration_sec"]),
                "X-Inference-Sec": str(stats["inference_sec"]),
                "X-RTF": str(stats["rtf"]),
                "X-Cached": str(stats["cached"]).lower(),
                "X-TTS-Precision": stats["precision"],
                "X-TTS-Model": eff_model,
                "X-Audio-Format": stats["format"],
                "X-History-Id": history_entry["id"],
                "X-History-Audio-Url": history_entry["audio_url"],
                "X-History-Filename": history_entry["filename"],
            },
        )
    except ValueError as exc:
        progress_store.update(req.job_id, 100, "Tham số không hợp lệ", str(exc), done=True, error=str(exc))
        push_log("WARN", "TTS", f"Tham số không hợp lệ [{eff_model}] voice='{eff_voice}': {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        if cancel_ev and cancel_ev.is_set():
            progress_store.update(req.job_id, 100, "Đã hủy tác vụ", str(exc), done=True, error="cancelled")
            raise HTTPException(status_code=409, detail="Đã hủy tác vụ tạo giọng nói.") from exc
        progress_store.update(req.job_id, 100, "Lỗi tổng hợp giọng nói", str(exc), done=True, error=str(exc))
        push_log("ERROR", "TTS", f"Lỗi tổng hợp [{eff_model}] voice='{eff_voice}': {exc}")
        raise HTTPException(status_code=500, detail=f"Lỗi tổng hợp giọng nói: {exc}") from exc
    except Exception as exc:
        progress_store.update(req.job_id, 100, "Lỗi tổng hợp giọng nói", str(exc), done=True, error=str(exc))
        push_log("ERROR", "TTS", f"Lỗi tổng hợp [{eff_model}] voice='{eff_voice}': {exc}")
        raise HTTPException(status_code=500, detail=f"Lỗi tổng hợp giọng nói: {exc}") from exc


# ── Lịch sử phát lại tức thì (WAV / MP3 / M4A + Đổi nhãn) ─────────────────────

@app.get("/v1/history")
async def list_history():
    return {
        "items": _history_mgr.list_items(),
        "history_dir": str(settings.history_dir),
    }


@app.patch("/v1/history/{item_id}")
async def patch_history_item(item_id: str, req: HistoryPatchRequest):
    try:
        updated = _history_mgr.update_label(item_id, req.label)
        return {"status": "updated", "item": updated}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v1/history/{item_id}/audio")
async def get_history_audio(item_id: str):
    info = _history_mgr.get_audio_info(item_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy file audio lịch sử '{item_id}'.")
    audio_path, mime = info
    return FileResponse(str(audio_path), media_type=mime, filename=audio_path.name)


@app.delete("/v1/history/{item_id}")
async def delete_history_item(item_id: str):
    removed = _history_mgr.delete_item(item_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy bản ghi lịch sử '{item_id}'.")
    push_log("INFO", "HISTORY", f"Đã xóa bản ghi lịch sử '{item_id}'.")
    return {"status": "deleted", "id": item_id}


@app.delete("/v1/history")
async def clear_history():
    count = _history_mgr.clear_all()
    push_log("INFO", "HISTORY", f"Đã xóa toàn bộ {count} bản ghi trong thư mục history/.")
    return {"status": "cleared", "deleted": count}


# ── Đọc sách nói / Tài liệu dài theo hàng đợi ────────────────────────────────

@app.get("/v1/audiobooks")
async def list_audiobooks():
    return {"books": _audiobook_mgr.list_books()}


@app.post("/v1/audiobooks")
async def create_audiobook(
    title: str = Form(default=""),
    raw_text: str = Form(default=""),
    voice: str = Form(default="default"),
    model: str = Form(default="vieneu"),
    speed: float = Form(default=1.0),
    temperature: float = Form(default=0.40),
    top_k: int = Form(default=12),
    top_p: float = Form(default=0.80),
    repetition_penalty: float = Form(default=1.2),
    output_format: str = Form(default="mp3"),
    normalize_lufs: bool = Form(default=True),
    voice_enhance: bool = Form(default=False),
    bgm_name: str = Form(default=""),
    bgm_volume: float = Form(default=0.12),
    file: Optional[UploadFile] = File(default=None),
):
    _require_ready()
    extracted = raw_text.strip()
    doc_title = title.strip()

    if file is not None and file.filename:
        suffix = Path(file.filename).suffix or ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)
        try:
            extracted = extract_text_from_document(tmp_path, file.filename)
            if not doc_title:
                doc_title = Path(file.filename).stem
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Không đọc được file tài liệu: {exc}") from exc
        finally:
            tmp_path.unlink(missing_ok=True)

    if not extracted:
        raise HTTPException(status_code=400, detail="Vui lòng tải lên tài liệu (.txt, .docx, .epub, .pdf) hoặc dán văn bản dài.")

    try:
        book = _audiobook_mgr.create_book(
            title=doc_title or "Sách nói mới",
            raw_text=extracted,
            voice=voice,
            model=model,
            speed=speed,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            output_format=output_format,
            normalize_lufs=normalize_lufs,
            voice_enhance=voice_enhance,
            bgm_name=bgm_name,
            bgm_volume=bgm_volume,
        )
        push_log("OK", "AUDIOBOOK", f"Tạo hàng đợi sách nói '{book['title']}' ({book['total_chapters']} chương, định dạng {book['format'].upper()}).")
        return {"status": "created", "book": book}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v1/audiobooks/{book_id}")
async def get_audiobook(book_id: str):
    book = _audiobook_mgr.get_book(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Không tìm thấy sách nói.")
    return {"book": book}


@app.post("/v1/audiobooks/{book_id}/pause")
async def pause_audiobook(book_id: str):
    try:
        return {"book": _audiobook_mgr.pause_book(book_id)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/v1/audiobooks/{book_id}/resume")
async def resume_audiobook(book_id: str):
    try:
        return {"book": _audiobook_mgr.resume_book(book_id)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/v1/audiobooks/{book_id}/cancel")
async def cancel_audiobook(book_id: str):
    try:
        return {"book": _audiobook_mgr.cancel_book(book_id)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v1/audiobooks/{book_id}/chapters/{chapter_idx}/audio")
async def get_audiobook_chapter_audio(book_id: str, chapter_idx: int):
    info = _audiobook_mgr.get_chapter_audio(book_id, chapter_idx)
    if info is None:
        raise HTTPException(status_code=404, detail="Chương này chưa được tạo hoặc không tồn tại.")
    p, mime = info
    return FileResponse(str(p), media_type=mime, filename=p.name)


@app.get("/v1/audiobooks/{book_id}/zip")
async def download_audiobook_zip(book_id: str):
    try:
        zip_bytes = _audiobook_mgr.build_zip_bytes(book_id)
        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{book_id}.zip"'},
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/v1/audiobooks/{book_id}")
async def delete_audiobook(book_id: str):
    if not _audiobook_mgr.delete_book(book_id):
        raise HTTPException(status_code=404, detail="Không tìm thấy sách nói.")
    return {"status": "deleted", "id": book_id}


# ── Nhận diện lời thoại mẫu tiếng Việt (PhoWhisper-base) ─────────────────────

@app.post("/v1/audio/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    from .stt_engine import STTEngine

    suffix = Path(file.filename or "sample.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        push_log("INFO", "STT", f"Whisper STT bắt đầu nhận diện file '{file.filename}'...")
        stt = STTEngine.get_instance()
        text = await asyncio.to_thread(stt.transcribe, tmp_path)
        push_log("OK", "STT", f"Whisper STT nhận diện xong ({len(text)} ký tự): \"{text[:60]}{'...' if len(text) > 60 else ''}\"")
        return {"status": "ok", "text": text}
    except Exception as exc:
        push_log("ERROR", "STT", f"Lỗi Whisper STT cho '{file.filename}': {exc}")
        raise HTTPException(status_code=500, detail=f"Lỗi nhận diện âm thanh: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)
