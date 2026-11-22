"""Đọc sách nói / tài liệu dài (.txt, .docx, .epub, .pdf) theo hàng đợi nền & xuất ZIP."""

from __future__ import annotations

import io
import re
import shutil
import threading
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET

import orjson

from .audio_postprocess import FORMAT_META, BgmManager
from .normalizer import preprocess_text
from .tts_engine import TTSEngine
from .voice_manager import VoiceManager

_CHAPTER_HEADING_RE = re.compile(
    r"^\s*(?:chương|phần|mục|hồi|bài|chapter|part)\s+[\dIVXLCDM]+[^\n]{0,80}$",
    re.IGNORECASE | re.MULTILINE,
)


def extract_text_from_document(file_path: Path, filename: str) -> str:
    """Trích xuất văn bản sạch từ .txt, .docx, .epub hoặc .pdf."""
    ext = Path(filename or file_path.name).suffix.lower()

    if ext in (".txt", ".md", ".csv"):
        raw = file_path.read_bytes()
        for enc in ("utf-8-sig", "utf-8", "utf-16", "cp1258", "latin-1"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace")

    if ext == ".docx":
        paragraphs: List[str] = []
        with zipfile.ZipFile(file_path, "r") as zf:
            xml_bytes = zf.read("word/document.xml")
            root = ET.fromstring(xml_bytes)
            for p_elem in root.iter():
                if p_elem.tag.endswith("}p"):
                    texts = [t.text for t in p_elem.iter() if t.tag.endswith("}t") and t.text]
                    line = "".join(texts).strip()
                    if line:
                        paragraphs.append(line)
        return "\n\n".join(paragraphs)

    if ext == ".epub":
        from lxml import html as lxml_html

        sections: List[str] = []
        with zipfile.ZipFile(file_path, "r") as zf:
            names = zf.namelist()
            html_files: List[str] = []
            opf_candidates = [n for n in names if n.lower().endswith(".opf")]
            if opf_candidates:
                try:
                    opf_path = opf_candidates[0]
                    opf_dir = str(Path(opf_path).parent).replace("\\", "/")
                    opf_dir = "" if opf_dir == "." else opf_dir + "/"
                    opf_root = ET.fromstring(zf.read(opf_path))
                    manifest = {}
                    for item in opf_root.iter():
                        if item.tag.endswith("}item"):
                            iid = item.attrib.get("id")
                            href = item.attrib.get("href")
                            if iid and href:
                                manifest[iid] = opf_dir + href.lstrip("/")
                    for itemref in opf_root.iter():
                        if itemref.tag.endswith("}itemref"):
                            idref = itemref.attrib.get("idref")
                            if idref and idref in manifest and manifest[idref] in names:
                                html_files.append(manifest[idref])
                except Exception:
                    html_files = []

            if not html_files:
                html_files = sorted(
                    n for n in names if n.lower().endswith((".xhtml", ".html", ".htm"))
                )

            for hf in html_files:
                try:
                    doc = lxml_html.fromstring(zf.read(hf))
                    txt = doc.text_content()
                    cleaned = "\n".join(ln.strip() for ln in txt.splitlines() if ln.strip())
                    if len(cleaned) >= 20:
                        sections.append(cleaned)
                except Exception:
                    continue
        return "\n\n".join(sections)

    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        pages: List[str] = []
        for page in reader.pages:
            txt = (page.extract_text() or "").strip()
            if txt:
                pages.append(txt)
        return "\n\n".join(pages)

    raise ValueError(f"Chưa hỗ trợ định dạng '{ext}'. Hỗ trợ: .txt, .docx, .epub, .pdf.")


def split_document_to_chapters(text: str, max_chapter_chars: int = 1600) -> List[Dict[str, str]]:
    """Chia văn bản dài thành danh sách chương/phần hợp lý để đọc theo hàng đợi."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    matches = list(_CHAPTER_HEADING_RE.finditer(normalized))
    raw_sections: List[Dict[str, str]] = []

    if len(matches) >= 2:
        if matches[0].start() > 40:
            intro = normalized[: matches[0].start()].strip()
            if intro:
                raw_sections.append({"title": "Phần mở đầu", "text": intro})
        for i, m in enumerate(matches):
            end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(normalized)
            title = m.group(0).strip()
            body = normalized[m.end() : end_idx].strip()
            full_text = f"{title}.\n{body}" if body else title
            if len(full_text) > 5:
                raw_sections.append({"title": title[:80], "text": full_text})
    else:
        raw_sections.append({"title": "Phần 1", "text": normalized})

    # Tách nhỏ các chương quá dài (> max_chapter_chars) theo ranh giới đoạn văn để tạo file nhanh & mượt
    chapters: List[Dict[str, str]] = []
    for sec in raw_sections:
        sec_text = sec["text"]
        if len(sec_text) <= max_chapter_chars:
            chapters.append(sec)
            continue

        paras = [p.strip() for p in re.split(r"\n{2,}", sec_text) if p.strip()]
        buf = ""
        part_idx = 1
        for p in paras:
            if len(p) > max_chapter_chars:
                if buf:
                    chapters.append({"title": f"{sec['title']} (Phần {part_idx})", "text": buf.strip()})
                    buf = ""
                    part_idx += 1
                # Chia thêm theo câu
                sentences = re.split(r"(?<=[.!?])\s+", p)
                for s in sentences:
                    if len(buf) + len(s) + 1 > max_chapter_chars and buf:
                        chapters.append({"title": f"{sec['title']} (Phần {part_idx})", "text": buf.strip()})
                        buf = s
                        part_idx += 1
                    else:
                        buf = f"{buf} {s}".strip()
                continue

            if len(buf) + len(p) + 2 > max_chapter_chars and buf:
                chapters.append({"title": f"{sec['title']} (Phần {part_idx})", "text": buf.strip()})
                buf = p
                part_idx += 1
            else:
                buf = f"{buf}\n\n{p}".strip()
        if buf:
            suffix = f" (Phần {part_idx})" if part_idx > 1 else ""
            chapters.append({"title": f"{sec['title']}{suffix}", "text": buf.strip()})

    return chapters


class AudiobookManager:
    """Quản lý hàng đợi đọc sách nói dài (nhiều chương), hỗ trợ Tạm dừng / Tiếp tục / Hủy / Tải ZIP."""

    def __init__(self, audiobooks_dir: Path, bgm_mgr: BgmManager) -> None:
        self.audiobooks_dir = audiobooks_dir
        self.audiobooks_dir.mkdir(parents=True, exist_ok=True)
        self.bgm_mgr = bgm_mgr
        self.index_path = self.audiobooks_dir / "index.json"
        self._lock = threading.Lock()
        self._pause_flags: Dict[str, threading.Event] = {}
        self._cancel_flags: Dict[str, threading.Event] = {}
        self._books: List[Dict[str, Any]] = []
        self._engine: Optional[TTSEngine] = None
        self._voice_mgr: Optional[VoiceManager] = None
        self._load_unlocked()

    def attach_runtime(self, engine: TTSEngine, voice_mgr: VoiceManager) -> None:
        self._engine = engine
        self._voice_mgr = voice_mgr

    def _load_unlocked(self) -> None:
        if not self.index_path.is_file():
            self._books = []
            return
        try:
            data = orjson.loads(self.index_path.read_bytes())
            if isinstance(data, list):
                for b in data:
                    if b.get("status") in ("running", "queued"):
                        b["status"] = "paused"
                    for ch in b.get("chapters", []):
                        if ch.get("status") == "running":
                            ch["status"] = "pending"
                self._books = data
        except Exception:
            self._books = []

    def _save_unlocked(self) -> None:
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_bytes(orjson.dumps(self._books, option=orjson.OPT_INDENT_2))
        tmp.replace(self.index_path)

    def list_books(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [orjson.loads(orjson.dumps(b)) for b in self._books]

    def get_book(self, book_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            for b in self._books:
                if b.get("id") == book_id:
                    return orjson.loads(orjson.dumps(b))
        return None

    def create_book(
        self,
        *,
        title: str,
        raw_text: str,
        voice: str,
        model: str,
        speed: float = 1.0,
        temperature: float = 0.40,
        top_k: int = 12,
        top_p: float = 0.80,
        repetition_penalty: float = 1.2,
        output_format: str = "mp3",
        normalize_lufs: bool = True,
        voice_enhance: bool = False,
        bgm_name: str = "",
        bgm_volume: float = 0.12,
        max_chapter_chars: int = 1500,
    ) -> Dict[str, Any]:
        chapters_raw = split_document_to_chapters(raw_text, max_chapter_chars=max_chapter_chars)
        if not chapters_raw:
            raise ValueError("Không trích xuất được nội dung văn bản hợp lệ từ tài liệu.")

        book_id = f"book_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:5]}"
        book_dir = self.audiobooks_dir / book_id
        book_dir.mkdir(parents=True, exist_ok=True)
        _, ext = FORMAT_META.get((output_format or "mp3").lower(), ("audio/mpeg", "mp3"))

        chapters = []
        for idx, ch in enumerate(chapters_raw):
            chapters.append(
                {
                    "index": idx,
                    "title": ch["title"],
                    "text": ch["text"],
                    "chars": len(ch["text"]),
                    "status": "pending",
                    "filename": f"chuong_{idx + 1:03d}.{ext}",
                    "duration_sec": 0.0,
                    "error": "",
                }
            )

        book: Dict[str, Any] = {
            "id": book_id,
            "title": (title or "Sách nói mới").strip()[:100],
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "voice": voice or "default",
            "model": model or "vieneu",
            "speed": round(float(speed), 2),
            "temperature": round(float(temperature), 2),
            "top_k": int(top_k),
            "top_p": round(float(top_p), 2),
            "repetition_penalty": round(float(repetition_penalty), 2),
            "format": ext,
            "normalize_lufs": bool(normalize_lufs),
            "voice_enhance": bool(voice_enhance),
            "bgm_name": (bgm_name or "").strip(),
            "bgm_volume": round(float(bgm_volume), 2),
            "status": "running",
            "total_chapters": len(chapters),
            "completed_chapters": 0,
            "total_chars": sum(c["chars"] for c in chapters),
            "chapters": chapters,
        }

        with self._lock:
            self._books.insert(0, book)
            self._pause_flags[book_id] = threading.Event()
            self._cancel_flags[book_id] = threading.Event()
            self._save_unlocked()

        t = threading.Thread(target=self._worker_loop, args=(book_id,), daemon=True)
        t.start()
        return self.get_book(book_id) or book

    def pause_book(self, book_id: str) -> Dict[str, Any]:
        with self._lock:
            for b in self._books:
                if b["id"] == book_id:
                    ev = self._pause_flags.setdefault(book_id, threading.Event())
                    ev.set()
                    if b["status"] in ("running", "queued"):
                        b["status"] = "paused"
                    self._save_unlocked()
                    return orjson.loads(orjson.dumps(b))
        raise KeyError(f"Không tìm thấy sách nói '{book_id}'.")

    def resume_book(self, book_id: str) -> Dict[str, Any]:
        need_spawn = False
        with self._lock:
            for b in self._books:
                if b["id"] == book_id:
                    self._pause_flags.setdefault(book_id, threading.Event()).clear()
                    self._cancel_flags.setdefault(book_id, threading.Event()).clear()
                    if b["status"] in ("paused", "cancelled", "error"):
                        b["status"] = "running"
                        need_spawn = True
                    self._save_unlocked()
                    if need_spawn:
                        t = threading.Thread(target=self._worker_loop, args=(book_id,), daemon=True)
                        t.start()
                    return orjson.loads(orjson.dumps(b))
        raise KeyError(f"Không tìm thấy sách nói '{book_id}'.")

    def cancel_book(self, book_id: str) -> Dict[str, Any]:
        with self._lock:
            for b in self._books:
                if b["id"] == book_id:
                    self._cancel_flags.setdefault(book_id, threading.Event()).set()
                    b["status"] = "cancelled"
                    self._save_unlocked()
                    return orjson.loads(orjson.dumps(b))
        raise KeyError(f"Không tìm thấy sách nói '{book_id}'.")

    def delete_book(self, book_id: str) -> bool:
        with self._lock:
            before = len(self._books)
            self._cancel_flags.setdefault(book_id, threading.Event()).set()
            self._books = [b for b in self._books if b["id"] != book_id]
            if len(self._books) == before:
                return False
            self._save_unlocked()
        shutil.rmtree(self.audiobooks_dir / book_id, ignore_errors=True)
        return True

    def get_chapter_audio(self, book_id: str, chapter_idx: int) -> Optional[tuple[Path, str]]:
        book = self.get_book(book_id)
        if not book:
            return None
        chapters = book.get("chapters", [])
        if chapter_idx < 0 or chapter_idx >= len(chapters):
            return None
        ch = chapters[chapter_idx]
        p = self.audiobooks_dir / book_id / ch["filename"]
        if not p.is_file():
            return None
        mime, _ = FORMAT_META.get(book.get("format", "mp3"), ("audio/mpeg", "mp3"))
        return p, mime

    def build_zip_bytes(self, book_id: str) -> bytes:
        book = self.get_book(book_id)
        if not book:
            raise KeyError(f"Không tìm thấy sách nói '{book_id}'.")
        buf = io.BytesIO()
        book_dir = self.audiobooks_dir / book_id
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for ch in book.get("chapters", []):
                p = book_dir / ch["filename"]
                if p.is_file():
                    zf.write(p, arcname=ch["filename"])
        return buf.getvalue()

    def _worker_loop(self, book_id: str) -> None:
        if self._engine is None or self._voice_mgr is None:
            return
        book_dir = self.audiobooks_dir / book_id

        while True:
            with self._lock:
                book = next((b for b in self._books if b["id"] == book_id), None)
                if not book:
                    return
                pause_ev = self._pause_flags.setdefault(book_id, threading.Event())
                cancel_ev = self._cancel_flags.setdefault(book_id, threading.Event())
                if cancel_ev.is_set() or book["status"] == "cancelled":
                    book["status"] = "cancelled"
                    self._save_unlocked()
                    return
                if pause_ev.is_set() or book["status"] == "paused":
                    book["status"] = "paused"
                    self._save_unlocked()
                    return

                next_ch = next((c for c in book["chapters"] if c["status"] in ("pending", "error")), None)
                if next_ch is None:
                    book["status"] = "completed"
                    book["completed_chapters"] = sum(1 for c in book["chapters"] if c["status"] == "done")
                    self._save_unlocked()
                    return

                next_ch["status"] = "running"
                next_ch["error"] = ""
                book["status"] = "running"
                self._save_unlocked()
                cfg = dict(book)
                ch_idx = int(next_ch["index"])
                ch_text = str(next_ch["text"])
                ch_filename = str(next_ch["filename"])

            try:
                bgm_path = self.bgm_mgr.get_path(cfg.get("bgm_name", "")) if cfg.get("bgm_name") else None
                audio_bytes, stats = self._engine.synthesize_wav(
                    ch_text,
                    model=cfg.get("model", "vieneu"),
                    voice=cfg.get("voice", "default"),
                    speed=float(cfg.get("speed", 1.0)),
                    temperature=float(cfg.get("temperature", 0.4)),
                    top_k=int(cfg.get("top_k", 12)),
                    top_p=float(cfg.get("top_p", 0.8)),
                    repetition_penalty=float(cfg.get("repetition_penalty", 1.2)),
                    voice_manager=self._voice_mgr,
                    response_format=cfg.get("format", "mp3"),
                    normalize_lufs=bool(cfg.get("normalize_lufs", True)),
                    voice_enhance=bool(cfg.get("voice_enhance", False)),
                    bgm_path=bgm_path,
                    bgm_volume=float(cfg.get("bgm_volume", 0.12)),
                    cancel_event=cancel_ev,
                )
                (book_dir / ch_filename).write_bytes(audio_bytes)
                with self._lock:
                    book = next((b for b in self._books if b["id"] == book_id), None)
                    if book and ch_idx < len(book["chapters"]):
                        book["chapters"][ch_idx]["status"] = "done"
                        book["chapters"][ch_idx]["duration_sec"] = float(stats.get("audio_duration_sec", 0.0))
                        book["completed_chapters"] = sum(1 for c in book["chapters"] if c["status"] == "done")
                        self._save_unlocked()
            except Exception as exc:
                with self._lock:
                    book = next((b for b in self._books if b["id"] == book_id), None)
                    if book and ch_idx < len(book["chapters"]):
                        if cancel_ev.is_set():
                            book["chapters"][ch_idx]["status"] = "pending"
                            book["status"] = "cancelled"
                        else:
                            book["chapters"][ch_idx]["status"] = "error"
                            book["chapters"][ch_idx]["error"] = str(exc)
                        self._save_unlocked()
                if cancel_ev.is_set():
                    return
                time.sleep(0.5)
