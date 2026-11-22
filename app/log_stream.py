"""In-memory thread-safe ring buffer cho bảng Terminal Log thời gian thực."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Deque, Dict, List


class _LogStore:
    def __init__(self, maxlen: int = 500) -> None:
        self._lock = threading.Lock()
        self._entries: Deque[Dict[str, object]] = deque(maxlen=maxlen)
        self._seq = 0

    def push(self, level: str, source: str, message: str) -> Dict[str, object]:
        lvl = (level or "INFO").upper()
        src = (source or "SERVER").upper()
        msg = str(message or "").strip()
        with self._lock:
            self._seq += 1
            entry = {
                "id": self._seq,
                "ts": time.strftime("%H:%M:%S"),
                "level": lvl,
                "source": src,
                "origin": "BE",
                "message": msg,
            }
            self._entries.append(entry)
            return entry

    def get_since(self, since_id: int = 0) -> Dict[str, object]:
        with self._lock:
            items: List[Dict[str, object]] = [
                e for e in self._entries if int(e["id"]) > since_id
            ]
            return {"logs": items, "latest_id": self._seq}

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


log_store = _LogStore(maxlen=500)


def push_log(level: str, source: str, message: str) -> Dict[str, object]:
    return log_store.push(level, source, message)


class _PythonLogBridge(logging.Handler):
    """Chuyển tiếp các cảnh báo / lỗi từ Python logging & uvicorn vào Terminal Log."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if record.levelno < logging.WARNING:
                return
            msg = self.format(record)
            if "unauthenticated requests to the HF Hub" in msg:
                return
            lvl = "ERROR" if record.levelno >= logging.ERROR else "WARN"
            log_store.push(lvl, record.name.split(".")[0][:12], msg)
        except Exception:
            pass


_bridge_attached = False


def attach_python_logging() -> None:
    global _bridge_attached
    if _bridge_attached:
        return
    handler = _PythonLogBridge()
    handler.setLevel(logging.WARNING)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger("uvicorn.error").addHandler(handler)
    _bridge_attached = True
