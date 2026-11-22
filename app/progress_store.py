"""Theo dõi tiến trình (%) & trạng thái hủy thời gian thực cho các tác vụ TTS / Clone."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Optional


class _ProgressStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._cancel_events: Dict[str, threading.Event] = {}

    def _prune_unlocked(self, now: float) -> None:
        stale = [k for k, v in self._jobs.items() if now - float(v.get("updated_at", 0.0)) > 600.0]
        for k in stale:
            self._jobs.pop(k, None)
            self._cancel_events.pop(k, None)

    def get_cancel_event(self, job_id: Optional[str]) -> Optional[threading.Event]:
        if not job_id:
            return None
        with self._lock:
            ev = self._cancel_events.get(job_id)
            if ev is None:
                ev = threading.Event()
                self._cancel_events[job_id] = ev
            return ev

    def cancel(self, job_id: str) -> bool:
        if not job_id:
            return False
        with self._lock:
            ev = self._cancel_events.get(job_id)
            if ev is None:
                ev = threading.Event()
                self._cancel_events[job_id] = ev
            ev.set()
            if job_id in self._jobs:
                self._jobs[job_id]["stage"] = "Đang hủy theo yêu cầu..."
                self._jobs[job_id]["cancelled"] = True
                self._jobs[job_id]["updated_at"] = time.time()
            return True

    def is_cancelled(self, job_id: Optional[str]) -> bool:
        if not job_id:
            return False
        with self._lock:
            ev = self._cancel_events.get(job_id)
            return bool(ev and ev.is_set())

    def update(
        self,
        job_id: Optional[str],
        percent: int,
        stage: str,
        detail: str = "",
        *,
        estimated_sec: Optional[float] = None,
        done: bool = False,
        error: str = "",
    ) -> None:
        if not job_id:
            return
        now = time.time()
        clamped = max(0, min(100, int(percent)))
        with self._lock:
            self._prune_unlocked(now)
            prev = self._jobs.get(job_id, {})
            est = estimated_sec if estimated_sec is not None else prev.get("estimated_sec", 0.0)
            ev = self._cancel_events.get(job_id)
            self._jobs[job_id] = {
                "job_id": job_id,
                "percent": clamped,
                "stage": stage,
                "detail": detail,
                "estimated_sec": round(float(est or 0.0), 1),
                "done": bool(done),
                "cancelled": bool(ev and ev.is_set()),
                "error": error,
                "updated_at": now,
            }

    def get(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            item = self._jobs.get(job_id)
            if not item:
                return {
                    "job_id": job_id,
                    "percent": 0,
                    "stage": "Đang khởi tạo tác vụ...",
                    "detail": "",
                    "estimated_sec": 0.0,
                    "done": False,
                    "cancelled": False,
                    "error": "",
                }
            return dict(item)


progress_store = _ProgressStore()
ProgressCallback = Optional[Callable[[int, str, str], None]]
