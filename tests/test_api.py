"""Kiểm thử tự động FastAPI cho upload giọng mẫu và tổng hợp song song v3 + v2."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.main import app  # noqa: E402

SAMPLE_DIR = (
    BASE_DIR / ".venv" / "Lib" / "site-packages" / "vieneu" / "assets" / "samples"
)
SAMPLE_WAV = SAMPLE_DIR / "Ngọc (nữ miền Bắc).wav"
SAMPLE_TXT = (SAMPLE_DIR / "Ngọc (nữ miền Bắc).txt").read_text(encoding="utf-8").strip()


def run_tests() -> None:
    with TestClient(app) as client:
        # 1. Health check
        r = client.get("/health")
        assert r.status_code == 200, r.text
        health = r.json()
        assert health["status"] == "ok"
        assert "vieneu" in health["models"] and "vieneu-v2" in health["models"]
        print("[PASS] /health:", health)

        # 2. Preview chunks
        long_text = (
            "Hôm nay là ngày 17 tháng 11 năm 2026, lúc 9 giờ 45 phút. "
            "Đơn hàng trị giá 1.250.000 đồng, giảm 15% còn 1.062.500 đồng. "
            "Anh ấy sống tại TP. Hồ Chí Minh và mở file PDF qua Wi-Fi."
        )
        r = client.post("/v1/text/preview-chunks", json={"input": long_text})
        assert r.status_code == 200, r.text
        assert r.json()["count"] >= 1
        print("[PASS] /v1/text/preview-chunks:", r.json()["chunks"])

        # 3. Upload voice without consent -> 400
        with SAMPLE_WAV.open("rb") as f:
            r = client.post(
                "/v1/voices",
                data={"name": "test_no_consent", "consent_confirmed": "false"},
                files={"file": ("ngoc.wav", f, "audio/wav")},
            )
        assert r.status_code == 400, r.text
        print("[PASS] Reject upload without consent")

        # 4. Upload voice with consent (initially without ref_text) -> 200
        with SAMPLE_WAV.open("rb") as f:
            r = client.post(
                "/v1/voices",
                data={
                    "name": "test_ngoc_clone",
                    "description": "Test voice clone",
                    "denoise": "false",
                    "consent_confirmed": "true",
                },
                files={"file": ("ngoc.wav", f, "audio/wav")},
            )
        assert r.status_code == 200, r.text
        assert r.json()["voice"]["has_ref_text"] is False
        print("[PASS] POST /v1/voices:", r.json())

        # 5. Calling v2 without ref_text -> clean 400
        r_missing = client.post(
            "/v1/audio/speech",
            json={
                "model": "vieneu-v2",
                "voice": "test_ngoc_clone",
                "input": "Xin chào.",
            },
        )
        assert r_missing.status_code == 400, r_missing.text
        print("[PASS] v2 rejects voice missing ref_text with 400")

        # 6. PATCH /v1/voices/{id} to add ref_text
        r_patch = client.patch(
            "/v1/voices/test_ngoc_clone",
            json={"ref_text": SAMPLE_TXT},
        )
        assert r_patch.status_code == 200, r_patch.text
        assert r_patch.json()["voice"]["has_ref_text"] is True
        print("[PASS] PATCH /v1/voices/test_ngoc_clone")

        # 7. Synthesize WAV with v3 Turbo
        r_v3 = client.post(
            "/v1/audio/speech",
            json={
                "model": "vieneu",
                "voice": "test_ngoc_clone",
                "input": "Dạ, em nghe rõ ạ. Xong rồi, anh kiểm tra thử đi ạ.",
                "speed": 1.0,
                "temperature": 0.45,
                "top_k": 15,
                "top_p": 0.85,
                "repetition_penalty": 1.2,
                "use_ref_codes": True,
                "response_format": "wav",
            },
        )
        assert r_v3.status_code == 200, r_v3.text
        assert r_v3.content[:4] == b"RIFF"
        assert r_v3.headers.get("X-Audio-Sample-Rate") == "48000"
        print(
            "[PASS] POST /v1/audio/speech (vieneu v3):",
            f"dur={r_v3.headers.get('X-Audio-Duration-Sec')}s, rtf={r_v3.headers.get('X-RTF')}",
        )

        # 8. Cache hit on identical v3 request
        r_v3_cached = client.post(
            "/v1/audio/speech",
            json={
                "model": "vieneu",
                "voice": "test_ngoc_clone",
                "input": "Dạ, em nghe rõ ạ. Xong rồi, anh kiểm tra thử đi ạ.",
                "speed": 1.0,
                "temperature": 0.45,
                "top_k": 15,
                "top_p": 0.85,
                "repetition_penalty": 1.2,
                "use_ref_codes": True,
                "response_format": "wav",
            },
        )
        assert r_v3_cached.status_code == 200
        assert r_v3_cached.headers.get("X-Cached") == "true"
        print("[PASS] v3 cache hit verified")

        # 9. Synthesize WAV with v2 GGUF Q8 + ref_text
        r_v2 = client.post(
            "/v1/audio/speech",
            json={
                "model": "vieneu-v2",
                "voice": "test_ngoc_clone",
                "input": "Xin chào Việt Nam.",
                "temperature": 0.5,
                "top_k": 20,
                "response_format": "wav",
            },
        )
        assert r_v2.status_code == 200, r_v2.text
        assert r_v2.content[:4] == b"RIFF"
        assert r_v2.headers.get("X-Audio-Sample-Rate") == "24000"
        print(
            "[PASS] POST /v1/audio/speech (vieneu-v2):",
            f"dur={r_v2.headers.get('X-Audio-Duration-Sec')}s, rtf={r_v2.headers.get('X-RTF')}",
        )

        # 10. Delete test voice to leave workspace clean
        r_del = client.delete("/v1/voices/test_ngoc_clone")
        assert r_del.status_code == 200, r_del.text
        print("[PASS] DELETE /v1/voices/test_ngoc_clone")


if __name__ == "__main__":
    run_tests()
