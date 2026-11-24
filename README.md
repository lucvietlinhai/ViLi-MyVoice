# ViLi-MyVoice — Clone Giọng Nói Tiếng Việt (Tác giả: linhlv)

> **ViLi-MyVoice (Tác giả: linhlv) — Hệ thống Clone Giọng Nói Tiếng Việt (Vietnamese TTS Voice Cloning) — 100% Miễn Phí, Chạy Hoàn Toàn Local Offline trên Windows.**

Dự án sử dụng công nghệ mô hình tiếng Việt chuyên sâu **VieNeu-TTS**, hỗ trợ chạy song song 2 engine tối ưu:
1. **VieNeu v3 Turbo (ONNX fp32 — 48.000 Hz):** Tốc độ suy luận siêu tốc (RTF ~0.7x trên CPU, nhanh hơn thời gian thực), âm thanh trong trẻo, không phụ thuộc PyTorch khi sinh giọng.
2. **VieNeu v2 GGUF Q8 (24.000 Hz):** Bám sát 100% ngữ điệu và phương ngữ vùng miền (Bắc / Trung / Nam) thông qua cơ chế khóa mã âm thanh kèm câu mẫu (`ref_text`).

---

## 🌟 Điểm nổi bật của dự án

- **Clone giọng tức thì (Zero-shot Cloning):** Chỉ cần tải lên 1 đoạn ghi âm mẫu ngắn từ **3–8 giây** (WAV, MP3, M4A), không cần micro, không cần huấn luyện lại mô hình phức tạp.
- **Giao diện Web Studio cao cấp (Obsidian & Crimson):** Thiết kế theo phong cách Đỏ - Đen sang trọng với đầy đủ thanh điều hướng, bảng chọn giọng, bộ lọc vùng miền, lịch sử tạo giọng (Generation History) và các kịch bản mẫu tiếng Việt chuyên sâu.
- **Chuẩn hóa tiếng Việt chuyên sâu:** Tự động đọc đúng số tiền tệ (`1.250.000 đồng`), ngày tháng (`17/11/2026`), phần trăm (`15%`), công suất (`22.5W`), và bảo toàn từ tiếng Anh / mã hiệu (`PDF`, `Wi-Fi`, `OH-P102`).
- **Khởi động 1-Click tự động (`run.bat`):**
  - Tự động kiểm tra cấu hình thiết bị đối chiếu với tiêu chuẩn tối thiểu (CPU 4C/8T, AVX2, RAM 16GB, 15GB ổ đĩa).
  - Tự động tải và cài đặt Python 3.11, eSpeak NG, FFmpeg và thư viện cần thiết qua `winget` nếu máy chưa có.
  - Tự động mở trình duyệt Web UI tại `http://127.0.0.1:8000/`.

---

## 🖥 Cấu hình phần cứng khuyến nghị (Baseline)

Hệ thống được tối ưu hóa để chạy mượt mà trên CPU thông thường:

| Phần cứng | Tiêu chuẩn tối thiểu khuyến nghị |
| :--- | :--- |
| **Hệ điều hành** | Windows 10/11 64-bit (`x64`) |
| **Vi xử lý (CPU)** | Tối thiểu 4 Nhân / 8 Luồng (Intel Core i5 / Xeon / Ryzen), hỗ trợ tập lệnh **AVX2** |
| **Bộ nhớ RAM** | Tối thiểu **16 GB** (để giữ song song cả v3 Turbo và v2 GGUF trong RAM mà không bị tràn bộ nhớ) |
| **Ổ đĩa trống** | Còn trống $\ge$ **15 GB** (để chứa môi trường Python, mô hình ONNX và GGUF cache) |

---

## 🚀 Hướng dẫn khởi chạy nhanh

### Cách 1: Khởi động 1-Click bằng file Batch (Khuyên dùng)
Nhấp đúp chuột vào file:
```cmd
run.bat
```
- Nếu là máy mới: File batch sẽ tự kiểm tra phần cứng, tải các gói tài nguyên cần thiết và mở trình duyệt.
- Nếu đã cài đặt: Tự động khởi chạy server và vào ngay ứng dụng trong 5–10 giây.

### Cách 2: Khởi chạy thủ công qua Terminal
1. Kích hoạt môi trường ảo:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
2. Khởi chạy server FastAPI:
   ```powershell
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```
3. Truy cập vào trình duyệt: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 📡 API Endpoints

Hệ thống cung cấp RESTful API tương thích OpenAI Audio TTS format:

| Endpoint | Method | Chức năng |
| :--- | :--- | :--- |
| `/health` | `GET` | Kiểm tra tình trạng engine và số lượng giọng |
| `/v1/voices` | `GET` | Liệt kê toàn bộ giọng clone và giọng preset |
| `/v1/voices` | `POST` | Tải lên file audio mẫu để đăng ký giọng clone mới |
| `/v1/voices/{id}/audio` | `GET` | Nghe lại file âm thanh mẫu gốc đã lưu |
| `/v1/voices/{id}` | `PATCH` | Cập nhật lại lời thoại mẫu (`ref_text`) cho v2 |
| `/v1/voices/{id}` | `DELETE` | Xóa giọng clone |
| `/v1/text/preview-chunks` | `POST` | Xem trước cách tách câu và chuẩn hóa văn bản |
| `/v1/audio/speech` | `POST` | Tổng hợp giọng nói từ văn bản (trả về file WAV hoặc stream PCM) |

---

## 📁 Cấu trúc thư mục dự án

```text
ViLi-MyVoice/
├── app/
│   ├── config.py           # Quản lý cấu hình, tham số v3/v2 và thư mục voices
│   ├── main.py             # FastAPI Server & REST API endpoints
│   ├── normalizer.py       # Bộ tách câu & chuẩn hóa tiếng Việt qua sea-g2p
│   ├── tts_engine.py       # Lõi suy luận VieNeu v3 Turbo ONNX & v2 GGUF Q8
│   ├── voice_manager.py    # Quản lý lưu trữ giọng mẫu, trích xuất đặc trưng audio
│   └── static/
│       └── index.html      # Giao diện Web Studio Obsidian & Crimson
├── voices/                 # Thư mục lưu các file audio mẫu và metadata
├── outputs/                # Thư mục lưu các kết quả xuất audio
├── tests/                  # Bộ kiểm thử tích hợp tự động API & phần cứng
├── check_hardware.ps1      # Script kiểm tra cấu hình thiết bị đối chiếu chuẩn tối thiểu
├── requirements.txt        # Danh sách thư viện Python tương thích Windows 64-bit
├── run.bat                 # Script khởi động nhanh và tự động cài đặt 1-click
└── PLAN.md                 # Tài liệu quy hoạch kiến trúc chi tiết
```

---

## ⚖️ Giấy phép & Trách nhiệm (License & Ethics)
- **VieNeu-TTS:** Phát hành theo giấy phép **Apache-2.0** — mã nguồn mở, cho phép sử dụng tự do và thương mại.
- **Trách nhiệm sử dụng giọng nói:** Chỉ sử dụng công nghệ clone giọng cho giọng nói của chính bạn hoặc khi đã có sự đồng ý rõ ràng của chủ sở hữu giọng. Nghiêm cấm sử dụng cho mục đích giả mạo danh tính hoặc lừa đảo.
