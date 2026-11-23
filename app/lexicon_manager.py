"""Quản lý từ điển phát âm tùy chỉnh (Lexicon) lưu bằng Rust orjson."""

from __future__ import annotations

import re
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import orjson

from .config import settings

STARTER_TEMPLATES: List[Dict[str, Any]] = [
    # ── 1. Mạng xã hội, Nền tảng & Ứng dụng phổ biến ─────────────────────────
    {"pattern": "TikTok", "replacement": "Tích Tóc", "whole_word": True, "case_sensitive": False, "note": "Mạng xã hội video TikTok"},
    {"pattern": "YouTube", "replacement": "Diu Túp", "whole_word": True, "case_sensitive": False, "note": "Nền tảng video YouTube"},
    {"pattern": "Facebook", "replacement": "Phây Búc", "whole_word": True, "case_sensitive": False, "note": "Mạng xã hội Facebook"},
    {"pattern": "Zalo", "replacement": "Da Lô", "whole_word": True, "case_sensitive": False, "note": "Ứng dụng nhắn tin Zalo"},
    {"pattern": "Google", "replacement": "Gút Gồ", "whole_word": True, "case_sensitive": False, "note": "Công cụ tìm kiếm Google"},
    {"pattern": "Shopee", "replacement": "Sô Pi", "whole_word": True, "case_sensitive": False, "note": "Sàn thương mại điện tử Shopee"},
    {"pattern": "Lazada", "replacement": "La Da Đa", "whole_word": True, "case_sensitive": False, "note": "Sàn thương mại điện tử Lazada"},
    {"pattern": "Grab", "replacement": "Gờ Ráp", "whole_word": True, "case_sensitive": False, "note": "Ứng dụng đặt xe Grab"},
    {"pattern": "Netflix", "replacement": "Nét Phờ Lích", "whole_word": True, "case_sensitive": False, "note": "Dịch vụ xem phim trực tuyến Netflix"},
    {"pattern": "Spotify", "replacement": "Sờ Po Ti Phai", "whole_word": True, "case_sensitive": False, "note": "Ứng dụng nghe nhạc Spotify"},
    {"pattern": "Instagram", "replacement": "In Xờ Ta Gờ Ram", "whole_word": True, "case_sensitive": False, "note": "Mạng xã hội hình ảnh Instagram"},
    {"pattern": "Messenger", "replacement": "Méc Xên Gơ", "whole_word": True, "case_sensitive": False, "note": "Ứng dụng nhắn tin Messenger"},
    {"pattern": "Telegram", "replacement": "Te Lê Gờ Ram", "whole_word": True, "case_sensitive": False, "note": "Ứng dụng nhắn tin Telegram"},
    {"pattern": "CapCut", "replacement": "Cắp Cắt", "whole_word": True, "case_sensitive": False, "note": "Ứng dụng chỉnh sửa video CapCut"},
    {"pattern": "ChatGPT", "replacement": "Chát Di Pi Ti", "whole_word": True, "case_sensitive": False, "note": "Trợ lý trí tuệ nhân tạo ChatGPT"},

    # ── 2. Thiết bị & Hệ điều hành thông dụng ────────────────────────────────
    {"pattern": "iPhone", "replacement": "Ai Phôn", "whole_word": True, "case_sensitive": False, "note": "Điện thoại Apple iPhone"},
    {"pattern": "iPad", "replacement": "Ai Pát", "whole_word": True, "case_sensitive": False, "note": "Máy tính bảng Apple iPad"},
    {"pattern": "MacBook", "replacement": "Mác Búc", "whole_word": True, "case_sensitive": False, "note": "Máy tính xách tay Apple MacBook"},
    {"pattern": "Samsung", "replacement": "Sam Sung", "whole_word": True, "case_sensitive": False, "note": "Thương hiệu điện tử Samsung"},
    {"pattern": "Windows", "replacement": "Uyn Đâu", "whole_word": True, "case_sensitive": False, "note": "Hệ điều hành Microsoft Windows"},
    {"pattern": "Android", "replacement": "An Đờ Roi", "whole_word": True, "case_sensitive": False, "note": "Hệ điều hành di động Google Android"},
    {"pattern": "iOS", "replacement": "Ai Ô Ét", "whole_word": True, "case_sensitive": False, "note": "Hệ điều hành di động Apple iOS"},

    # ── 3. Thuật ngữ Công nghệ thông tin, Internet & AI ─────────────────────
    {"pattern": "AI", "replacement": "ây ai", "whole_word": True, "case_sensitive": True, "note": "Trí tuệ nhân tạo (Artificial Intelligence)"},
    {"pattern": "GPT", "replacement": "di pi ti", "whole_word": True, "case_sensitive": True, "note": "Mô hình ngôn ngữ tự nhiên GPT"},
    {"pattern": "UI", "replacement": "iu ai", "whole_word": True, "case_sensitive": True, "note": "Giao diện người dùng (User Interface)"},
    {"pattern": "UX", "replacement": "iu ích", "whole_word": True, "case_sensitive": True, "note": "Trải nghiệm người dùng (User Experience)"},
    {"pattern": "API", "replacement": "ây pi ai", "whole_word": True, "case_sensitive": True, "note": "Cổng giao tiếp lập trình ứng dụng"},
    {"pattern": "TTS", "replacement": "ti ti ét", "whole_word": True, "case_sensitive": True, "note": "Chuyển văn bản thành giọng nói (Text-to-Speech)"},
    {"pattern": "STT", "replacement": "ét ti ti", "whole_word": True, "case_sensitive": True, "note": "Chuyển giọng nói thành văn bản (Speech-to-Text)"},
    {"pattern": "IT", "replacement": "ai ti", "whole_word": True, "case_sensitive": True, "note": "Công nghệ thông tin"},
    {"pattern": "Wi-Fi", "replacement": "Oai Phai", "whole_word": True, "case_sensitive": False, "note": "Mạng internet không dây Wi-Fi"},
    {"pattern": "WiFi", "replacement": "Oai Phai", "whole_word": True, "case_sensitive": False, "note": "Mạng internet không dây WiFi"},
    {"pattern": "Website", "replacement": "Oép Sai", "whole_word": True, "case_sensitive": False, "note": "Trang thông tin điện tử"},
    {"pattern": "Web", "replacement": "Oép", "whole_word": True, "case_sensitive": False, "note": "Mạng toàn cầu / trang web"},
    {"pattern": "Email", "replacement": "I Meo", "whole_word": True, "case_sensitive": False, "note": "Thư điện tử"},
    {"pattern": "Gmail", "replacement": "Gút Gồ Meo", "whole_word": True, "case_sensitive": False, "note": "Dịch vụ thư điện tử Google"},
    {"pattern": "Online", "replacement": "On Lai", "whole_word": True, "case_sensitive": False, "note": "Trực tuyến trên mạng"},
    {"pattern": "Offline", "replacement": "Ọp Lai", "whole_word": True, "case_sensitive": False, "note": "Ngoại tuyến, đời thực"},
    {"pattern": "Livestream", "replacement": "Lai Sờ Trim", "whole_word": True, "case_sensitive": False, "note": "Phát sóng trực tiếp trên mạng"},
    {"pattern": "Podcast", "replacement": "Pót Cát", "whole_word": True, "case_sensitive": False, "note": "Chương trình âm thanh Podcast"},
    {"pattern": "Video", "replacement": "Vi Dé O", "whole_word": True, "case_sensitive": False, "note": "Tệp hình ảnh chuyển động"},
    {"pattern": "Audio", "replacement": "Ô Đi Ô", "whole_word": True, "case_sensitive": False, "note": "Tệp âm thanh, giọng nói"},
    {"pattern": "App", "replacement": "Áp", "whole_word": True, "case_sensitive": False, "note": "Ứng dụng di động"},
    {"pattern": "Link", "replacement": "Linh", "whole_word": True, "case_sensitive": False, "note": "Đường dẫn liên kết mạng"},
    {"pattern": "Cloud", "replacement": "Cờ Lao", "whole_word": True, "case_sensitive": False, "note": "Điện toán đám mây"},
    {"pattern": "Server", "replacement": "Xơ Vơ", "whole_word": True, "case_sensitive": False, "note": "Máy chủ hệ thống"},
    {"pattern": "Data", "replacement": "Đa Ta", "whole_word": True, "case_sensitive": False, "note": "Dữ liệu thông tin"},
    {"pattern": "Code", "replacement": "Cốt", "whole_word": True, "case_sensitive": False, "note": "Mã nguồn lập trình"},
    {"pattern": "Bug", "replacement": "Bắc", "whole_word": True, "case_sensitive": False, "note": "Lỗi phần mềm"},
    {"pattern": "Fix", "replacement": "Phích", "whole_word": True, "case_sensitive": False, "note": "Sửa lỗi kỹ thuật"},
    {"pattern": "Setup", "replacement": "Xét Ắp", "whole_word": True, "case_sensitive": False, "note": "Cài đặt, thiết lập hệ thống"},
    {"pattern": "Update", "replacement": "Ắp Đét", "whole_word": True, "case_sensitive": False, "note": "Cập nhật phiên bản mới"},
    {"pattern": "Test", "replacement": "Tét", "whole_word": True, "case_sensitive": False, "note": "Kiểm thử chức năng"},
    {"pattern": "RAM", "replacement": "Ram", "whole_word": True, "case_sensitive": True, "note": "Bộ nhớ truy xuất ngẫu nhiên máy tính"},
    {"pattern": "CPU", "replacement": "Xê Pê U", "whole_word": True, "case_sensitive": True, "note": "Bộ vi xử lý trung tâm máy tính"},
    {"pattern": "GPU", "replacement": "Gờ Pê U", "whole_word": True, "case_sensitive": True, "note": "Bộ xử lý đồ họa máy tính"},
    {"pattern": "SSD", "replacement": "Ét Ét Đê", "whole_word": True, "case_sensitive": True, "note": "Ổ cứng thể rắn tốc độ cao"},
    {"pattern": "HDD", "replacement": "Hắt Đê Đê", "whole_word": True, "case_sensitive": True, "note": "Ổ đĩa cứng cơ truyền thống"},
    {"pattern": "USB", "replacement": "U Ét Bê", "whole_word": True, "case_sensitive": True, "note": "Cổng và thiết bị lưu trữ USB"},
    {"pattern": "SIM", "replacement": "Sim", "whole_word": True, "case_sensitive": True, "note": "Thẻ SIM điện thoại"},
    {"pattern": "4G", "replacement": "Bốn Gờ", "whole_word": True, "case_sensitive": False, "note": "Mạng di động băng thông rộng 4G"},
    {"pattern": "5G", "replacement": "Năm Gờ", "whole_word": True, "case_sensitive": False, "note": "Mạng di động tốc độ cao 5G"},
    {"pattern": "SMS", "replacement": "Ét Em Ét", "whole_word": True, "case_sensitive": True, "note": "Tin nhắn văn bản điện thoại"},
    {"pattern": "OTP", "replacement": "Ô Tê Pê", "whole_word": True, "case_sensitive": True, "note": "Mã xác thực bảo mật một lần"},
    {"pattern": "QR Code", "replacement": "Kiu A Cốt", "whole_word": True, "case_sensitive": False, "note": "Mã quét phản hồi nhanh"},
    {"pattern": "QR", "replacement": "Kiu A", "whole_word": True, "case_sensitive": True, "note": "Mã quét hình vuông QR"},

    # ── 4. Kinh doanh, Thương mại điện tử & Marketing ───────────────────────
    {"pattern": "CEO", "replacement": "Xi I Ô", "whole_word": True, "case_sensitive": True, "note": "Tổng giám đốc điều hành"},
    {"pattern": "KOL", "replacement": "Cây O Eo", "whole_word": True, "case_sensitive": True, "note": "Người có tầm ảnh hưởng truyền thông"},
    {"pattern": "KOC", "replacement": "Cây Ô Xi", "whole_word": True, "case_sensitive": True, "note": "Người tiêu dùng có ảnh hưởng đánh giá"},
    {"pattern": "Sale", "replacement": "Seo", "whole_word": True, "case_sensitive": False, "note": "Bán hàng, chương trình giảm giá"},
    {"pattern": "Flash Sale", "replacement": "Phờ Lát Seo", "whole_word": True, "case_sensitive": False, "note": "Giảm giá sốc trong khung giờ chớp nhoáng"},
    {"pattern": "Voucher", "replacement": "Vao Chờ", "whole_word": True, "case_sensitive": False, "note": "Phiếu ưu đãi giảm giá"},
    {"pattern": "Shipper", "replacement": "Síp Pơ", "whole_word": True, "case_sensitive": False, "note": "Người giao nhận hàng"},
    {"pattern": "Ship", "replacement": "Síp", "whole_word": True, "case_sensitive": False, "note": "Giao nhận vận chuyển hàng"},
    {"pattern": "COD", "replacement": "Xê Ô Đê", "whole_word": True, "case_sensitive": True, "note": "Thanh toán tiền mặt khi nhận hàng"},
    {"pattern": "Fanpage", "replacement": "Phan Pếch", "whole_word": True, "case_sensitive": False, "note": "Trang bán hàng / cộng đồng mạng xã hội"},
    {"pattern": "Content", "replacement": "Con Ten", "whole_word": True, "case_sensitive": False, "note": "Nội dung truyền thông bài viết"},
    {"pattern": "Marketing", "replacement": "Mác Két Tinh", "whole_word": True, "case_sensitive": False, "note": "Tiếp thị và quảng bá kinh doanh"},
    {"pattern": "Admin", "replacement": "Át Min", "whole_word": True, "case_sensitive": False, "note": "Quản trị viên hệ thống hoặc hội nhóm"},
    {"pattern": "Feedback", "replacement": "Phít Bách", "whole_word": True, "case_sensitive": False, "note": "Ý kiến phản hồi đánh giá của khách hàng"},
    {"pattern": "Review", "replacement": "Ri Viu", "whole_word": True, "case_sensitive": False, "note": "Đánh giá, nhận xét trải nghiệm"},
    {"pattern": "Check-in", "replacement": "Chếch In", "whole_word": True, "case_sensitive": False, "note": "Ghi nhận địa điểm đã đến"},
    {"pattern": "Check in", "replacement": "Chếch In", "whole_word": True, "case_sensitive": False, "note": "Ghi nhận địa điểm đã đến"},
    {"pattern": "Follow", "replacement": "Pho Lâu", "whole_word": True, "case_sensitive": False, "note": "Theo dõi kênh hoặc trang cá nhân"},
    {"pattern": "Subscribe", "replacement": "Súp Rai", "whole_word": True, "case_sensitive": False, "note": "Đăng ký theo dõi kênh"},
    {"pattern": "Like", "replacement": "Lai", "whole_word": True, "case_sensitive": False, "note": "Bấm thích bài đăng"},
    {"pattern": "Share", "replacement": "Se", "whole_word": True, "case_sensitive": False, "note": "Chia sẻ bài viết"},
    {"pattern": "Comment", "replacement": "Com Men", "whole_word": True, "case_sensitive": False, "note": "Bình luận nhận xét"},
    {"pattern": "Inbox", "replacement": "In Bóc", "whole_word": True, "case_sensitive": False, "note": "Nhắn tin riêng tư vào hòm thư"},
    {"pattern": "Seeding", "replacement": "Xít Đinh", "whole_word": True, "case_sensitive": False, "note": "Tạo tương tác tự nhiên lan tỏa"},
    {"pattern": "Deadline", "replacement": "Đét Lai", "whole_word": True, "case_sensitive": False, "note": "Hạn chót hoàn thành nhiệm vụ"},
    {"pattern": "Brief", "replacement": "Bờ Ríp", "whole_word": True, "case_sensitive": False, "note": "Bản mô tả tóm tắt yêu cầu"},
    {"pattern": "KPI", "replacement": "Cây Pi Ai", "whole_word": True, "case_sensitive": True, "note": "Chỉ số đánh giá kết quả công việc"},
    {"pattern": "Staff", "replacement": "Sờ Tát", "whole_word": True, "case_sensitive": False, "note": "Đội ngũ nhân viên"},
    {"pattern": "Menu", "replacement": "Mê Niu", "whole_word": True, "case_sensitive": False, "note": "Bảng thực đơn món ăn hoặc dịch vụ"},
    {"pattern": "Brand", "replacement": "Bờ Ran", "whole_word": True, "case_sensitive": False, "note": "Thương hiệu nhãn hàng"},
    {"pattern": "Hotline", "replacement": "Hót Lai", "whole_word": True, "case_sensitive": False, "note": "Đường dây nóng hỗ trợ khách hàng"},
    {"pattern": "Store", "replacement": "Sờ To", "whole_word": True, "case_sensitive": False, "note": "Cửa hàng kinh doanh"},

    # ── 5. Từ viết tắt hành chính, Y tế, Giáo dục & Địa danh tiếng Việt ───────
    {"pattern": "VNĐ", "replacement": "Việt Nam đồng", "whole_word": False, "case_sensitive": False, "note": "Đơn vị tiền tệ Việt Nam"},
    {"pattern": "VND", "replacement": "Việt Nam đồng", "whole_word": True, "case_sensitive": False, "note": "Mã quốc tế tiền tệ Việt Nam đồng"},
    {"pattern": "USD", "replacement": "đô la Mỹ", "whole_word": True, "case_sensitive": True, "note": "Đơn vị tiền tệ đô la Mỹ"},
    {"pattern": "EUR", "replacement": "ơ rô", "whole_word": True, "case_sensitive": True, "note": "Đơn vị tiền tệ đồng Euro châu Âu"},
    {"pattern": "TP.HCM", "replacement": "Thành phố Hồ Chí Minh", "whole_word": False, "case_sensitive": False, "note": "Địa danh Thành phố Hồ Chí Minh"},
    {"pattern": "TP. HCM", "replacement": "Thành phố Hồ Chí Minh", "whole_word": False, "case_sensitive": False, "note": "Địa danh Thành phố Hồ Chí Minh"},
    {"pattern": "TP.HN", "replacement": "Thành phố Hà Nội", "whole_word": False, "case_sensitive": False, "note": "Địa danh Thủ đô Hà Nội"},
    {"pattern": "TP. Hà Nội", "replacement": "Thành phố Hà Nội", "whole_word": False, "case_sensitive": False, "note": "Địa danh Thủ đô Hà Nội"},
    {"pattern": "TP. Đà Nẵng", "replacement": "Thành phố Đà Nẵng", "whole_word": False, "case_sensitive": False, "note": "Địa danh Thành phố Đà Nẵng"},
    {"pattern": "TP. Cần Thơ", "replacement": "Thành phố Cần Thơ", "whole_word": False, "case_sensitive": False, "note": "Địa danh Thành phố Cần Thơ"},
    {"pattern": "TP. Hải Phòng", "replacement": "Thành phố Hải Phòng", "whole_word": False, "case_sensitive": False, "note": "Địa danh Thành phố Hải Phòng"},
    {"pattern": "UBND", "replacement": "Ủy ban nhân dân", "whole_word": True, "case_sensitive": True, "note": "Cơ quan hành chính nhà nước địa phương"},
    {"pattern": "HĐND", "replacement": "Hội đồng nhân dân", "whole_word": True, "case_sensitive": True, "note": "Cơ quan quyền lực nhà nước ở địa phương"},
    {"pattern": "BHYT", "replacement": "Bảo hiểm y tế", "whole_word": True, "case_sensitive": True, "note": "Chế độ bảo hiểm y tế toàn dân"},
    {"pattern": "BHXH", "replacement": "Bảo hiểm xã hội", "whole_word": True, "case_sensitive": True, "note": "Chế độ bảo hiểm xã hội"},
    {"pattern": "CCCD", "replacement": "Căn cước công dân", "whole_word": True, "case_sensitive": True, "note": "Thẻ căn cước công dân gắn chip"},
    {"pattern": "CMND", "replacement": "Chứng minh nhân dân", "whole_word": True, "case_sensitive": True, "note": "Chứng minh nhân dân (cũ)"},
    {"pattern": "CSKH", "replacement": "Chăm sóc khách hàng", "whole_word": True, "case_sensitive": True, "note": "Bộ phận chăm sóc hỗ trợ khách hàng"},
    {"pattern": "PCCC", "replacement": "Phòng cháy chữa cháy", "whole_word": True, "case_sensitive": True, "note": "Công tác phòng cháy và chữa cháy"},
    {"pattern": "TNHH", "replacement": "Trách nhiệm hữu hạn", "whole_word": True, "case_sensitive": True, "note": "Loại hình doanh nghiệp Trách nhiệm hữu hạn"},
    {"pattern": "CP", "replacement": "Cổ phần", "whole_word": True, "case_sensitive": True, "note": "Loại hình công ty Cổ phần"},
    {"pattern": "BV", "replacement": "Bệnh viện", "whole_word": True, "case_sensitive": True, "note": "Cơ sở y tế Bệnh viện"},
    {"pattern": "ĐH", "replacement": "Đại học", "whole_word": True, "case_sensitive": True, "note": "Cơ sở giáo dục bậc Đại học"},
    {"pattern": "THPT", "replacement": "Trung học phổ thông", "whole_word": True, "case_sensitive": True, "note": "Trường cấp 3 Trung học phổ thông"},
    {"pattern": "THCS", "replacement": "Trung học cơ sở", "whole_word": True, "case_sensitive": True, "note": "Trường cấp 2 Trung học cơ sở"},
    {"pattern": "BS.", "replacement": "Bác sĩ", "whole_word": False, "case_sensitive": False, "note": "Danh xưng Bác sĩ chuyên môn"},
    {"pattern": "ThS.", "replacement": "Thạc sĩ", "whole_word": False, "case_sensitive": False, "note": "Học vị Thạc sĩ"},
    {"pattern": "TS.", "replacement": "Tiến sĩ", "whole_word": False, "case_sensitive": False, "note": "Học vị Tiến sĩ"},
    {"pattern": "PGS.", "replacement": "Phó Giáo sư", "whole_word": False, "case_sensitive": False, "note": "Học hàm Phó Giáo sư"},
    {"pattern": "GS.", "replacement": "Giáo sư", "whole_word": False, "case_sensitive": False, "note": "Học hàm Giáo sư"},
]


class LexiconManager:
    """Lưu trữ, quản lý và áp dụng quy tắc phát âm tiếng Việt/từ mượn."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._compiled_cache: List[tuple[re.Pattern, str]] = []
        self._rules: List[Dict[str, Any]] = []
        self._load_unlocked()

    def _install_starter_unlocked(self) -> List[Dict[str, Any]]:
        existing_patterns = {str(r.get("pattern", "")).lower() for r in self._rules}
        added = []
        for tpl in STARTER_TEMPLATES:
            if tpl["pattern"].lower() in existing_patterns:
                continue
            rule = {
                "id": uuid.uuid4().hex[:10],
                "pattern": tpl["pattern"],
                "replacement": tpl["replacement"],
                "whole_word": bool(tpl.get("whole_word", True)),
                "case_sensitive": bool(tpl.get("case_sensitive", False)),
                "enabled": True,
                "note": tpl.get("note", ""),
            }
            self._rules.append(rule)
            added.append(rule)
        if added:
            self._save_unlocked()
            self._rebuild_compiled_unlocked()
        return added

    def _load_unlocked(self) -> None:
        if not self.path.is_file():
            self._rules = []
        else:
            try:
                data = orjson.loads(self.path.read_bytes())
                if isinstance(data, list):
                    self._rules = data
                else:
                    self._rules = []
            except Exception:
                self._rules = []

        if not self._rules:
            self._install_starter_unlocked()
        else:
            self._rebuild_compiled_unlocked()

    def _save_unlocked(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_bytes(orjson.dumps(self._rules, option=orjson.OPT_INDENT_2))
        tmp.replace(self.path)

    def _rebuild_compiled_unlocked(self) -> None:
        compiled: List[tuple[re.Pattern, str]] = []
        # Ưu tiên cụm dài trước để tránh rule ngắn cắt ngang rule dài
        enabled_rules = [r for r in self._rules if r.get("enabled", True) and r.get("pattern")]
        enabled_rules.sort(key=lambda r: len(str(r.get("pattern", ""))), reverse=True)
        for r in enabled_rules:
            pat = str(r["pattern"])
            rep = str(r.get("replacement", ""))
            whole = bool(r.get("whole_word", True))
            case_sens = bool(r.get("case_sensitive", False))
            escaped = re.escape(pat)
            if whole:
                expr = rf"(?<!\w){escaped}(?!\w)"
            else:
                expr = escaped
            flags = 0 if case_sens else re.IGNORECASE
            try:
                compiled.append((re.compile(expr, flags), rep))
            except re.error:
                continue
        self._compiled_cache = compiled

    def list_rules(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._rules]

    def add_rule(
        self,
        *,
        pattern: str,
        replacement: str,
        whole_word: bool = True,
        case_sensitive: bool = False,
        enabled: bool = True,
        note: str = "",
    ) -> Dict[str, Any]:
        pat = (pattern or "").strip()
        rep = (replacement or "").strip()
        if not pat or not rep:
            raise ValueError("Từ gốc và cách đọc thay thế không được để trống.")
        with self._lock:
            rule = {
                "id": uuid.uuid4().hex[:10],
                "pattern": pat,
                "replacement": rep,
                "whole_word": bool(whole_word),
                "case_sensitive": bool(case_sensitive),
                "enabled": bool(enabled),
                "note": (note or "").strip(),
            }
            self._rules.insert(0, rule)
            self._save_unlocked()
            self._rebuild_compiled_unlocked()
            return dict(rule)

    def update_rule(self, rule_id: str, **fields: Any) -> Dict[str, Any]:
        with self._lock:
            for r in self._rules:
                if r.get("id") == rule_id:
                    for k in ("pattern", "replacement", "whole_word", "case_sensitive", "enabled", "note"):
                        if k in fields and fields[k] is not None:
                            val = fields[k]
                            if isinstance(val, str):
                                val = val.strip()
                            r[k] = val
                    if not r.get("pattern") or not r.get("replacement"):
                        raise ValueError("Từ gốc và cách đọc không được để trống.")
                    self._save_unlocked()
                    self._rebuild_compiled_unlocked()
                    return dict(r)
            raise KeyError(f"Không tìm thấy quy tắc phát âm '{rule_id}'.")

    def delete_rule(self, rule_id: str) -> bool:
        with self._lock:
            before = len(self._rules)
            self._rules = [r for r in self._rules if r.get("id") != rule_id]
            if len(self._rules) == before:
                return False
            self._save_unlocked()
            self._rebuild_compiled_unlocked()
            return True

    def install_starter_presets(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._install_starter_unlocked()

    def apply(self, text: str) -> str:
        if not text:
            return text
        with self._lock:
            compiled = list(self._compiled_cache)
        out = text
        for regex, rep in compiled:
            out = regex.sub(lambda _m, r=rep: r, out)
        return out


lexicon_manager = LexiconManager(settings.lexicon_path)
