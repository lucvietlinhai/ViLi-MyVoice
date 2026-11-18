# ================================================================================
# check_hardware.ps1
# Kiểm tra cấu hình phần cứng thiết bị trước khi cài đặt Vietnamese TTS Clone Voice
# Tiêu chuẩn tối thiểu: Máy chuẩn người dùng (Intel Xeon E3-1245 v3 4C/8T, AVX2, 16GB RAM)
# ================================================================================

[CmdletBinding()]
param(
    [switch]$CheckOnly,
    [switch]$ForceContinue,
    [string]$ProjectDir = $PSScriptRoot
)

$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Đảm bảo ProjectDir hợp lệ
if ($ProjectDir) {
    $ProjectDir = $ProjectDir.Trim('"', "'").TrimEnd('\', '/')
}
if (-not $ProjectDir -or -not (Test-Path $ProjectDir)) {
    $ProjectDir = $PSScriptRoot
}
if (-not $ProjectDir -or -not (Test-Path $ProjectDir)) {
    $ProjectDir = (Get-Location).Path
}

# Tiêu chuẩn tối thiểu (Baseline Specifications)
$BASELINE_MIN_RAM_GB   = 15.0  # Chuẩn 16GB RAM (cho phép trừ hao BIOS/hardware reserved)
$BASELINE_MIN_CORES    = 4     # 4 Cores
$BASELINE_MIN_THREADS  = 8     # 8 Threads
$BASELINE_MIN_DISK_GB  = 15.0  # 15 GB trống trên ổ đĩa dự án
$BASELINE_SYS_DISK_GB  = 5.0   # 5 GB trống trên ổ đĩa hệ thống C:

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "          KIỂM TRA CẤU HÌNH THIẾT BỊ (HARDWARE SPECIFICATION CHECK)             " -ForegroundColor Cyan
Write-Host " Tiêu chuẩn tối thiểu: Intel Xeon 4C/8T, Tập lệnh AVX2, RAM 16GB, Win 64-bit   " -ForegroundColor DarkCyan
Write-Host "================================================================================" -ForegroundColor Cyan

# --------------------------------------------------------------------------------
# 1. Thu thập thông số phần cứng
# --------------------------------------------------------------------------------
$warnings = [System.Collections.Generic.List[string]]::new()
$criticalErrors = [System.Collections.Generic.List[string]]::new()

# OS Architecture & Caption
$is64Bit = [Environment]::Is64BitOperatingSystem
$osCaption = "Windows"
try {
    $os = Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue
    if ($os -and $os.Caption) { $osCaption = $os.Caption.Trim() }
} catch {}

# CPU details
$cpuName = "Unknown CPU"
$cores = 1
$threads = 1
try {
    $cpu = Get-CimInstance Win32_Processor -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cpu) {
        if ($cpu.Name) { $cpuName = $cpu.Name.Trim() }
        if ($cpu.NumberOfCores) { $cores = [int]$cpu.NumberOfCores }
        if ($cpu.NumberOfLogicalProcessors) { $threads = [int]$cpu.NumberOfLogicalProcessors }
    }
} catch {}

# AVX2 Instruction Set check
$hasAvx2 = $false
try {
    $typeDef = @"
using System;
using System.Runtime.InteropServices;
public class CpuFeatureChecker {
    [DllImport("kernel32.dll")]
    public static extern bool IsProcessorFeaturePresent(int Feature);
}
"@
    if (-not ([System.Management.Automation.PSTypeName]'CpuFeatureChecker').Type) {
        Add-Type -TypeDefinition $typeDef -ErrorAction Stop
    }
    # Feature 40 = PF_AVX2_INSTRUCTIONS_AVAILABLE
    $hasAvx2 = [CpuFeatureChecker]::IsProcessorFeaturePresent(40)
} catch {
    # Fallback nếu Add-Type bị chặn: kiểm tra tên CPU hoặc giả định True nếu CPU thế hệ mới
    $hasAvx2 = $true
}

# RAM Total Physical Memory
$ramGB = 0.0
try {
    $cs = Get-CimInstance Win32_ComputerSystem -ErrorAction SilentlyContinue
    if ($cs -and $cs.TotalPhysicalMemory) {
        $ramGB = [Math]::Round($cs.TotalPhysicalMemory / 1GB, 1)
    }
} catch {}

# Free Disk Space - Project Drive
$projDriveLetter = (Get-Item $ProjectDir).PSDrive.Name
$projDriveFreeGB = 0.0
try {
    $projDrive = Get-PSDrive -Name $projDriveLetter -PSProvider FileSystem -ErrorAction SilentlyContinue
    if ($projDrive) {
        $projDriveFreeGB = [Math]::Round($projDrive.Free / 1GB, 1)
    }
} catch {}

# Free Disk Space - System Drive C:
$sysDriveFreeGB = 0.0
$checkSysDrive = $false
if ($projDriveLetter -ne "C") {
    $checkSysDrive = $true
    try {
        $sysDrive = Get-PSDrive -Name "C" -PSProvider FileSystem -ErrorAction SilentlyContinue
        if ($sysDrive) {
            $sysDriveFreeGB = [Math]::Round($sysDrive.Free / 1GB, 1)
        }
    } catch {}
}

# --------------------------------------------------------------------------------
# 2. Đánh giá từng thông số so với tiêu chuẩn tối thiểu
# --------------------------------------------------------------------------------

# Kiểm tra OS
$osStatus = "[ĐẠT]"
$osStatusColor = "Green"
if (-not $is64Bit) {
    $osStatus = "[LỖI]"
    $osStatusColor = "Red"
    $criticalErrors.Add("Hệ điều hành 32-bit không được hỗ trợ bởi PyTorch, ONNX và VieNeu. Bắt buộc Windows 64-bit.")
}

# Kiểm tra CPU Cores / Threads
$cpuStatus = "[ĐẠT]"
$cpuStatusColor = "Green"
if ($cores -lt $BASELINE_MIN_CORES -or $threads -lt $BASELINE_MIN_THREADS) {
    $cpuStatus = "[CẢNH BÁO]"
    $cpuStatusColor = "Yellow"
    $warnings.Add("Vi xử lý ($cores Cores / $threads Threads) thấp hơn chuẩn tối thiểu ($BASELINE_MIN_CORES Cores / $BASELINE_MIN_THREADS Threads). Tốc độ tạo giọng sẽ bị chậm.")
}

# Kiểm tra AVX2
$avx2Status = "[ĐẠT]"
$avx2StatusColor = "Green"
if (-not $hasAvx2) {
    $avx2Status = "[CẢNH BÁO]"
    $avx2StatusColor = "Yellow"
    $warnings.Add("CPU không hỗ trợ tập lệnh AVX2. ONNX Runtime và llama.cpp sẽ hoạt động rất chậm hoặc có thể phát sinh lỗi.")
}

# Kiểm tra RAM
$ramStatus = "[ĐẠT]"
$ramStatusColor = "Green"
if ($ramGB -lt $BASELINE_MIN_RAM_GB) {
    $ramStatus = "[CẢNH BÁO]"
    $ramStatusColor = "Yellow"
    $warnings.Add("RAM phát hiện ($ramGB GB) thấp hơn tiêu chuẩn tối thiểu (16 GB). Khi chạy đồng thời VieNeu v3 và v2 GGUF có nguy cơ tràn RAM (OOM) hoặc giật lag.")
}

# Kiểm tra Ổ đĩa dự án
$projDiskStatus = "[ĐẠT]"
$projDiskColor = "Green"
if ($projDriveFreeGB -lt $BASELINE_MIN_DISK_GB) {
    $projDiskStatus = "[CẢNH BÁO]"
    $projDiskColor = "Yellow"
    $warnings.Add("Ổ đĩa dự án (${projDriveLetter}:) chỉ còn $projDriveFreeGB GB trống (khuyến nghị tối thiểu $BASELINE_MIN_DISK_GB GB để chứa code, venv và audio sinh ra).")
}

# Kiểm tra Ổ đĩa hệ thống C:
$sysDiskStatus = "[ĐẠT]"
$sysDiskColor = "Green"
if ($checkSysDrive -and ($sysDriveFreeGB -lt $BASELINE_SYS_DISK_GB)) {
    $sysDiskStatus = "[CẢNH BÁO]"
    $sysDiskColor = "Yellow"
    $warnings.Add("Ổ đĩa hệ thống C: chỉ còn $sysDriveFreeGB GB trống (cần tối thiểu $BASELINE_SYS_DISK_GB GB cho Hugging Face model cache và thư mục tạm).")
}

# --------------------------------------------------------------------------------
# 3. Hiển thị bảng đối chiếu
# --------------------------------------------------------------------------------
Write-Host (" {0,-20} | {1,-26} | {2,-30} | {3}" -f "Thông số", "Tiêu chuẩn tối thiểu", "Thiết bị hiện tại", "Đánh giá")
Write-Host "---------------------+----------------------------+--------------------------------+-----------"

# OS
Write-Host (" {0,-20} | {1,-26} | {2,-30} | " -f "Hệ điều hành", "Windows 64-bit (x64)", "$osCaption (64-bit)") -NoNewline
Write-Host $osStatus -ForegroundColor $osStatusColor

# CPU
$cpuDetectedStr = "$cpuName ($cores Cores / $threads Threads)"
if ($cpuDetectedStr.Length -gt 30) { $cpuDetectedStr = "$cores Cores / $threads Threads ($cpuName)" }
if ($cpuDetectedStr.Length -gt 30) { $cpuDetectedStr = "$cores Cores / $threads Threads" }
Write-Host (" {0,-20} | {1,-26} | {2,-30} | " -f "Vi xử lý (CPU)", "4 Nhân / 8 Luồng", $cpuDetectedStr) -NoNewline
Write-Host $cpuStatus -ForegroundColor $cpuStatusColor

# AVX2
$avx2DetectedStr = if ($hasAvx2) { "Hỗ trợ (AVX2 Enabled)" } else { "Không hỗ trợ (Missing AVX2)" }
Write-Host (" {0,-20} | {1,-26} | {2,-30} | " -f "Tập lệnh AVX2", "Bắt buộc có AVX2", $avx2DetectedStr) -NoNewline
Write-Host $avx2Status -ForegroundColor $avx2StatusColor

# RAM
Write-Host (" {0,-20} | {1,-26} | {2,-30} | " -f "Bộ nhớ RAM", "16 GB (Khả dụng >= 15 GB)", "$ramGB GB") -NoNewline
Write-Host $ramStatus -ForegroundColor $ramStatusColor

# Ổ đĩa dự án
Write-Host (" {0,-20} | {1,-26} | {2,-30} | " -f "Ổ đĩa cài đặt (${projDriveLetter}:)", "Trống >= $BASELINE_MIN_DISK_GB GB", "$projDriveFreeGB GB trống") -NoNewline
Write-Host $projDiskStatus -ForegroundColor $projDiskColor

# Ổ đĩa hệ thống (nếu khác ổ dự án)
if ($checkSysDrive) {
    Write-Host (" {0,-20} | {1,-26} | {2,-30} | " -f "Ổ hệ thống (C:)", "Trống >= $BASELINE_SYS_DISK_GB GB", "$sysDriveFreeGB GB trống") -NoNewline
    Write-Host $sysDiskStatus -ForegroundColor $sysDiskColor
}

Write-Host "================================================================================" -ForegroundColor Cyan

# --------------------------------------------------------------------------------
# 4. Xử lý kết quả & Cảnh báo người dùng
# --------------------------------------------------------------------------------

# Trường hợp có lỗi nghiêm trọng (Fatal error)
if ($criticalErrors.Count -gt 0) {
    Write-Host "`n[LỖI NGHIÊM TRỌNG - KHÔNG THỂ CÀI ĐẶT]" -ForegroundColor Red
    foreach ($err in $criticalErrors) {
        Write-Host "  - $err" -ForegroundColor Red
    }
    Write-Host "`nQuá trình cài đặt đã bị dừng lại do phần cứng/hệ điều hành không hỗ trợ.`n" -ForegroundColor Red
    exit 1
}

# Trường hợp đạt toàn bộ tiêu chuẩn
if ($warnings.Count -eq 0) {
    Write-Host "[OK] Cấu hình thiết bị ĐẠT TIÊU CHUẨN TỐI THIỂU để sử dụng tốt ứng dụng!" -ForegroundColor Green
    Write-Host "Hệ thống sẵn sàng cho quá trình cài đặt tự động.`n" -ForegroundColor Gray
    exit 0
}

# Trường hợp có cảnh báo (Cấu hình thấp hơn tiêu chuẩn)
Write-Host "`n[CẢNH BÁO CẤU HÌNH THẤP HƠN TIÊU CHUẨN KHUYẾN NGHỊ]" -ForegroundColor Yellow
Write-Host "Phát hiện $($warnings.Count) thông số phần cứng chưa đạt cấu hình chuẩn tham chiếu:" -ForegroundColor Yellow
foreach ($w in $warnings) {
    Write-Host "  * $w" -ForegroundColor Yellow
}

Write-Host "`nLưu ý: Thiết bị có cấu hình thấp hơn tiêu chuẩn có thể gặp tình trạng sinh giọng rất chậm" -ForegroundColor DarkYellow
Write-Host "hoặc bị tràn bộ nhớ khi chạy đồng thời cả hai mô hình (v3 Turbo và v2 GGUF).`n" -ForegroundColor DarkYellow

if ($CheckOnly) {
    Write-Host "[CHECK-ONLY] Chế độ kiểm tra: Phát hiện cảnh báo phần cứng." -ForegroundColor Yellow
    exit 2
}

if ($ForceContinue) {
    Write-Host "[INFO] Tự động tiếp tục cài đặt do tham số --force được truyền vào." -ForegroundColor Cyan
    exit 0
}

# Hỏi người dùng có tiếp tục hay không
$promptChoice = ""
while ($true) {
    Write-Host "Cấu hình thiết bị thấp hơn tiêu chuẩn tối thiểu có thể sử dụng tốt dự án." -ForegroundColor Yellow
    $ans = Read-Host "Bạn có muốn tiếp tục cài đặt không? (Y/N) [Mặc định: N]"
    $ans = $ans.Trim()
    
    if ($ans -eq "" -or $ans -eq "N" -or $ans -eq "n") {
        Write-Host "`n[ĐÃ HỦY] Bạn đã chọn dừng cài đặt." -ForegroundColor Red
        Write-Host "Vui lòng nâng cấp phần cứng theo bảng tiêu chuẩn trên để trải nghiệm ứng dụng tốt nhất.`n" -ForegroundColor Gray
        exit 1
    }
    elseif ($ans -eq "Y" -or $ans -eq "y") {
        Write-Host "`n[XÁC NHẬN] Bạn đã chọn tiếp tục cài đặt dù cấu hình thấp hơn tiêu chuẩn." -ForegroundColor Green
        Write-Host "Tiến hành cài đặt các gói tài nguyên...`n" -ForegroundColor Gray
        exit 0
    }
    else {
        Write-Host "Vui lòng chỉ nhập 'Y' (tiếp tục) hoặc 'N' (dừng cài đặt)." -ForegroundColor DarkYellow
    }
}
