@echo off
chcp 65001 >nul
setlocal EnableExtensions

cd /d "%~dp0"

set "APP_URL=http://127.0.0.1:8000"
set "VENV_PY=.venv\Scripts\python.exe"
set "VENV_PIP=.venv\Scripts\pip.exe"
set "SETUP_MARKER=.venv\.setup_done"
set "CHECK_ONLY=0"
if /i "%~1"=="--check-only" set "CHECK_ONLY=1"
if /i "%~1"=="--check-hardware" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0check_hardware.ps1" -CheckOnly
    pause
    exit /b %errorlevel%
)

echo ============================================================
echo   Vietnamese TTS Clone Voice - Khoi dong nhanh (Local)
echo ============================================================

if not exist ".env" (
    if exist ".env.example" (
        copy /y ".env.example" ".env" >nul
        echo [INFO] Da tao file .env mac dinh tu .env.example.
    )
)

call :ENSURE_ESPEAK_PATH

:: Kiem tra nhanh: da cai du moi truong chua?
if exist "%VENV_PY%" (
    if exist "%SETUP_MARKER%" (
        where espeak-ng >nul 2>&1
        if not errorlevel 1 (
            echo [OK] Moi truong da duoc cai dat san. Bo qua buoc setup.
            goto :START_SERVER
        )
    )
)

:: Thu kiem tra lai neu .venv da co san tu truoc nhung chua co marker
if exist "%VENV_PY%" (
    echo [INFO] Phat hien .venv co san, dang kiem tra thu vien...
    "%VENV_PY%" -c "import vieneu, fastapi, uvicorn, soundfile, llama_cpp, neucodec, torch" >nul 2>&1
    if not errorlevel 1 (
        where espeak-ng >nul 2>&1
        if not errorlevel 1 (
            echo ok> "%SETUP_MARKER%"
            echo [OK] Thu vien day du. Bo qua cai dat.
            goto :START_SERVER
        )
    )
)

echo [SETUP] Phat hien may chua cai dat day du. Bat dau tu dong setup...

:: ------------------------------------------------------------
:: 0. Kiem tra cau hinh phan cung (Doi chieu tieu chuan toi thieu)
:: ------------------------------------------------------------
if exist "%~dp0check_hardware.ps1" (
    if "%CHECK_ONLY%"=="1" (
        powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0check_hardware.ps1" -CheckOnly
    ) else (
        powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0check_hardware.ps1"
    )
    if errorlevel 1 (
        echo [THONG BAO] Tien trinh cai dat da dung lai theo yeu cau cua ban.
        pause
        exit /b 1
    )
)

:: ------------------------------------------------------------
:: 1. Kiem tra & cai dat Python 3.11 (64-bit)
:: ------------------------------------------------------------
set "PYTHON_EXE="
call :FIND_PYTHON
if not defined PYTHON_EXE (
    echo [SETUP] Chua co Python 3.10+ 64-bit. Dang cai dat Python 3.11 qua winget...
    where winget >nul 2>&1
    if errorlevel 1 (
        echo [LOI] May khong co winget. Vui long cai thu cong Python 3.11 64-bit tu python.org roi chay lai run.bat.
        pause
        exit /b 1
    )
    winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo [LOI] Cai dat Python 3.11 that bai. Vui long cai thu cong roi chay lai.
        pause
        exit /b 1
    )
    call :REFRESH_PATH
    call :FIND_PYTHON
)
if not defined PYTHON_EXE (
    echo [LOI] Khong tim thay Python 64-bit sau khi cai dat. Hay mo cua so moi hoac khoi dong lai may va chay lai run.bat.
    pause
    exit /b 1
)
echo [OK] Su dung Python: %PYTHON_EXE%

:: ------------------------------------------------------------
:: 2. Kiem tra & cai dat eSpeak NG (bat buoc cho VieNeu G2P)
:: ------------------------------------------------------------
where espeak-ng >nul 2>&1
if errorlevel 1 (
    echo [SETUP] Chua co eSpeak NG. Dang cai dat qua winget...
    where winget >nul 2>&1
    if errorlevel 1 (
        echo [LOI] May khong co winget. Hay cai thu cong eSpeak NG tu GitHub Releases roi chay lai.
        pause
        exit /b 1
    )
    winget install -e --id eSpeak-NG.eSpeak-NG --accept-package-agreements --accept-source-agreements --silent
    call :ENSURE_ESPEAK_PATH
    where espeak-ng >nul 2>&1
    if errorlevel 1 (
        echo [LOI] Khong tim thay espeak-ng sau khi cai. Hay kiem tra quyen Administrator va chay lai.
        pause
        exit /b 1
    )
)
echo [OK] eSpeak NG san sang.

:: ------------------------------------------------------------
:: 3. Kiem tra & cai dat FFmpeg (ho tro upload MP3/M4A)
:: ------------------------------------------------------------
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo [SETUP] Chua co FFmpeg. Dang cai dat qua winget...
    where winget >nul 2>&1
    if not errorlevel 1 (
        winget install -e --id Gyan.FFmpeg --accept-package-agreements --accept-source-agreements
        call :REFRESH_PATH
    ) else (
        echo [CANH BAO] Khong co winget de tu cai FFmpeg. Ban chi upload duoc file .WAV neu khong co FFmpeg.
    )
)

:: ------------------------------------------------------------
:: 4. Tao .venv va cai thu vien tu requirements.txt
:: ------------------------------------------------------------
if not exist "%VENV_PY%" (
    echo [SETUP] Dang tao moi truong ao .venv...
    "%PYTHON_EXE%" -m venv .venv
    if errorlevel 1 (
        echo [LOI] Khong tao duoc .venv.
        pause
        exit /b 1
    )
)

echo [SETUP] Dang nang cap pip, setuptools, wheel...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [LOI] Nang cap pip that bai.
    pause
    exit /b 1
)

echo [SETUP] Dang cai dat cac goi Python (qua trinh nay mat vai phut o lan dau)...
"%VENV_PIP%" install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --only-binary llama_cpp_python
if errorlevel 1 (
    echo [LOI] Cai dat requirements.txt that bai. Kiem tra ket noi mang va chay lai run.bat.
    pause
    exit /b 1
)

echo ok> "%SETUP_MARKER%"
echo [OK] Cai dat hoan tat!

:START_SERVER
if "%CHECK_ONLY%"=="1" (
    echo [CHECK] Kiem tra moi truong thanh cong.
    exit /b 0
)

:: Kiem tra xem server da chay san chua
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri '%APP_URL%/health' -TimeoutSec 2 -UseBasicParsing; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Server da dang chay tai %APP_URL%. Dang mo trinh duyet...
    start "" "%APP_URL%/"
    exit /b 0
)

echo [INFO] Dang khoi dong server TTS tai %APP_URL% ...
echo [INFO] Trinh duyet se tu dong mo ngay khi model nap xong (khoang 8-15 giay).
echo [INFO] Nhan Ctrl+C trong cua so nay de tat server khi dung xong.

:: Chay tien trinh nen cho /health tra 200 OK roi moi mo trinh duyet
start /b "" powershell -NoProfile -WindowStyle Hidden -Command "for ($i=0; $i -lt 180; $i++) { Start-Sleep -Seconds 1; try { $r = Invoke-WebRequest -Uri '%APP_URL%/health' -TimeoutSec 2 -UseBasicParsing; if ($r.StatusCode -eq 200) { Start-Process '%APP_URL%/'; break } } catch {} }"

"%VENV_PY%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
set "EXIT_CODE=%errorlevel%"
if not "%EXIT_CODE%"=="0" (
    echo [LOI] Server dung voi ma loi %EXIT_CODE%.
    pause
)
exit /b %EXIT_CODE%

:: ============================================================
:: Ham phu tro
:: ============================================================
:ENSURE_ESPEAK_PATH
if exist "C:\Program Files\eSpeak NG\espeak-ng.exe" (
    set "PATH=C:\Program Files\eSpeak NG;%PATH%"
)
if exist "C:\Program Files (x86)\eSpeak NG\espeak-ng.exe" (
    set "PATH=C:\Program Files (x86)\eSpeak NG;%PATH%"
)
exit /b 0

:REFRESH_PATH
for /f "usebackq delims=" %%P in (`powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')"`) do set "PATH=%%P"
call :ENSURE_ESPEAK_PATH
exit /b 0

:FIND_PYTHON
set "PYTHON_EXE="
where py >nul 2>&1
if not errorlevel 1 (
    py -3.11 -c "import sys, struct; sys.exit(0 if struct.calcsize('P')*8==64 else 1)" >nul 2>&1
    if not errorlevel 1 (
        for /f "usebackq delims=" %%I in (`py -3.11 -c "import sys; print(sys.executable)"`) do set "PYTHON_EXE=%%I"
        exit /b 0
    )
)
where python >nul 2>&1
if not errorlevel 1 (
    python -c "import sys, struct; sys.exit(0 if sys.version_info[:2] in ((3,10),(3,11),(3,12)) and struct.calcsize('P')*8==64 else 1)" >nul 2>&1
    if not errorlevel 1 (
        for /f "usebackq delims=" %%I in (`python -c "import sys; print(sys.executable)"`) do set "PYTHON_EXE=%%I"
        exit /b 0
    )
)
if exist "%LocalAppData%\Programs\Python\Python311\python.exe" (
    set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python311\python.exe"
    exit /b 0
)
if exist "C:\Program Files\Python311\python.exe" (
    set "PYTHON_EXE=C:\Program Files\Python311\python.exe"
    exit /b 0
)
exit /b 0
