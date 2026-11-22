$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$hfHome = Join-Path $PSScriptRoot ".hf_cache"
$env:HF_HOME = $hfHome
$env:HF_HUB_CACHE = Join-Path $hfHome "hub"
$env:TORCH_HOME = Join-Path $hfHome "torch"
Write-Host "Dang khoi dong Vietnamese TTS Clone Voice Server tai http://127.0.0.1:8000 (HF_HOME=$hfHome) ..."
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
