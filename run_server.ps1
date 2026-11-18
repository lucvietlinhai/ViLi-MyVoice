$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
Write-Host "Dang khoi dong Vietnamese TTS Clone Voice Server tai http://127.0.0.1:8000 ..."
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
