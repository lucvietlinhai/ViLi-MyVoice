# Automated test for check_hardware.ps1
$ErrorActionPreference = "Stop"

Write-Host "--- Test 1: Real hardware on this machine (expect Pass, code 0) ---"
& powershell -NoProfile -ExecutionPolicy Bypass -File "e:\tts-clone-voice\check_hardware.ps1" -CheckOnly
$code1 = $LASTEXITCODE
if ($code1 -ne 0) {
    Write-Error "Test 1 failed: expected 0, got $code1"
} else {
    Write-Host "Test 1 Passed: code 0" -ForegroundColor Green
}

Write-Host "`n--- Test 2: CheckOnly with warning (expect Warning, code 2) ---"
$tempScript = "e:\tts-clone-voice\tests\temp_check.ps1"
$content = Get-Content "e:\tts-clone-voice\check_hardware.ps1" -Raw
# Set baseline RAM to 32GB to trigger warning
$modContent = $content.Replace('$BASELINE_MIN_RAM_GB   = 15.0', '$BASELINE_MIN_RAM_GB   = 32.0')
Set-Content -Path $tempScript -Value $modContent -Encoding UTF8

& powershell -NoProfile -ExecutionPolicy Bypass -File $tempScript -CheckOnly
$code2 = $LASTEXITCODE
if ($code2 -ne 2) {
    Write-Error "Test 2 failed: expected 2, got $code2"
} else {
    Write-Host "Test 2 Passed: code 2" -ForegroundColor Green
}

Write-Host "`n--- Test 3: Warning triggered + User enters 'N' (expect Abort, code 1) ---"
Set-Content -Path "e:\tts-clone-voice\tests\input_n.txt" -Value "N"
$p = Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$tempScript`"" -RedirectStandardInput "e:\tts-clone-voice\tests\input_n.txt" -PassThru -NoNewWindow -Wait
$code3 = $p.ExitCode
if ($code3 -ne 1) {
    Write-Error "Test 3 failed: expected 1, got $code3"
} else {
    Write-Host "Test 3 Passed: code 1 (aborted as requested by user)" -ForegroundColor Green
}

Write-Host "`n--- Test 4: Warning triggered + User enters 'Y' (expect Proceed, code 0) ---"
Set-Content -Path "e:\tts-clone-voice\tests\input_y.txt" -Value "Y"
$p4 = Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$tempScript`"" -RedirectStandardInput "e:\tts-clone-voice\tests\input_y.txt" -PassThru -NoNewWindow -Wait
$code4 = $p4.ExitCode
if ($code4 -ne 0) {
    Write-Error "Test 4 failed: expected 0, got $code4"
} else {
    Write-Host "Test 4 Passed: code 0 (proceeded as confirmed by user)" -ForegroundColor Green
}

# Cleanup temp files
Remove-Item $tempScript, "e:\tts-clone-voice\tests\input_n.txt", "e:\tts-clone-voice\tests\input_y.txt" -Force -ErrorAction SilentlyContinue

Write-Host "`nALL 4 HARDWARE CHECK TESTS PASSED!" -ForegroundColor Green
