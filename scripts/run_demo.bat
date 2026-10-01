@echo off
chcp 65001 >nul
REM 시연 시작: (1) Ollama가 없으면 MAX_LOADED_MODELS=1 로 새 창에서 띄움 (2) 사전 점검 (3) 서버
cd /d "%~dp0\.."
curl -s -m 2 http://127.0.0.1:11434/api/version >nul 2>&1
if errorlevel 1 (
  start "ollama" cmd /k scripts\start_ollama.bat
  timeout /t 5 >nul
)
python -m app.preflight --profile gpu_4060
python -m app.server --profile gpu_4060 %*
