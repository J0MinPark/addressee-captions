@echo off
chcp 65001 >nul
REM Ollama를 "한 번에 모델 1개만" 로드하도록 띄운다(VRAM 8GB에서 Whisper·AST와 공존).
REM 트레이의 Ollama 앱이 이미 떠 있으면 이 설정이 적용되지 않으므로 먼저 종료한다.
taskkill /IM "ollama app.exe" /F >nul 2>&1
taskkill /IM ollama.exe /F >nul 2>&1
REM ollama.exe만 죽이면 llama-server.exe(모델 러너)가 고아로 남아 VRAM을 쥐고 있는다 → 같이 종료
taskkill /IM llama-server.exe /F >nul 2>&1
set OLLAMA_MAX_LOADED_MODELS=1
set OLLAMA_KEEP_ALIVE=30m
echo [start_ollama] OLLAMA_MAX_LOADED_MODELS=1 로 ollama serve 시작 (이 창은 열어 두세요)
if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
  "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
) else (
  ollama serve
)
