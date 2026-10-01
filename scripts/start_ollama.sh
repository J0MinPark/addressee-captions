#!/usr/bin/env bash
# Ollama를 "한 번에 모델 1개만" 로드하도록 띄운다(VRAM 8GB에서 Whisper·AST와 공존).
# systemd 서비스로 설치됐다면:  sudo systemctl edit ollama  →  [Service] Environment="OLLAMA_MAX_LOADED_MODELS=1"
pkill -f "ollama serve" 2>/dev/null || true
pkill -f "ollama runner" 2>/dev/null || true; pkill -f llama-server 2>/dev/null || true
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=30m
echo "[start_ollama] OLLAMA_MAX_LOADED_MODELS=1 로 ollama serve 시작 (이 터미널은 열어 두세요)"
exec ollama serve
