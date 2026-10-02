#!/usr/bin/env bash
# 우리 Ollama 인스턴스만 시작·종료·상태 확인한다(공용 서버용). 이름으로 프로세스를 죽이지 않는다.
#
#   scripts/start_ollama.sh start  [--port 11435] [--gpu 2]   # 백그라운드 시작, pid 파일 기록
#   scripts/start_ollama.sh stop   [--port 11435]             # pid 파일의 프로세스(와 그 자식 러너)만 종료
#   scripts/start_ollama.sh status [--port 11435]
#   scripts/start_ollama.sh restart [...]
#   scripts/start_ollama.sh run    [...]                      # 포그라운드(systemd 사용자 서비스용)
#
# 서버 기본값: GPU 2(PCI 순서), 127.0.0.1:11435, 모델 models/ollama, 한 번에 모델 1개, keep_alive 30m,
# Vulkan 끔(켜 두면 CUDA_VISIBLE_DEVICES와 무관하게 다른 GPU를 잡는다), 컨텍스트 4096(판정 프롬프트는 1k 토큰 미만).
# Windows 노트북은 scripts\start_ollama.bat 를 쓴다.
set -euo pipefail

CMD="${1:-status}"; shift || true
source "$(dirname "${BASH_SOURCE[0]}")/server_env.sh"
PORT="$HEARME_OLLAMA_PORT"; GPU="$HEARME_GPU"
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    *) echo "알 수 없는 인자: $1" >&2; exit 2 ;;
  esac
done

PIDF="$HEARME_RUN/ollama_${PORT}.pid"
LOGF="$HEARME_ROOT/logs/ollama_${PORT}.log"
OLLAMA_BIN="${OLLAMA_BIN:-$(command -v ollama)}"

ollama_env() {
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="$GPU"
  export OLLAMA_VULKAN=0
  export GGML_VK_VISIBLE_DEVICES=-1
  export OLLAMA_HOST="127.0.0.1:${PORT}"
  export OLLAMA_MODELS="$HEARME_ROOT/models/ollama"
  export OLLAMA_MAX_LOADED_MODELS=1
  export OLLAMA_KEEP_ALIVE=30m
  export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-4096}"
}

# pid 파일의 프로세스가 살아 있고, 내 것이고, ollama serve 인가
ours() {
  local pid="$1"
  [ -n "$pid" ] && [ -d "/proc/$pid" ] || return 1
  [ "$(stat -c %U "/proc/$pid")" = "$(id -un)" ] || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q "ollama serve"
}

cur_pid() { [ -f "$PIDF" ] && cat "$PIDF" || true; }

port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; }

do_status() {
  local pid; pid="$(cur_pid)"
  if ours "$pid"; then
    echo "[ollama:$PORT] 실행 중 pid=$pid GPU=$GPU"
    curl -s -m 2 "http://127.0.0.1:$PORT/api/version" && echo
    curl -s -m 2 "http://127.0.0.1:$PORT/api/ps" | head -c 400 && echo
    grep "inference compute" "$LOGF" 2>/dev/null | tail -n 3 | sed 's/^/  /' || true
    return 0
  fi
  echo "[ollama:$PORT] 꺼짐 (pid 파일: ${pid:-없음})"
  return 3
}

do_stop() {
  local pid; pid="$(cur_pid)"
  if ! ours "$pid"; then
    echo "[ollama:$PORT] 우리 인스턴스가 떠 있지 않음 (pid 파일: ${pid:-없음}) — 아무것도 종료하지 않음"
    rm -f "$PIDF"
    return 0
  fi
  local kids; kids="$(ps -o pid= --ppid "$pid" | tr -d ' ' || true)"   # 이 인스턴스의 러너만(부모 pid로)
  kill -TERM "$pid"
  for _ in $(seq 50); do [ -d "/proc/$pid" ] || break; sleep 0.2; done
  [ -d "/proc/$pid" ] && kill -KILL "$pid" || true
  for k in $kids; do
    if [ -d "/proc/$k" ] && [ "$(stat -c %U "/proc/$k")" = "$(id -un)" ]; then kill -TERM "$k" 2>/dev/null || true; fi
  done
  rm -f "$PIDF"
  echo "[ollama:$PORT] 종료 pid=$pid (러너: ${kids:-없음})"
}

wait_up() {
  for _ in $(seq 150); do
    curl -s -m 1 "http://127.0.0.1:$PORT/api/version" >/dev/null 2>&1 && return 0
    sleep 0.2
  done
  return 1
}

do_start() {
  local pid; pid="$(cur_pid)"
  if ours "$pid"; then echo "[ollama:$PORT] 이미 실행 중 pid=$pid"; return 0; fi
  if port_busy; then echo "[ollama:$PORT] 포트가 다른 프로세스에 사용 중 — 시작하지 않음" >&2; return 1; fi
  mkdir -p "$HEARME_ROOT/models/ollama" "$(dirname "$LOGF")"
  ( ollama_env; exec setsid "$OLLAMA_BIN" serve ) >"$LOGF" 2>&1 </dev/null &
  echo $! > "$PIDF"
  if wait_up; then
    echo "[ollama:$PORT] 시작 pid=$(cur_pid) GPU=$GPU 로그=$LOGF"
    grep "inference compute" "$LOGF" | sed 's/^/  /' || true
  else
    echo "[ollama:$PORT] 30초 안에 응답 없음 — 로그 확인: $LOGF" >&2; return 1
  fi
}

do_run() {   # systemd: 포그라운드, pid 파일은 자기 pid
  if port_busy; then echo "[ollama:$PORT] 포트 사용 중" >&2; exit 1; fi
  mkdir -p "$HEARME_ROOT/models/ollama"
  echo $$ > "$PIDF"
  ollama_env
  exec "$OLLAMA_BIN" serve >"$LOGF" 2>&1
}

case "$CMD" in
  start) do_start ;;
  stop) do_stop ;;
  restart) do_stop; do_start ;;
  status) do_status ;;
  run) do_run ;;
  *) echo "사용법: $0 start|stop|restart|status|run [--port P] [--gpu G]" >&2; exit 2 ;;
esac
