#!/usr/bin/env bash
# 원격 시연 앱 서버(공용 리눅스 서버). GPU 2 고정, localhost 전용(대시보드 8000, 음성 수신 8765).
#
#   scripts/run_server.sh start  [서버 인자...]   # 백그라운드(tmux 세션 hearme-server), pid 파일 기록
#   scripts/run_server.sh stop                    # pid 파일의 프로세스만 종료
#   scripts/run_server.sh status
#   scripts/run_server.sh restart [서버 인자...]
#   scripts/run_server.sh run    [서버 인자...]   # 포그라운드(systemd 사용자 서비스용)
#
# 기본 인자: --profile server --network  (시연 구성은 app/demo_config.yaml, 없으면 v1)
# 예) scripts/run_server.sh run --no-selected        # v1 구성 강제
#     scripts/run_server.sh run --replay data/demo --realtime --loop   # 클라이언트 없이 녹음 재생
set -euo pipefail

CMD="${1:-status}"; shift || true
source "$(dirname "${BASH_SOURCE[0]}")/server_env.sh"
PIDF="$HEARME_RUN/server.pid"
LOGF="$HEARME_ROOT/logs/server.log"
SESSION="hearme-server"

ARGS=("$@")
has() { local x; for x in "${ARGS[@]:-}"; do [ "$x" = "$1" ] && return 0; done; return 1; }
has --profile || ARGS=(--profile server "${ARGS[@]}")
has --replay || has --network || ARGS+=(--network)

ours() {
  local pid="$1"
  [ -n "$pid" ] && [ -d "/proc/$pid" ] || return 1
  [ "$(stat -c %U "/proc/$pid")" = "$(id -un)" ] || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q "app.server"
}
cur_pid() { [ -f "$PIDF" ] && cat "$PIDF" || true; }

do_run() {
  cd "$HEARME_ROOT"
  if ! curl -s -m 2 "$OLLAMA_URL/api/version" >/dev/null; then
    echo "[server] 경고: Ollama($OLLAMA_URL) 응답 없음 → scripts/start_ollama.sh start (LLM 없이 시작하고 켜지면 자동 복구)"
  fi
  echo $$ > "$PIDF"
  echo "[server] GPU=$CUDA_VISIBLE_DEVICES OLLAMA_URL=$OLLAMA_URL HEARME_DATA=$HEARME_DATA 인자: ${ARGS[*]}"
  exec "$HEARME_PY" -u -m app.server "${ARGS[@]}"
}

do_start() {
  local pid; pid="$(cur_pid)"
  if ours "$pid"; then echo "[server] 이미 실행 중 pid=$pid"; return 0; fi
  if tmux has-session -t "$SESSION" 2>/dev/null; then tmux kill-session -t "$SESSION"; fi   # 우리 세션 이름만
  local q; q="$(printf '%q ' "${ARGS[@]}")"
  tmux new-session -d -s "$SESSION" "bash -c '$HEARME_ROOT/scripts/run_server.sh run $q 2>&1 | tee $LOGF'"
  for _ in $(seq 300); do
    grep -q "Uvicorn running\|대시보드:" "$LOGF" 2>/dev/null && break
    sleep 0.5
  done
  sleep 1
  do_status
}

do_stop() {
  local pid; pid="$(cur_pid)"
  if ours "$pid"; then
    kill -TERM "$pid"
    for _ in $(seq 50); do [ -d "/proc/$pid" ] || break; sleep 0.2; done
    [ -d "/proc/$pid" ] && kill -KILL "$pid" || true
    echo "[server] 종료 pid=$pid"
  else
    echo "[server] 실행 중 아님 (pid 파일: ${pid:-없음})"
  fi
  rm -f "$PIDF"
  if tmux has-session -t "$SESSION" 2>/dev/null; then tmux kill-session -t "$SESSION"; fi
}

do_status() {
  local pid; pid="$(cur_pid)"
  if ours "$pid"; then
    echo "[server] 실행 중 pid=$pid  포트: $(cat "$HEARME_RUN/ports.json" 2>/dev/null || echo '?')"
    grep -E "\[구성\]|llm=|음성 수신|대시보드" "$LOGF" 2>/dev/null | tail -n 5 | sed 's/^/  /' || true
  else
    echo "[server] 꺼짐"; return 3
  fi
}

case "$CMD" in
  start) do_start ;;
  stop) do_stop ;;
  restart) do_stop; do_start ;;
  status) do_status ;;
  run) do_run ;;
  *) echo "사용법: $0 start|stop|restart|status|run [서버 인자...]" >&2; exit 2 ;;
esac
