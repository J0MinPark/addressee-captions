# 리눅스 공용 서버(연구실 RTX PRO 6000) 공통 환경. 서버 스크립트가 source 한다.
#   source scripts/server_env.sh
# 값은 미리 export 해 두면 그 값을 쓴다(예: HEARME_GPU=3 scripts/run_server.sh ...).
# 노트북(Windows)은 이 파일을 쓰지 않는다.

_HEARME_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HEARME_ROOT="${HEARME_ROOT:-$_HEARME_ROOT}"

# GPU 고정: nvidia-smi 번호와 같게 PCI 순서로 센다. 다른 GPU는 보이지 않게 한다.
export HEARME_GPU="${HEARME_GPU:-2}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$HEARME_GPU"

# 우리 Ollama 인스턴스(시스템 Ollama 11434는 다른 사용자 것 — 건드리지 않는다)
export HEARME_OLLAMA_PORT="${HEARME_OLLAMA_PORT:-11435}"
export OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:${HEARME_OLLAMA_PORT}}"

# 저장소 밖 데이터 루트(AI Hub 원본·파생 파일, DEMAND, 시나리오). 라이선스상 저장소에 넣지 않는다.
export HEARME_DATA="${HEARME_DATA:-$(cd "$HEARME_ROOT/.." && pwd)/hearme_data}"
mkdir -p "$HEARME_DATA"

export HEARME_PY="${HEARME_PY:-$HEARME_ROOT/.venv/bin/python}"
export HEARME_RUN="$HEARME_ROOT/logs/run"
mkdir -p "$HEARME_RUN"
