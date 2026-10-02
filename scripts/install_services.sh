#!/usr/bin/env bash
# systemd 사용자 서비스 등록(재부팅 후 자동 시작). 필요: loginctl enable-linger $USER (이 서버에서 Linger=yes 확인됨)
#   scripts/install_services.sh          # 등록 + 시작
#   systemctl --user status hearme-ollama hearme-server
#   systemctl --user restart hearme-server       # 구성 바꾼 뒤
#   journalctl --user -u hearme-ollama -n 50
# 저장소 경로가 ~/jm/addressee-captions 가 아니면 유닛 파일의 경로를 고칠 것.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/server_env.sh
# 수동으로 띄운 인스턴스가 있으면 pid 파일로만 내린다(이름으로 죽이지 않음)
scripts/run_server.sh stop || true
scripts/start_ollama.sh stop || true
mkdir -p ~/.config/systemd/user
cp scripts/systemd/hearme-*.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now hearme-ollama.service
sleep 3
systemctl --user enable --now hearme-server.service
loginctl show-user "$(id -un)" -p Linger
systemctl --user --no-pager status hearme-ollama hearme-server | grep -E "●|Active:"
