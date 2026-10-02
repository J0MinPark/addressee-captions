#!/usr/bin/env bash
# pre-commit 훅(전체 테스트) 설치
set -euo pipefail
cd "$(dirname "$0")/.."
git config core.hooksPath .githooks
echo "core.hooksPath=.githooks 설정 완료"
