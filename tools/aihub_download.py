"""AI Hub 데이터셋 464(주요 영역별 회의 음성인식) 내려받기. aihubshell(v0.6)과 같은 API를 쓰되 진행률·이어받기·디스크 검사를 붙였다.

    source scripts/server_env.sh
    python tools/aihub_download.py --list                        # 파일 트리(키 불필요)
    python tools/aihub_download.py --labels                      # TL1~TL7 + VL1 라벨(약 280MB)
    python tools/aihub_download.py --filekey 31658               # 원천 zip 하나(예: TS5) — 디스크 검사 후

- API 키는 환경변수 AIHUB_API_KEY에서만 읽고 HTTP 헤더로만 보낸다(명령줄·로그·출력에 쓰지 않는다).
- 저장 위치: $HEARME_DATA/aihub/download/<filekey>/ (저장소 밖). 응답은 tar(큰 파일은 .partN 조각)이다.
  풀면서 조각을 합치고, 다 풀리면 tar를 지운다.
- 이어받기: 같은 명령을 다시 실행하면 받은 바이트 뒤부터 Range로 이어받는다.
- 디스크: tar와 푼 파일이 잠깐 같이 있으므로 여유 공간 < 2.2×크기 이면 멈춘다(사전 기준 1.2× 보다 엄격).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

import requests

BASE = "https://api.aihub.or.kr"
VER = "0.6"
DATASET = 464
LABELS = {"TL1": 61634, "TL2": 61635, "TL3": 61636, "TL4": 61637, "TL5": 61638, "TL6": 61639, "TL7": 61640,
          "VL1": 563185}
SOURCES = {"TS1": 31654, "TS2": 31655, "TS3": 31656, "TS4": 31657, "TS5": 31658, "TS6": 31659, "TS7": 31660}
SIZE_GB = {31654: 90, 31655: 60, 31656: 65, 31657: 75, 31658: 41, 31659: 86, 31660: 83}

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def root() -> Path:
    d = os.environ.get("HEARME_DATA")
    if not d:
        raise SystemExit("HEARME_DATA 미설정 → source scripts/server_env.sh")
    p = Path(d) / "aihub"
    p.mkdir(parents=True, exist_ok=True)
    return p


def key() -> str:
    k = os.environ.get("AIHUB_API_KEY", "")
    if not k:
        raise SystemExit("AIHUB_API_KEY 미설정 (~/.config/hearme/aihub.env 에 export AIHUB_API_KEY=... 후 source scripts/server_env.sh)")
    return k


def fmt_t(s: float) -> str:
    s = int(max(s, 0))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def download(filekey: int, expect_gb: float | None = None) -> Path:
    out = root() / "download" / str(filekey)
    out.mkdir(parents=True, exist_ok=True)
    done = out / ".done"
    if done.exists():
        print(f"[aihub] {filekey}: 이미 받음 ({out})")
        return out
    tarp = out / "download.tar"
    if expect_gb:
        free = shutil.disk_usage(out).free / 1e9
        have = tarp.stat().st_size / 1e9 if tarp.exists() else 0
        need = 2.2 * expect_gb - have
        print(f"[aihub] 디스크 여유 {free:.0f}GB, 필요 ≈ {need:.0f}GB (크기 {expect_gb}GB × 2.2, 받은 {have:.1f}GB 제외)")
        if free < need or free < 1.2 * expect_gb:
            raise SystemExit(f"[aihub] 디스크 부족 → 멈춤 (여유 {free:.0f}GB < 필요 {need:.0f}GB)")
    url = f"{BASE}/down/{VER}/{DATASET}.do"
    pos = tarp.stat().st_size if tarp.exists() else 0
    headers = {"apikey": key()}
    if pos:
        headers["Range"] = f"bytes={pos}-"
        print(f"[aihub] 이어받기: {pos / 1e9:.2f}GB부터")
    with requests.get(url, params={"fileSn": filekey}, headers=headers, stream=True, timeout=60) as r:
        if r.status_code == 416:      # 이미 끝까지 받음
            pass
        elif r.status_code not in (200, 206):
            body = r.text[:300]
            raise SystemExit(f"[aihub] HTTP {r.status_code}: {body}")
        else:
            if r.status_code == 200 and pos:   # 서버가 Range를 무시 → 처음부터
                print("[aihub] 서버가 이어받기를 지원하지 않음 → 처음부터")
                pos = 0
            total = int(r.headers.get("Content-Length", 0)) + pos
            mode = "ab" if pos else "wb"
            t0, got0, last = time.monotonic(), pos, 0.0
            with open(tarp, mode) as f:
                for chunk in r.iter_content(chunk_size=1 << 22):
                    f.write(chunk)
                    pos += len(chunk)
                    now = time.monotonic()
                    if now - last >= 10:
                        last = now
                        rate = (pos - got0) / max(now - t0, 1e-6)
                        eta = (total - pos) / rate if (total and rate > 0) else 0
                        pct = f"{100 * pos / total:5.1f}%" if total else "?"
                        print(f"[aihub] {filekey}: {pos / 1e9:7.2f}/{total / 1e9:.2f}GB {pct} "
                              f"{rate / 1e6:6.1f}MB/s 남은 시간 {fmt_t(eta)}", flush=True)
    print(f"[aihub] {filekey}: 받기 완료 {tarp.stat().st_size / 1e9:.2f}GB → 풀기")
    with tarfile.open(tarp) as tf:
        tf.extractall(out, filter="data")
    merge_parts(out)
    tarp.unlink()
    done.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"[aihub] {filekey}: 완료 → {out}")
    return out


def merge_parts(d: Path) -> None:
    """aihubshell과 같이 NAME.part0, part1 ... 을 숫자 순서로 이어 붙인다."""
    groups: dict[Path, list[Path]] = {}
    for p in d.rglob("*.part*"):
        stem, _, n = p.name.rpartition(".part")
        if n.isdigit():
            groups.setdefault(p.with_name(stem), []).append(p)
    for target, parts in groups.items():
        parts.sort(key=lambda p: int(p.name.rpartition(".part")[2]))
        print(f"[aihub] 병합 {target.name} ({len(parts)}조각)")
        with open(target, "wb") as w:
            for p in parts:
                with open(p, "rb") as r:
                    shutil.copyfileobj(r, w, 1 << 24)
        for p in parts:
            p.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--labels", action="store_true", help="TL1~TL7, VL1")
    ap.add_argument("--filekey", type=int, nargs="*", default=[])
    a = ap.parse_args()
    if a.list:
        print(requests.get(f"{BASE}/info/{DATASET}.do", timeout=30).text)
        return
    keys = list(LABELS.values()) if a.labels else []
    keys += a.filekey
    for k in keys:
        download(k, SIZE_GB.get(k))


if __name__ == "__main__":
    main()
