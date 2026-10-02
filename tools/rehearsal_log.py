"""리허설 기록: 회차마다 시연 단계별 성공/실패를 키 입력으로 남기고 누적 성공률 표를 만든다.

    python tools/rehearsal_log.py                 # 리허설 시작 전에 실행 → Enter(시작) … 시연 … Enter(끝) → 단계별 y/n/s
    python tools/rehearsal_log.py --report        # 표만 다시 만들기
    python tools/rehearsal_log.py --log logs/run_20261003_101500.jsonl   # 서버 이벤트 로그 지정

- 단계: 자동 등록, 함정 거르기, 사이렌 알림, 호명 알림, 판정 근거 표시. 키: y=성공, n=실패, s=이번 회차에서 안 함.
- 서버 이벤트 로그(logs/run_*.jsonl)가 이 컴퓨터에 있으면(서버에서 실행) 시작~끝 시각의 이벤트로 단서를 보여 준다
  (대화 상대 등록, 사이렌·호명 알림, 판정 근거 칩 수). 최종 판단은 사람이 한다. 로그 파일 이름은 회차 기록에 남긴다.
- 기록: results/rehearsal.jsonl(회차별), results/rehearsal.md(회차 표 + 단계별 누적 성공률).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REC = ROOT / "results" / "rehearsal.jsonl"
MD = ROOT / "results" / "rehearsal.md"
STEPS = [("register", "자동 등록", "상대가 대답하면 '대화 상대 #N 추가' 배너, 화자 목록이 '대화 상대'"),
         ("trap", "함정 거르기", "착용자 질문 직후 옆 사람이 다른 사람에게 한 말이 회색 한 줄(짝 아님)"),
         ("siren", "사이렌 알림", "사이렌 소리에 빨간 깜박임(+ 폰 진동)"),
         ("namecall", "호명 알림", "'민수야' 호명에 노란 배너"),
         ("evidence", "판정 근거 표시", "상대 자막 아래 근거 칩(응답 간격 · 질문→대답 · 화자 #N)")]

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def latest_log() -> Path | None:
    logs = sorted((ROOT / "logs").glob("run_*.jsonl"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def hints(log: Path | None, t0: float, t1: float) -> dict:
    """이벤트 로그에서 [t0, t1](벽시계) 범위의 단서."""
    h = {"partner_added": 0, "siren": 0, "name": 0, "chips": 0, "captions": 0, "other_after_wearer": 0}
    if log is None or not log.exists():
        return h
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = e.get("ts")
        if ts is None or not (t0 <= ts <= t1):
            continue
        typ = e.get("type")
        if typ == "partner_added":
            h["partner_added"] += 1
        elif typ == "alert":
            h["name" if e.get("kind") == "name" else "siren"] += 1
        elif typ in ("caption", "caption_update"):
            h["captions"] += typ == "caption"
            if e.get("role") == "partner" and e.get("chip"):
                h["chips"] += 1
            ev = e.get("evidence") or {}
            if e.get("role") in ("other", "unknown") and ev.get("T", 0) >= 0.5 and typ == "caption_update":
                h["other_after_wearer"] += 1
    return h


def ask(prompt: str, keys: str) -> str:
    while True:
        v = input(prompt).strip().lower()
        if v in keys:
            return v
        print(f"  {'/'.join(keys)} 중 하나")


def report() -> None:
    rows = [json.loads(l) for l in REC.read_text(encoding="utf-8").splitlines()] if REC.exists() else []
    md = ["# 리허설 기록", "", f"회차 {len(rows)}개. y=성공 n=실패 -=안 함. (tools/rehearsal_log.py)", "",
          "| 회차 | 시각 | " + " | ".join(n for _, n, _ in STEPS) + " | 서버 로그 | 메모 |",
          "|---:|---|" + ":-:|" * len(STEPS) + "---|---|"]
    for i, r in enumerate(rows, 1):
        marks = " | ".join({"y": "✓", "n": "✗"}.get(r["steps"].get(k), "–") for k, _, _ in STEPS)
        md.append(f"| {i} | {r['start'][5:16]} | {marks} | {r.get('log') or '–'} | {r.get('note', '')} |")
    md += ["", "## 단계별 누적 성공률", "", "| 단계 | 성공 / 시도 | 성공률 | 최근 5회 |", "|---|---:|---:|---|"]
    for k, n, _ in STEPS:
        v = [r["steps"].get(k) for r in rows if r["steps"].get(k) in ("y", "n")]
        ok = sum(x == "y" for x in v)
        recent = "".join({"y": "✓", "n": "✗"}[x] for x in v[-5:])
        rate = f"{100 * ok / len(v):.0f}%" if v else "–"
        md.append(f"| {n} | {ok} / {len(v)} | {rate} | {recent or '–'} |")
    allv = [r for r in rows if all(r["steps"].get(k) in ("y", "s") for k, _, _ in STEPS)]
    md += ["", f"모든 단계 성공(안 한 단계 제외) 회차: {len(allv)} / {len(rows)}"]
    MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--log", default=None, help="서버 이벤트 로그(기본: logs/ 의 가장 최근 run_*.jsonl)")
    a = ap.parse_args()
    if a.report:
        report()
        return
    n = (len(REC.read_text(encoding="utf-8").splitlines()) if REC.exists() else 0) + 1
    input(f"[리허설 {n}회차] 시작할 때 Enter ")
    t0 = time.time()
    print("  기록 중… 시연이 끝나면 Enter")
    input()
    t1 = time.time()
    log = Path(a.log) if a.log else latest_log()
    h = hints(log, t0, t1)
    if log is not None and log.exists():
        print(f"  서버 로그 {log.name} 단서: 대화 상대 등록 {h['partner_added']} · 사이렌/경보 알림 {h['siren']} · "
              f"호명 알림 {h['name']} · 근거 칩이 붙은 상대 자막 {h['chips']} · 착용자 직후 접힌 말 {h['other_after_wearer']} · "
              f"자막 {h['captions']}")
    else:
        print("  (서버 로그를 이 컴퓨터에서 찾을 수 없음 — 서버에서 실행하거나 --log 로 지정. 판단은 화면으로)")
    steps = {}
    for k, name, desc in STEPS:
        steps[k] = ask(f"  {name} ({desc}) 성공? [y/n/s] ", "yns")
    note = input("  메모(엔터=없음): ").strip()
    rec = {"start": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)), "dur_s": round(t1 - t0),
           "steps": steps, "log": log.name if log is not None and log.exists() else (a.log or None), "hints": h, "note": note}
    REC.parent.mkdir(parents=True, exist_ok=True)
    with open(REC, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    report()


if __name__ == "__main__":
    main()
