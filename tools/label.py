"""터미널 라벨링: 구간을 하나씩 보여주고 키 하나로 라벨.

    python tools/label.py data/demo          (또는 results/demo.segments.jsonl, 또는 이름 demo)

키:  y = 착용자에게 한 말   n = 아님   w = 착용자 본인   s = 건너뛰기
     p = 다시 듣기   b = 이전으로   q = 저장하고 종료
결과: results/NAME.labels.csv (키를 누를 때마다 저장, 다시 실행하면 이어서)
착용자 구간(본인 발화 검출)은 자동으로 w 라벨이 붙고 건너뛴다(--all 이면 함께 보여줌).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

C = {"y": "\033[92m", "n": "\033[90m", "w": "\033[94m", "s": "\033[93m", "end": "\033[0m", "b": "\033[1m"}


def getch() -> str:
    if os.name == "nt":
        import msvcrt
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            msvcrt.getwch()
            return ""
        return ch.lower()
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1).lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def name_of(arg: str) -> str:
    p = Path(arg)
    n = p.name
    for suf in (".segments.jsonl", "_A.wav", "_B.wav", ".wav"):
        if n.endswith(suf):
            n = n[: -len(suf)]
    return n


def load_segments(results: Path, name: str) -> list[dict]:
    path = results / f"{name}.segments.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} 없음. 먼저: python tools/replay.py data/{name}")
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if not r.get("_meta"):
            out.append(r)
    return out


def load_labels(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {r["seg_id"]: r["label"] for r in csv.DictReader(f)}


def save_labels(path: Path, segs: list[dict], labels: dict[str, str]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seg_id", "label", "t_start", "t_end", "speaker_id", "is_wearer", "text"])
        for s in segs:
            if s["seg_id"] in labels:
                w.writerow([s["seg_id"], labels[s["seg_id"]], s["t_start"], s["t_end"], s.get("speaker_id"),
                            int(bool(s.get("is_wearer"))), s.get("text") or ""])
    os.replace(tmp, path)


def player(data_dir: Path, name: str):
    try:
        import sounddevice as sd
        from app.audio_source import read_wav
        a = read_wav(data_dir / f"{name}_A.wav")
        pb = data_dir / f"{name}_B.wav"
        b = read_wav(pb) if pb.exists() else a
    except Exception:
        return lambda s: None

    def play(s):
        x = a if s.get("is_wearer") else b
        seg = x[int(s["t_start"] * 16000): int(s["t_end"] * 16000)]
        try:
            sd.stop()
            sd.play(seg * min(1.0, 0.3 / (abs(seg).max() + 1e-6)) if len(seg) else seg, 16000)
        except Exception:
            pass
    return play


def main():
    from app.config import load_config, resolve_path
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--all", action="store_true", help="착용자 구간도 보여줌")
    ap.add_argument("--no-audio", action="store_true", help="자동 재생 끄기")
    ap.add_argument("--auto", action="store_true", help="data/NAME.truth.json(합성 시나리오 정답)으로 자동 라벨")
    args = ap.parse_args()
    cfg = load_config()
    results = resolve_path(cfg, "results_dir")
    name = name_of(args.scenario)
    segs = load_segments(results, name)
    path = results / f"{name}.labels.csv"
    labels = load_labels(path)
    for s in segs:
        if s.get("is_wearer") and s["seg_id"] not in labels:
            labels[s["seg_id"]] = "w"
    if args.auto:
        truth_p = resolve_path(cfg, "data_dir") / f"{name}.truth.json"
        truth = [u for u in json.loads(truth_p.read_text(encoding="utf-8")) if u["label"] in ("w", "y", "n")]
        n_auto = 0
        for s in segs:
            if s.get("is_wearer") or s.get("skip"):
                continue
            ov = [(min(s["t_end"], u["t_end"]) - max(s["t_start"], u["t_start"]), u) for u in truth]
            best, u = max(ov, key=lambda x: x[0]) if ov else (0, None)
            labels[s["seg_id"]] = u["label"] if u is not None and best > 0.25 * (s["t_end"] - s["t_start"]) else "s"
            n_auto += 1
        save_labels(path, segs, labels)
        print(f"자동 라벨(정답 파일 {truth_p.name}): {n_auto}개 → {path}")
        return
    todo = [s for s in segs if (args.all or not s.get("is_wearer")) and not s.get("skip")]
    play = (lambda s: None) if args.no_audio else player(resolve_path(cfg, "data_dir"), name)
    prev_wearer = {}
    last_w = None
    for s in segs:
        if s.get("is_wearer") and s.get("text"):
            last_w = s
        prev_wearer[s["seg_id"]] = last_w

    i = next((k for k, s in enumerate(todo) if s["seg_id"] not in labels), len(todo))
    print(f"{name}: 라벨할 구간 {len(todo)}개 (완료 {sum(1 for s in todo if s['seg_id'] in labels)})")
    print("y=나에게  n=아님  w=본인  s=건너뜀  p=듣기  b=이전  q=종료\n")
    while i < len(todo):
        s = todo[i]
        pw = prev_wearer.get(s["seg_id"])
        gap = s.get("gap")
        cur = labels.get(s["seg_id"], "")
        print(f"{C['b']}[{i + 1}/{len(todo)}] {s['seg_id']}  {s['t_start']:.1f}-{s['t_end']:.1f}s  "
              f"화자 #{s.get('speaker_id')} (sim {s.get('sim', 0):.2f})  "
              f"간격 {'-' if gap is None else f'{gap:+.1f}s'}{C['end']}"
              + (f"  현재={cur}" if cur else ""))
        if pw is not None and pw is not s:
            print(f"   {C['w']}나: {pw.get('text')}{C['end']}")
        llm = s.get("llm")
        llm_s = f"  (LLM: {'짝' if llm['pair'] else '아님'}/{llm['confidence']})" if llm else ""
        print(f"   ▶ {s.get('text') or '(텍스트 없음)'}{llm_s}")
        play(s)
        while True:
            k = getch()
            if k in ("y", "n", "w", "s"):
                labels[s["seg_id"]] = k
                save_labels(path, segs, labels)
                print(f"   → {C[k]}{k}{C['end']}")
                i += 1
                break
            if k == "p":
                play(s)
            elif k == "b":
                i = max(0, i - 1)
                break
            elif k in ("q", "\x03", "\x1b"):
                save_labels(path, segs, labels)
                print(f"저장: {path}")
                return
    save_labels(path, segs, labels)
    print(f"완료. 저장: {path}\n다음: python tools/evaluate.py")


if __name__ == "__main__":
    main()
