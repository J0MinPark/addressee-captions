"""자막 지연 측정(발화 종료 → 대시보드 표시). 클라이언트에서 녹음을 실시간 속도로 보내고 대시보드 이벤트를 받는다.

    python tools/latency_bench.py --remote --replay data/demo_trap        # Book5: SSH 터널 경유(사람이 실행)
    python tools/latency_bench.py --replay $HEARME_DATA/data/demo_trap   # 서버 localhost(잠정 측정)

- 음성: ws://localhost:8765 (client_capture와 같은 전송), 이벤트: ws://localhost:8000/ws (대시보드와 같은 이벤트)
- 지연 = 대시보드 이벤트 수신 시각 − 그 자막 구간의 마지막 음성 블록을 클라이언트가 보낸 시각(같은 시계).
  서버 스트림 시간 ↔ 블록 seq 대응은 연결할 때 서버가 알려 준 stream_t로 맞춘다.
  first = 처음 표시(caption), final = LLM 판정 반영 후(caption_update, 없으면 first).
- 시작 전에 대시보드 '초기화'(화자·상태 리셋)를 보낸다(--no-reset 으로 끔). 시연 중에는 돌리지 말 것.
- .truth.json(합성 시나리오)이 있으면 대화 상대 등록·자막 표시가 정답과 맞는지도 요약한다.
결과: results/latency_<녹음>_<remote|local>_<시각>.csv (구간별) + 콘솔 요약(평균, p95). 클라이언트 패키지만 필요.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client_capture import BLOCK, SR, ReplayCapture, Streamer, replay_paths  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


async def bench(args) -> dict:
    import websockets
    cap = ReplayCapture(args.replay)
    sent_t: dict[int, float] = {}
    sess = {"stream_t": None, "seq0": None}

    def on_connect(welcome, seq):
        if sess["stream_t"] is None:
            sess["stream_t"], sess["seq0"] = float(welcome.get("stream_t", 0.0)), seq

    def on_sent(seq, t, pos):
        sent_t[seq] = t

    caps: dict[str, dict] = {}
    partners: list[dict] = []
    t_wall0 = time.monotonic()

    def block_time(t_stream: float):
        """서버 스트림 시간 → 그 시각까지의 음성을 담은 마지막 블록을 보낸 클라이언트 시각."""
        if sess["stream_t"] is None:
            return None
        idx = int(np.ceil((t_stream - sess["stream_t"]) * SR / BLOCK)) - 1
        return sent_t.get(sess["seq0"] + max(idx, 0))

    async with websockets.connect(args.dashboard, max_size=2**22, ping_interval=None) as dws:
        snap = json.loads(await dws.recv())
        cfg_name = None
        if not args.no_reset:
            await dws.send(json.dumps({"cmd": "reset"}))
            await asyncio.sleep(0.5)
        st = Streamer(cap, args.server, f"latency_bench:{socket.gethostname()}", on_sent=on_sent,
                      on_connect=on_connect, quiet=True)
        stream_task = asyncio.create_task(st.run())

        async def events():
            async for msg in dws:
                now = time.monotonic()
                ev = json.loads(msg)
                typ = ev.get("type")
                if typ == "caption" and sess["stream_t"] is not None and ev.get("t_start", -1) >= sess["stream_t"]:
                    c = caps.setdefault(ev["id"], {})
                    c.update(id=ev["id"], role=ev.get("role"), first_role=ev.get("role"), speaker_id=ev.get("speaker_id"),
                             text=ev.get("text"), t_start=ev["t_start"], t_end=ev["t_end"], prob=ev.get("prob"),
                             pending=ev.get("pending_llm"), server_latency_ms=ev.get("latency_ms"), recv_first=now,
                             recv_final=now)
                elif typ == "caption_update" and ev.get("id") in caps:
                    c = caps[ev["id"]]
                    c.update(role=ev.get("role", c["role"]), prob=ev.get("prob", c.get("prob")), recv_final=now,
                             llm_ms=ev.get("llm_ms"))
                elif typ == "partner_added":
                    partners.append({"speaker_id": ev.get("speaker_id"), "label": ev.get("label"),
                                     "t_wall": round(now - t_wall0, 2)})
                elif typ == "snapshot":
                    pass

        ev_task = asyncio.create_task(events())
        await stream_task
        await asyncio.sleep(args.tail)        # 마지막 판정(LLM)까지 기다림
        ev_task.cancel()
        _ = snap, cfg_name

    rows = []
    for c in sorted(caps.values(), key=lambda c: c["t_end"]):
        ts = block_time(c["t_end"])
        if ts is None:
            continue
        c["lat_first_ms"] = round((c["recv_first"] - ts) * 1000, 1)
        c["lat_final_ms"] = round((c["recv_final"] - ts) * 1000, 1)
        c["t_start_rel"] = round(c["t_start"] - sess["stream_t"], 3)
        c["t_end_rel"] = round(c["t_end"] - sess["stream_t"], 3)
        rows.append(c)
    return {"rows": rows, "partners": partners, "sent": st.sent, "reconnects": st.reconnects,
            "duration": cap.duration}


def truth_check(rows: list[dict], truth_path: Path) -> list[str]:
    """정답 발화(합성 시나리오)와 시간 겹침으로 연결: y가 크게(partner) 표시됐는지, n이 크게 표시됐는지."""
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    out = {"y": [0, 0], "n": [0, 0], "w": [0, 0]}
    for tr in truth:
        lab = tr["label"]
        if lab not in out:
            continue
        hit = [r for r in rows if min(r["t_end_rel"], tr["t_end"]) - max(r["t_start_rel"], tr["t_start"]) > 0.3]
        out[lab][1] += 1
        if lab == "w":
            out[lab][0] += any(r["role"] == "wearer" for r in hit)
        else:
            out[lab][0] += any(r["role"] == "partner" for r in hit)
    return [f"정답 y(나에게) {out['y'][1]}개 중 크게 표시 {out['y'][0]}",
            f"정답 n(다른 대화·함정) {out['n'][1]}개 중 크게 표시(오표시) {out['n'][0]}",
            f"착용자 발화 {out['w'][1]}개 중 착용자 자막 {out['w'][0]}"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="자막 지연(발화 종료 → 대시보드) 측정")
    ap.add_argument("--replay", required=True, help="녹음 접두사(예: data/demo_trap)")
    ap.add_argument("--remote", action="store_true", help="원격 경로 측정으로 표시(Book5에서 SSH 터널 경유)")
    ap.add_argument("--server", default="ws://localhost:8765")
    ap.add_argument("--dashboard", default="ws://localhost:8000/ws")
    ap.add_argument("--no-reset", action="store_true")
    ap.add_argument("--tail", type=float, default=6.0, help="재생 후 판정을 기다리는 시간(초)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    res = asyncio.run(bench(args))
    rows = res["rows"]
    name = Path(replay_paths(args.replay)[0]).name[:-6]
    tag = "remote" if args.remote else "local"
    out = Path(args.out) if args.out else ROOT / "results" / f"latency_{name}_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["id", "role", "first_role", "speaker_id", "t_start_rel", "t_end_rel", "prob", "lat_first_ms",
            "lat_final_ms", "server_latency_ms", "llm_ms", "text"]
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    others = [r for r in rows if r["role"] != "wearer"]
    lf = [r["lat_first_ms"] for r in others]
    ll = [r["lat_final_ms"] for r in others]
    print(f"\n[{tag}] {name}: 재생 {res['duration']:.0f}s, 보낸 블록 {res['sent']}, 재연결 {res['reconnects']}, "
          f"자막 {len(rows)}개(착용자 제외 {len(others)})")
    print("| 지연(발화 종료 → 표시) | 평균 | p95 | 최대 |\n|---|---:|---:|---:|")
    if lf:
        print(f"| 처음 표시 | {np.mean(lf):.0f} ms | {pct(lf, 95):.0f} ms | {max(lf):.0f} ms |")
        print(f"| 최종(LLM 반영) | {np.mean(ll):.0f} ms | {pct(ll, 95):.0f} ms | {max(ll):.0f} ms |")
    print(f"대화 상대 등록: {[(p['speaker_id'], p['t_wall']) for p in res['partners']] or '없음'}")
    for r in rows:
        print(f"  {r['t_start_rel']:6.1f}-{r['t_end_rel']:6.1f}s {r['role']:<8} #{r['speaker_id']} "
              f"{(r['prob'] if r['prob'] is not None else float('nan')):.2f} {r['lat_final_ms']:6.0f}ms  {r['text'][:40]}")
    pj = replay_paths(args.replay)[0]
    truth = Path(str(pj)[:-6] + ".truth.json")
    if truth.exists():
        print("정답 대조:", " · ".join(truth_check(rows, truth)))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
