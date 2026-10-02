"""행사장 사전 점검: 클라이언트 ↔ 서버(SSH 터널) 네트워크를 30초 동안 잰다. 파이프라인(자막)에는 영향 없음.

    python tools/net_check.py                     # ws://localhost:8765, 30초
    python tools/net_check.py --seconds 60 --server ws://localhost:8765

시연과 같은 크기의 음성 블록(20ms, 2채널 16kHz int16 ≈ 520kbps)을 실시간 속도로 보내면서
클라이언트 측 RTT(200ms마다 cping)와 서버 측 RTT·지터·처리량을 표로 낸다. probe 모드라 서버는 받은 음성을 버린다.
판정 기준(대시보드 배너와 같다): RTT p95 ≤ 300ms, 처리량 ≥ 필요량의 95%, 지터 ≤ 30ms.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client_capture import BLOCK, SR, pack_block  # noqa: E402


async def run(url: str, seconds: float) -> dict:
    import websockets
    rtts: list[float] = []
    server: dict = {}
    sent = 0
    async with websockets.connect(url, max_size=2**20, ping_interval=None, compression=None, open_timeout=5) as ws:
        await ws.send(json.dumps({"type": "hello", "mode": "probe", "client": "net_check"}))
        json.loads(await asyncio.wait_for(ws.recv(), 5))

        async def recv():
            async for msg in ws:
                m = json.loads(msg)
                if m.get("type") == "cpong":
                    rtts.append((time.monotonic() - float(m["t"])) * 1000)
                elif m.get("type") == "ping":
                    await ws.send(json.dumps({"type": "pong", "id": m.get("id"), "t": m.get("t")}))
                elif m.get("type") == "stats":
                    server.update(m)
        rt = asyncio.create_task(recv())
        rng = np.random.default_rng(0)
        noise = (rng.standard_normal((2, BLOCK)) * 0.05).astype(np.float32)
        t0 = time.monotonic()
        next_ping = t0
        i = 0
        while time.monotonic() - t0 < seconds:
            due = t0 + (sent + 1) * BLOCK / SR
            now = time.monotonic()
            if now >= next_ping:
                i += 1
                await ws.send(json.dumps({"type": "cping", "id": i, "t": now}))
                next_ping = now + 0.2
            if now >= due:
                await ws.send(pack_block(sent, now, noise[0], noise[1]))
                sent += 1
            else:
                await asyncio.sleep(min(due - now, 0.005))
        await ws.send(json.dumps({"type": "stats?"}))
        await asyncio.sleep(1.0)
        rt.cancel()
    return {"rtts": rtts, "server": server, "sent": sent, "seconds": seconds}


def main(argv=None):
    ap = argparse.ArgumentParser(description="원격 시연 네트워크 점검(RTT·지터·처리량)")
    ap.add_argument("--server", default="ws://localhost:8765")
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args(argv)
    print(f"[net_check] {args.server} 에 {args.seconds:.0f}초 동안 시연 크기 음성을 보냅니다...")
    r = asyncio.run(run(args.server, args.seconds))
    rt = np.array(r["rtts"]) if r["rtts"] else np.array([np.nan])
    need_kbps = (2 * BLOCK * 2 + 20) * 8 * (SR / BLOCK) / 1000
    s = r["server"]
    got = s.get("kbps") or 0.0
    rows = [
        ("RTT (클라이언트 측) 중앙값", f"{np.nanmedian(rt):.0f} ms", ""),
        ("RTT p95", f"{np.nanpercentile(rt, 95):.0f} ms", "PASS" if np.nanpercentile(rt, 95) <= 300 else "FAIL"),
        ("RTT 최대", f"{np.nanmax(rt):.0f} ms", ""),
        ("RTT (서버 측) 중앙값", f"{s.get('server_rtt_ms')} ms" if s.get("server_rtt_ms") is not None else "–", ""),
        ("지터 (서버 측, RFC 3550)", f"{s.get('jitter_ms', float('nan')):.1f} ms",
         "PASS" if (s.get("jitter_ms") or 0) <= 30 else "WARN"),
        ("처리량 (서버 수신)", f"{got:.0f} kbps / 필요 {need_kbps:.0f} kbps", "PASS" if got >= 0.95 * need_kbps else "FAIL"),
        ("받은 블록 / 보낸 블록", f"{s.get('blocks', 0)} / {r['sent']}",
         "PASS" if s.get("blocks", 0) >= 0.99 * r["sent"] else "FAIL"),
        ("RTT 측정 수", f"{len(r['rtts'])}", ""),
    ]
    w = max(len(a) for a, _, _ in rows)
    print(f"\n| {'항목':<{w}} | 값 | 판정 |\n|---|---:|---|")
    for a, b, c in rows:
        print(f"| {a:<{w}} | {b} | {c} |")
    bad = [a for a, _, c in rows if c == "FAIL"]
    print("\n→ " + ("모두 통과 — 원격 시연 가능" if not bad else f"FAIL: {', '.join(bad)} → 핫스팟/유선 전환 또는 서버 로컬 재생 백업"))


if __name__ == "__main__":
    main()
