"""NetworkSource: 원격 클라이언트(tools/client_capture.py)가 WebSocket으로 보낸 2채널 음성을 받는다.

구조: 클라이언트(마이크 2개) → SSH 터널 → 이 서버 127.0.0.1:8765 → NetworkSource → 기존 파이프라인(Block 동일).

프로토콜(바이너리 프레임 = 20ms 블록 하나)
  헤더 20바이트 "<4sIdHBB": magic b"HMA1", seq(uint32), 클라이언트 단조 시간(float64 초), 채널당 샘플 수(uint16),
  채널 수(uint8, 1|2), 플래그(uint8, 예약). 뒤에 int16 PCM, 채널 순서대로(A 전체 → B 전체). 16kHz.
텍스트 프레임(JSON)
  클라이언트 → 서버: hello {mode: "audio"|"probe", client, devices, single_mic}, pong {id, t}, cping {id, t}
  서버 → 클라이언트: welcome {sr, block, jitter_ms}, ping {id, t}(1초마다, 서버 RTT 측정), cpong {id, t}, stats {...}

지터 버퍼: seq 순서로 내보낸다. 다음 seq가 없으면, 그 뒤 seq 블록이 도착한 지 jitter_ms(100ms)가 지날 때까지
기다렸다가 무음으로 채우고 누락으로 센다. 늦게 온 블록(이미 지나간 seq)은 버리고 센다.
재연결하면 새 세션: 끊긴 동안의 오디오는 없고(클라이언트가 버림), 스트림 시간은 끊김 없이 이어진다.
probe 모드(tools/net_check.py)는 파이프라인에 넣지 않고 지연·처리량만 잰다.
"""
from __future__ import annotations

import asyncio
import json
import struct
import threading
import time
from collections import deque
from typing import Optional

import numpy as np

from app.audio_source import AudioSource, Block

MAGIC = b"HMA1"
HDR = struct.Struct("<4sIdHBB")


def pack_block(seq: int, t_client: float, a: np.ndarray, b: Optional[np.ndarray]) -> bytes:
    chans = [a] if b is None else [a, b]
    pcm = np.concatenate([np.clip(np.asarray(c, np.float32), -1, 1) for c in chans])
    return HDR.pack(MAGIC, seq & 0xFFFFFFFF, t_client, len(a), len(chans), 0) + (pcm * 32767).astype("<i2").tobytes()


def unpack_block(buf: bytes):
    magic, seq, tc, n, nch, _ = HDR.unpack_from(buf, 0)
    if magic != MAGIC:
        raise ValueError("bad magic")
    pcm = np.frombuffer(buf, dtype="<i2", offset=HDR.size, count=n * nch).astype(np.float32) / 32767.0
    a = pcm[:n]
    b = pcm[n:2 * n] if nch >= 2 else None
    return seq, tc, a, b


class NetworkSource(AudioSource):
    realtime = True

    def __init__(self, cfg: dict, log=print):
        n = cfg.get("network", {}) or {}
        self.log = log
        self.sr = cfg["audio"]["sample_rate"]
        self.block = int(self.sr * cfg["audio"]["block_ms"] / 1000)
        self.host = n.get("host", "127.0.0.1")
        self.port = int(n.get("port", 8765))
        self.jitter_s = float(n.get("jitter_ms", 100)) / 1000
        self.max_buffer = int(n.get("max_buffer_blocks", 250))
        self.rtt_warn = float(n.get("rtt_warn_ms", 300))
        self.missing_warn = float(n.get("missing_warn_pct", 1.0))
        self._recent: deque = deque(maxlen=1500)   # 최근 30초 블록: (시각, 누락 여부)
        self.single_mic = bool(cfg["audio"].get("single_mic"))
        self.cv = threading.Condition()
        self.buf: dict[int, tuple] = {}     # seq -> (a, b, 도착 시각)
        self.next_seq: Optional[int] = None
        self.n_out = 0
        self.session = 0
        self.connected = False
        self.client: dict = {}
        self.last_disconnect: Optional[float] = None
        self.t_start = time.monotonic()
        # 통계(세션 누적)
        self.received = self.missing = self.late = self.dups = self.overflow = 0
        self.bytes_in = 0
        self.rtt_ms: deque = deque(maxlen=60)
        self.jitter_ms = 0.0
        self._prev_transit: Optional[float] = None
        self._rate: deque = deque(maxlen=200)   # (도착 시각, 바이트)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._active_ws = None
        self.bound_port: Optional[int] = None

    # --------------------------------------------------------- 수신 서버
    def start(self) -> None:
        ready = threading.Event()

        def run():
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._serve(ready))
        self._thread = threading.Thread(target=run, name="net-source", daemon=True)
        self._thread.start()
        ready.wait(10)
        if self.bound_port is None:
            raise RuntimeError(f"음성 수신 포트 {self.host}:{self.port} 열기 실패")
        self.log(f"[net] 음성 수신 대기 ws://{self.host}:{self.bound_port}  (지터 버퍼 {self.jitter_s * 1000:.0f}ms)")

    async def _serve(self, ready: threading.Event):
        import websockets
        try:
            self._server = await websockets.serve(self._handler, self.host, self.port, max_size=2**20,
                                                  ping_interval=None, compression=None)
            self.bound_port = self._server.sockets[0].getsockname()[1]
        except OSError as e:
            self.log(f"[net] 포트 열기 실패: {e}")
            ready.set()
            return
        ready.set()
        while not self._stop.is_set():
            await asyncio.sleep(0.2)
        self._server.close()
        await self._server.wait_closed()

    async def _handler(self, ws):
        hello = {"mode": "audio"}
        try:
            first = await asyncio.wait_for(ws.recv(), timeout=10)
            if isinstance(first, str):
                hello = json.loads(first)
        except Exception:
            return
        mode = hello.get("mode", "audio")
        if mode == "probe":
            await ws.send(json.dumps({"type": "welcome", "sr": self.sr, "block": self.block,
                                      "jitter_ms": round(self.jitter_s * 1000), "mode": mode}))
            await self._probe(ws)
            return
        if self._active_ws is not None:   # 새 연결이 이긴다(이전 클라이언트는 끊음)
            try:
                await self._active_ws.close()
            except Exception:
                pass
        self._active_ws = ws
        with self.cv:
            # stream_t: 이 세션 첫 블록의 서버 스트림 시간(latency_bench가 자막 시각을 보낸 블록에 맞출 때 쓴다).
            # 버퍼를 비웠으므로 다음에 내보내는 블록이 이 세션의 seq 첫 블록이다.
            t0_stream = self.n_out / self.sr
            self.session += 1
            self.next_seq = None      # 새 세션: 끊긴 동안의 오디오는 없다
            self.buf.clear()
            self._prev_transit = None
            self.connected = True
            self.client = {k: hello.get(k) for k in ("client", "devices", "single_mic", "replay")}
            self.client["since"] = time.time()
        await ws.send(json.dumps({"type": "welcome", "sr": self.sr, "block": self.block, "mode": mode,
                                  "jitter_ms": round(self.jitter_s * 1000), "stream_t": t0_stream,
                                  "session": self.session}))
        self.log(f"[net] 클라이언트 연결: {self.client.get('client')} 장치={self.client.get('devices')}")
        pinger = asyncio.create_task(self._pinger(ws))
        try:
            async for msg in ws:
                if isinstance(msg, bytes):
                    self._on_block(msg)
                else:
                    self._on_text(ws, msg)
        except Exception:
            pass
        finally:
            pinger.cancel()
            if self._active_ws is ws:
                self._active_ws = None
                with self.cv:
                    self.connected = False
                    self.last_disconnect = time.time()
                    self.cv.notify_all()
                self.log("[net] 클라이언트 연결 끊김")

    async def _pinger(self, ws):
        i = 0
        while True:
            i += 1
            try:
                await ws.send(json.dumps({"type": "ping", "id": i, "t": time.monotonic()}))
            except Exception:
                return
            await asyncio.sleep(1.0)

    def _on_text(self, ws, msg: str) -> None:
        try:
            m = json.loads(msg)
        except json.JSONDecodeError:
            return
        if m.get("type") == "pong" and m.get("t") is not None:
            self.rtt_ms.append((time.monotonic() - float(m["t"])) * 1000)
        elif m.get("type") == "cping":
            asyncio.ensure_future(ws.send(json.dumps({"type": "cpong", "id": m.get("id"), "t": m.get("t")})))

    def _on_block(self, data: bytes) -> None:
        now = time.monotonic()
        try:
            seq, tc, a, b = unpack_block(data)
        except Exception:
            return
        if len(a) != self.block:
            return
        if b is None:
            b = a
        transit = now - tc            # RFC 3550 도착 간 지터(시계 차이는 상쇄된다)
        if self._prev_transit is not None:
            d = abs(transit - self._prev_transit) * 1000
            self.jitter_ms += (d - self.jitter_ms) / 16
        self._prev_transit = transit
        self.bytes_in += len(data)
        self._rate.append((now, len(data)))
        with self.cv:
            if self.next_seq is not None and seq < self.next_seq:
                self.late += 1
                return
            if seq in self.buf:
                self.dups += 1
                return
            self.buf[seq] = (a, b, now)
            self.received += 1
            if len(self.buf) > self.max_buffer:   # 파이프라인이 못 따라옴: 가장 오래된 것부터 버린다
                for s in sorted(self.buf)[: len(self.buf) - self.max_buffer]:
                    del self.buf[s]
                    self.overflow += 1
                self.next_seq = min(self.buf)
            self.cv.notify_all()

    async def _probe(self, ws):
        """net_check: 지연(클라이언트 cping/cpong + 서버 ping/pong)과 처리량만. 파이프라인에는 넣지 않는다."""
        n = nbytes = 0
        prev = None
        jit = 0.0
        rtts: list[float] = []
        t0 = time.monotonic()
        pinger = asyncio.create_task(self._pinger(ws))
        try:
            async for msg in ws:
                now = time.monotonic()
                if isinstance(msg, bytes):
                    try:
                        _, tc, _, _ = unpack_block(msg)
                    except Exception:
                        continue
                    n += 1
                    nbytes += len(msg)
                    tr = now - tc
                    if prev is not None:
                        jit += (abs(tr - prev) * 1000 - jit) / 16
                    prev = tr
                    continue
                m = json.loads(msg)
                if m.get("type") == "cping":
                    await ws.send(json.dumps({"type": "cpong", "id": m.get("id"), "t": m.get("t")}))
                elif m.get("type") == "pong":
                    rtts.append((now - float(m["t"])) * 1000)
                elif m.get("type") == "stats?":
                    dt = max(now - t0, 1e-6)
                    await ws.send(json.dumps({"type": "stats", "blocks": n, "kbps": round(nbytes * 8 / dt / 1000, 1),
                                              "jitter_ms": round(jit, 2),
                                              "server_rtt_ms": round(float(np.median(rtts)), 1) if rtts else None}))
        except Exception:
            pass
        finally:
            pinger.cancel()

    # --------------------------------------------------------- 파이프라인 쪽
    def read(self, timeout: float = 1.0) -> Optional[Block]:
        deadline = time.monotonic() + timeout
        with self.cv:
            while True:
                now = time.monotonic()
                if self.next_seq is None and self.buf:
                    self.next_seq = min(self.buf)
                if self.next_seq is not None:
                    item = self.buf.pop(self.next_seq, None)
                    if item is not None:
                        self.next_seq += 1
                        self._recent.append((now, False))
                        return self._emit(item[0], item[1])
                    later = [v[2] for s, v in self.buf.items() if s > self.next_seq]
                    if later and now - min(later) >= self.jitter_s:
                        self.missing += 1          # 기다려도 안 옴 → 무음으로 채움
                        self.next_seq += 1
                        self._recent.append((now, True))
                        z = np.zeros(self.block, np.float32)
                        return self._emit(z, z)
                if now >= deadline or self._stop.is_set():
                    return None
                self.cv.wait(min(0.005, deadline - now))

    def _emit(self, a: np.ndarray, b: np.ndarray) -> Block:
        t = self.n_out / self.sr
        self.n_out += self.block
        return Block(t=t, a=a, b=a.copy() if self.single_mic else b, wall=time.monotonic())

    def stats(self) -> dict:
        now = time.monotonic()
        rate = [x for x in self._rate if now - x[0] <= 5.0]
        kbps = round(sum(x[1] for x in rate) * 8 / 5.0 / 1000, 1) if rate else 0.0
        total = self.received + self.missing
        rtt = list(self.rtt_ms)
        rec = [m for t, m in self._recent if now - t <= 30.0]
        return {
            "connected": self.connected, "session": self.session, "client": self.client.get("client"),
            "rtt_ms": round(rtt[-1], 1) if rtt else None,
            "rtt_p95_ms": round(float(np.percentile(rtt, 95)), 1) if rtt else None,
            "jitter_ms": round(self.jitter_ms, 1), "received": self.received, "missing": self.missing,
            "missing_pct": round(100 * self.missing / total, 2) if total else 0.0,
            "missing_pct_30s": round(100 * sum(rec) / len(rec), 2) if rec else 0.0,
            "rtt_warn_ms": self.rtt_warn, "missing_warn_pct": self.missing_warn,
            "late": self.late, "dups": self.dups, "overflow": self.overflow, "kbps": kbps,
            "buffered": len(self.buf), "port": self.bound_port,
            "disconnected_s": round(time.time() - self.last_disconnect, 1)
            if (not self.connected and self.last_disconnect) else None,
        }

    def stop(self) -> None:
        self._stop.set()
        with self.cv:
            self.cv.notify_all()
