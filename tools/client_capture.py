"""원격 시연 클라이언트(Galaxy Book5 등 GPU 없는 노트북): 마이크 2개 → 20ms 블록 → WebSocket → 서버.

    pip install -r requirements-client.txt          # numpy, sounddevice, websockets 만
    python tools/client_capture.py --list           # 입력 장치 목록
    python tools/client_capture.py --wearer "Lavalier" --ambient "USB Audio"
    python tools/client_capture.py --single-mic --wearer "마이크"        # 마이크 하나(서버도 --single-mic)
    python tools/client_capture.py --replay data/demo                   # 마이크 대신 녹음 파일(백업)
    python tools/client_capture.py --wearer ... --ambient ... --record rec/reh01   # 보낸 음성을 rec/reh01_A.wav, _B.wav 로 저장

기본 서버 주소는 ws://localhost:8765 (SSH 터널: ssh -L 8000:localhost:8000 -L 8765:localhost:8765 사용자@서버).
연결이 끊기면 2초마다 다시 연결한다. 끊긴 동안의 오디오는 버린다(밀린 음성을 나중에 보내지 않는다).
이 파일은 저장소의 다른 코드에 의존하지 않는다(프로토콜은 app/net_source.py와 같다).
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import struct
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Callable, Optional

import numpy as np

SR = 16000
BLOCK = 320                      # 20ms
MAGIC = b"HMA1"
HDR = struct.Struct("<4sIdHBB")  # magic, seq, 클라이언트 단조 시간, 채널당 샘플 수, 채널 수, 플래그

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def pack_block(seq: int, t_client: float, a: np.ndarray, b: Optional[np.ndarray]) -> bytes:
    chans = [a] if b is None else [a, b]
    pcm = np.concatenate([np.clip(np.asarray(c, np.float32), -1, 1) for c in chans])
    return HDR.pack(MAGIC, seq & 0xFFFFFFFF, t_client, len(a), len(chans), 0) + (pcm * 32767).astype("<i2").tobytes()


# ------------------------------------------------------------------ 리샘플(numpy만)
class Resampler:
    """정수배 다운샘플(48k→16k)은 상태 유지 FIR, 그 외(44.1k)는 상태 유지 선형 보간."""

    def __init__(self, sr_in: int, sr_out: int = SR):
        self.sr_in, self.sr_out = int(sr_in), int(sr_out)
        self.ratio = self.sr_in // self.sr_out if self.sr_in % self.sr_out == 0 else None
        if self.ratio and self.ratio > 1:
            n = 63
            k = np.arange(n) - (n - 1) / 2
            fc = 0.45 / self.ratio
            self.taps = (2 * fc * np.sinc(2 * fc * k) * np.hamming(n)).astype(np.float32)
            self.taps /= self.taps.sum()
            self.hist = np.zeros(n - 1, np.float32)
            self.phase = 0
        self.pos = 0.0
        self.prev = np.zeros(0, np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float32)
        if self.sr_in == self.sr_out:
            return x
        if self.ratio:
            buf = np.concatenate([self.hist, x])
            y = np.convolve(buf, self.taps, mode="valid")
            self.hist = buf[-(len(self.taps) - 1):]
            out = y[self.phase::self.ratio]
            self.phase = (self.phase - len(y)) % self.ratio
            return out.astype(np.float32)
        buf = np.concatenate([self.prev, x])
        if len(buf) < 2:
            self.prev = buf
            return np.zeros(0, np.float32)
        step = self.sr_in / self.sr_out
        idx = np.arange(self.pos, len(buf) - 1, step)
        out = np.interp(idx, np.arange(len(buf)), buf).astype(np.float32)
        self.pos = (idx[-1] + step) - (len(buf) - 1) if len(idx) else self.pos - (len(buf) - 1)
        self.prev = buf[-1:]
        return out


# ------------------------------------------------------------------ 장치
def list_devices() -> None:
    import sounddevice as sd
    apis = sd.query_hostapis()
    print("입력 장치 (--wearer / --ambient 에 이름 일부를 적으세요)")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"  [{i:2d}] {d['name']}  ({apis[d['hostapi']]['name']}, {int(d['default_samplerate'])}Hz)")
    try:
        print(f"기본 입력: [{sd.default.device[0]}]")
    except Exception:
        pass


def find_input(name: str, prefer_wasapi: bool = True) -> Optional[int]:
    import sounddevice as sd
    if not name:
        return None
    if name.isdigit():
        return int(name)
    apis = sd.query_hostapis()
    hits = [(i, d) for i, d in enumerate(sd.query_devices())
            if d["max_input_channels"] > 0 and name.lower() in d["name"].lower()]
    if not hits:
        raise SystemExit(f"입력 장치 '{name}'를 찾을 수 없음 → --list 로 이름 확인")
    if prefer_wasapi:
        w = [h for h in hits if "WASAPI" in apis[h[1]["hostapi"]]["name"]]
        if w:
            return w[0][0]
    return hits[0][0]


class _Chan:
    def __init__(self):
        self.q = collections.deque()
        self.n = 0
        self.lock = threading.Lock()

    def put(self, x):
        with self.lock:
            self.q.append(x)
            self.n += len(x)

    def take(self, n) -> np.ndarray:
        with self.lock:
            parts, need = [], n
            while need > 0 and self.q:
                x = self.q[0]
                if len(x) <= need:
                    parts.append(x)
                    self.q.popleft()
                    need -= len(x)
                else:
                    parts.append(x[:need])
                    self.q[0] = x[need:]
                    need = 0
            self.n -= (n - need)
            return np.concatenate(parts) if parts else np.zeros(0, np.float32)

    def clear(self):
        with self.lock:
            self.q.clear()
            self.n = 0


class MicCapture:
    """마이크 1~2개를 16kHz float32로. blocks()가 20ms (a, b) 를 낸다."""

    def __init__(self, wearer: str, ambient: str, single_mic: bool, drift_max_s: float = 0.5):
        import sounddevice as sd
        self.sd = sd
        self.single = single_mic
        self.dev = [find_input(wearer)] + ([] if single_mic else [find_input(ambient)])
        if not single_mic and self.dev[0] == self.dev[1]:
            print("[client] 경고: 두 마이크가 같은 장치 → --single-mic 으로 실행하세요")
        self.ch = [_Chan() for _ in self.dev]
        self.streams = []
        self.names = []
        self.status_errors = 0
        self.drift_max = int(drift_max_s * SR)

    def _open(self, dev, ch: _Chan):
        sd = self.sd
        info = sd.query_devices(dev if dev is not None else sd.default.device[0], "input")
        api = sd.query_hostapis(info["hostapi"])["name"]
        sr, extra = SR, None
        try:
            if "WASAPI" in api:
                extra = sd.WasapiSettings(auto_convert=True)
            sd.check_input_settings(device=dev, samplerate=SR, channels=1, dtype="float32", extra_settings=extra)
        except Exception:
            extra, sr = None, int(info["default_samplerate"])
        rs = Resampler(sr, SR)

        def cb(indata, frames, t, status):
            if status:
                self.status_errors += 1
            ch.put(rs(indata[:, 0].copy()))

        st = sd.InputStream(device=dev, samplerate=sr, channels=1, dtype="float32", blocksize=int(sr * 0.02),
                            callback=cb, extra_settings=extra)
        self.names.append(f"{info['name']} ({api}, {sr}Hz)")
        return st

    def start(self):
        for d, c in zip(self.dev, self.ch):
            self.streams.append(self._open(d, c))
        for s in self.streams:
            s.start()
        print("[client] 열린 장치:", " | ".join(self.names))

    def ready(self) -> bool:
        return all(c.n >= BLOCK for c in self.ch)

    def take(self):
        if not self.single:
            diff = self.ch[0].n - self.ch[1].n
            if abs(diff) > self.drift_max:   # 장치 간 클럭 드리프트
                (self.ch[0] if diff > 0 else self.ch[1]).take(abs(diff) - BLOCK)
        a = self.ch[0].take(BLOCK)
        b = None if self.single else self.ch[1].take(BLOCK)
        return a, b

    def discard(self):
        for c in self.ch:
            c.clear()

    def stop(self):
        for s in self.streams:
            try:
                s.stop()
                s.close()
            except Exception:
                pass


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        fs, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw != 2:
        raise SystemExit(f"{path}: 16비트 PCM WAV만 지원")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if fs != SR:
        x = Resampler(fs, SR)(x)
    return x


def replay_paths(prefix: str) -> tuple[Path, Optional[Path]]:
    p = str(prefix)
    for suf in ("_A.wav", "_B.wav", ".wav"):
        if p.endswith(suf):
            p = p[: -len(suf)]
    pa, pb = Path(p + "_A.wav"), Path(p + "_B.wav")
    if not pa.exists():
        raise SystemExit(f"{pa} 없음")
    return pa, (pb if pb.exists() else None)


class ReplayCapture:
    """녹음 파일을 실시간 속도로(마이크 문제 시 백업, latency_bench). 끝나면 loop가 아니면 종료."""

    def __init__(self, prefix: str, loop: bool = False, single_mic: bool = False):
        pa, pb = replay_paths(prefix)
        self.a = read_wav(pa)
        self.b = read_wav(pb) if (pb and not single_mic) else None
        if self.b is not None:
            n = min(len(self.a), len(self.b))
            self.a, self.b = self.a[:n], self.b[:n]
        self.single = self.b is None
        self.loop = loop
        self.pos = 0
        self.t0 = None
        self.sent = 0
        self.names = [f"replay:{pa.name}" + (f"+{pb.name}" if self.b is not None else "")]
        self.duration = len(self.a) / SR
        self.status_errors = 0

    def start(self):
        self.t0 = time.monotonic()

    @property
    def finished(self) -> bool:
        return not self.loop and self.pos + BLOCK > len(self.a)

    def ready(self) -> bool:
        if self.finished:
            return False
        return time.monotonic() >= self.t0 + (self.sent + 1) * BLOCK / SR

    def take(self):
        if self.pos + BLOCK > len(self.a):
            self.pos = 0
        a = self.a[self.pos:self.pos + BLOCK]
        b = None if self.b is None else self.b[self.pos:self.pos + BLOCK]
        self.pos += BLOCK
        self.sent += 1
        return a, b

    def discard(self):
        # 끊긴 동안 흘러간 시간만큼 건너뛴다(실시간 마이크와 같게: 밀린 음성을 몰아서 보내지 않는다)
        due = int((time.monotonic() - self.t0) * SR / BLOCK)
        skip = max(0, due - self.sent)
        self.pos += skip * BLOCK
        self.sent += skip

    def stop(self):
        pass


class Recorder:
    """보낸 블록을 그대로 wav(16kHz 16비트)로 저장한다: PREFIX_A.wav, PREFIX_B.wav(--replay 로 다시 보낼 수 있는 형식).
    연결이 끊겨 버린 음성은 저장되지 않는다(서버가 받은 것과 같다)."""

    def __init__(self, prefix: str, two_channels: bool):
        p = Path(prefix)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.paths = [Path(f"{p}_A.wav")] + ([Path(f"{p}_B.wav")] if two_channels else [])
        for q in self.paths:
            if q.exists():
                raise SystemExit(f"{q} 이미 있음 — 다른 이름으로(--record)")
        self.ws = []
        for q in self.paths:
            w = wave.open(str(q), "wb")
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            self.ws.append(w)
        self.n = 0

    def write(self, a: np.ndarray, b: Optional[np.ndarray]) -> None:
        for w, x in zip(self.ws, (a, b)):
            if x is not None:
                w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
        self.n += len(a)

    def close(self) -> None:
        for w in self.ws:
            try:
                w.close()
            except Exception:
                pass
        print(f"[client] 녹음 저장: {', '.join(str(p) for p in self.paths)} ({self.n / SR:.0f}s)")


# ------------------------------------------------------------------ 전송
class Streamer:
    """캡처 → WebSocket. 재연결, 서버 ping 응답(pong), 상태 출력.
    on_sent(seq, t_mono, pos_samples): 블록을 보낸 직후 콜백(latency_bench가 쓴다)."""

    def __init__(self, cap, url: str, client_name: str = "client", on_sent: Optional[Callable] = None,
                 quiet: bool = False, on_connect: Optional[Callable] = None, recorder: Optional[Recorder] = None):
        self.cap = cap
        self.url = url
        self.name = client_name
        self.on_sent = on_sent
        self.on_connect = on_connect   # on_connect(welcome, 다음 seq): 서버 스트림 시간 ↔ seq 대응(latency_bench)
        self.welcome: dict = {}
        self.recorder = recorder
        self.quiet = quiet
        self.seq = 0
        self.connected = False
        self.sent = self.discarded_s = 0
        self.reconnects = 0
        self.rtt: collections.deque = collections.deque(maxlen=30)
        self.stop_flag = False

    async def run(self, duration: Optional[float] = None):
        import websockets
        self.cap.start()
        t_end = None if duration is None else time.monotonic() + duration
        status_t = time.monotonic()
        while not self.stop_flag:
            try:
                async with websockets.connect(self.url, max_size=2**20, ping_interval=None,
                                              compression=None, open_timeout=5) as ws:
                    await ws.send(json.dumps({"type": "hello", "mode": "audio", "client": self.name,
                                              "devices": self.cap.names, "single_mic": self.cap.single,
                                              "replay": isinstance(self.cap, ReplayCapture)}))
                    welcome = json.loads(await asyncio.wait_for(ws.recv(), 5))
                    self.connected = True
                    self.welcome = welcome
                    self.cap.discard()   # 연결 전·끊긴 동안 쌓인 오디오는 버린다
                    if self.on_connect:
                        self.on_connect(welcome, self.seq)
                    if not self.quiet:
                        print(f"[client] 연결됨 {self.url} (서버 sr={welcome.get('sr')}, "
                              f"지터 버퍼 {welcome.get('jitter_ms')}ms)")
                    recv_task = asyncio.create_task(self._recv(ws))
                    try:
                        while not self.stop_flag:
                            if t_end and time.monotonic() >= t_end:
                                self.stop_flag = True
                                break
                            if getattr(self.cap, "finished", False):
                                self.stop_flag = True
                                break
                            if recv_task.done():
                                raise ConnectionError("수신 종료")
                            sent_any = False
                            while self.cap.ready():
                                a, b = self.cap.take()
                                t = time.monotonic()
                                await ws.send(pack_block(self.seq, t, a, b))
                                if self.recorder is not None:
                                    self.recorder.write(a, b)
                                if self.on_sent:
                                    self.on_sent(self.seq, t, getattr(self.cap, "pos", None))
                                self.seq += 1
                                self.sent += 1
                                sent_any = True
                            if not sent_any:
                                await asyncio.sleep(0.004)
                            if not self.quiet and time.monotonic() - status_t >= 5:
                                status_t = time.monotonic()
                                rtt = f"{np.median(self.rtt):.0f}ms" if self.rtt else "–"
                                print(f"[client] 전송 {self.sent}블록 ({self.sent * BLOCK / SR:.0f}s) · RTT {rtt} · "
                                      f"장치 오류 {self.cap.status_errors}")
                    finally:
                        recv_task.cancel()
            except Exception as e:
                if self.stop_flag:
                    break
                if self.connected and not self.quiet:
                    print(f"[client] 연결 끊김: {type(e).__name__} {e}")
                elif not self.quiet:
                    print(f"[client] 연결 실패({type(e).__name__}) → 2초 후 재시도 ({self.url})")
                self.connected = False
                self.reconnects += 1
                t0 = time.monotonic()
                while time.monotonic() - t0 < 2.0 and not self.stop_flag:
                    if getattr(self.cap, "finished", False):
                        self.stop_flag = True
                    self.cap.discard()   # 끊긴 동안의 오디오는 버린다
                    await asyncio.sleep(0.1)
        self.connected = False
        self.cap.stop()

    async def _recv(self, ws):
        async for msg in ws:
            if isinstance(msg, bytes):
                continue
            m = json.loads(msg)
            if m.get("type") == "ping":
                await ws.send(json.dumps({"type": "pong", "id": m.get("id"), "t": m.get("t")}))
            elif m.get("type") == "cpong" and m.get("t") is not None:
                self.rtt.append((time.monotonic() - float(m["t"])) * 1000)


def main(argv=None):
    ap = argparse.ArgumentParser(description="원격 시연 클라이언트: 마이크 → 서버")
    ap.add_argument("--server", default="ws://localhost:8765", help="음성 수신 주소(SSH 터널 경유)")
    ap.add_argument("--list", action="store_true", help="입력 장치 목록")
    ap.add_argument("--wearer", default="", help="착용자 마이크(채널 A) 이름 일부 또는 번호. 비우면 기본 입력")
    ap.add_argument("--ambient", default="", help="주변 마이크(채널 B) 이름 일부 또는 번호")
    ap.add_argument("--single-mic", action="store_true", help="마이크 하나(채널 A만 보냄)")
    ap.add_argument("--replay", default=None, help="마이크 대신 녹음 접두사(예: data/demo → demo_A.wav, demo_B.wav)")
    ap.add_argument("--loop", action="store_true", help="--replay 반복")
    ap.add_argument("--name", default=None, help="대시보드에 보일 클라이언트 이름")
    ap.add_argument("--record", default=None, help="보낸 음성을 PREFIX_A.wav, PREFIX_B.wav 로 저장(리허설 백업용)")
    args = ap.parse_args(argv)
    if args.list:
        list_devices()
        return
    import socket
    if args.replay:
        cap = ReplayCapture(args.replay, loop=args.loop, single_mic=args.single_mic)
        print(f"[client] 재생: {cap.names[0]} ({cap.duration:.0f}s{', 반복' if args.loop else ''})")
    else:
        cap = MicCapture(args.wearer, args.ambient, args.single_mic)
    rec = Recorder(args.record, not cap.single) if args.record else None
    import signal
    try:   # 종료 신호(SIGTERM)에도 녹음 파일을 정상적으로 닫는다(Ctrl+C 와 같게)
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    except Exception:
        pass
    st = Streamer(cap, args.server, args.name or socket.gethostname(), recorder=rec)
    try:
        asyncio.run(st.run())
    except KeyboardInterrupt:
        pass
    finally:
        if rec is not None:
            rec.close()
    print(f"[client] 종료: 보낸 블록 {st.sent}, 재연결 {st.reconnects}회")


if __name__ == "__main__":
    main()
