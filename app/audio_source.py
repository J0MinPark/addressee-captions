"""오디오 입력. LiveSource(마이크 2개)와 ReplaySource(WAV 2개)는 같은 Block을 내보내므로
실시간/재생이 같은 파이프라인을 탄다.

시간 규약: Block.t 는 '스트림 시간'(첫 샘플부터의 샘플 카운터 / sr, 초).
정책·평가는 모두 스트림 시간을 쓴다. Block.wall 은 time.monotonic() 기준 생성 시각(지연 측정용).
"""
from __future__ import annotations

import collections
import json
import os
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

SR = 16000


@dataclass
class Block:
    t: float            # 스트림 시간(초), 블록 첫 샘플
    a: np.ndarray       # 채널 A: 착용자 핀마이크 (float32)
    b: np.ndarray       # 채널 B: 주변 마이크 (float32)
    wall: float         # time.monotonic()


class RingBuffer:
    """최근 N초 샘플 링버퍼. 스트림 시간으로 잘라낸다."""

    def __init__(self, seconds: float, sr: int = SR):
        self.sr = sr
        self.cap = int(seconds * sr)
        self.buf = np.zeros(self.cap, dtype=np.float32)
        self.total = 0  # 지금까지 쓴 샘플 수
        self.lock = threading.Lock()

    def push(self, x: np.ndarray) -> None:
        with self.lock:
            n = len(x)
            if n >= self.cap:
                self.buf[:] = x[-self.cap:]
                self.total += n
                return
            pos = self.total % self.cap
            first = min(n, self.cap - pos)
            self.buf[pos:pos + first] = x[:first]
            if first < n:
                self.buf[:n - first] = x[first:]
            self.total += n

    @property
    def t_now(self) -> float:
        return self.total / self.sr

    def get(self, t0: float, t1: float) -> np.ndarray:
        with self.lock:
            s0 = max(int(round(t0 * self.sr)), self.total - self.cap, 0)
            s1 = min(int(round(t1 * self.sr)), self.total)
            if s1 <= s0:
                return np.zeros(0, dtype=np.float32)
            idx = np.arange(s0, s1) % self.cap
            return self.buf[idx].copy()

    def clear(self) -> None:
        with self.lock:
            self.buf[:] = 0
            self.total = 0


class StreamResampler:
    """정수배 다운샘플(48k->16k 등)은 상태 유지 FIR, 그 외는 블록 단위 resample_poly."""

    def __init__(self, sr_in: int, sr_out: int = SR):
        self.sr_in, self.sr_out = int(sr_in), int(sr_out)
        self.ratio = None
        self.zi = None
        if self.sr_in != self.sr_out and self.sr_in % self.sr_out == 0:
            from scipy.signal import firwin, lfilter_zi
            self.ratio = self.sr_in // self.sr_out
            self.taps = firwin(63, 0.9 / self.ratio).astype(np.float32)
            self.zi = lfilter_zi(self.taps, [1.0]).astype(np.float32) * 0
            self.phase = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.sr_in == self.sr_out:
            return x.astype(np.float32)
        if self.ratio:
            from scipy.signal import lfilter
            y, self.zi = lfilter(self.taps, [1.0], x, zi=self.zi)
            out = y[self.phase::self.ratio]
            self.phase = (self.phase - len(x)) % self.ratio
            return out.astype(np.float32)
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(self.sr_in, self.sr_out)
        return resample_poly(x, self.sr_out // g, self.sr_in // g).astype(np.float32)


class AudioSource:
    realtime = True
    single_mic = False
    block = int(SR * 0.02)

    def start(self) -> None: ...
    def read(self, timeout: float = 1.0) -> Optional[Block]: ...
    def stop(self) -> None: ...
    @property
    def finished(self) -> bool:
        return False


class _Channel:
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

    def drop(self, n):
        self.take(n)


class LiveSource(AudioSource):
    """마이크 2개(또는 single_mic면 1개)를 연다."""

    def __init__(self, cfg: dict):
        from app.devices import find_input
        import sounddevice as sd
        self.sd = sd
        a = cfg["audio"]
        self.sr = a["sample_rate"]
        self.block = int(self.sr * a["block_ms"] / 1000)
        self.single_mic = bool(a.get("single_mic"))
        self.drift_max = int(a.get("drift_max_s", 0.5) * self.sr)
        pref = a.get("prefer_wasapi", True)
        self.dev_a = find_input(a.get("device_wearer", ""), pref)
        self.dev_b = None if self.single_mic else find_input(a.get("device_ambient", ""), pref)
        if not self.single_mic and self.dev_a == self.dev_b:
            print("[audio] 경고: 착용자/주변 마이크가 같은 장치입니다. single_mic 모드로 전환합니다.")
            self.single_mic = True
        self.ch = [_Channel(), _Channel()]
        self.streams = []
        self.n_out = 0
        self.t0_wall = None
        self.overflows = 0
        self.device_names = []

    def _open(self, dev, ch: _Channel):
        sd = self.sd
        info = sd.query_devices(dev if dev is not None else sd.default.device[0], "input")
        api = sd.query_hostapis(info["hostapi"])["name"]
        extra = None
        sr = self.sr
        try:
            if "WASAPI" in api:
                extra = sd.WasapiSettings(auto_convert=True)
            sd.check_input_settings(device=dev, samplerate=sr, channels=1, dtype="float32",
                                    extra_settings=extra)
        except Exception:
            extra = None
            sr = int(info["default_samplerate"])
        rs = StreamResampler(sr, self.sr)

        def cb(indata, frames, t, status):
            if status:
                self.overflows += 1
            ch.put(rs(indata[:, 0].copy()))

        st = sd.InputStream(device=dev, samplerate=sr, channels=1, dtype="float32",
                            blocksize=int(sr * 0.02), callback=cb, extra_settings=extra)
        self.device_names.append(f"{info['name']} ({api}, {sr}Hz)")
        return st

    def start(self):
        self.streams.append(self._open(self.dev_a, self.ch[0]))
        if not self.single_mic:
            self.streams.append(self._open(self.dev_b, self.ch[1]))
        for s in self.streams:
            s.start()
        print("[audio] 열린 장치:", " | ".join(self.device_names))

    def read(self, timeout: float = 1.0) -> Optional[Block]:
        deadline = time.monotonic() + timeout
        chans = self.ch[:1] if self.single_mic else self.ch
        while True:
            if all(c.n >= self.block for c in chans):
                break
            if time.monotonic() > deadline:
                return None
            time.sleep(0.003)
        if not self.single_mic:  # 장치 간 클럭 드리프트 보정
            diff = self.ch[0].n - self.ch[1].n
            if abs(diff) > self.drift_max:
                (self.ch[0] if diff > 0 else self.ch[1]).drop(abs(diff) - self.block)
        a = self.ch[0].take(self.block)
        b = a.copy() if self.single_mic else self.ch[1].take(self.block)
        if self.t0_wall is None:
            self.t0_wall = time.monotonic()
        t = self.n_out / self.sr
        self.n_out += self.block
        return Block(t=t, a=a, b=b, wall=time.monotonic())

    def stop(self):
        for s in self.streams:
            try:
                s.stop()
                s.close()
            except Exception:
                pass
        self.streams = []


def read_wav(path: str | Path, sr: int = SR) -> np.ndarray:
    path = str(path)
    try:
        import soundfile as sf
        x, fs = sf.read(path, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
    except Exception:
        with wave.open(path, "rb") as w:
            fs = w.getframerate()
            ch = w.getnchannels()
            raw = w.readframes(w.getnframes())
            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if ch > 1:
                x = x.reshape(-1, ch).mean(axis=1)
    if fs != sr:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(fs, sr)
        x = resample_poly(x, sr // g, fs // g).astype(np.float32)
    return x.astype(np.float32)


def write_wav(path: str | Path, x: np.ndarray, sr: int = SR) -> None:
    x = np.clip(np.asarray(x, dtype=np.float32), -1, 1)
    try:
        import soundfile as sf
        sf.write(str(path), x, sr, subtype="PCM_16")
    except Exception:
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes((x * 32767).astype(np.int16).tobytes())


def scenario_paths(prefix: str | Path):
    """data/demo -> (data/demo_A.wav, data/demo_B.wav, data/demo.json)."""
    p = Path(prefix)
    if p.suffix.lower() == ".wav":
        stem = p.with_suffix("")
        s = str(stem)
        if s.endswith("_A") or s.endswith("_B"):
            stem = Path(s[:-2])
        p = stem
    return Path(f"{p}_A.wav"), Path(f"{p}_B.wav"), Path(f"{p}.json")


class ReplaySource(AudioSource):
    """WAV 두 개를 실시간 속도(realtime=True) 또는 최대 속도로 흘려보낸다.
    _B.wav가 없으면 단일 마이크 녹음으로 간주(b=a)."""

    def __init__(self, prefix: str | Path, cfg: dict, realtime: bool = False,
                 single_mic: Optional[bool] = None, loop: bool = False):
        self.loop = loop
        self.offset = 0
        self.sr = cfg["audio"]["sample_rate"]
        self.block = int(self.sr * cfg["audio"]["block_ms"] / 1000)
        pa, pb, pj = scenario_paths(prefix)
        if not pa.exists():
            raise FileNotFoundError(f"{pa} 없음")
        self.a = read_wav(pa, self.sr)
        if pb.exists() and not single_mic:
            self.b = read_wav(pb, self.sr)
            n = min(len(self.a), len(self.b))
            self.a, self.b = self.a[:n], self.b[:n]
            self.single_mic = False
        else:
            self.b = self.a
            self.single_mic = True
        self.meta = json.loads(pj.read_text(encoding="utf-8")) if pj.exists() else {}
        self.name = Path(str(pa)[:-6]).name
        self.realtime = realtime
        self.pos = 0
        self.t0_wall = None
        self.duration = len(self.a) / self.sr

    def start(self):
        self.t0_wall = time.monotonic()

    def read(self, timeout: float = 1.0) -> Optional[Block]:
        if self.pos + self.block > len(self.a):
            if not self.loop:
                return None
            self.offset += self.pos   # 처음으로 되감기, 스트림 시간은 계속 증가
            self.pos = 0
        t = (self.offset + self.pos) / self.sr
        if self.realtime:
            wait = self.t0_wall + t + self.block / self.sr - time.monotonic()
            if wait > 0:
                time.sleep(wait)
        a = self.a[self.pos:self.pos + self.block]
        b = self.b[self.pos:self.pos + self.block]
        self.pos += self.block
        return Block(t=t, a=a, b=b, wall=time.monotonic())

    @property
    def finished(self) -> bool:
        return not self.loop and self.pos + self.block > len(self.a)
