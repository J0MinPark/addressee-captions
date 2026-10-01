"""VAD와 주변 마이크(채널 B) 발화 구간 분할.

- SileroVAD: silero-vad pip 패키지. 16kHz에서 512샘플(32ms) 청크 단위, 상태 유지.
- EnergyVAD: Silero 로딩 실패 시 폴백(적응형 잡음 바닥 + 마진).
- StreamingVAD: 20ms 블록을 512샘플 청크로 모아 확률을 낸다.
- Segmenter: 확률 시퀀스 -> 구간(시작/끝). 순수 로직.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.ownvoice import rms_db

CHUNK = 512


class SileroVAD:
    name = "silero"

    def __init__(self):
        import torch
        from silero_vad import load_silero_vad
        torch.set_num_threads(1)
        self.torch = torch
        self.model = load_silero_vad()

    def __call__(self, chunk: np.ndarray) -> float:
        with self.torch.no_grad():
            return float(self.model(self.torch.from_numpy(chunk), 16000).item())

    def reset(self):
        self.model.reset_states()


class EnergyVAD:
    name = "energy"

    def __init__(self, margin_db: float = 10.0):
        self.margin = margin_db
        self.floor = -60.0

    def __call__(self, chunk: np.ndarray) -> float:
        db = rms_db(chunk)
        # 잡음 바닥: 내려갈 땐 빠르게, 올라갈 땐 천천히
        self.floor = db if db < self.floor else self.floor + 0.002 * (db - self.floor)
        return float(1.0 / (1.0 + np.exp(-(db - self.floor - self.margin) / 2.0)))

    def reset(self):
        self.floor = -60.0


def make_vad(cfg: dict, log=print):
    if cfg["vad"].get("backend", "silero") == "silero":
        try:
            return SileroVAD()
        except Exception as e:
            log(f"[vad] Silero 로딩 실패 → energy VAD 폴백: {e}")
    return EnergyVAD(cfg["vad"].get("energy_margin_db", 10.0))


class StreamingVAD:
    """블록을 받아 512샘플 청크마다 (청크 시작 시간, 확률, 길이) 를 낸다."""

    def __init__(self, vad, sr: int = 16000):
        self.vad = vad
        self.sr = sr
        self.buf = np.zeros(0, dtype=np.float32)
        self.buf_t = 0.0
        self.last_prob = 0.0
        self.total_ms = 0.0
        self.calls = 0

    def push(self, t: float, x: np.ndarray) -> list[tuple[float, float, float]]:
        if len(self.buf) == 0:
            self.buf_t = t
        self.buf = np.concatenate([self.buf, x])
        out = []
        while len(self.buf) >= CHUNK:
            c = self.buf[:CHUNK]
            t0 = time.perf_counter()
            try:
                p = self.vad(c)
            except Exception:
                p = 0.0
            self.total_ms += (time.perf_counter() - t0) * 1000
            self.calls += 1
            out.append((self.buf_t, p, CHUNK / self.sr))
            self.last_prob = p
            self.buf = self.buf[CHUNK:]
            self.buf_t += CHUNK / self.sr
        return out

    def reset(self):
        self.buf = np.zeros(0, dtype=np.float32)
        self.last_prob = 0.0
        try:
            self.vad.reset()
        except Exception:
            pass


@dataclass
class Segment:
    seg_id: str
    t_start: float
    t_end: float
    forced: bool = False         # 최대 길이로 강제 절단됨
    closed_wall: float = field(default_factory=time.monotonic)

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


class Segmenter:
    def __init__(self, cfg: dict, prefix: str = "s"):
        v = cfg["vad"]
        self.thr = v["threshold"]
        self.neg = v.get("neg_threshold", max(0.0, self.thr - 0.15))
        self.min_speech = v["min_speech_s"]
        self.end_sil = v["end_silence_s"]
        self.max_seg = v["max_segment_s"]
        self.pad = v.get("pad_s", 0.0)
        self.prefix = prefix
        self.n = 0
        self.active = False
        self.start = 0.0
        self.last_voice = 0.0

    def reset(self):
        self.active = False

    def _emit(self, t0: float, t1: float, forced: bool = False) -> Optional[Segment]:
        if t1 - t0 < self.min_speech:
            return None
        self.n += 1
        return Segment(f"{self.prefix}{self.n:05d}", max(0.0, t0 - self.pad), t1 + self.pad, forced)

    def update(self, t: float, prob: float, dur: float) -> Optional[Segment]:
        end = t + dur
        if not self.active:
            if prob >= self.thr:
                self.active = True
                self.start = t
                self.last_voice = end
            return None
        if prob >= self.neg:
            self.last_voice = end
        if end - self.start >= self.max_seg:
            seg = self._emit(self.start, end, forced=True)
            # 계속 말하는 중이면 바로 다음 구간 시작
            self.active = prob >= self.neg
            self.start = end
            self.last_voice = end
            return seg
        if end - self.last_voice >= self.end_sil:
            self.active = False
            return self._emit(self.start, self.last_voice)
        return None

    def cut(self, t: float) -> Optional[Segment]:
        """t 시점에서 열린 구간을 닫는다(착용자가 말을 시작함)."""
        if not self.active:
            return None
        self.active = False
        return self._emit(self.start, min(self.last_voice, t))

    def flush(self) -> Optional[Segment]:
        if not self.active:
            return None
        self.active = False
        return self._emit(self.start, self.last_voice)
