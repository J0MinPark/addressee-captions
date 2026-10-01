"""착용자 본인 발화 검출.

20ms 프레임마다: 채널 A에서 VAD가 말소리이고 A_dB - B_dB >= own_margin_db 이면 본인 프레임.
본인 구간은 hangover(기본 300ms) 동안 이어 붙인다. 순수 로직(VAD 확률은 밖에서 넣어준다).
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


def rms_db(x: np.ndarray, floor_db: float = -100.0) -> float:
    if len(x) == 0:
        return floor_db
    r = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return max(20.0 * np.log10(r + 1e-10), floor_db)


class OwnVoiceDetector:
    def __init__(self, cfg: dict):
        o = cfg["ownvoice"]
        self.margin = float(o["own_margin_db"])
        self.hangover = float(o["hangover_s"])
        self.min_own = float(o["min_own_s"])
        self.vad_thr = float(cfg["vad"]["threshold"])
        self.active = False
        self.start = 0.0
        self.last_own = 0.0
        self.intervals: deque[tuple[float, float]] = deque(maxlen=200)

    def reset(self) -> None:
        self.active = False
        self.intervals.clear()

    def is_own_frame(self, a_db: float, b_db: float, a_speech_prob: float) -> bool:
        return a_speech_prob >= self.vad_thr and (a_db - b_db) >= self.margin

    def update(self, t: float, dur: float, a_db: float, b_db: float,
               a_speech_prob: float) -> Optional[tuple[float, float]]:
        """프레임 하나 처리. 본인 구간이 끝나면 (시작, 끝)을 반환."""
        own = self.is_own_frame(a_db, b_db, a_speech_prob)
        if own:
            if not self.active:
                self.active = True
                self.start = t
            self.last_own = t + dur
            return None
        if self.active and (t + dur) - self.last_own > self.hangover:
            return self._close()
        return None

    def _close(self) -> Optional[tuple[float, float]]:
        self.active = False
        seg = (self.start, self.last_own)
        if seg[1] - seg[0] >= self.min_own:
            self.intervals.append(seg)
            return seg
        return None

    def flush(self) -> Optional[tuple[float, float]]:
        return self._close() if self.active else None

    def overlap_ratio(self, t0: float, t1: float, now: Optional[float] = None) -> float:
        """[t0,t1] 중 본인 발화(진행 중인 구간 포함)와 겹치는 비율."""
        if t1 <= t0:
            return 0.0
        ivs = list(self.intervals)
        if self.active:
            ivs.append((self.start, now if now is not None else self.last_own))
        ov = 0.0
        for a, b in ivs:
            ov += max(0.0, min(b, t1) - max(a, t0))
        return min(1.0, ov / (t1 - t0))
