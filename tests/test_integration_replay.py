"""합성 통합 테스트: 녹음 파일 하나를 ReplaySource로 파이프라인에 통과시켜 이벤트 순서를 검증한다.

모델 없이 돈다(EnergyVAD, 스펙트럼 임베딩, 주파수로 텍스트를 고르는 가짜 ASR, 가짜 LLM).
시나리오(“화자”는 서로 다른 기본 주파수의 배음 신호):
  1.0-2.5  착용자 "지금 몇 시예요?"           (A 크게, B 작게)
  2.8-4.6  상대(220Hz) "세 시 반이요"          → 판정 중 → LLM yes → partner 등록
  6.0-7.5  착용자 "점심 뭐 드실래요?"
  7.8-9.6  옆 사람(400Hz) "야 어제 경기 봤어?"  → LLM no → other (함정)
  12.0-14.0 상대(220Hz) 독백                    → T=0, S=1 → partner
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np
import pytest

from app.audio_source import ReplaySource, write_wav
from app.config import load_config
from app.pipeline import Models, Pipeline
from app.segmenter import EnergyVAD
from app.speaker import SpectralEmbedder

SR = 16000
VOICES = {"wearer": 130.0, "partner": 220.0, "trap": 400.0}
TEXT = {"wearer": None, "partner": None, "trap": "야 어제 경기 봤어?"}


def voice(f0, dur, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    x = sum((1.0 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6)) for k in range(1, 12))
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 4 * t) ** 2
    ramp = np.minimum(1, np.minimum(t / 0.03, (dur - t) / 0.03))
    return (x * env * ramp / 4).astype(np.float32)


def make_scenario(tmp_path):
    n = int(16 * SR)
    rng = np.random.default_rng(0)
    a = rng.normal(0, 3e-4, n).astype(np.float32)
    b = rng.normal(0, 3e-4, n).astype(np.float32)
    plan = [("wearer", 1.0, 2.5), ("partner", 2.8, 4.6), ("wearer", 6.0, 7.5), ("trap", 7.8, 9.6),
            ("partner", 12.0, 14.0)]
    for i, (who, t0, t1) in enumerate(plan):
        v = voice(VOICES[who], t1 - t0, i)
        s = slice(int(t0 * SR), int(t0 * SR) + len(v))
        if who == "wearer":
            a[s] += 0.5 * v
            b[s] += 0.12 * v
        else:
            a[s] += 0.06 * v
            b[s] += 0.4 * v
    prefix = tmp_path / "synth_trap"
    write_wav(f"{prefix}_A.wav", a)
    write_wav(f"{prefix}_B.wav", b)
    return prefix


class FakeASR:
    desc = "fake"
    wearer_lines = deque(["지금 몇 시예요?", "점심 뭐 드실래요?"])

    def transcribe(self, audio):
        spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
        f = np.argmax(spec[20:]) * SR / len(audio) + 20 * SR / len(audio)
        who = min(VOICES, key=lambda k: abs(VOICES[k] - f))
        if who == "wearer":
            return (self.wearer_lines.popleft() if self.wearer_lines else "음"), []
        if who == "partner":
            return "세 시 반이요", []
        return TEXT["trap"], []


class FakeJudge:
    available = True
    model = "fake"

    def __init__(self):
        self.inflight = 0
        self.timeouts = 0
        self.lat_ms = deque(maxlen=10)
        self.calls = []

    def judge_async(self, prev, a, b, cb):
        self.calls.append((a, b))
        pair = "시" in b and "반" in b
        cb({"pair": pair, "confidence": "high", "type": "질문-대답" if pair else "없음",
            "prob": 0.95 if pair else 0.05, "latency_ms": 1.0})


@pytest.fixture
def cfg():
    c = load_config(overrides={"vad": {"backend": "energy"}, "llm": {"always_call": False}})
    return c


def test_replay_event_order(tmp_path, cfg):
    prefix = make_scenario(tmp_path)
    models = Models(vad_a=EnergyVAD(10), vad_b=EnergyVAD(10), embedder=SpectralEmbedder(), asr=FakeASR(),
                    sound=None, judge=FakeJudge())
    src = ReplaySource(prefix, cfg, realtime=False)
    pipe = Pipeline(cfg, src, models, mode="full", log_events=False, record_segments=True)
    events = []
    pipe.add_listener(lambda e: events.append(e))
    pipe.start()
    assert pipe.wait_finished(timeout=60)
    pipe.stop()
    time.sleep(0.2)

    caps = [e for e in events if e["type"] == "caption"]
    roles = [(c["role"], c["text"]) for c in caps]
    assert roles[0] == ("wearer", "지금 몇 시예요?"), roles
    partner_first = next(c for c in caps if c["text"] == "세 시 반이요")
    assert partner_first["pending_llm"] is True and partner_first["evidence"]["T"] == 1.0
    # 순서: 상대 caption → caption_update(partner) → partner_added
    i_cap = events.index(partner_first)
    i_upd = next(i for i, e in enumerate(events) if e["type"] == "caption_update" and e["id"] == partner_first["id"])
    i_add = next(i for i, e in enumerate(events) if e["type"] == "partner_added")
    assert i_cap < i_upd < i_add
    assert events[i_upd]["role"] == "partner"
    # 함정: 타이밍은 맞지만 LLM no → other
    trap = next(c for c in caps if c["text"] == TEXT["trap"])
    trap_upd = next(e for e in events if e["type"] == "caption_update" and e["id"] == trap["id"])
    assert trap_upd["role"] == "other"
    # 독백: T=0, S=1 → partner, LLM 호출 없음
    mono = [c for c in caps if c["text"] == "세 시 반이요"][-1]
    assert mono["id"] != partner_first["id"]
    assert mono["role"] == "partner" and mono["evidence"]["T"] == 0 and mono["evidence"]["S"] == 1.0
    assert mono["pending_llm"] is False
    assert len(models.judge.calls) == 2
    partners = [e["speaker_id"] for e in events if e["type"] == "partner_added"]
    assert partners == [partner_first["speaker_id"]]

    out = tmp_path / "synth.segments.jsonl"
    n = pipe.write_segments(out)
    assert n >= 5
