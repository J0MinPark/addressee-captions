import numpy as np

from app.config import load_config
from app.ownvoice import OwnVoiceDetector, rms_db


def run(det, frames, dt=0.02):
    out = []
    for i, (a_db, b_db, p) in enumerate(frames):
        seg = det.update(i * dt, dt, a_db, b_db, p)
        if seg:
            out.append(seg)
    return out


def test_rms_db():
    assert abs(rms_db(np.ones(320, np.float32) * 0.1) - (-20.0)) < 0.01
    assert rms_db(np.zeros(320, np.float32)) <= -100


def test_own_speech_detected_with_margin():
    det = OwnVoiceDetector(load_config())
    frames = [(-60, -60, 0.0)] * 10 + [(-20, -30, 0.9)] * 50 + [(-60, -60, 0.0)] * 30
    segs = run(det, frames)
    assert len(segs) == 1
    t0, t1 = segs[0]
    assert abs(t0 - 0.2) < 1e-6 and abs(t1 - 1.2) < 1e-6


def test_other_speaker_not_own():
    det = OwnVoiceDetector(load_config())
    # 상대가 말함: A에도 들리지만 B가 더 큼
    frames = [(-30, -22, 0.9)] * 60 + [(-60, -60, 0.0)] * 30
    assert run(det, frames) == []


def test_hangover_bridges_short_pause():
    det = OwnVoiceDetector(load_config())
    frames = [(-20, -30, 0.9)] * 30 + [(-50, -52, 0.1)] * 10 + [(-20, -30, 0.9)] * 30 + [(-60, -60, 0)] * 30
    segs = run(det, frames)   # 200ms 쉼 < hangover 300ms → 한 구간
    assert len(segs) == 1
    assert abs(segs[0][1] - segs[0][0] - 1.4) < 1e-6


def test_overlap_ratio():
    det = OwnVoiceDetector(load_config())
    det.intervals.append((1.0, 2.0))
    assert det.overlap_ratio(1.5, 2.5) == 0.5
    assert det.overlap_ratio(3.0, 4.0) == 0.0
