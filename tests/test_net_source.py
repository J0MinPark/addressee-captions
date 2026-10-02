"""NetworkSource: 지터 버퍼(재정렬·누락 무음 채움·늦은 블록 버림)와 실제 WebSocket 전송·재연결."""
import asyncio
import sys
import threading
import time
from pathlib import Path

import numpy as np

from app.config import load_config
from app.net_source import NetworkSource, pack_block

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))


def _src(**net):
    cfg = load_config(overrides={"network": {"port": 0, **net}})
    return NetworkSource(cfg, log=lambda *a: None)


def _blk(v):
    return np.full(320, v, np.float32)


def test_reorder_within_jitter_buffer():
    s = _src(jitter_ms=100)
    for seq in (0, 2, 1, 3):
        s._on_block(pack_block(seq, time.monotonic(), _blk(seq / 10), _blk(-seq / 10)))
    out = [s.read(0.2) for _ in range(4)]
    assert [round(float(b.a[0]), 2) for b in out] == [0.0, 0.1, 0.2, 0.3]
    assert [round(float(b.b[0]), 2) for b in out] == [0.0, -0.1, -0.2, -0.3]
    assert [b.t for b in out] == [0.0, 0.02, 0.04, 0.06]
    assert s.missing == 0


def test_missing_block_filled_with_silence_after_jitter():
    s = _src(jitter_ms=50)
    for seq in (0, 1, 3):
        s._on_block(pack_block(seq, time.monotonic(), _blk(0.5), _blk(0.5)))
    assert s.read(0.2).a[0] > 0.4 and s.read(0.2).a[0] > 0.4
    t0 = time.monotonic()
    gap = s.read(1.0)               # 2번은 오지 않음 → 50ms 기다린 뒤 무음
    assert time.monotonic() - t0 >= 0.04
    assert np.all(gap.a == 0) and s.missing == 1
    assert s.read(0.2).a[0] > 0.4
    s._on_block(pack_block(2, time.monotonic(), _blk(0.5), _blk(0.5)))   # 늦게 온 2번은 버린다
    assert s.late == 1 and s.read(0.05) is None
    st = s.stats()
    assert st["missing"] == 1 and st["received"] == 3


def test_websocket_stream_and_reconnect(tmp_path):
    from client_capture import ReplayCapture, Streamer
    from app.audio_source import write_wav
    x = (np.sin(np.arange(16000) / 5) * 0.3).astype(np.float32)   # 1초
    write_wav(tmp_path / "r_A.wav", x)
    write_wav(tmp_path / "r_B.wav", -x)
    s = _src()
    s.start()
    url = f"ws://127.0.0.1:{s.bound_port}"
    got = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            b = s.read(0.1)
            if b is not None:
                got.append(b)
    th = threading.Thread(target=reader, daemon=True)
    th.start()
    welcomes = []
    for _ in range(2):   # 두 번 연결(재연결 = 새 세션, 스트림 시간은 이어진다)
        st = Streamer(ReplayCapture(str(tmp_path / "r")), url, "test", quiet=True,
                      on_connect=lambda w, seq: welcomes.append(w))
        asyncio.run(st.run())
        time.sleep(0.3)
        assert not s.connected
    stop.set()
    th.join(1)
    s.stop()
    assert len(got) >= 98 and s.missing == 0
    assert [w["session"] for w in welcomes] == [1, 2]
    assert abs(welcomes[1]["stream_t"] - 50 * 0.02) < 0.05     # 두 번째 세션은 첫 세션 50블록 뒤에서 시작
    ts = [b.t for b in got]
    assert np.allclose(np.diff(ts), 0.02)
    assert np.allclose(got[10].b, -got[10].a, atol=1e-3)       # 채널 A/B 유지
