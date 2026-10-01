"""시나리오 녹음: 두 마이크 동시 → data/NAME_A.wav, data/NAME_B.wav, data/NAME.json

    python tools/record.py --scenario cafe_trap1            # Ctrl+C로 종료
    python tools/record.py --scenario demo --seconds 90
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import winsetup  # noqa: E402

winsetup.setup()

import numpy as np  # noqa: E402

from app.audio_source import LiveSource, write_wav  # noqa: E402
from app.config import load_config, resolve_path  # noqa: E402
from app.ownvoice import rms_db  # noqa: E402


def meter(db: float, width: int = 20) -> str:
    n = int(np.clip((db + 60) / 60, 0, 1) * width)
    return "█" * n + "·" * (width - n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--seconds", type=float, default=0, help="0이면 Ctrl+C까지")
    ap.add_argument("--profile", default=None)
    args = ap.parse_args()
    cfg = load_config(args.profile)
    data = resolve_path(cfg, "data_dir")
    src = LiveSource(cfg)
    src.start()
    a_parts, b_parts = [], []
    t_start = time.time()
    print(f"녹음 중: {args.scenario}  (Ctrl+C로 종료)")
    try:
        last = 0
        while True:
            blk = src.read(2.0)
            if blk is None:
                print("\n오디오가 들어오지 않습니다. 장치를 확인하세요.")
                break
            a_parts.append(blk.a)
            b_parts.append(blk.b)
            t = blk.t
            if t - last >= 0.1:
                last = t
                print(f"\r{t:6.1f}s  A {meter(rms_db(blk.a))}  B {meter(rms_db(blk.b))}  "
                      f"A-B {rms_db(blk.a) - rms_db(blk.b):+5.1f}dB", end="", flush=True)
            if args.seconds and t >= args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        src.stop()
    a = np.concatenate(a_parts) if a_parts else np.zeros(0, np.float32)
    b = np.concatenate(b_parts) if b_parts else np.zeros(0, np.float32)
    pa, pb, pj = data / f"{args.scenario}_A.wav", data / f"{args.scenario}_B.wav", data / f"{args.scenario}.json"
    write_wav(pa, a)
    if not src.single_mic:
        write_wav(pb, b)
    meta = {"scenario": args.scenario, "start_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t_start)),
            "duration_s": round(len(a) / 16000, 2), "devices": src.device_names, "single_mic": src.single_mic,
            "overflows": src.overflows, "sample_rate": 16000}
    pj.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {pa}" + ("" if src.single_mic else f", {pb}") + f", {pj}  ({meta['duration_s']}s)")


if __name__ == "__main__":
    main()
