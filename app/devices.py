"""입력 장치 목록과 이름 기반 장치 선택.

    python -m app.devices
"""
from __future__ import annotations

import os
from typing import Optional


def _sd():
    import sounddevice as sd
    return sd


def list_inputs():
    sd = _sd()
    hostapis = sd.query_hostapis()
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            out.append({
                "index": i,
                "name": d["name"],
                "hostapi": hostapis[d["hostapi"]]["name"],
                "channels": d["max_input_channels"],
                "default_sr": int(d["default_samplerate"]),
            })
    return out


def find_input(name_part: str, prefer_wasapi: bool = True) -> Optional[int]:
    """이름 일부로 입력 장치를 찾는다. Windows에서는 WASAPI를 우선한다.
    name_part가 비어 있으면 기본 입력 장치(None)."""
    if not name_part:
        return None
    key = name_part.lower()
    cands = [d for d in list_inputs() if key in d["name"].lower()]
    if not cands:
        names = "\n  ".join(f"[{d['index']}] {d['name']} ({d['hostapi']})" for d in list_inputs())
        raise RuntimeError(f"입력 장치 '{name_part}'를 찾을 수 없음. 사용 가능:\n  {names}")

    def rank(d):
        api = d["hostapi"].lower()
        if os.name == "nt" and prefer_wasapi:
            order = ["wasapi", "directsound", "mme", "wdm-ks"]
        else:
            order = ["alsa", "pulse", "jack", "core"]
        for i, o in enumerate(order):
            if o in api:
                return i
        return len(order)

    cands.sort(key=rank)
    return cands[0]["index"]


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        devs = list_inputs()
    except Exception as e:  # pragma: no cover
        print("sounddevice 로딩 실패:", e)
        return
    sd = _sd()
    try:
        default_in = sd.default.device[0]
    except Exception:
        default_in = None
    print("입력 장치 목록 (config.yaml audio.device_wearer / device_ambient 에 이름 일부를 적으세요)")
    print("-" * 78)
    if not devs:
        print("  (입력 장치 없음 — 마이크를 연결하세요. 마이크 없이 시연: python tools/replay.py data/demo --realtime)")
    for d in devs:
        mark = "*" if d["index"] == default_in else " "
        print(f"{mark}[{d['index']:>2}] {d['name'][:44]:<44} {d['hostapi']:<22} ch={d['channels']} sr={d['default_sr']}")
    print("-" * 78)
    print("* = 기본 입력 장치. Windows에서는 같은 장치가 여러 API로 보이면 WASAPI가 우선 선택됩니다.")


if __name__ == "__main__":
    main()
