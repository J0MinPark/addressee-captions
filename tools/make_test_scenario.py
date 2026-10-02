"""[개발용] 실제 녹음 전에 파이프라인을 끝까지 시험하기 위한 합성 2채널 시나리오 생성기.

Windows 내장 한국어 음성(SAPI, Microsoft Heami)으로 대사를 만들고, 리샘플링으로 음높이/음색을 바꿔
서로 다른 '화자'를 흉내 낸다. 제품 기능이 아니라 테스트 데이터 도구다(시스템은 음성을 출력하지 않는다).
리눅스(서버)에서는 edge-tts(온라인, 한국어 뉴럴 음성 3개 + 음높이 변경)를 쓴다: pip install edge-tts

    python tools/make_test_scenario.py --name demo_trap
      → data/demo_trap_A.wav, data/demo_trap_B.wav, data/demo_trap.json, data/demo_trap.truth.json
    python tools/label.py demo_trap --auto      # 정답(truth)으로 자동 라벨
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from app.audio_source import read_wav, write_wav  # noqa: E402

SR = 16000
# (화자, 대사, 앞 간격(초: 이전 발화 끝 기준), 라벨)  라벨: w=착용자, y=착용자에게, n=아님
SCRIPT = [
    ("W", "안녕하세요, 혹시 여기 자리 있어요?", 0.8, "w"),
    ("P", "아니요, 비어 있어요. 앉으세요.", 0.35, "y"),
    ("W", "감사합니다. 오늘 사람 진짜 많네요.", 0.5, "w"),
    ("P", "그러게요, 주말이라 그런가 봐요.", 0.4, "y"),
    ("C", "내일 회의 몇 시였는지 기억나?", 3.0, "n"),
    ("D", "아마 열 시였을 거야.", 0.5, "n"),
    ("W", "혹시 와이파이 비밀번호 아세요?", 2.5, "w"),
    ("C", "야 지훈아, 너 어제 그 경기 봤어? 진짜 대박이더라.", 0.3, "n"),     # 함정
    ("P", "카운터 옆에 적혀 있던데요.", 0.6, "y"),
    ("W", "아 그렇구나, 고마워요.", 0.5, "w"),
    ("P", "저는 여기 자주 오는데 오늘처럼 붐비는 건 처음이에요.", 0.4, "y"),
    ("SIREN", "", 2.0, "-"),
    ("C", "민수야, 이거 좀 봐봐.", 1.0, "n"),
    ("W", "네? 저 부르셨어요?", 0.6, "w"),
    ("D", "아니 너 말고, 저쪽 민수.", 0.4, "y"),   # 착용자에게 한 말(“너 말고”)
    ("W", "커피는 어떤 게 맛있어요?", 3.0, "w"),
    ("P", "여기는 라떼가 제일 괜찮아요.", 0.4, "y"),
    ("W", "그럼 라떼 마셔 볼게요.", 0.5, "w"),
    ("D", "근데 우리 몇 시에 나가야 돼?", 0.35, "n"),                          # 함정 2
    ("C", "한 시간쯤 뒤에.", 0.5, "n"),
    ("P", "여기 디저트도 맛있어요. 케이크 추천해요.", 4.0, "y"),
]
# 화자별 (리샘플 비율: >1 높은 목소리, SAPI 속도)
VOICE = {"W": (1.00, 0), "P": (1.22, 1), "C": (0.84, -1), "D": (1.42, 2)}


def sapi(lines: list[tuple[str, int]], outdir: Path) -> list[Path]:
    ps = ["Add-Type -AssemblyName System.Speech",
          "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer",
          "$v = $s.GetInstalledVoices() | ? { $_.VoiceInfo.Culture.Name -eq 'ko-KR' } | select -First 1",
          "if ($v) { $s.SelectVoice($v.VoiceInfo.Name) } else { throw 'no ko-KR voice' }",
          "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, "
          "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)"]
    paths = []
    for i, (text, rate) in enumerate(lines):
        p = outdir / f"l{i:03d}.wav"
        paths.append(p)
        t = text.replace("'", "''")
        ps += [f"$s.Rate = {rate}", f"$s.SetOutputToWaveFile('{p}', $f)", f"$s.Speak('{t}')"]
    ps.append("$s.SetOutputToNull()")
    script = outdir / "gen.ps1"
    script.write_text("\n".join(ps), encoding="utf-8-sig")
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)], check=True)
    return paths


# edge-tts: 화자별 (음성, 음높이). 리샘플 비율은 쓰지 않는다(서로 다른 실제 음성)
EDGE_VOICE = {"W": ("ko-KR-InJoonNeural", "+0Hz"), "P": ("ko-KR-SunHiNeural", "+0Hz"),
              "C": ("ko-KR-HyunsuMultilingualNeural", "-10Hz"), "D": ("ko-KR-SunHiNeural", "+25Hz")}


def edge(lines: list[tuple[str, str]], outdir: Path) -> list[np.ndarray]:
    """[(화자, 대사)] → 16kHz 클립. 인터넷 필요(개발용)."""
    import asyncio
    import io

    import edge_tts
    import soundfile as sf
    from scipy.signal import resample_poly

    async def one(who, text):
        voice, pitch = EDGE_VOICE[who]
        buf = b""
        async for ch in edge_tts.Communicate(text, voice, pitch=pitch).stream():
            if ch["type"] == "audio":
                buf += ch["data"]
        x, fs = sf.read(io.BytesIO(buf), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        return resample_poly(x, 2, 3).astype(np.float32) if fs == 24000 else resample_poly(x, SR, fs).astype(np.float32)

    async def all_():
        return [await one(w, t) for w, t in lines]
    return asyncio.run(all_())


def shift(x: np.ndarray, ratio: float) -> np.ndarray:
    """ratio>1: 음높이·포먼트 상승(짧아짐)."""
    if ratio == 1.0:
        return x
    from scipy.signal import resample_poly
    up, down = 100, int(round(100 * ratio))
    return resample_poly(x, up, down).astype(np.float32)


def trim(x: np.ndarray, thr: float = 0.01) -> np.ndarray:
    idx = np.where(np.abs(x) > thr)[0]
    return x[idx[0]: idx[-1] + 1] if len(idx) else x


def siren(dur: float) -> np.ndarray:
    """yelp 사이렌(빠른 상하 스윕). 느린 wail은 1초 창 AST에서 점수가 낮다(README 참고)."""
    t = np.arange(int(dur * SR)) / SR
    f = 700 + 800 * np.abs(((t * 3) % 1) * 2 - 1)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) + 0.4 * np.sin(2 * ph) + 0.2 * np.sin(3 * ph)
    return (x * 0.3).astype(np.float32)


def pink(n: int, rng) -> np.ndarray:
    w = rng.standard_normal(n)
    f = np.fft.rfft(w)
    f /= np.sqrt(np.maximum(np.arange(len(f)), 1))
    x = np.fft.irfft(f, n)
    return (x / np.abs(x).max()).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="demo_trap")
    ap.add_argument("--noise-db", type=float, default=-38.0, help="카페 잡음 크기")
    ap.add_argument("--tts", choices=("auto", "sapi", "edge"), default="auto", help="auto: Windows=sapi, 그 외 edge")
    ap.add_argument("--out", default=None, help="출력 폴더(기본 config paths.data_dir — 서버는 $HEARME_DATA/data)")
    args = ap.parse_args()
    tts = args.tts if args.tts != "auto" else ("sapi" if os.name == "nt" else "edge")
    if tts == "sapi" and os.name != "nt":
        raise SystemExit("Windows SAPI 음성이 필요합니다(리눅스는 --tts edge).")
    from app.config import load_config, resolve_path
    data = Path(args.out) if args.out else resolve_path(load_config(), "data_dir")
    data.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    if tts == "sapi":
        speech = [(VOICE[w][1], txt) for w, txt, _, _ in SCRIPT if w != "SIREN"]
        with tempfile.TemporaryDirectory() as td:
            paths = sapi([(t, r) for r, t in speech], Path(td))
            clips = [trim(read_wav(p)) for p in paths]
    else:
        clips = [trim(x) for x in edge([(w, txt) for w, txt, _, _ in SCRIPT if w != "SIREN"], data)]
    items, k = [], 0
    for who, txt, gap, lab in SCRIPT:
        if who == "SIREN":
            items.append((who, txt, gap, lab, siren(3.0)))
        else:
            x = shift(clips[k], VOICE[who][0]) if tts == "sapi" else clips[k]
            x = x / (np.abs(x).max() + 1e-6) * 0.5
            items.append((who, txt, gap, lab, x))
            k += 1
    total = sum(g + len(x) / SR for _, _, g, _, x in items) + 2.0
    n = int(total * SR)
    a = np.zeros(n, np.float32)
    b = np.zeros(n, np.float32)
    # 채널 이득: 착용자는 핀마이크(A)에 크게, 다른 사람은 주변 마이크(B)에 크게
    gain = {"W": (1.0, 0.30), "P": (0.16, 0.85), "C": (0.08, 0.55), "D": (0.10, 0.60), "SIREN": (0.25, 0.9)}
    truth, t = [], 0.0
    for who, txt, gap, lab, x in items:
        t += gap
        s = int(t * SR)
        ga, gb = gain[who]
        a[s:s + len(x)] += ga * x
        # 주변 마이크는 약간 늦게(거리) + 살짝 울림
        d = int(0.002 * SR)
        b[s + d:s + d + len(x)] += gb * x
        b[s + d + 800:s + d + 800 + len(x)] += 0.15 * gb * x
        truth.append({"who": who, "text": txt, "t_start": round(t, 3), "t_end": round(t + len(x) / SR, 3),
                      "label": lab})
        t += len(x) / SR
    noise = 10 ** (args.noise_db / 20)
    a += noise * 0.4 * pink(n, rng) + rng.normal(0, 1e-4, n).astype(np.float32)
    b += noise * pink(n, rng) + rng.normal(0, 1e-4, n).astype(np.float32)
    write_wav(data / f"{args.name}_A.wav", a)
    write_wav(data / f"{args.name}_B.wav", b)
    (data / f"{args.name}.json").write_text(json.dumps(
        {"scenario": args.name, "synthetic": True, "duration_s": round(total, 2),
         "devices": ["synthetic (Windows SAPI ko-KR, pitch-shifted speakers)" if tts == "sapi" else
                     "synthetic (edge-tts ko-KR InJoon/SunHi/Hyunsu, 4 speakers)"]}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    (data / f"{args.name}.truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"생성: {data / args.name}_A.wav, _B.wav ({total:.1f}s), truth {len(truth)}개 발화")


if __name__ == "__main__":
    main()
