"""발표 전 사전 점검. 항목마다 PASS/WARN/FAIL을 표로 출력한다(하나가 실패해도 끝까지 진행).

    python -m app.preflight --profile gpu_4060
    python -m app.preflight --profile gpu_4060 --yes     # 녹음 안내에서 Enter 대기 없이 카운트다운만

점검 순서: 마이크 장치 → 마이크 녹음(dB 차) → GPU/VRAM → ASR 워밍업 → LLM 헬스체크 → 위험 소리 → LAN 주소/QR
"""
import argparse
import socket
import sys
import time
from pathlib import Path

from app import winsetup

winsetup.setup()

import numpy as np  # noqa: E402

from app.config import ROOT, load_config  # noqa: E402

ASSETS = ROOT / "assets"
ROWS: list[tuple[str, str, str]] = []
COLOR = {"PASS": "\033[92m", "WARN": "\033[93m", "FAIL": "\033[91m", "SKIP": "\033[90m", "END": "\033[0m"}


def _w(text: str) -> int:
    """터미널 표시 폭(한글 2칸)."""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _w(text))


def add(name: str, status: str, detail: str) -> None:
    ROWS.append((name, status, detail))
    print(f"  → {COLOR.get(status, '')}{status}{COLOR['END']}  {name}: {detail}", flush=True)


def head(n: int, title: str) -> None:
    print(f"\n[{n}/7] {title}", flush=True)


def db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-10))


def peak_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.max(np.abs(x)) + 1e-10)) if len(x) else -100.0


# ---------------------------------------------------------------- 1. 장치
def check_devices(cfg) -> bool:
    from app.devices import find_input, list_inputs
    a = cfg["audio"]
    try:
        devs = list_inputs()
    except Exception as e:
        add("마이크 장치", "FAIL", f"sounddevice 오류: {e}")
        return False
    if not devs:
        add("마이크 장치", "FAIL", "입력 장치가 하나도 없음 — 마이크 연결/드라이버 확인")
        return False
    ok = True
    need = [("착용자(A)", a.get("device_wearer", ""))]
    if not a.get("single_mic"):
        need.append(("주변(B)", a.get("device_ambient", "")))
    found = []
    for label, name in need:
        try:
            idx = find_input(name, a.get("prefer_wasapi", True))
            dname = next((d["name"] for d in devs if d["index"] == idx), "기본 입력 장치") if idx is not None \
                else "기본 입력 장치(config에 이름 없음)"
            found.append(f"{label}={dname}")
        except Exception as e:
            ok = False
            found.append(f"{label}='{name}' 없음")
    if not a.get("single_mic") and a.get("device_wearer", "") == a.get("device_ambient", ""):
        add("마이크 장치", "FAIL", "착용자/주변 마이크가 같은 장치로 설정됨 — config.yaml audio.device_* 를 서로 다르게")
        return False
    add("마이크 장치", "PASS" if ok else "FAIL", ", ".join(found))
    return ok


# ---------------------------------------------------------------- 2. 녹음
def record(cfg, seconds: float) -> tuple[np.ndarray, np.ndarray]:
    from app.audio_source import LiveSource
    src = LiveSource(cfg)
    src.start()
    a, b = [], []
    try:
        n = 0
        while n < seconds * 16000:
            blk = src.read(2.0)
            if blk is None:
                raise RuntimeError("오디오가 들어오지 않음")
            a.append(blk.a)
            b.append(blk.b)
            n += len(blk.a)
    finally:
        src.stop()
    return np.concatenate(a), np.concatenate(b)


def frame_diff(a: np.ndarray, b: np.ndarray, gate_db: float = -45.0) -> tuple[float, int]:
    """A가 말소리 크기인 20ms 프레임들의 (A_dB - B_dB) 중앙값."""
    n = 320
    diffs = []
    for i in range(0, min(len(a), len(b)) - n, n):
        da, dbb = db(a[i:i + n]), db(b[i:i + n])
        if max(da, dbb) > gate_db:
            diffs.append(da - dbb)
    return (float(np.median(diffs)) if diffs else float("nan")), len(diffs)


def check_recording(cfg, devices_ok: bool, auto: bool) -> None:
    margin = cfg["ownvoice"]["own_margin_db"]
    if not devices_ok:
        add("마이크 녹음", "FAIL", "장치 점검 실패로 건너뜀")
        return
    if cfg["audio"].get("single_mic"):
        add("마이크 녹음", "SKIP", "단일 마이크 모드 — dB 차 점검 없음(시작 시 목소리 등록)")
        return
    results = {}
    for who, prompt in (("wearer", "착용자가 5초 동안 평소처럼 말하세요"), ("other", "이번엔 상대방(다른 사람)이 5초 동안 말하세요")):
        print(f"    ▶ {prompt}", flush=True)
        if not auto:
            try:
                input("      준비되면 Enter… ")
            except EOFError:
                pass
        for k in (3, 2, 1):
            print(f"      {k}…", end=" ", flush=True)
            time.sleep(0.7)
        print("녹음", flush=True)
        try:
            a, b = record(cfg, 5.0)
        except Exception as e:
            add("마이크 녹음", "FAIL", f"녹음 실패: {e}")
            return
        d, nfr = frame_diff(a, b)
        results[who] = d
        print(f"      핀마이크(A) 평균 {db(a):6.1f}dB 피크 {peak_db(a):6.1f}dBFS | 주변(B) 평균 {db(b):6.1f}dB 피크 "
              f"{peak_db(b):6.1f}dBFS | 말소리 프레임 A-B 중앙값 {d:+.1f}dB ({nfr}프레임)", flush=True)
        if peak_db(a) > -1 or peak_db(b) > -1:
            print("      ⚠ 피크가 0dBFS 근처(클리핑) — 입력 게인을 낮추세요", flush=True)
    w, o = results.get("wearer", float("nan")), results.get("other", float("nan"))
    if np.isnan(w):
        add("마이크 녹음", "FAIL", "말소리가 감지되지 않음(게인/음소거 확인)")
    elif w >= margin and (np.isnan(o) or o < margin):
        add("마이크 녹음", "PASS", f"착용자 A-B {w:+.1f}dB ≥ {margin}dB, 상대 A-B {o:+.1f}dB < {margin}dB")
    elif w >= margin:
        add("마이크 녹음", "WARN", f"착용자 {w:+.1f}dB 는 OK지만 상대도 {o:+.1f}dB ≥ {margin}dB — 주변 마이크를 상대 쪽으로")
    else:
        add("마이크 녹음", "FAIL", f"착용자 A-B {w:+.1f}dB < own_margin_db {margin} — 핀마이크를 입 가까이, "
                               f"또는 tools/calibrate.py --ownvoice 로 보정")


# ---------------------------------------------------------------- 3. GPU
def check_gpu(cfg) -> None:
    want_gpu = str(cfg["asr"]["device"]).startswith("cuda") or str(cfg["sound"].get("device", "")).startswith("cuda")
    try:
        import torch
        has = torch.cuda.is_available()
    except Exception as e:
        has, torch = False, None
        err = str(e)
    if not has:
        add("GPU/VRAM", "FAIL" if want_gpu else "PASS",
            "GPU 없음 — gpu 프로필이면 --profile cpu_light 로" if want_gpu else "CPU 프로필(GPU 불필요)")
        return
    free, total = torch.cuda.mem_get_info()
    name = torch.cuda.get_device_name(0)
    llm_on_gpu = 0.0
    try:
        import requests
        s = requests.Session()
        s.trust_env = False
        ps = s.get(cfg["llm"]["url"].rstrip("/") + "/api/ps", timeout=2).json().get("models", [])
        llm_on_gpu = sum(m.get("size_vram", 0) for m in ps) / 1e9
    except Exception:
        pass
    # 대략치: Whisper large-v3-turbo int8 ~1.0GB + AST fp16 ~0.3GB + CUDA 컨텍스트 ~0.6GB, LLM 4B Q4 ~3.2GB
    need = (1.9 if want_gpu else 0.0) + (0.0 if llm_on_gpu > 0 else 3.2)
    detail = f"{name} · 여유 {free / 1e9:.1f}GB / {total / 1e9:.1f}GB · Ollama가 쓰는 VRAM {llm_on_gpu:.1f}GB · 필요 ~{need:.1f}GB"
    if not want_gpu:
        add("GPU/VRAM", "PASS", detail + " (CPU 프로필)")
    elif free / 1e9 < need:
        add("GPU/VRAM", "FAIL", detail + " — 게임·브라우저 등 GPU 쓰는 프로그램 종료, 다른 Ollama 모델 내리기")
    elif free / 1e9 < need + 0.5:
        add("GPU/VRAM", "WARN", detail + " — 여유가 빠듯함")
    else:
        add("GPU/VRAM", "PASS", detail)


# ---------------------------------------------------------------- 4. ASR
def check_asr(cfg) -> None:
    from app.asr import WhisperASR
    from app.audio_source import read_wav
    t0 = time.perf_counter()
    asr = WhisperASR(cfg, log=lambda *a: print("     ", *a))
    load = time.perf_counter() - t0
    if asr.model is None:
        add("ASR 워밍업", "FAIL", "Whisper 로딩 실패(GPU·CPU 모두)")
        return
    x = read_wav(ASSETS / "sample_speech_ko.wav")
    lat = []
    text = ""
    for _ in range(2):
        t1 = time.perf_counter()
        text, _ = asr.transcribe(x)
        lat.append((time.perf_counter() - t1) * 1000)
    want = cfg["asr"]["model"]
    gpu = cfg["asr"]["device"].startswith("cuda")
    limit = 1500 if gpu else 4000
    detail = f"{asr.desc} · 로딩 {load:.1f}s · 3초 음성 {lat[-1]:.0f}ms · \"{text}\""
    if not text:
        add("ASR 워밍업", "FAIL", detail + " — 텍스트가 비었음")
    elif not asr.desc.startswith(want):
        add("ASR 워밍업", "WARN", detail + f" — 폴백 모델로 동작({want} 실패)")
    elif lat[-1] > limit:
        add("ASR 워밍업", "WARN", detail + f" — {limit}ms 초과")
    else:
        add("ASR 워밍업", "PASS", detail)


# ---------------------------------------------------------------- 5. LLM
def check_llm(cfg) -> None:
    from app.llm_judge import LLMJudge
    j = LLMJudge(cfg, log=lambda *a: print("     ", *a))
    j.setup()
    st = j.status
    line = j.health_line()
    if st["state"] == "ok" and st.get("device") == "GPU":
        add("LLM 헬스체크", "PASS", line + (f" · {st['reason']}" if st.get("reason") else ""))
    elif st["state"] == "ok":
        add("LLM 헬스체크", "WARN", line + " — GPU에 다 안 올라감(느릴 수 있음)")
    elif st["state"] == "off":
        add("LLM 헬스체크", "WARN", line)
    else:
        add("LLM 헬스체크", "FAIL", line + " — scripts\\start_ollama.bat 로 Ollama 재시작")


# ---------------------------------------------------------------- 6. 위험 소리
def check_sound(cfg) -> None:
    from app.audio_source import read_wav
    from app.sound_events import AlertDebouncer, make_sound_classifier
    clf = make_sound_classifier(cfg, log=lambda *a: print("     ", *a))
    if clf is None:
        add("위험 소리", "FAIL", "분류기 로딩 실패")
        return
    x = read_wav(ASSETS / "sample_siren.wav")
    s = cfg["sound"]
    deb = AlertDebouncer(cfg)
    win, step = int(s["window_s"] * 16000), int(s["interval_s"] * 16000)
    alerts, best = [], 0.0
    t0 = time.perf_counter()
    n = 0
    for end in range(win, len(x) + 1, step):
        sc = clf(x[end - win:end])
        n += 1
        best = max([best] + list(sc.values()))
        alerts += deb.update(end / 16000, sc)
    ms = (time.perf_counter() - t0) * 1000 / max(n, 1)
    name = getattr(clf, "name", "?")
    if alerts:
        add("위험 소리", "PASS", f"{name} · 샘플 사이렌 감지: {alerts[0]['label']} {alerts[0]['score']:.2f} "
                               f"(창당 {ms:.0f}ms)")
    else:
        add("위험 소리", "FAIL", f"{name} · 샘플 사이렌 미감지(최고 점수 {best:.2f} < {s['score_threshold']})")


# ---------------------------------------------------------------- 7. LAN
def check_lan(cfg) -> None:
    from app.server import lan_addresses, print_qr
    port = cfg["server"]["port"]
    ips = lan_addresses()
    busy = False
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            busy = s.connect_ex(("127.0.0.1", port)) == 0
    except Exception:
        pass
    for ip in ips:
        print(f"      대시보드 http://{ip}:{port}/   폰 http://{ip}:{port}/phone", flush=True)
    print_qr(f"http://{ips[0]}:{port}/phone")
    if ips == ["127.0.0.1"]:
        add("LAN 주소/QR", "FAIL", "네트워크 없음 — 휴대폰 핫스팟에 연결")
    elif busy:
        add("LAN 주소/QR", "WARN", f"{', '.join(ips)} · 포트 {port}이 이미 사용 중(서버가 떠 있거나 다른 프로그램) — --port 로 변경")
    else:
        add("LAN 주소/QR", "PASS", f"{', '.join(ips)} · 포트 {port} 사용 가능 · 폰과 같은 핫스팟인지 확인")


def main():
    import os
    if os.name == "nt":
        os.system("")   # Windows 콘솔 ANSI 색 켜기
    ap = argparse.ArgumentParser(description="발표 전 사전 점검")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--yes", action="store_true", help="녹음 안내에서 Enter 대기 없음")
    args = ap.parse_args()
    cfg = load_config(args.profile)
    print(f"사전 점검 · 프로필 {cfg['_profile']} · 착용자 이름 '{cfg['wearer']['name']}'")
    t_all = time.perf_counter()
    head(1, "마이크 장치")
    dev_ok = check_devices(cfg)
    head(2, "마이크 녹음 (5초씩)")
    check_recording(cfg, dev_ok, args.yes)
    for i, (title, fn) in enumerate([("GPU / VRAM", check_gpu), ("ASR 워밍업", check_asr),
                                     ("LLM 헬스체크", check_llm), ("위험 소리 분류", check_sound),
                                     ("LAN 주소 / QR", check_lan)], start=3):
        head(i, title)
        try:
            fn(cfg)
        except Exception as e:
            add(title, "FAIL", f"점검 중 오류: {type(e).__name__}: {e}")

    w = max(len(n) for n, _, _ in ROWS) + 2
    print("\n" + "=" * 100)
    print(f"{'항목':<{w}} 결과   상세")
    print("-" * 100)
    for n, st, d in ROWS:
        print(f"{n:<{w}} {COLOR.get(st, '')}{st:<5}{COLOR['END']}  {d}")
    print("=" * 100)
    fails = [n for n, st, _ in ROWS if st == "FAIL"]
    print(f"총 {time.perf_counter() - t_all:.0f}초 · PASS {sum(st == 'PASS' for _, st, _ in ROWS)} · "
          f"WARN {sum(st == 'WARN' for _, st, _ in ROWS)} · FAIL {len(fails)}"
          + (f"  → 실패: {', '.join(fails)}" if fails else "  → 시연 준비 완료"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
