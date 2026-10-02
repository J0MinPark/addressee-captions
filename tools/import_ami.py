"""AMI Meeting Corpus → 수신자 판별 평가용 시나리오.

    python tools/import_ami.py --list                 # addressee 주석이 있는 회의 목록만
    python tools/import_ami.py                        # 기본: 회의 3개, 앞 15분, clean/snr10/snr5
    python tools/import_ami.py --meetings 1 --minutes 10 --conds clean
    python tools/import_ami.py --ids ES2008a IS1008a TS3005a --calib TS3005a

- 공식 수동 주석(NXT, ami_public_manual_1.6.2)에서 dialogue act의 addressee 속성이 있는 회의를 찾는다.
- 회의마다 개별 헤드셋 4개 + 원거리 Array1-01 을 공식 미러에서 받는다(있으면 건너뜀).
- 참가자 문자(A–D) ↔ 헤드셋 채널은 corpusResources/meetings.xml 의 speaker@channel 로 확인한다(추측하지 않음).
- 착용자 4명 각각: A 채널 = 착용자 헤드셋, B 채널 = Array1-01. 16kHz mono, 앞 N분.
- 소음: DEMAND PCAFETER(카페테리아)를 B 채널에 SNR 10/5 dB로 섞은 버전. 소음 파일이 없고 받을 수도 없으면 경고 후 건너뜀.
- 출력: <data>/ami_<ID>_w<L>_<clean|snr10|snr5>_<take1|take2>_{A,B}.wav + .json, 회의별 <data>/ami_<ID>.das.json(대화행위)
  회의 하나는 보정용(take1), 나머지는 평가용(take2). 한 회의는 한쪽에만 속한다.
원본 다운로드는 --raw(기본 ~/hearme_data/ami), 시나리오는 config paths.data_dir(HEARME_DATA_DIR로 변경 가능).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
import os
import shutil
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402

ANN_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
AUDIO_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{m}/audio/{m}.{ch}.wav"
DEMAND_URL = "https://zenodo.org/records/1227121/files/PCAFETER_16k.zip?download=1"
NITE = "{http://nite.sourceforge.net/}"
SR = 16000
LETTERS = "ABCD"
FAR_FIELD_ORDER = ("Array1-01", "Array2-01", "Array2-02", "Array1-02")


# ---------------------------------------------------------------- 다운로드
def download(url: str, dst: Path, label: str) -> bool:
    if dst.exists() and dst.stat().st_size > 0:
        print(f"  ✓ 있음  {label} ({dst.stat().st_size / 1e6:.1f}MB)")
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "hearme-ami-import"})
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            got, t0, last = 0, time.time(), 0.0
            while True:
                buf = r.read(1 << 20)
                if not buf:
                    break
                f.write(buf)
                got += len(buf)
                if time.time() - last > 1.0:
                    last = time.time()
                    pct = f"{got / total * 100:5.1f}%" if total else ""
                    print(f"\r  ↓ {label}: {got / 1e6:7.1f}/{total / 1e6:.1f}MB {pct} "
                          f"{got / 1e6 / max(time.time() - t0, 1e-3):.1f}MB/s", end="", flush=True)
        tmp.replace(dst)
        print(f"\r  ✓ 받음  {label} ({got / 1e6:.1f}MB, {time.time() - t0:.0f}s)" + " " * 20)
        return True
    except Exception as e:
        print(f"\n  ✗ 실패  {label}: {e}")
        tmp.unlink(missing_ok=True)
        return False


def ensure_annotations(raw: Path) -> Path:
    ann = raw / "ann"
    if not (ann / "corpusResources" / "meetings.xml").exists():
        z = raw / "ami_public_manual_1.6.2.zip"
        if not download(ANN_URL, z, "AMI 수동 주석(NXT)"):
            raise SystemExit("AMI 주석을 받을 수 없습니다.")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(ann)
    return ann


# ---------------------------------------------------------------- NXT 파싱
def parse_meetings(ann: Path) -> dict:
    """meetings.xml → {회의ID: {"duration": 초, "channels": {문자: 헤드셋 채널}, "global": {문자: 참가자ID}}}"""
    out = {}
    for m in ET.parse(ann / "corpusResources" / "meetings.xml").getroot().iter("meeting"):
        mid = m.get("observation")
        ch, gl = {}, {}
        for s in m.iter("speaker"):
            ch[s.get("nxt_agent")] = int(s.get("channel"))
            gl[s.get("nxt_agent")] = s.get("global_name")
        out[mid] = {"duration": float(m.get("duration") or 0), "channels": ch, "global": gl}
    return out


def _words(ann: Path, mid: str, spk: str) -> tuple[list[str], dict]:
    p = ann / "words" / f"{mid}.{spk}.words.xml"
    order, info = [], {}
    if not p.exists():
        return order, info
    for el in ET.parse(p).getroot():
        wid = el.get(f"{NITE}id")
        if wid is None:
            continue
        order.append(wid)
        st, en = el.get("starttime"), el.get("endtime")
        info[wid] = (float(st) if st else None, float(en) if en else None,
                     (el.text or "") if el.tag == "w" and el.get("punc") != "true" else "")
    return order, info


def load_das(ann: Path, mid: str) -> list[dict]:
    """대화행위 목록 [{speaker, start, end, addressee: [문자], text, id}] (시간순)."""
    das = []
    for spk in LETTERS:
        p = ann / "dialogueActs" / f"{mid}.{spk}.dialog-act.xml"
        if not p.exists():
            continue
        order, info = _words(ann, mid, spk)
        pos = {w: i for i, w in enumerate(order)}
        for d in ET.parse(p).getroot().iter("dact"):
            ch = d.find(f"{NITE}child")
            if ch is None:
                continue
            ids = re.findall(r"id\(([^)]+)\)", ch.get("href", ""))
            if not ids or ids[0] not in pos:
                continue
            i0, i1 = pos[ids[0]], pos.get(ids[-1], pos[ids[0]])
            ws = [info[order[i]] for i in range(i0, i1 + 1)]
            ts = [t for w in ws for t in w[:2] if t is not None]
            if not ts:
                continue
            addr = d.get("addressee")
            das.append({"id": d.get(f"{NITE}id"), "speaker": spk, "start": min(ts), "end": max(ts),
                        "addressee": addr.split(",") if addr else [],
                        "text": " ".join(w[2] for w in ws if w[2]).strip()})
    das.sort(key=lambda x: (x["start"], x["end"]))
    return das


def addressee_meetings(ann: Path, meets: dict, minutes: float) -> list[dict]:
    files = sorted((ann / "dialogueActs").glob("*.dialog-act.xml"))
    ids = sorted({f.name.split(".")[0] for f in files if "addressee=" in f.read_text(encoding="latin-1")})
    rows = []
    for mid in ids:
        das = load_das(ann, mid)
        win = [d for d in das if d["start"] < minutes * 60]
        rows.append({"id": mid, "duration_min": round(meets.get(mid, {}).get("duration", 0) / 60, 1),
                     "dacts": len(das), "addressed": sum(bool(d["addressee"]) for d in das),
                     "win_dacts": len(win), "win_addressed": sum(bool(d["addressee"]) for d in win)})
    for r in rows:
        r["coverage"] = r["addressed"] / r["dacts"] if r["dacts"] else 0
        r["win_coverage"] = r["win_addressed"] / r["win_dacts"] if r["win_dacts"] else 0
    return rows


# ---------------------------------------------------------------- 오디오
def rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12))


def speech_mask(das: list[dict], n: int) -> np.ndarray:
    m = np.zeros(n, bool)
    for d in das:
        a, b = int(d["start"] * SR), int(d["end"] * SR)
        m[max(a, 0):min(b, n)] = True
    return m


def normalize(x: np.ndarray, mask: np.ndarray, target_db: float = -26.0) -> tuple[np.ndarray, float]:
    """말소리 구간(정답 대화행위 시간)의 RMS를 target_db로 맞춘다. 반환: (신호, 이득 dB)."""
    ref = x[mask] if mask.any() else x
    g = target_db - rms_db(ref)
    y = x * (10 ** (g / 20))
    peak = np.abs(y).max()
    if peak > 0.99:   # 클리핑 방지
        y *= 0.99 / peak
        g += 20 * np.log10(0.99 / peak)
    return y.astype(np.float32), round(float(g), 2)


def mix_noise(b: np.ndarray, noise: np.ndarray, mask: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, max(1, len(noise) - 1)))
    reps = int(np.ceil((len(b) + start) / len(noise))) + 1
    nz = np.tile(noise, reps)[start:start + len(b)]
    p_s = np.mean(np.square(b[mask], dtype=np.float64)) if mask.any() else np.mean(np.square(b, dtype=np.float64))
    p_n = np.mean(np.square(nz, dtype=np.float64)) + 1e-12
    nz = nz * np.sqrt(p_s / (p_n * 10 ** (snr_db / 10)))
    y = b + nz
    peak = np.abs(y).max()
    return (y * (0.99 / peak if peak > 0.99 else 1.0)).astype(np.float32)


def ensure_noise(raw: Path) -> Path | None:
    wav = raw / "demand" / "PCAFETER" / "ch01.wav"
    if wav.exists():
        return wav
    z = raw / "demand" / "PCAFETER_16k.zip"
    if not download(DEMAND_URL, z, "DEMAND PCAFETER (16kHz)"):
        return None
    try:
        with zipfile.ZipFile(z) as zf:
            zf.extractall(raw / "demand")
    except Exception as e:
        print(f"  ✗ DEMAND 압축 해제 실패: {e}")
        return None
    return wav if wav.exists() else next((raw / "demand").rglob("ch01.wav"), None)


def main():
    from app.audio_source import read_wav, write_wav
    from app.config import load_config, resolve_path
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="addressee 주석 회의 목록만 출력")
    ap.add_argument("--meetings", type=int, default=3, help="사용할 회의 수(기본 3)")
    ap.add_argument("--ids", nargs="*", default=None, help="회의 ID 직접 지정")
    ap.add_argument("--calib", default=None, help="보정용(take1) 회의 ID(기본: 선택한 회의 중 주석 비율이 가장 낮은 것)")
    ap.add_argument("--minutes", type=float, default=15.0, help="회의 앞 몇 분을 쓸지")
    ap.add_argument("--conds", default="clean,snr10,snr5")
    ap.add_argument("--raw", default=str(Path.home() / "hearme_data" / "ami"), help="원본 다운로드 위치")
    args = ap.parse_args()

    cfg = load_config("ami")
    raw = Path(args.raw)
    data = resolve_path(cfg, "data_dir")
    ann = ensure_annotations(raw)
    meets = parse_meetings(ann)
    rows = addressee_meetings(ann, meets, args.minutes)
    print(f"\naddressee 속성이 있는 회의 {len(rows)}개 (앞 {args.minutes:g}분 기준 주석 비율):")
    print(f"  {'회의':<9}{'길이(분)':>8}{'DA':>6}{'주석':>6}{'비율':>7}{'앞N분 DA':>10}{'앞N분 비율':>11}")
    for r in sorted(rows, key=lambda r: -r["win_coverage"]):
        print(f"  {r['id']:<9}{r['duration_min']:>8}{r['dacts']:>6}{r['addressed']:>6}{r['coverage']:>7.0%}"
              f"{r['win_dacts']:>10}{r['win_coverage']:>11.0%}")
    if args.list:
        return

    ok = [r for r in rows if r["duration_min"] >= args.minutes and r["win_coverage"] >= 0.5]
    if args.ids:
        sel = args.ids
    else:
        # 같은 그룹(같은 참가자, 예: IS1008a/b)은 하나만 → 보정·평가에 같은 사람이 겹치지 않게
        sel, groups = [], set()
        for r in sorted(ok, key=lambda r: (-r["win_coverage"], r["id"])):
            if r["id"][:-1] not in groups and len(sel) < args.meetings:
                sel.append(r["id"])
                groups.add(r["id"][:-1])
    if not sel:
        raise SystemExit("조건에 맞는 회의가 없습니다(--minutes 를 줄이거나 --ids 지정).")
    cov = {r["id"]: r["win_coverage"] for r in rows}
    calib = args.calib or (min(sel, key=lambda m: cov.get(m, 0)) if len(sel) > 1 else None)
    print(f"\n선택: {', '.join(sel)} · 보정용(take1): {calib or '없음(회의 1개 → 전부 평가용)'}")

    conds = [c.strip() for c in args.conds.split(",") if c.strip()]
    noise = None
    if any(c.startswith("snr") for c in conds):
        noise_wav = ensure_noise(raw)
        if noise_wav is None:
            print("⚠ 경고: DEMAND PCAFETER 소음 파일이 없어 소음 조건(snr10/snr5)을 건너뜁니다.")
            conds = [c for c in conds if not c.startswith("snr")]
        else:
            noise = read_wav(noise_wav)

    log = {"meetings": {}, "conds": conds, "minutes": args.minutes, "calib": calib}
    for mid in sel:
        info = meets[mid]
        print(f"\n[{mid}] 참가자–헤드셋 채널 대응(meetings.xml): " +
              ", ".join(f"{L}→Headset-{info['channels'][L]} ({info['global'][L]})" for L in sorted(info["channels"])))
        files = {f"Headset-{c}": raw / "audio" / mid / f"{mid}.Headset-{c}.wav" for c in sorted(info["channels"].values())}
        for ch, dst in files.items():
            if not download(AUDIO_URL.format(m=mid, ch=ch), dst, f"{mid}.{ch}"):
                raise SystemExit(f"{mid}.{ch} 다운로드 실패")
        # 원거리 마이크: 기본 Array1-01. 미러에 없으면(예: IS1003b) 정해진 순서로 대체하고 기록한다.
        far = None
        for ch in FAR_FIELD_ORDER:
            dst = raw / "audio" / mid / f"{mid}.{ch}.wav"
            if download(AUDIO_URL.format(m=mid, ch=ch), dst, f"{mid}.{ch}"):
                far = ch
                break
        if far is None:
            raise SystemExit(f"{mid}: 원거리 마이크({FAR_FIELD_ORDER}) 없음")
        if far != "Array1-01":
            print(f"  ⚠ {mid}: Array1-01 없음 → {far} 사용")
        files["far"] = raw / "audio" / mid / f"{mid}.{far}.wav"
        n = int(args.minutes * 60 * SR)
        das = [d for d in load_das(ann, mid) if d["start"] < args.minutes * 60]
        (data / f"ami_{mid}.das.json").write_text(json.dumps(
            {"meeting": mid, "minutes": args.minutes, "channels": info["channels"], "das": das},
            ensure_ascii=False), encoding="utf-8")
        b_raw = read_wav(files["far"])[:n]
        mask_all = speech_mask(das, len(b_raw))
        b_clean, gb = normalize(b_raw, mask_all)
        bs = {"clean": b_clean}
        for c in conds:
            if c.startswith("snr"):
                bs[c] = mix_noise(b_clean, noise, mask_all, float(c[3:]), seed=zlib.crc32(mid.encode()) % 1000)
        take = "take1" if mid == calib else "take2"
        log["meetings"][mid] = {"channels": info["channels"], "global": info["global"], "take": take, "far_field": far,
                                "das": len(das), "array_gain_db": gb, "wearers": {}}
        for L in sorted(info["channels"]):
            a_raw = read_wav(files[f"Headset-{info['channels'][L]}"])[:n]
            m_w = speech_mask([d for d in das if d["speaker"] == L], len(a_raw))
            a, ga = normalize(a_raw, m_w)
            log["meetings"][mid]["wearers"][L] = {"headset_gain_db": ga}
            for c in conds:
                name = f"ami_{mid}_w{L}_{c}_{take}"
                write_wav(data / f"{name}_A.wav", a)
                # B 채널은 착용자 4명이 똑같다 → 한 번만 쓰고 하드링크(디스크 절약, 실패하면 복사)
                shared = data / f"ami_{mid}_{c}_Bshared.wav"
                if not shared.exists():
                    write_wav(shared, bs[c][:len(a)])
                dst = data / f"{name}_B.wav"
                dst.unlink(missing_ok=True)
                try:
                    os.link(shared, dst)
                except OSError:
                    shutil.copyfile(shared, dst)
                (data / f"{name}.json").write_text(json.dumps({
                    "scenario": name, "source": "AMI Meeting Corpus", "synthetic": False, "meeting": mid,
                    "wearer": L, "wearer_global": info["global"][L], "headset_channel": info["channels"][L],
                    "far_field": far, "condition": c, "split": take, "minutes": args.minutes,
                    "noise": None if c == "clean" else f"DEMAND PCAFETER @ SNR {c[3:]}dB",
                    "gain_db": {"A": ga, "B": gb}, "duration_s": round(len(a) / SR, 1),
                    "devices": [f"AMI {mid} Headset-{info['channels'][L]}", f"AMI {mid} {far}"]},
                    ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"  ✓ 착용자 {L} (Headset-{info['channels'][L]}): {', '.join(conds)} → ami_{mid}_w{L}_*_{take}")
    out = data / f"ami_import_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n완료. 시나리오 {len(sel) * 4 * len(conds)}개 · 기록: {out}\n다음: python tools/run_ami.py")


if __name__ == "__main__":
    main()
