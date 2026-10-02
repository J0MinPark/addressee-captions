"""임계값 보정.

1) 화자 유사도: 화자별 WAV(한 사람만 말한 녹음)를 0.5/1/2초 구간으로 잘라 ECAPA 유사도 분포(동일인 vs 타인)
   python tools/calibrate.py --speaker 민수=data/spk_minsu.wav --speaker 지영=data/spk_jiyoung.wav
   (또는 라벨된 시나리오의 캐시 임베딩: --scenario demo  → results/demo.emb.npz + labels 로 화자 묶음)

2) 본인 발화 dB 차이: 2채널 녹음에서 A가 말소리인 프레임의 A_dB - B_dB 분포 → own_margin_db 추천
   python tools/calibrate.py --ownvoice data/demo

결과를 보고 app/config.yaml 의 speaker.spk_threshold, ownvoice.own_margin_db 를 고친다.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import winsetup  # noqa: E402

winsetup.setup()

import numpy as np  # noqa: E402

from app.config import load_config, resolve_path  # noqa: E402


def hist_line(vals, lo, hi, bins=20, width=40):
    if len(vals) == 0:
        return "(없음)"
    h, _ = np.histogram(vals, bins=bins, range=(lo, hi))
    m = h.max() or 1
    chars = " ▁▂▃▄▅▆▇█"
    return "".join(chars[int(round(v / m * 8))] for v in h)


def best_threshold(same: np.ndarray, diff: np.ndarray) -> tuple[float, float]:
    """균형 정확도 최대 임계값."""
    best = (0.5, 0.0)
    for t in np.linspace(0, 1, 201):
        acc = 0.5 * ((same >= t).mean() + (diff < t).mean())
        if acc > best[1]:
            best = (float(t), float(acc))
    return best


def voiced_chunks(x: np.ndarray, vad, win_s: float, sr=16000) -> list[np.ndarray]:
    """VAD로 말소리만 이어 붙인 뒤 win_s 길이로 자른다."""
    keep = [x[i:i + 512] for i in range(0, len(x) - 512, 512) if vad(x[i:i + 512]) >= 0.5]
    try:
        vad.reset()
    except Exception:
        pass
    if not keep:
        return []
    v = np.concatenate(keep)
    n = int(win_s * sr)
    return [v[i:i + n] for i in range(0, len(v) - n + 1, n)]


def speaker_calibration(cfg, speakers: dict[str, list[np.ndarray]] | None, embs_by_spk=None):
    from app.speaker import make_embedder
    from app.segmenter import make_vad
    print("\n=== 화자 유사도 (ECAPA) ===")
    recs = {}
    if speakers:
        emb = make_embedder(cfg)
        vad = make_vad(cfg)
        for win in (0.5, 1.0, 2.0):
            E = {name: [emb(c) for c in voiced_chunks(np.concatenate(xs), vad, win)][:40]
                 for name, xs in speakers.items()}
            recs[win] = E
    if embs_by_spk:
        recs["cached"] = embs_by_spk
    thr_now = cfg["speaker"]["spk_threshold"]
    suggestions = {}
    for win, E in recs.items():
        same, diff = [], []
        names = [k for k in E if len(E[k]) >= 2]
        for k in names:
            for a, b in itertools.combinations(E[k], 2):
                same.append(float(np.dot(a, b)))
        for k1, k2 in itertools.combinations([k for k in E if E[k]], 2):
            for a in E[k1][:20]:
                for b in E[k2][:20]:
                    diff.append(float(np.dot(a, b)))
        same, diff = np.array(same), np.array(diff)
        label = f"{win}초 구간" if win != "cached" else "캐시 구간(실제 구간 길이)"
        print(f"\n[{label}]  화자 {len(E)}명, 동일 쌍 {len(same)}, 타인 쌍 {len(diff)}")
        if len(same) and len(diff):
            print(f"  동일인  평균 {same.mean():.2f}  p5 {np.percentile(same, 5):.2f}  |{hist_line(same, -0.2, 1)}|")
            print(f"  타인    평균 {diff.mean():.2f}  p95 {np.percentile(diff, 95):.2f} |{hist_line(diff, -0.2, 1)}|")
            print(f"           -0.2{' ' * 34}1.0")
            t, acc = best_threshold(same, diff)
            suggestions[win] = t
            print(f"  추천 임계값 {t:.2f} (균형 정확도 {acc * 100:.0f}%) · 현재 spk_threshold={thr_now}")
            print(f"  현재 임계값에서: 동일인 통과 {(same >= thr_now).mean() * 100:.0f}%, "
                  f"타인 오통과 {(diff >= thr_now).mean() * 100:.0f}%")
    if 1.0 in suggestions or 2.0 in suggestions:
        s = suggestions.get(1.0, suggestions.get(2.0))
        print(f"\n→ 제안: speaker.spk_threshold: {s:.2f}  (1초 이상 구간 기준. 0.5초 결과가 나쁘면 short_s 유지)")


def otsu(x: np.ndarray, lo: float, hi: float) -> float:
    h, edges = np.histogram(x, bins=120, range=(lo, hi))
    c = (edges[:-1] + edges[1:]) / 2
    w = h.astype(float) / max(h.sum(), 1)
    best, bt = -1, (lo + hi) / 2
    for i in range(1, len(c)):
        w0, w1 = w[:i].sum(), w[i:].sum()
        if w0 == 0 or w1 == 0:
            continue
        m0, m1 = (w[:i] * c[:i]).sum() / w0, (w[i:] * c[i:]).sum() / w1
        v = w0 * w1 * (m0 - m1) ** 2
        if v > best:
            best, bt = v, c[i]
    return float(bt)


def ownvoice_calibration(cfg, prefix: str):
    from app.audio_source import read_wav, scenario_paths
    from app.ownvoice import rms_db
    from app.segmenter import make_vad
    pa, pb, _ = scenario_paths(prefix)
    a, b = read_wav(pa), read_wav(pb)
    n = min(len(a), len(b))
    vad = make_vad(cfg)
    diffs = []
    for i in range(0, n - 512, 512):
        ca, cb = a[i:i + 512], b[i:i + 512]
        if vad(ca) >= cfg["vad"]["threshold"]:
            diffs.append(rms_db(ca) - rms_db(cb))
    diffs = np.array(diffs)
    print(f"\n=== 본인 발화 dB 차이 (A_dB - B_dB, A가 말소리인 프레임 {len(diffs)}개) ===")
    if len(diffs) < 20:
        print("말소리 프레임이 너무 적습니다.")
        return
    lo, hi = np.percentile(diffs, 1) - 1, np.percentile(diffs, 99) + 1
    print(f"  {lo:+.0f}dB |{hist_line(diffs, lo, hi, bins=30, width=30)}| {hi:+.0f}dB")
    t = otsu(diffs, lo, hi)
    own, other = diffs[diffs >= t], diffs[diffs < t]
    print(f"  두 봉우리 경계(Otsu) {t:+.1f}dB · 위쪽(본인 추정) 중앙값 {np.median(own):+.1f}dB, "
          f"아래쪽(타인 추정) 중앙값 {np.median(other) if len(other) else float('nan'):+.1f}dB")
    margin = float(np.clip(t, 2.0, 15.0))
    print(f"  현재 own_margin_db={cfg['ownvoice']['own_margin_db']}  → 제안: ownvoice.own_margin_db: {margin:.1f}")
    if len(other) < 0.05 * len(diffs):
        print("  (주의: 타인 발화가 거의 없는 녹음입니다. 상대가 말하는 부분이 있는 녹음으로 다시 재세요.)")
    return margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", action="append", default=[], help="이름=경로.wav (여러 번)")
    ap.add_argument("--scenario", action="append", default=[], help="라벨된 시나리오(캐시 임베딩 사용)")
    ap.add_argument("--ownvoice", default=None, help="2채널 녹음 접두사 (data/NAME)")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--allow-untagged", action="store_true",
                    help="_take1 표시 없는 시나리오도 보정에 사용(기록돼서 이후 평가에서는 제외됨)")
    args = ap.parse_args()
    cfg = load_config(args.profile)
    if not (args.speaker or args.scenario or args.ownvoice):
        ap.print_help()
        return
    # 데이터 분할: 보정은 보정용(_take1)만. 쓴 녹음은 기록해서 evaluate.py 가 평가에서 거부한다.
    sys.path.insert(0, str(ROOT / "tools"))
    from datasplit import check_calibration_inputs, register_calibration, role, scenario_name
    scen = [scenario_name(n) for n in args.scenario] + ([scenario_name(args.ownvoice)] if args.ownvoice else [])
    check_calibration_inputs(scen, args.allow_untagged)
    spk_files = [s.split("=", 1)[1] for s in args.speaker]
    bad = [f for f in spk_files if role(scenario_name(f)) == "eval"]
    if bad:
        raise SystemExit(f"[분할 오류] 평가용(_take2) 녹음은 보정에 쓸 수 없습니다: {', '.join(bad)}")
    register_calibration(scen + [scenario_name(f) for f in spk_files])
    if args.speaker:
        from app.audio_source import read_wav
        spk = {}
        for s in args.speaker:
            name, path = s.split("=", 1)
            spk.setdefault(name, []).append(read_wav(path))
        speaker_calibration(cfg, spk)
    if args.scenario:
        results = resolve_path(cfg, "results_dir")
        by = {}
        for n in args.scenario:
            emb = np.load(results / f"{n}.emb.npz")
            segs = {}
            for line in (results / f"{n}.segments.jsonl").read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                if not r.get("_meta"):
                    segs[r["seg_id"]] = r
            for sid in emb.files:
                spk = segs.get(sid, {}).get("speaker_id")
                if spk is not None:
                    by.setdefault(f"{n}#{spk}", []).append(emb[sid])
        print("(캐시 모드: 화자 ID는 온라인 군집 결과이므로 군집 오류는 측정되지 않습니다. 정확한 보정은 --speaker 사용)")
        speaker_calibration(cfg, None, by)
    if args.ownvoice:
        ownvoice_calibration(cfg, args.ownvoice)


if __name__ == "__main__":
    main()
