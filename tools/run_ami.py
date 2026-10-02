"""AMI 평가 전체 실행: 보정(take1) → 재생(take2, 최대 속도) → 정답 라벨 → ami_ablation.md

    python tools/run_ami.py                                   # import_ami.py로 만든 시나리오 전부
    python tools/run_ami.py --meeting ES2008b --wearer A --conds clean   # 빠른 확인(회의 1·착용자 1·clean)

- own_margin_db 는 보정용(take1, clean) 녹음에서만 측정(tools/calibrate.py와 같은 방법)해 평가 재생에 쓴다.
  보정용 녹음은 results/calibration_used.json 에 기록되어 평가에서 거부된다.
- 같은 회의·같은 소음 조건의 B 채널 처리 결과(VAD 확률, ASR, 화자 임베딩)는 착용자 4명이 공유
  (results/cache/ami_<회의>_<조건>.featcache.pkl).
- 시작할 때 예상 시간을 출력하고, 60분이 넘으면 줄이는 옵션을 제안한다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from app import winsetup  # noqa: E402

winsetup.setup()

import numpy as np  # noqa: E402

from app.config import load_config, resolve_path  # noqa: E402

# 예상 시간 모델(RTX 4060 실측 기반 대략치): 오디오 1분당 처리 초
SEC_PER_MIN_FIRST = 9.0    # 회의·조건의 첫 착용자(B 채널 ASR·임베딩 계산)
SEC_PER_MIN_CACHED = 4.0   # 같은 B 채널을 공유하는 나머지 착용자(LLM 판정 위주)


def scenarios(data: Path, args) -> list[dict]:
    out = []
    for p in sorted(data.glob("ami_*_take*.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        if args.meeting and m["meeting"] not in args.meeting:
            continue
        if args.wearer and m["wearer"] not in args.wearer:
            continue
        if m["condition"] not in args.conds:
            continue
        out.append(m)
    return out


def calibrate_margin(cfg, data: Path, calib: list[dict], results: Path) -> float | None:
    from calibrate import ownvoice_calibration
    from datasplit import register_calibration
    clean = [m for m in calib if m["condition"] == "clean"]
    if not clean:
        return None
    vals = []
    for m in clean:
        print(f"\n[보정] {m['scenario']}")
        v = ownvoice_calibration(cfg, str(data / m["scenario"]))
        if v is not None:
            vals.append(v)
    register_calibration([m["scenario"] for m in calib])
    if not vals:
        return None
    margin = float(np.median(vals))
    (results / "ami_calibration.json").write_text(json.dumps(
        {"own_margin_db": margin, "per_wearer": vals, "from": [m["scenario"] for m in clean]}, indent=1),
        encoding="utf-8")
    return margin


def main(argv=None, _final_test_ok: bool = False):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=None, choices=["dev", "test"],
                    help="splits.json 집합만 실행. test 는 tools/final_test.py 를 통해서만(한 번)")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--meeting", nargs="*", default=None)
    ap.add_argument("--wearer", nargs="*", default=None)
    ap.add_argument("--conds", default="clean,snr10,snr5")
    ap.add_argument("--no-llm-cache", action="store_true")
    ap.add_argument("--yes", action="store_true", help="예상 시간이 60분을 넘어도 그대로 진행")
    args = ap.parse_args(argv)
    args.conds = [c.strip() for c in args.conds.split(",") if c.strip()]
    from splits import guard, meetings as split_meetings
    if args.split == "test" and not _final_test_ok:
        raise SystemExit("[splits] 시험 세트 재생은 tools/final_test.py 로만(한 번) 실행합니다.")
    if args.split:
        args.meeting = (args.meeting or []) + split_meetings(args.split) if not args.meeting else args.meeting

    # 재생은 '특징 추출'이다: dev·test 모두 같은 고정 구성(v1 판정기 P1c·qwen3:4b, 손 가중치, 플래그 끔)으로 돌리고,
    # 후보 구성(판정기 변형·융합·플래그)은 기록된 입력으로 오프라인 적용한다(judge_offline / tune_dev / final_test).
    over = {"llm": {"always_call": True, "variant": "P1c", "models": ["qwen3:4b", "qwen3:1.7b", "qwen2.5:3b"]},
            "policy": {"fusion": {"type": "hand"}, "candidate_rejudge": False, "short_skip_llm": False}}
    if args.no_llm_cache:
        over["llm"]["cache"] = False
    cfg = load_config("ami", overrides=over)
    data, results = resolve_path(cfg, "data_dir"), resolve_path(cfg, "results_dir")
    allsc = scenarios(data, argparse.Namespace(meeting=None, wearer=None, conds=["clean"]))
    calib = [m for m in allsc if m["split"] == "take1"]
    todo = [m for m in scenarios(data, args) if m["split"] == "take2"]
    if args.split:
        guard([m["scenario"] for m in todo], "tune" if args.split == "dev" else "test")
    if not todo:
        raise SystemExit("평가용(take2) AMI 시나리오가 없습니다. 먼저: python tools/import_ami.py")

    # ---- 예상 시간
    bkeys, est = set(), 0.0
    for m in todo:
        k = (m["meeting"], m["condition"])
        est += m["minutes"] * (SEC_PER_MIN_CACHED if k in bkeys else SEC_PER_MIN_FIRST)
        bkeys.add(k)
    est += 60 * len({m["meeting"] for m in calib}) if calib else 0
    print(f"평가 시나리오 {len(todo)}개 (회의 {sorted({m['meeting'] for m in todo})}, 조건 {args.conds}) · "
          f"B 채널 공유 그룹 {len(bkeys)}개 · 예상 {est / 60:.0f}분")
    if est > 3600:
        print("⚠ 60분을 넘을 것 같습니다. 줄이는 방법:\n"
              "   python tools/import_ami.py --meetings 2 --minutes 10      # 회의 수·사용 길이 줄여 다시 만들기\n"
              "   python tools/run_ami.py --conds clean,snr5                # 소음 조건 줄이기\n"
              "   python tools/run_ami.py --meeting ES2008b                 # 회의 하나만")
        if not args.yes:
            print("   그대로 진행하려면 --yes")
            return

    # ---- 보정(take1만)
    calp = results / "ami_calibration.json"
    margin = calibrate_margin(cfg, data, calib, results) if calib else None
    if margin is None and calp.exists():
        margin = json.loads(calp.read_text(encoding="utf-8"))["own_margin_db"]
    if margin is not None:
        cfg["ownvoice"]["own_margin_db"] = round(margin, 1)
    print(f"\nown_margin_db = {cfg['ownvoice']['own_margin_db']} "
          f"({'보정용 take1에서 측정' if margin is not None else '기본값 — 보정용 데이터 없음'})")

    from ami_labels import make_labels
    from app.audio_source import ReplaySource
    from app.featcache import FeatureCache
    from app.pipeline import Pipeline, load_models
    models = load_models(cfg)
    if models.judge is None or not models.judge.available:
        print("⚠ LLM이 붙지 않았습니다 — 의미 판정 없이 진행합니다(full/semantic 결과가 타이밍 규칙으로만 나옴).")
    caches: dict = {}
    t_all = time.perf_counter()
    for i, m in enumerate(todo, 1):
        name = m["scenario"]
        key = f"{m['meeting']}_{m['condition']}"
        if key not in caches:
            caches[key] = FeatureCache(results / "cache" / f"ami_{key}.featcache.pkl")
        fc = caches[key]
        t0 = time.perf_counter()
        src = ReplaySource(data / name, cfg, realtime=False)
        pipe = Pipeline(cfg, src, models, record_segments=True, log_events=False, run_name=name,
                        feature_cache=fc, cache_key=key, log=lambda *a: None)
        pipe.start()
        ok = pipe.wait_finished(timeout=max(1800, src.duration * 3))
        pipe.stop()
        n = pipe.write_segments(results / f"{name}.segments.jsonl")
        fc.save()
        st = make_labels(name, data, results)
        dt = time.perf_counter() - t0
        el = time.perf_counter() - t_all
        print(f"[{i}/{len(todo)}] {name}: {src.duration / 60:.0f}분 → {dt:.0f}s · 구간 {n} · unmatched {st['unmatched']} · "
              f"캐시 {fc.stats()} · LLM 호출 {getattr(models.judge, 'calls', 0)} · 경과 {el / 60:.1f}분, "
              f"남은 예상 {el / i * (len(todo) - i) / 60:.0f}분" + ("" if ok else " [시간 초과]"), flush=True)

    if args.no_report:
        return [m["scenario"] for m in todo]
    from ami_report import ami_report
    from evaluate import TO_ME
    ami_report(cfg, [m["scenario"] for m in todo], results, list(TO_ME))
    return [m["scenario"] for m in todo]


if __name__ == "__main__":
    main()
