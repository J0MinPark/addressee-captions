"""AI Hub 라벨 시뮬레이션 dev 평가와 시연 구성 선택(사전 등록 개정 1). 시험(test)은 읽지 않는다.

    source scripts/server_env.sh
    python tools/aihub_tune.py [--latency-csv results/latency_..._remote_*.csv]

- 재시뮬레이션·학습된 융합·교차검증은 tools/tune_dev.py 함수를 그대로 쓴다(새 정책 로직 없음).
- 후보: 판정기 P1c/P1/P2/P3 × 융합 hand/learned(겹 밖 교차검증) × 플래그 4 + timing, timing_speaker. 기준선 all.
- 규칙: 제약 1 자연 함정 오표시율 < timing, 제약 2 원격 자막 지연 p95 ≤ 2초 → F0.5 최대 → 동률(소수 셋째 자리)이면 단순한 쪽.
  만족하는 후보가 없으면 timing.
- 지연(제약 2): --latency-csv(Book5 latency_bench --remote, "최종" 지연)가 있으면 그 값. 없으면 서버 localhost 측정
  (results/latency_*_local_*.csv 최신, v1)으로 잠정. 판정기가 다르면 v1 p95 − P1c 판정 p95 + 그 판정기 판정 p95로 추정(잠정 표시).
출력: results/aihub_sim_dev_results.md, results/demo_selection.md, results/demo_selection.json (수치만)
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

os.environ["HEARME_NO_SELECTED"] = "1"   # dev 평가는 배포 구성과 무관한 기본값(v1) 위에서

ROOT = Path(__file__).resolve().parent.parent
# 출력 위치(가짜 데이터 점검 때 저장소를 더럽히지 않게 바꿀 수 있다)
RES = Path(os.environ.get("AIHUB_RESULTS_DIR") or ROOT / "results")
SPLITS = Path(os.environ.get("AIHUB_SPLITS") or ROOT / "splits_aihub.json")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402

from app.config import load_config  # noqa: E402
from app.policy import FUSION_FEATURES  # noqa: E402
from evaluate import load_labels, load_segments  # noqa: E402
from stats_boot import bootstrap, fmt, paired  # noqa: E402
from tune_dev import FLAGS, learned_fusion, run_config, run_learned_cv, with_policy  # noqa: E402

MODEL = "qwen3:4b"
LAT_MAX_MS = 2000.0


def base_cfg():
    return load_config("server", overrides={"namecall": {"enabled": False}, "sound": {"enabled": False}})


def load_sim(split: str, _final_test_ok: bool = False) -> tuple[dict, dict]:
    if split == "test" and not _final_test_ok:
        raise SystemExit("[aihub] 시험 세트는 tools/final_test_aihub_sim.py 에서만")
    d = Path(os.environ["HEARME_DATA"]) / "aihub" / "sim" / split
    names = sorted(p.name[: -len(".segments.jsonl")] for p in d.glob("aihubsim_*.segments.jsonl"))
    if not names or any(not n.endswith(f"_{split}") for n in names):
        raise SystemExit(f"[aihub] {split} 시나리오 없음/섞임: {d}")
    data = {n: (load_segments(d / f"{n}.segments.jsonl")[1], load_labels(d / f"{n}.labels.csv", "single")) for n in names}
    judges = {f"P1c-{MODEL}": None}
    for p in sorted((d / "llm_v2").glob("*.json")):
        model, var = p.stem.split("__")
        judges[f"{var}-{model.replace('-', ':', 1)}"] = json.loads(p.read_text(encoding="utf-8"))
    return data, judges


def judge_lat_p95(data, judge) -> float | None:
    if judge is None:
        v = [s["llm"]["latency_ms"] for segs, _ in data.values() for s in segs if s.get("llm") and s["llm"].get("latency_ms")]
    else:
        v = [r["latency_ms"] for sc in judge.values() for r in sc.values() if r and r.get("latency_ms")]
    return float(np.percentile(v, 95)) if v else None


def latency_source(path: str | None) -> dict:
    """제약 2용 지연(최종 p95, ms)."""
    if path:
        rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
        v = [float(r["lat_final_ms"]) for r in rows if r["role"] != "wearer" and r.get("lat_final_ms")]
        return {"p95_ms": float(np.percentile(v, 95)), "src": f"원격(Book5) {Path(path).name}", "provisional": False}
    loc = sorted(glob.glob(str(RES / "latency_*_local_*.csv")))
    if not loc:
        return {"p95_ms": None, "src": "측정 없음", "provisional": True}
    rows = list(csv.DictReader(open(loc[-1], encoding="utf-8-sig")))
    v = [float(r["lat_final_ms"]) for r in rows if r["role"] != "wearer" and r.get("lat_final_ms")]
    f = [float(r["lat_first_ms"]) for r in rows if r["role"] != "wearer" and r.get("lat_first_ms")]
    return {"p95_ms": float(np.percentile(v, 95)), "p95_first_ms": float(np.percentile(f, 95)),
            "src": f"서버 localhost(잠정) {Path(loc[-1]).name}", "provisional": True}


def disp(k: str) -> str:
    """표 안에서 열 구분자와 겹치지 않게."""
    return k.replace(" | ", " · ")


def simple_key(k: str) -> tuple:
    """동률일 때 단순한 쪽: timing 계열 < hand < learned, 플래그 적은 쪽."""
    if "|" not in k:
        return (0, 0, 0)
    _, fu, fl = k.split(" | ")
    return (1, 0 if fu == "hand" else 1, 0 if fl == "none" else (2 if "+" in fl else 1))


def evaluate_dev(lat_csv: str | None = None) -> dict:
    cfg = base_cfg()
    data, judges = load_sim("dev")
    print(f"[aihub] dev 시나리오 {len(data)}개 · 판정기 {list(judges)}", flush=True)
    runs = {"all": run_config(cfg, data, None, mode="all"),
            "timing": run_config(cfg, data, None, mode="timing"),
            "timing_speaker": run_config(cfg, data, None, mode="timing_speaker")}
    fusions, insample = {}, {}
    for jn, judge in judges.items():
        fusions[jn] = learned_fusion(cfg, data, judge)
        for fl, pol in FLAGS.items():
            runs[f"{jn} | hand | {fl}"] = run_config(with_policy(cfg, **pol), data, judge)
            runs[f"{jn} | learned | {fl}"] = run_learned_cv(cfg, data, judge, fusions[jn][0]["C"], pol)
            insample[f"{jn} | learned | {fl}"] = run_config(with_policy(cfg, fusion=fusions[jn][0], **pol), data, judge)
        print(f"  {jn}: hand {runs[f'{jn} | hand | none']['pooled']['f05']:.3f} · learned(CV) "
              f"{runs[f'{jn} | learned | none']['pooled']['f05']:.3f}", flush=True)

    lat = latency_source(lat_csv)
    jl = {jn: judge_lat_p95(data, j) for jn, j in judges.items()}
    def lat_of(k):
        if "|" not in k:      # LLM 없는 모드: 처음 표시 = 최종
            return lat.get("p95_first_ms", lat["p95_ms"])
        jn = k.split(" | ")[0]
        if lat["p95_ms"] is None:
            return None
        if lat["provisional"] and jn != f"P1c-{MODEL}" and jl.get(jn) and jl.get(f"P1c-{MODEL}"):
            return lat["p95_ms"] - jl[f"P1c-{MODEL}"] + jl[jn]
        return lat["p95_ms"]
    t_trap = runs["timing"]["pooled"]["trap_shown_rate"]
    cands = [k for k in runs if k != "all"]
    rows = {}
    for k in cands:
        r = runs[k]["pooled"]
        c1 = r["trap_shown_rate"] is not None and t_trap is not None and r["trap_shown_rate"] < t_trap
        lk = lat_of(k)
        c2 = lk is not None and lk <= LAT_MAX_MS
        rows[k] = {"f05": r["f05"], "c1": c1, "c2": c2, "lat": lk}
    ok = [k for k in cands if rows[k]["c1"] and rows[k]["c2"]]
    if ok:
        best = max(ok, key=lambda k: (round(rows[k]["f05"], 3), tuple(-x for x in simple_key(k))))
        fallback = False
    else:
        best, fallback = "timing", True
    return {"cfg": cfg, "data": data, "judges": judges, "runs": runs, "insample": insample, "fusions": fusions,
            "rows": rows, "best": best, "fallback": fallback, "lat": lat, "judge_lat": jl, "t_trap": t_trap}


def selection_of(R: dict) -> dict:
    best = R["best"]
    if "|" not in best:
        return {"config": best, "mode": best, "variant": "P1c", "model": MODEL, "fusion": {"type": "hand"}, "flags": {},
                "config_name": f"{best}(LLM 미사용)"}
    jn, fu, fl = best.split(" | ")
    var = jn.split("-", 1)[0]
    fusion = {"type": "hand"}
    if fu == "learned":   # 배포 가중치: dev 전체로 다시 학습(플래그가 있으면 그 플래그로 임계값을 맞춘다)
        fusion = learned_fusion(R["cfg"], R["data"], R["judges"][jn], C=R["fusions"][jn][0]["C"], flags=FLAGS[fl])[0]
    return {"config": best, "mode": "full", "variant": var, "model": MODEL, "fusion": fusion, "flags": FLAGS[fl],
            "config_name": f"{var}-{MODEL}-{fu}" + "".join(f"+{x}" for x in fl.split("+") if x != "none")}


def report(R: dict) -> dict:
    runs, rows, best = R["runs"], R["rows"], R["best"]
    v1 = f"P1c-{MODEL} | hand | none"
    def line(k, src=runs):
        r, b = src[k]["pooled"], bootstrap(src[k]["units"])
        rr = rows.get(k, {})
        lat = "–" if rr.get("lat") is None else f"{rr['lat'] / 1000:.2f}s"
        mr = "–" if r["misreg_rate"] is None else f"{r['misreg_rate'] * 100:.1f}%"
        c1 = "✓" if rr.get("c1") else ("✗" if rr else "")
        c2 = "✓" if rr.get("c2") else ("✗" if rr else "")
        return (f"| {disp(k)} | {fmt(b['f05'], False)} | {fmt(b['precision'])} | {fmt(b['recall'])} | "
                f"{r['contamination'] * 100:.1f}% | {fmt(b['trap_shown_rate'])} | {mr} ({r['misreg']}) | "
                f"{lat} | {c1} | {c2} |")
    hdr = ["| 구성 (판정기 · 융합 · 플래그) | F0.5 [95% CI] | 정밀도 | 재현율 | 자막 오염도 | 자연 함정 오표시율 | "
           "옆 대화 화자 오등록률 | 지연 p95 | 제약1 | 제약2 |", "|---|---:|---:|---:|---:|---:|---:|---:|:-:|:-:|"]
    data = R["data"]
    n_units = len(runs["timing"]["units"])
    pr = runs["timing"]["pooled"]
    md = ["# AI Hub 라벨 시뮬레이션 dev 결과 (사전 등록 개정 1)", "",
          "**AI Hub 주요 영역별 회의 음성인식 데이터셋 활용** · 라벨 기반 대화 시뮬레이션(원천 음성 미사용: ASR·화자 임베딩·분할 오류 없음, 판정 계층만)", "",
          f"dev 시나리오 {len(data)}개 · (세션, 착용자) {n_units}단위 · 양성(x) {pr['n_pos']} · 음성(y) {pr['n_neg']} · "
          f"부트스트랩 1000회(세션 단위) · 판정 모델 {MODEL} · 한국어 프롬프트", "",
          f"제약 2 지연 출처: {R['lat']['src']}" + (" — **잠정**" if R["lat"]["provisional"] else ""), "",
          "## 기준", ""] + hdr + [line_all(runs["all"]), line("timing"), line("timing_speaker"), line(v1)]
    md += ["", "## 판정기 × 융합 (플래그 없음)", ""] + hdr
    for jn in R["judges"]:
        md += [line(f"{jn} | hand | none"), line(f"{jn} | learned | none")]
    md += ["", "learned 행은 (세션, 착용자) 단위 6겹 교차검증 겹 밖 추정. in-sample(낙관적):", "",
           "| 구성 | F0.5 |", "|---|---:|"] + [f"| {disp(k)} | {v['pooled']['f05']:.3f} |" for k, v in R["insample"].items() if k.endswith("| none")]
    md += ["", "## 플래그 효과 (F0.5)", "", "| 판정기 · 융합 | none | rejudge | shortskip | rejudge+shortskip |", "|---|---:|---:|---:|---:|"]
    for jn in R["judges"]:
        for fu in ("hand", "learned"):
            md.append(f"| {jn} · {fu} | " + " | ".join(f"{runs[f'{jn} | {fu} | {fl}']['pooled']['f05']:.3f}" for fl in FLAGS) + " |")
    md += ["", "## 학습된 융합 계수 (dev 전체, 플래그 없음)", "",
           "| 판정기 | C | 임계값 | 절편 | " + " | ".join(FUSION_FEATURES) + " | 표본(양성) |",
           "|---|---:|---:|---:|" + "---:|" * len(FUSION_FEATURES) + "---:|"]
    for jn, (f, info) in R["fusions"].items():
        md.append(f"| {jn} | {f['C']} | {f['threshold']} | {f['intercept']:+.2f} | " +
                  " | ".join(f"{f['coef'][k]:+.2f}" for k in FUSION_FEATURES) + f" | {info['n']} ({info['pos']}) |")
    md += ["", "판정기 판정 지연 p95(ms): " + ", ".join(f"{k} {v:.0f}" for k, v in R["judge_lat"].items() if v)]
    d_t = paired(runs[best]["units"], runs["timing"]["units"], "f05") if best != "timing" else None
    d_v = paired(runs[best]["units"], runs[v1]["units"], "f05") if best != v1 else None
    tr_t = paired(runs[best]["units"], runs["timing"]["units"], "trap_shown_rate") if best != "timing" else None
    md += ["", "## 핵심 비교 (짝지은 부트스트랩, 같은 재표본)", ""]
    if d_t:
        md.append(f"- 선택 − timing · F0.5: {d_t[0]:+.3f} [{d_t[1]:+.3f}, {d_t[2]:+.3f}] (선택이 더 큰 재표본 {d_t[3]:.0%})")
        md.append(f"- 선택 − timing · 자연 함정 오표시율: {tr_t[0] * 100:+.1f}%p [{tr_t[1] * 100:+.1f}, {tr_t[2] * 100:+.1f}]")
    if d_v:
        md.append(f"- 선택 − v1 · F0.5: {d_v[0]:+.3f} [{d_v[1]:+.3f}, {d_v[2]:+.3f}] ({d_v[3]:.0%})")
    (RES / "aihub_sim_dev_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    sel = selection_of(R)
    r = runs[best]["pooled"]
    sel.update(dev_f05=r["f05"], dev_precision=r["precision"], dev_recall=r["recall"],
               dev_trap_shown_rate=r["trap_shown_rate"], timing_trap_shown_rate=R["t_trap"],
               latency_p95_ms=rows[best]["lat"], latency_source=R["lat"]["src"], provisional=R["lat"]["provisional"],
               fallback_timing=R["fallback"], rule="results/preregistration.md 개정 1")
    if sel["provisional"]:
        sel["config_name"] += " (잠정)"
    (RES / "demo_selection.json").write_text(json.dumps(sel, ensure_ascii=False, indent=1), encoding="utf-8")
    top = sorted([k for k in rows], key=lambda k: -rows[k]["f05"])[:8]
    lat_s = "–" if rows[best]["lat"] is None else f"{rows[best]['lat'] / 1000:.2f}초"
    smd = ["# 시연 구성 선택 (AI Hub 라벨 시뮬레이션 dev, 사전 등록 개정 1)", "",
           "**규칙**: 제약 1(자연 함정 오표시율 < timing) · 제약 2(원격 자막 지연 p95 ≤ 2초)를 만족하는 후보 중 dev F0.5(정답 = x) 최대. "
           "동률이면 단순한 쪽. 없으면 timing.", "",
           f"**선택: `{sel['config_name']}`** — `{disp(best)}`" + (" (제약을 만족하는 후보가 없어 timing으로 대체 → 런북에서 함정 장면 제외)" if R["fallback"] else ""), "",
           f"- dev F0.5 {r['f05']:.3f} · 정밀도 {r['precision']:.1%} · 재현율 {r['recall']:.1%} · 자연 함정 오표시율 "
           f"{(r['trap_shown_rate'] or 0):.1%} (timing {(R['t_trap'] or 0):.1%})",
           f"- 지연 p95 {lat_s} · 출처 {R['lat']['src']}"
           + (" → **잠정**: Book5 원격 측정(`latency_bench.py --remote`) 후 `--latency-csv`로 다시 돌리면 확정" if R["lat"]["provisional"] else ""),
           ]
    if d_t:
        smd.append(f"- timing 대비 F0.5 {d_t[0]:+.3f} [{d_t[1]:+.3f}, {d_t[2]:+.3f}], 자연 함정 오표시율 "
                   f"{tr_t[0] * 100:+.1f}%p [{tr_t[1] * 100:+.1f}, {tr_t[2] * 100:+.1f}]")
    smd += ["", "F0.5 상위 후보와 제약:", "", "| 후보 | F0.5 | 제약1 | 제약2 |", "|---|---:|:-:|:-:|"] + \
           [f"| {disp(k)} | {rows[k]['f05']:.3f} | {'✓' if rows[k]['c1'] else '✗'} | {'✓' if rows[k]['c2'] else '✗'} |" for k in top] + \
           ["", "전체 표: `results/aihub_sim_dev_results.md`. 적용: `python tools/apply_selection.py` → `app/demo_config.yaml`.",
            "한계: 라벨 기반 시뮬레이션이라 받아쓰기·화자 임베딩·분할·소음 오류가 없다. 실제 시연 성능은 더 낮을 수 있다."]
    (RES / "demo_selection.md").write_text("\n".join(smd) + "\n", encoding="utf-8")
    print("\n".join(smd))
    return sel


def line_all(run):
    r, b = run["pooled"], bootstrap(run["units"])
    mr = "–" if r["misreg_rate"] is None else f"{r['misreg_rate'] * 100:.1f}%"
    return (f"| all (전부 표시) | {fmt(b['f05'], False)} | {fmt(b['precision'])} | {fmt(b['recall'])} | "
            f"{r['contamination'] * 100:.1f}% | {fmt(b['trap_shown_rate'])} | {mr} ({r['misreg']}) | – | | |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency-csv", default=None, help="Book5 latency_bench --remote 결과 CSV")
    a = ap.parse_args()
    report(evaluate_dev(a.latency_csv))


if __name__ == "__main__":
    main()
