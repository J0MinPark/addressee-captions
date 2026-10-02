"""개발 세트 튜닝·선택(시험 세트는 읽지 않음 — splits.guard 'tune').

    python tools/tune_dev.py

1. 판정기: v1(P1c, 재생 기록) + tools/judge_offline.py 결과(results/llm_v2/dev/*.json)
2. 융합: hand(현재 손 가중치, 그대로) vs learned(L2 로지스틱 회귀, 착용자 단위 교차검증으로 C 선택,
   결정 임계값은 dev F0.5 최대)
3. 플래그: --candidate-rejudge, --short-skip-llm (끔/켬)
4. 모든 구성: dev 3개 조건 합산 지표 + (회의, 착용자) 부트스트랩 95% CI, timing 대비 짝지은 차이
5. 선택 규칙(사전 고정): dev 합산 F0.5 점추정 최대. → results/selection.md, selection.json
"""
from __future__ import annotations

import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402

from app.config import load_config, resolve_path  # noqa: E402
from app.policy import FUSION_FEATURES, fusion_features, is_question  # noqa: E402
from evaluate import load_labels, load_segments, metrics, simulate, summarize  # noqa: E402
from splits import guard, meetings  # noqa: E402
from stats_boot import bootstrap, fmt, paired, unit_of  # noqa: E402

FLAGS = {"none": {}, "rejudge": {"candidate_rejudge": True}, "shortskip": {"short_skip_llm": True},
         "rejudge+shortskip": {"candidate_rejudge": True, "short_skip_llm": True}}
CS = [0.01, 0.1, 1.0, 10.0, 100.0]
THRESHOLDS = [round(x, 2) for x in np.arange(0.30, 0.86, 0.05)]


# ---------------------------------------------------------------- 데이터
def load_split(results: Path, split: str, _final_test_ok: bool = False) -> dict:
    ms = set(meetings(split))
    names = sorted(p.name[: -len(".segments.jsonl")] for p in results.glob("ami_*.segments.jsonl")
                   if p.name.split("_")[1] in ms and "_take2" in p.name)
    guard(names, "tune" if split == "dev" else "test")
    if split == "test" and not _final_test_ok:
        raise SystemExit("[splits] 시험 세트는 tools/final_test.py 에서만")
    return {n: (load_segments(results / f"{n}.segments.jsonl")[1], load_labels(results / f"{n}.labels.csv", "single"))
            for n in names if (results / f"{n}.labels.csv").exists()}


def load_judges(results: Path, split: str) -> dict:
    """{판정기 이름: None(재생 기록 v1) | {시나리오: {seg_id: 결과}}}"""
    out = {"P1c-qwen3:4b": None}
    for p in sorted((results / "llm_v2" / split).glob("*.json")):
        model, var = p.stem.split("__")
        out[f"{var}-{model.replace('-', ':', 1)}"] = json.loads(p.read_text(encoding="utf-8"))
    return out


def override_for(judge, name):
    return None if judge is None else judge.get(name, {})


# ---------------------------------------------------------------- 평가
def run_config(cfg, data, judge, mode="full") -> dict:
    """→ {"units": {(회의,착용자): [metrics...]}, "pooled": summarize, "per_cond": {조건: summarize}}"""
    units, per_cond = defaultdict(list), defaultdict(list)
    for n, (segs, labels) in data.items():
        m = metrics(segs, labels, simulate(cfg, segs, mode, override_for(judge, n)))
        units[unit_of(n)].append(m)
        per_cond[n.split("_")[3]].append(m)
    return {"units": dict(units), "pooled": summarize([m for v in units.values() for m in v]),
            "per_cond": {c: summarize(v) for c, v in per_cond.items()}}


def with_policy(cfg, **pol):
    c = copy.deepcopy(cfg)
    c["policy"].update(pol)
    return c


# ---------------------------------------------------------------- 학습된 융합
def T_of(gap, p):
    if gap is None:
        return 0.0
    if 0 <= gap <= p["timing_full_max_s"]:
        return 1.0
    if p["timing_early_s"] <= gap < 0 or p["timing_full_max_s"] < gap <= p["timing_half_max_s"]:
        return 0.5
    return 0.0


def training_rows(cfg, data, judge):
    X, y, g = [], [], []
    for n, (segs, labels) in data.items():
        sim = simulate(cfg, segs, "full", override_for(judge, n))
        wtext = {s["seg_id"]: s.get("text") for s in segs if s.get("is_wearer")}
        ov = override_for(judge, n)
        for s in segs:
            if s.get("is_wearer") or s.get("skip") or not s.get("text"):
                continue
            lab = labels.get(s["seg_id"])
            if lab not in ("y", "n"):
                continue
            L = None
            if s.get("llm_eligible"):
                r = ov.get(s["seg_id"]) if ov is not None else s.get("llm")
                L = r.get("prob") if r else None
            f = fusion_features(T_of(s.get("gap"), cfg["policy"]), s.get("gap"), s.get("sim", 0.0),
                                sim["pre_state"].get(s["seg_id"]) == "partner", L, s["t_end"] - s["t_start"],
                                is_question(wtext.get(s.get("wearer_seg"))))
            X.append([f[k] for k in FUSION_FEATURES])
            y.append(1 if lab == "y" else 0)
            g.append(unit_of(n))
    return np.array(X, float), np.array(y, float), g


def fit_logreg(X, y, C, iters=50):
    """L2 로지스틱 회귀(절편은 벌점 없음), Newton/IRLS."""
    Xb = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(Xb.shape[1])
    lam = np.ones(Xb.shape[1]) / C
    lam[0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        gr = Xb.T @ (p - y) + lam * w
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb + np.diag(lam) + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(H, gr)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return w


def logloss(w, X, y):
    p = np.clip(1 / (1 + np.exp(-np.hstack([np.ones((len(X), 1)), X]) @ w)), 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def cv_choose_C(X, y, groups, k=6, seed=0):
    units = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    fold = {u: i % k for i, u in enumerate(units)}
    fid = np.array([fold[u] for u in groups])
    scores = {}
    for C in CS:
        ll = []
        for f in range(k):
            tr, te = fid != f, fid == f
            if te.sum() == 0 or len(set(y[tr])) < 2:
                continue
            ll.append(logloss(fit_logreg(X[tr], y[tr], C), X[te], y[te]))
        scores[C] = float(np.mean(ll))
    return min(scores, key=scores.get), scores


def learned_fusion(cfg, data, judge):
    X, y, g = training_rows(cfg, data, judge)
    C, cv = cv_choose_C(X, y, g)
    w = fit_logreg(X, y, C)
    fusion = {"type": "logistic", "C": C, "intercept": float(w[0]),
              "coef": {k: float(v) for k, v in zip(FUSION_FEATURES, w[1:])}, "threshold": 0.5}
    # 결정 임계값: dev F0.5 최대 (정책 시뮬레이션으로)
    best = None
    for t in THRESHOLDS:
        f = dict(fusion, threshold=t)
        r = run_config(with_policy(cfg, fusion=f), data, judge)["pooled"]
        if best is None or r["f05"] > best[1]:
            best = (t, r["f05"])
    fusion["threshold"] = best[0]
    return fusion, {"n": len(y), "pos": int(y.sum()), "cv_logloss": cv, "thr_f05": best[1]}


# ---------------------------------------------------------------- 메인
def main():
    cfg = load_config("ami")
    cal = resolve_path(cfg, "results_dir") / "ami_calibration.json"
    if cal.exists():
        cfg["ownvoice"]["own_margin_db"] = round(json.loads(cal.read_text(encoding="utf-8"))["own_margin_db"], 1)
    results = resolve_path(cfg, "results_dir")
    data = load_split(results, "dev")
    judges = load_judges(results, "dev")
    print(f"[tune] dev 시나리오 {len(data)}개 · 판정기 {list(judges)}")

    runs = {}
    runs["timing"] = run_config(cfg, data, None, mode="timing")
    runs["timing_speaker"] = run_config(cfg, data, None, mode="timing_speaker")
    fusions = {}
    for jn, judge in judges.items():
        fusions[jn] = learned_fusion(cfg, data, judge)
        print(f"  learned fusion {jn}: C={fusions[jn][0]['C']} thr={fusions[jn][0]['threshold']} "
              f"(dev F0.5 {fusions[jn][1]['thr_f05']:.3f})", flush=True)
        for fl, pol in FLAGS.items():
            runs[f"{jn} | hand | {fl}"] = run_config(with_policy(cfg, **pol), data, judge)
            runs[f"{jn} | learned | {fl}"] = run_config(with_policy(cfg, fusion=fusions[jn][0], **pol), data, judge)
        print(f"  {jn}: hand {runs[f'{jn} | hand | none']['pooled']['f05']:.3f} · "
              f"learned {runs[f'{jn} | learned | none']['pooled']['f05']:.3f}", flush=True)

    # 선택(사전 고정 규칙): dev 합산 F0.5 점추정 최대, 동률이면 단순한 쪽(hand < learned, 플래그 적은 쪽)
    cands = [k for k in runs if "|" in k]
    simple = lambda k: (k.split(" | ")[1] == "learned", k.split(" | ")[2] != "none", "rejudge+shortskip" in k)  # noqa: E731
    best = max(cands, key=lambda k: (round(runs[k]["pooled"]["f05"], 4), tuple(not x for x in simple(k))))
    base = "P1c-qwen3:4b | hand | none"

    # ---- 보고서
    def row(k):
        b = bootstrap(runs[k]["units"])
        return (f"| {k} | {fmt(b['f05'], False)} | {fmt(b['precision'])} | {fmt(b['recall'])} | "
                f"{fmt(b['trap_shown_rate'])} | {fmt(b['trapspk_rate'])} |")
    hdr = ["| 구성 (판정기 · 융합 · 플래그) | F0.5 [95% CI] | 정밀도 | 재현율 | 자연 함정 오표시율 | 함정 화자 오등록률 |",
           "|---|---:|---:|---:|---:|---:|"]
    md = ["# 개발 세트 결과 (AMI dev, single 정의, clean+snr10+snr5 합산)", "",
          f"dev 회의 {meetings('dev')} · 시나리오 {len(data)}개 · (회의, 착용자) {len(runs['timing']['units'])}단위 · "
          f"부트스트랩 1000회", "", "## 기준", ""] + hdr + [row("timing"), row("timing_speaker"), row(base)]
    md += ["", "## 판정기 × 융합 (플래그 없음)", ""] + hdr
    for jn in judges:
        md += [row(f"{jn} | hand | none"), row(f"{jn} | learned | none")]
    md += ["", "## 플래그 효과 (각 판정기·융합 조합)", "",
           "| 판정기 · 융합 | none | rejudge | shortskip | rejudge+shortskip |", "|---|---:|---:|---:|---:|"]
    for jn in judges:
        for fu in ("hand", "learned"):
            md.append(f"| {jn} · {fu} | " + " | ".join(f"{runs[f'{jn} | {fu} | {fl}']['pooled']['f05']:.3f}" for fl in FLAGS) + " |")
    md += ["", "## 학습된 융합 계수 (L2 로지스틱, 특징은 policy.fusion_features)", "",
           "| 판정기 | C | 임계값 | 절편 | " + " | ".join(FUSION_FEATURES) + " | 학습 표본(양성) |",
           "|---|---:|---:|---:|" + "---:|" * len(FUSION_FEATURES) + "---:|"]
    for jn, (f, info) in fusions.items():
        md.append(f"| {jn} | {f['C']} | {f['threshold']} | {f['intercept']:+.2f} | " +
                  " | ".join(f"{f['coef'][k]:+.2f}" for k in FUSION_FEATURES) + f" | {info['n']} ({info['pos']}) |")
    md += ["", "손 가중치(hand, 변경 없음): z = −1.0 + 1.5·T + 2.5·S + 2.0·(2L−1), 임계 0.6", "",
           "## 조건별 F0.5 (선택 구성 vs timing vs v1)", "", "| 구성 | clean | snr10 | snr5 |", "|---|---:|---:|---:|"]
    for k in ("timing", base, best):
        md.append(f"| {k} | " + " | ".join(f"{runs[k]['per_cond'].get(c, {}).get('f05', 0):.3f}" for c in ("clean", "snr10", "snr5")) + " |")
    d = paired(runs[best]["units"], runs["timing"]["units"], "f05")
    d2 = paired(runs[best]["units"], runs[base]["units"], "f05")
    md += ["", "## 짝지은 부트스트랩 (F0.5 차이)", "",
           f"- 선택 구성 − timing: {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] (선택 구성이 더 높은 재표본 {d[3]:.0%})",
           f"- 선택 구성 − v1(baseline 구성): {d2[0]:+.3f} [{d2[1]:+.3f}, {d2[2]:+.3f}] ({d2[3]:.0%})"]
    (results / "dev_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    jn, fu, fl = best.split(" | ")
    var, model = jn.split("-", 1)
    sel = {"config": best, "variant": var, "model": model, "fusion": fusions[jn][0] if fu == "learned" else {"type": "hand"},
           "flags": FLAGS[fl], "dev_f05": runs[best]["pooled"]["f05"],
           "config_name": f"{var}-{model}-{fu}" + "".join(f"+{x}" for x in fl.split("+") if x != "none")}
    (results / "selection.json").write_text(json.dumps(sel, ensure_ascii=False, indent=1), encoding="utf-8")
    top = sorted(cands, key=lambda k: -runs[k]["pooled"]["f05"])[:5]
    smd = ["# 최종 구성 선택 (개발 세트만 사용)", "",
           "**규칙(사전 고정)**: dev(3개 조건 합산) single 정의 F0.5 점추정 최대. 동률이면 단순한 구성(손 가중치, 플래그 없음) 우선.", "",
           f"**선택: `{sel['config_name']}`** — {best}", "",
           f"- dev F0.5 = {runs[best]['pooled']['f05']:.3f} (timing {runs['timing']['pooled']['f05']:.3f}, "
           f"v1 구성 {runs[base]['pooled']['f05']:.3f})",
           f"- 정밀도 {runs[best]['pooled']['precision']:.1%}, 재현율 {runs[best]['pooled']['recall']:.1%}, "
           f"자연 함정 오표시율 {(runs[best]['pooled']['trap_shown_rate'] or 0):.1%}",
           f"- timing 대비 짝지은 차이 {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]", "",
           "상위 5개:", ""] + [f"{i + 1}. {k} — F0.5 {runs[k]['pooled']['f05']:.3f}" for i, k in enumerate(top)] + [
           "", "판정기 단독 비교는 `results/judge_dev.md`, 전체 표는 `results/dev_results.md`."]
    (results / "selection.md").write_text("\n".join(smd) + "\n", encoding="utf-8")
    print("\n".join(smd))


if __name__ == "__main__":
    main()
