"""qwen3:8b 를 dev 전 조건으로 확장할지 판정(결과를 보기 전에 고정한 기준).

    python tools/compare_8b.py

비교 범위: dev 세트 clean 조건의 착용자 직후 구간 중 라벨 y/n(single 정의) — 4b·8b가 같은 구간을 판정.
지표: 판정기 단독 AUC. 각 모델에서 변형 P1/P2/P3 중 AUC가 가장 높은 것을 그 모델의 대표로 쓴다.
기준("확실히 낫다"): AUC(8b 최고) − AUC(4b 최고) ≥ +0.02 이고,
                   (회의, 착용자) 단위 짝지은 부트스트랩 1000회의 차이 95% 신뢰구간 하한 > 0.
통과하면 8b 최고 변형만 dev snr10·snr5 로 확장해 최종 후보에 넣는다. 아니면 8b 는 후보에서 빠진다.
출력: results/compare_8b.md, 종료 코드 0 = 확장, 1 = 확장 안 함
"""
from __future__ import annotations

import json
import sys
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
from judge_offline import auc, dev_scenarios, items_of  # noqa: E402
from splits import guard  # noqa: E402
from stats_boot import unit_of  # noqa: E402

MIN_GAIN = 0.02
VARIANTS = ("P1", "P2", "P3")


def main():
    results = resolve_path(load_config("ami"), "results_dir")
    names = [n for n in dev_scenarios(results, "dev") if n.split("_")[3] == "clean"]
    guard(names, "tune")
    items = [it for it in items_of(results, names) if it["label"] in ("y", "n")]
    y = np.array([1 if it["label"] == "y" else 0 for it in items])
    units = np.array([hash(unit_of(it["scenario"])) for it in items])
    scores = {}
    for model, sub in (("qwen3:4b", "dev"), ("qwen3:8b", "dev_clean8b")):
        for v in VARIANTS:
            p = results / "llm_v2" / sub / f"{model.replace(':', '-')}__{v}.json"
            if not p.exists():
                continue
            res = json.loads(p.read_text(encoding="utf-8"))
            scores[(model, v)] = np.array([((res.get(it["scenario"], {}).get(it["seg_id"])) or {}).get("prob", 0.5)
                                           for it in items])
    aucs = {k: auc(s, y) for k, s in scores.items()}
    b4 = max((k for k in aucs if k[0] == "qwen3:4b"), key=lambda k: aucs[k])
    b8 = max((k for k in aucs if k[0] == "qwen3:8b"), key=lambda k: aucs[k])
    d0 = aucs[b8] - aucs[b4]
    # (회의, 착용자) 단위 짝지은 부트스트랩
    uk = np.unique(units)
    rng = np.random.default_rng(0)
    ds = []
    for _ in range(1000):
        pick = rng.choice(uk, len(uk))
        idx = np.concatenate([np.where(units == u)[0] for u in pick])
        a8, a4 = auc(scores[b8][idx], y[idx]), auc(scores[b4][idx], y[idx])
        if a8 is not None and a4 is not None:
            ds.append(a8 - a4)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    ok = d0 >= MIN_GAIN and lo > 0
    md = ["# qwen3:8b 확장 판정 (dev clean)", "",
          f"구간 {len(items)}개(y {int(y.sum())} / n {int(len(y) - y.sum())}) · (회의, 착용자) {len(uk)}단위", "",
          "| 모델 · 변형 | AUC |", "|---|---:|"] + [f"| {m} · {v} | {aucs[(m, v)]:.3f} |" for (m, v) in sorted(aucs)] + [
          "", f"4b 최고 {b4[1]} {aucs[b4]:.3f} · 8b 최고 {b8[1]} {aucs[b8]:.3f}",
          f"차이 {d0:+.3f} [95% CI {lo:+.3f}, {hi:+.3f}] · 기준: ≥ +{MIN_GAIN} 이고 하한 > 0",
          f"**판정: {'확장 — 8b ' + b8[1] + ' 을 dev 전 조건으로' if ok else '확장 안 함 — 8b 는 최종 후보에서 제외'}**"]
    (results / "compare_8b.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (results / "compare_8b.json").write_text(json.dumps({"extend": bool(ok), "variant": b8[1], "diff": d0,
                                                         "ci": [lo, hi]}), encoding="utf-8")
    print("\n".join(md))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
