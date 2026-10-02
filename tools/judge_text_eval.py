"""한국어 문장 쌍으로 판정기만 평가(사람이 데이터를 채운다).

    python tools/judge_text_eval.py data/judge_pairs_ko.csv              # config 의 판정기 변형(시연 구성)
    python tools/judge_text_eval.py data/judge_pairs_ko.csv --variant P2 --model qwen3:4b

CSV 열: a(착용자 말), b(직후 상대 말), label(y=착용자에게 한 반응 / n=아님, 1/0도 됨), type(자유 분류, 예: 질문-대답, 제3자 질문)
형식 예시: data/judge_pairs_ko.example.csv
출력: 정확도, 혼동행렬, 유형별 정확도 (+ 연속 점수 변형이면 AUC)
"""
from __future__ import annotations

import argparse
import csv
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


def main():
    from app.config import load_config
    from app.llm_judge import LLMJudge
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--profile", default=None)
    a = ap.parse_args()
    over = {"llm": {"prompt_lang": "ko", "cache": False}}
    if a.variant:
        over["llm"]["variant"] = a.variant
    cfg = load_config(a.profile, overrides=over)
    if a.model:
        cfg["llm"]["models"] = [a.model]
    j = LLMJudge(cfg, log=lambda *x: None)
    if not j.setup():
        raise SystemExit(f"LLM 사용 불가: {j.health_line()}")
    rows = list(csv.DictReader(open(a.csv, encoding="utf-8-sig")))
    conf = defaultdict(int)
    by_type = defaultdict(lambda: [0, 0])
    scores, ys = [], []
    for r in rows:
        y = str(r["label"]).strip().lower() in ("y", "1", "true", "yes")
        res = j.judge([], r["a"].strip(), r["b"].strip())
        pred = bool(res and res["pair"])
        conf[(y, "none" if res is None else pred)] += 1
        t = r.get("type", "").strip() or "(없음)"
        by_type[t][0] += pred == y and res is not None
        by_type[t][1] += 1
        if res is not None:
            scores.append(res["prob"])
            ys.append(int(y))
    n = len(rows)
    acc = sum(v[0] for v in by_type.values()) / max(n, 1)
    print(f"판정기 {j.model} · 변형 {j.variant} · 문장 쌍 {n}개 · 정확도 {acc:.1%}")
    print("\n혼동행렬 (행 = 정답, 열 = 판정)")
    print(f"{'':12}{'짝':>8}{'짝 아님':>10}{'실패':>8}")
    for y, name in ((True, "y(반응)"), (False, "n(아님)")):
        print(f"{name:12}{conf[(y, True)]:>8}{conf[(y, False)]:>10}{conf[(y, 'none')]:>8}")
    print("\n유형별 정확도")
    for t, (ok, tot) in sorted(by_type.items(), key=lambda x: -x[1][1]):
        print(f"  {t:<20} {ok}/{tot} ({ok / tot:.0%})")
    if j.variant != "P1c" and len(set(ys)) == 2:
        from judge_offline import auc
        import numpy as np
        print(f"\nAUC {auc(np.array(scores), np.array(ys)):.3f}")


if __name__ == "__main__":
    main()
