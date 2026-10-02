"""한국어 판정 프롬프트 점검(P1 버그 수정용). AI Hub 데이터를 쓰지 않는다(점검용 직접 작성 쌍).

    source scripts/server_env.sh && python tools/ko_prompt_check.py   → 콘솔 표 + results/ko_prompt_check.json

data/judge_pairs_ko_check.csv: 한국어 쌍 48개(짝 24 · 짝 아님 24)와 그 영어 역번역.
기준(결과를 보기 전에 고정): 한국어판은 영어판의 번역이어야 한다. 그래서 같은 변형의 영어판이 역번역 문장에 내는 점수
(AMI에서 검증된 판정기, 바꾸지 않는다)를 기준으로, 한국어판 점수와의 평균 절대 차이(MAD)가 가장 작은 P1 후보를 고른다.
동률(차이 0.01 이내)이면 AUC가 높은 쪽. 정확도(라벨)는 보고만 한다.
P1 후보(고정): K0 수정 전 · K1 few-shot 없음 · K2 영어 P1 few-shot의 1:1 번역 · K3 현재에서 '자리' 짝 아님 예시만 교체.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from app import llm_judge as L  # noqa: E402
from app.config import load_config  # noqa: E402

_T, _F = {"pair": True}, {"pair": False}
# 영어 P1 few-shot(FEW_SHOT_EN)의 한국어 1:1 번역(이름만 한국 이름으로)
SHOT_P1_KO_TR = [
    ({"prev": [], "a": "다음 회의 몇 시예요?", "b": "세 시일 거예요."}, _T),
    ({"prev": [], "a": "그럼 고무 케이스로 갈까요?", "b": "네, 좋은 것 같아요."}, _T),
    ({"prev": [("B", "그게 예산이에요.")], "a": "회의 끝나고 그 슬라이드 좀 보내 주실 수 있어요?",
      "b": "그럼요, 메일로 보내 드릴게요."}, _T),
    ({"prev": [], "a": "리모컨에 화면이 필요할까요?", "b": "민준 씨, 거기 그 펜 좀 건네줄래요?"}, _F),
    ({"prev": [], "a": "저는 노란색이 정말 좋아요.", "b": "잠깐, 프로젝터 아직 켜져 있어요?"}, _F),
    ({"prev": [], "a": "음성 인식을 넣으면 비용이 얼마나 늘까요?", "b": "수연 씨, 마케팅 쪽에서 온 메일 받았어요?"}, _F),
]
_K0 = L._SHOT_P1_KO_OLD   # 수정 전 한국어 P1 few-shot
SHOT_P1_KO_NOSEAT = [s if s[0]["a"] != "여기 앉아도 돼요?" else
                     ({"prev": [], "a": "이 발표 자료 언제까지 내야 돼요?", "b": s[0]["b"]}, s[1]) for s in _K0]
P1_CANDIDATES = {"K0 수정 전(원래)": _K0, "K1 few-shot 없음": [], "K2 영어 few-shot 1:1 번역": SHOT_P1_KO_TR,
                 "K3 자리 예시만 교체": SHOT_P1_KO_NOSEAT}


def auc(p, y):
    p, y = np.asarray(p, float), np.asarray(y, int)
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))


def score(variant: str, lang: str, rows: list[dict], shots=None) -> list[float]:
    cfg = load_config("server", overrides={"llm": {"variant": variant, "prompt_lang": lang, "cache": False,
                                                   "timeout_s": 30}})
    j = L.LLMJudge(cfg, lambda *a: None)
    saved = None
    if shots is not None:
        saved = L.PROMPTS_V2[(lang, variant)]
        L.PROMPTS_V2[(lang, variant)] = (saved[0], shots)
    try:
        assert j.setup(), j.health_line()
        out = []
        for r in rows:
            a, b = (r["a"], r["b"]) if lang == "ko" else (r["a_en"], r["b_en"])
            res = j.judge([], a, b)
            out.append(float(res["prob"]) if res else float("nan"))
        return out
    finally:
        if saved is not None:
            L.PROMPTS_V2[(lang, variant)] = saved


def summary(p, y, ref=None) -> dict:
    p, y = np.asarray(p), np.asarray(y)
    d = {"auc": round(auc(p, y), 3), "acc": round(float(np.mean((p >= 0.5) == (y == 1))), 3),
         "mean_pos": round(float(np.mean(p[y == 1])), 3), "mean_neg": round(float(np.mean(p[y == 0])), 3)}
    if ref is not None:
        d["mad_vs_en"] = round(float(np.mean(np.abs(p - np.asarray(ref)))), 3)
    return d


def main():
    rows = list(csv.DictReader(open(ROOT / "data" / "judge_pairs_ko_check.csv", encoding="utf-8")))
    y = [int(r["label"]) for r in rows]
    out = {"n": len(rows), "n_pos": sum(y), "variants": {}, "p1_candidates": {}}
    print(f"점검 쌍 {len(rows)}개 (짝 {sum(y)}) · qwen3:4b · {load_config('server')['llm']['url']}")
    print("\n| 변형 | 언어 | AUC | 정확도@0.5 | 짝 평균 | 짝 아님 평균 | 영어판과 MAD |\n|---|---|---:|---:|---:|---:|---:|")
    refs = {}
    for v in ("P1c", "P1", "P2", "P3"):
        en = score(v, "en", rows)
        ko = score(v, "ko", rows)
        refs[v] = en
        se, sk = summary(en, y), summary(ko, y, en)
        out["variants"][v] = {"en_backtranslated": se, "ko": sk, "p_en": en, "p_ko": ko}
        print(f"| {v} | 영어(역번역 문장) | {se['auc']} | {se['acc']} | {se['mean_pos']} | {se['mean_neg']} | – |")
        print(f"| {v} | 한국어(현재) | {sk['auc']} | {sk['acc']} | {sk['mean_pos']} | {sk['mean_neg']} | {sk['mad_vs_en']} |")
    print("\nP1 후보(기준: 영어 P1과의 MAD 최소, 동률이면 AUC)\n| 후보 | AUC | 정확도@0.5 | 짝 평균 | 짝 아님 평균 | 영어판과 MAD | 자리→아니요 |\n|---|---:|---:|---:|---:|---:|---:|")
    for name, shots in P1_CANDIDATES.items():
        p = score("P1", "ko", rows, shots)
        s = summary(p, y, refs["P1"])
        s["seat_no"] = round(p[0], 3)
        out["p1_candidates"][name] = {**s, "p": p}
        print(f"| {name} | {s['auc']} | {s['acc']} | {s['mean_pos']} | {s['mean_neg']} | {s['mad_vs_en']} | {s['seat_no']} |")
    c = out["p1_candidates"]
    best = min(c, key=lambda k: c[k]["mad_vs_en"])
    tie = [k for k in c if c[k]["mad_vs_en"] - c[best]["mad_vs_en"] <= 0.01]
    best = max(tie, key=lambda k: c[k]["auc"])
    out["chosen"] = best
    print(f"\n→ 선택: {best}")
    out["rows"] = rows
    (ROOT / "results" / "ko_prompt_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
