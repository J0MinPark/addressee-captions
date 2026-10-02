"""AI Hub 라벨 시뮬레이션 최종 시험(한 번만, 사전 등록 개정 1).

    source scripts/server_env.sh && python tools/final_test_aihub_sim.py

1. results/aihub_sim_final_test.lock 획득(이미 있으면 거부 — 두 번 돌리면 시험 세트로 고르는 셈)
2. test 시나리오 24개 생성 + 판정(P1c와 시연 선택 구성의 판정기만)
3. 비교: all, timing, v1(P1c · 손 가중치), 시연 선택 구성(results/demo_selection.json, dev에서만 고름)
   세션 단위 부트스트랩 1000회 95% CI + 짝지은 차이
→ results/aihub_sim_final_test.md, .csv (수치만). 결과를 보고 시연 구성을 바꾸지 않는다.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ["HEARME_NO_SELECTED"] = "1"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
RES = Path(os.environ.get("AIHUB_RESULTS_DIR") or ROOT / "results")
LOCK = RES / "aihub_sim_final_test.lock"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def acquire_lock(config_name: str) -> None:
    if LOCK.exists():
        raise SystemExit(f"[aihub] 시험 세트는 이미 한 번 평가됐습니다: {LOCK.read_text(encoding='utf-8').strip()}\n"
                         f"  (다시 돌리면 시험 세트로 선택하는 셈. 정말 필요하면 lock을 지우고 그 사실을 보고할 것)")
    LOCK.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} config={config_name}\n", encoding="utf-8")


def main():
    sel = json.loads((RES / "demo_selection.json").read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    acquire_lock(sel["config_name"])
    print(f"[final] 선택 구성 {sel['config_name']} ({sel['config']})")
    import aihub_sim
    from aihub_tune import MODEL, base_cfg, load_sim
    from stats_boot import bootstrap, fmt, paired
    from tune_dev import run_config, with_policy
    variants = ["P1c"] + ([sel["variant"]] if sel["variant"] != "P1c" and sel["mode"] == "full" else [])
    built = aihub_sim.build_split("test", variants, _final_test_ok=True)
    cfg = base_cfg()
    data, judges = load_sim("test", _final_test_ok=True)
    judge = judges.get(f"{sel['variant']}-{MODEL}") if sel["variant"] != "P1c" else None
    runs = {"all": run_config(cfg, data, None, mode="all"),
            "timing": run_config(cfg, data, None, mode="timing"),
            "v1 (P1c · hand)": run_config(cfg, data, None)}
    fin = f"선택: {sel['config_name']}"
    if sel["mode"] in ("timing", "timing_speaker"):
        runs[fin] = run_config(cfg, data, None, mode=sel["mode"])
    else:
        runs[fin] = run_config(with_policy(cfg, fusion=sel["fusion"], **sel["flags"]), data, judge)
    pr = runs["timing"]["pooled"]
    st = built["stats"]
    md = ["# 최종 시험 결과 — AI Hub 라벨 시뮬레이션 test (한 번만 실행)", "",
          "**AI Hub 주요 영역별 회의 음성인식 데이터셋 활용** · 라벨 기반 대화 시뮬레이션(사전 등록 개정 1: 판정 계층만 평가)", "",
          f"test 시나리오 {len(data)}개 · (세션, 착용자) {len(runs['timing']['units'])}단위 · 양성(x) {pr['n_pos']} · 음성(y) {pr['n_neg']} · "
          f"자연 함정 {sum(s['natural_traps'] for s in st)}개 · 부트스트랩 1000회(세션 단위)", "",
          f"시연 구성은 dev에서만 골랐다(`results/demo_selection.md`, dev F0.5 {sel['dev_f05']:.3f}).", "",
          "| 방식 | F0.5 [95% CI] | 정밀도 | 재현율 | 자막 오염도 | 자연 함정 오표시율 | 옆 대화 화자 오등록률 |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    rows = []
    for k, r in runs.items():
        b, p = bootstrap(r["units"]), r["pooled"]
        mr = None if p["misreg_rate"] is None else p["misreg_rate"]
        md.append(f"| {k} | {fmt(b['f05'], False)} | {fmt(b['precision'])} | {fmt(b['recall'])} | "
                  f"{p['contamination'] * 100:.1f}% | {fmt(b['trap_shown_rate'])} | "
                  f"{'–' if mr is None else f'{mr * 100:.1f}%'} ({p['misreg']}) |")
        rows.append({"method": k, "f05": b["f05"][0], "f05_lo": b["f05"][1], "f05_hi": b["f05"][2],
                     "precision": b["precision"][0], "recall": b["recall"][0], "contamination": p["contamination"],
                     "trap_shown_rate": b["trap_shown_rate"][0], "trap_lo": b["trap_shown_rate"][1],
                     "trap_hi": b["trap_shown_rate"][2], "misreg_rate": mr, "n_pos": p["n_pos"], "n_neg": p["n_neg"]})
    md += ["", "짝지은 부트스트랩(같은 재표본):", ""]
    for other in ("timing", "v1 (P1c · hand)", "all"):
        if other == fin:
            continue
        for m in ("f05", "precision", "recall", "trap_shown_rate"):
            d = paired(runs[fin]["units"], runs[other]["units"], m)
            scale, unit = (100, "%p") if m != "f05" else (1, "")
            md.append(f"- 선택 − {other} · {m}: {d[0] * scale:+.3f}{unit} [{d[1] * scale:+.3f}, {d[2] * scale:+.3f}] "
                      f"(선택이 더 큰 재표본 {d[3]:.0%})")
    md += ["", "한계: 정답 분할·전사·화자 ID를 쓰는 시뮬레이션이라 받아쓰기·화자 임베딩·분할·소음 오류가 없다.",
           f"", f"총 실행 시간 {(time.perf_counter() - t0) / 60:.1f}분"]
    (RES / "aihub_sim_final_test.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with open(RES / "aihub_sim_final_test.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n".join(md))


if __name__ == "__main__":
    main()
