"""최종 시험(한 번만): selection.json 의 구성으로 시험 세트를 평가 → results/final_test.md

    python tools/final_test.py

1. results/final_test.lock 획득(이미 있으면 오류 — 두 번 돌리면 시험 세트로 고르는 셈)
2. 시험 회의 재생(run_ami --split test, dev와 같은 고정 특징 추출 구성)
3. 선택된 판정기 변형으로 시험 구간 판정(judge_offline, 선택 모델·변형 하나만)
4. 비교: baseline-v1(P1c·손 가중치·플래그 없음), timing, 최종 구성 — 부트스트랩 95% CI + 짝지은 차이
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main():
    from app.config import load_config, resolve_path
    from splits import acquire_test_lock, meetings
    cfg = load_config("ami")
    results = resolve_path(cfg, "results_dir")
    sel = json.loads((results / "selection.json").read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    acquire_test_lock(sel["config_name"])
    print(f"[final] 구성 {sel['config_name']} · 시험 회의 {meetings('test')}")

    import run_ami
    run_ami.main(["--split", "test", "--no-report", "--yes"], _final_test_ok=True)
    if sel["variant"] != "P1c" and sel.get("mode", "full") in ("full", "semantic"):
        import judge_offline
        judge_offline.run("test", [sel["model"]], [sel["variant"]], out_md="judge_test.md", _final_test_ok=True)

    from stats_boot import bootstrap, fmt, paired
    from tune_dev import load_judges, load_split, run_config, with_policy
    cal = results / "ami_calibration.json"
    if cal.exists():
        cfg["ownvoice"]["own_margin_db"] = round(json.loads(cal.read_text(encoding="utf-8"))["own_margin_db"], 1)
    data = load_split(results, "test", _final_test_ok=True)
    judges = load_judges(results, "test")
    jkey = f"{sel['variant']}-{sel['model']}"
    judge = judges.get(jkey) if sel["variant"] != "P1c" else None
    runs = {
        "baseline-v1 (P1c · hand · 플래그 없음)": run_config(cfg, data, None),
        "timing": run_config(cfg, data, None, mode="timing"),
        f"최종: {sel['config_name']}": run_config(with_policy(cfg, fusion=sel["fusion"], **sel["flags"]), data, judge,
                                                 mode=sel.get("mode", "full")),
    }
    names = list(runs)
    md = ["# 최종 시험 결과 (AMI test, 한 번만 실행)", "",
          f"시험 회의 {meetings('test')} · 시나리오 {len(data)}개 · (회의, 착용자) {len(runs['timing']['units'])}단위 · "
          f"조건 clean+snr10+snr5 합산 · single 정의 · 부트스트랩 1000회", "",
          f"최종 구성은 개발 세트에서만 골랐다(`results/selection.md`, dev F0.5 {sel['dev_f05']:.3f}).", "",
          "| 방식 | F0.5 [95% CI] | 정밀도 | 재현율 | 자연 함정 오표시율 | 함정 화자 오등록률 |", "|---|---:|---:|---:|---:|---:|"]
    for k in names:
        b = bootstrap(runs[k]["units"])
        md.append(f"| {k} | {fmt(b['f05'], False)} | {fmt(b['precision'])} | {fmt(b['recall'])} | "
                  f"{fmt(b['trap_shown_rate'])} | {fmt(b['trapspk_rate'])} |")
    md += ["", "조건별 F0.5:", "", "| 방식 | clean | snr10 | snr5 |", "|---|---:|---:|---:|"]
    for k in names:
        md.append(f"| {k} | " + " | ".join(f"{runs[k]['per_cond'].get(c, {}).get('f05', 0):.3f}" for c in ("clean", "snr10", "snr5")) + " |")
    fin = names[2]
    md += ["", "짝지은 부트스트랩(같은 재표본):", ""]
    for other in names[:2]:
        for m in ("f05", "precision", "recall", "trap_shown_rate"):
            d = paired(runs[fin]["units"], runs[other]["units"], m)
            md.append(f"- 최종 − {other} · {m}: {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] (최종이 더 큰 재표본 {d[3]:.0%})")
    md += ["", f"총 실행 시간 {(time.perf_counter() - t0) / 60:.1f}분"]
    (results / "final_test.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
