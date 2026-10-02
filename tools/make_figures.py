"""발표용 그림·표(results/figures/). 새로 평가하지 않는다: 커밋된 AMI 시험 결과와 학습된 융합 계수만 읽는다.

    python tools/make_figures.py      (pip install matplotlib koreanize-matplotlib — 나눔고딕 포함, 한글 깨짐 방지)

입력: results/final_test.md(AMI 시험, 한 번 실행된 결과), results/selection.json(AMI dev 학습 융합 계수)
출력(각각 png + 원본 수치 csv):
  ami_test_table      baseline-v1 · timing · 최종 구성의 F0.5·정밀도·재현율·자연 함정 오표시율 [95% CI]
  ami_trap_rate       자연 함정 오표시율 timing vs 최종 구성(95% CI 오차막대)
  fusion_coef         학습된 융합 계수(응답 간격, 구간 길이, p_pair, 의문문 직후)
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import koreanize_matplotlib  # noqa: F401  (NanumGothic)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "figures"
# 참조 팔레트(dataviz references/palette.md, 라이트 모드): 표면·잉크·격자·계열 1(파랑)·2(주황)·발산(파랑↔빨강)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
BLUE, ORANGE, RED, NEUTRAL = "#2a78d6", "#eb6834", "#e34948", "#b5b3aa"

plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 12,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200,
                     "axes.unicode_minus": False})   # 나눔고딕에 U+2212 없음

NAMES = {"baseline-v1": "v1 (P1c · 손 가중치)", "timing": "timing (타이밍만)", "최종": "최종 (P1 · 학습된 융합)"}
CI = re.compile(r"([\d.]+)%?\s*\[([\d.]+),\s*([\d.]+)\]")


def parse_test() -> list[dict]:
    rows = []
    for line in (ROOT / "results" / "final_test.md").read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or "F0.5" in line or "---" in line or "clean" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 6:
            continue
        key = next((k for k in NAMES if cells[0].startswith(k)), None)
        if key is None:
            continue
        r = {"method": NAMES[key], "source_row": cells[0]}
        for name, cell, scale in (("f05", cells[1], 1), ("precision", cells[2], 100), ("recall", cells[3], 100),
                                  ("trap_shown_rate", cells[4], 100)):
            m = CI.match(cell)
            r[name], r[name + "_lo"], r[name + "_hi"] = (float(m.group(i)) / scale for i in (1, 2, 3))
        rows.append(r)
    assert len(rows) == 3, rows
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def fig_table(rows):
    write_csv(OUT / "ami_test_table.csv", rows)
    pc = lambda r, k: f"{r[k] * 100:.1f}%\n[{r[k + '_lo'] * 100:.1f}, {r[k + '_hi'] * 100:.1f}]"  # noqa: E731
    cell = [[r["method"], f"{r['f05']:.3f}\n[{r['f05_lo']:.3f}, {r['f05_hi']:.3f}]", pc(r, "precision"),
             pc(r, "recall"), pc(r, "trap_shown_rate")] for r in rows]
    fig, ax = plt.subplots(figsize=(10.5, 2.9))
    ax.axis("off")
    t = ax.table(cellText=cell, colLabels=["방식", "F0.5", "정밀도", "재현율", "자연 함정 오표시율 ↓"],
                 cellLoc="center", loc="center", colWidths=[0.28, 0.17, 0.18, 0.18, 0.19])
    t.auto_set_font_size(False)
    t.set_fontsize(12)
    t.scale(1, 2.6)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor(GRID)
        c.set_facecolor(SURF)
        if i == 0:
            c.set_text_props(weight="bold", color=INK2)
        if i == 3:
            c.set_text_props(weight="bold")
    ax.set_title("AMI 시험 세트 (회의 4개 · 착용자 16명 · 3개 조건 합산, 한 번만 실행) · [95% 신뢰구간]",
                 fontsize=12, color=INK2, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "ami_test_table.png")
    plt.close(fig)
    md = ["| 방식 | F0.5 [95% CI] | 정밀도 | 재현율 | 자연 함정 오표시율 |", "|---|---:|---:|---:|---:|"]
    md += [f"| {r['method']} | {r['f05']:.3f} [{r['f05_lo']:.3f}, {r['f05_hi']:.3f}] | "
           + " | ".join(f"{r[k] * 100:.1f}% [{r[k + '_lo'] * 100:.1f}, {r[k + '_hi'] * 100:.1f}]"
                        for k in ("precision", "recall", "trap_shown_rate")) + " |" for r in rows]
    (OUT / "ami_test_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def fig_trap(rows):
    sel = [r for r in rows if r["method"].startswith(("timing", "최종"))]
    data = [{"method": r["method"], "trap_shown_rate": r["trap_shown_rate"], "lo": r["trap_shown_rate_lo"],
             "hi": r["trap_shown_rate_hi"]} for r in sel]
    write_csv(OUT / "ami_trap_rate.csv", data)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = range(len(data))
    vals = [d["trap_shown_rate"] * 100 for d in data]
    err = [[(d["trap_shown_rate"] - d["lo"]) * 100 for d in data], [(d["hi"] - d["trap_shown_rate"]) * 100 for d in data]]
    ax.bar(x, vals, width=0.5, color=[NEUTRAL, BLUE], zorder=2)
    ax.errorbar(x, vals, yerr=err, fmt="none", ecolor=INK2, elinewidth=1.5, capsize=6, zorder=3)
    for i, d in enumerate(data):
        ax.text(i, d["hi"] * 100 + 2.5, f"{vals[i]:.1f}%", ha="center", va="bottom", fontsize=14, weight="bold", color=INK)
    ax.set_xticks(list(x), [d["method"] for d in data])
    ax.set_ylim(0, 105)
    ax.set_ylabel("자연 함정 오표시율 (%) ↓")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_title("착용자 말 직후 끼어든 '나에게 한 말이 아닌' 발화를\n크게 잘못 표시한 비율 (AMI 시험, 95% CI)", fontsize=12,
                 color=INK2, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "ami_trap_rate.png")
    plt.close(fig)


def fig_coef():
    sel = json.loads((ROOT / "results" / "selection.json").read_text(encoding="utf-8"))
    coef = sel["fusion"]["coef"]
    feats = [("gap", "응답 간격"), ("dur", "구간 길이"), ("p_pair", "p_pair (LLM 짝 점수)"), ("wq", "의문문 직후")]
    data = [{"feature": k, "label": n, "coef": coef[k]} for k, n in feats]
    write_csv(OUT / "fusion_coef.csv", data + [{"feature": "_all", "label": json.dumps(coef), "coef": ""}])
    data.sort(key=lambda d: d["coef"])
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    y = range(len(data))
    ax.barh(y, [d["coef"] for d in data], height=0.55, color=[RED if d["coef"] < 0 else BLUE for d in data], zorder=2)
    ax.axvline(0, color=INK2, linewidth=1)
    for i, d in enumerate(data):
        ax.text(d["coef"] + (0.08 if d["coef"] >= 0 else -0.08), i, f"{d['coef']:+.2f}", va="center",
                ha="left" if d["coef"] >= 0 else "right", fontsize=12, color=INK)
    ax.set_yticks(list(y), [d["label"] for d in data])
    lim = max(abs(d["coef"]) for d in data) * 1.35
    ax.set_xlim(-lim, lim)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_xlabel("계수 (+ 나에게 한 말 쪽 · - 아닌 쪽)")
    ax.set_title("학습된 융합 계수 (AMI dev, L2 로지스틱)\n짧고 바로 붙은 응답·의문문 직후·LLM 짝 점수가 '나에게 한 말' 쪽",
                 fontsize=12, color=INK2, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fusion_coef.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = parse_test()
    fig_table(rows)
    fig_trap(rows)
    fig_coef()
    for p in sorted(OUT.iterdir()):
        print(p.relative_to(ROOT))


if __name__ == "__main__":
    main()
