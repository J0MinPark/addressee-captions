"""두 소거 모드의 판정이 다른 구간을 모두 나열하고 오류 유형을 집계한다(캐시만 사용, 모델 호출 없음).

    python tools/diff_modes.py                       # 평가용(_take2) 시나리오 전부, full vs timing_speaker
    python tools/diff_modes.py demo_trap --a full --b timing_speaker

출력: results/diff_<a>_vs_<b>.md  (합성 데이터면 diff_<a>_vs_<b>_[SYNTHETIC].md)
끝에 '정책 개선 제안'(최대 3개)을 적는다 — 제안일 뿐 코드에는 반영하지 않는다.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from app.config import load_config, resolve_path  # noqa: E402
from app.policy import MODES  # noqa: E402
from datasplit import SYN, check_eval_inputs, is_synthetic, role, scenario_name  # noqa: E402
from evaluate import collect, simulate  # noqa: E402


def syllables(text: str) -> int:
    t = re.sub(r"[\s\.,!?~…·\"'“”‘’\-]+", "", text or "")
    return len(t)


def esc(s) -> str:
    return str(s if s is not None else "–").replace("|", "\\|").replace("\n", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="*", help="결과(시나리오 이름). 생략하면 평가용(_take2) 전부")
    ap.add_argument("--a", default="full", choices=MODES)
    ap.add_argument("--b", default="timing_speaker", choices=MODES)
    ap.add_argument("--profile", default=None)
    args = ap.parse_args()
    cfg = load_config(args.profile)
    results = resolve_path(cfg, "results_dir")
    if args.scenarios:
        names = [scenario_name(n) for n in args.scenarios]
    else:
        names = sorted(scenario_name(p.name) for p in results.glob("*.labels.csv")
                       if role(scenario_name(p.name)) == "eval")
    if not names:
        raise SystemExit("대상 시나리오가 없습니다.")
    warnings = check_eval_inputs(names)
    data = collect(names, results)
    if not data:
        raise SystemExit("segments/labels 가 있는 시나리오가 없습니다.")
    synthetic = any(meta.get("synthetic") or is_synthetic(n) for n, (meta, _, _) in data.items())
    A, B = args.a, args.b
    rows = []
    for n, (meta, segs, labels) in data.items():
        sa, sb = simulate(cfg, segs, A), simulate(cfg, segs, B)
        by_id = {s["seg_id"]: s for s in segs}
        for s in segs:
            sid = s["seg_id"]
            if sid not in sa["roles"] or sid not in sb["roles"]:
                continue
            pa, pb = sa["roles"][sid] == "partner", sb["roles"][sid] == "partner"
            if pa == pb:
                continue
            lab = labels.get(sid, "?")
            w = by_id.get(s.get("wearer_seg") or "", {})
            llm = s.get("llm") or {}
            rows.append({
                "scenario": n, "t": f"{s['t_start']:.1f}–{s['t_end']:.1f}", "label": lab,
                "a": f"{'표시' if pa else '접힘'} {sa['probs'][sid]:.2f}", "b": f"{'표시' if pb else '접힘'} {sb['probs'][sid]:.2f}",
                "a_wrong": lab in ("y", "n") and pa != (lab == "y"),
                "b_wrong": lab in ("y", "n") and pb != (lab == "y"),
                "text": s.get("text", ""), "wearer": w.get("text") if s.get("gap") is not None else None,
                "gap": s.get("gap"), "sim": s.get("sim"), "spk": s.get("speaker_id"),
                "llm": (f"{llm.get('pair')}/{llm.get('confidence')}/{llm.get('type')}" if llm else
                        ("호출 안 함" if not s.get("llm_eligible") else "결과 없음")),
                "llm_pair": llm.get("pair") if llm else None,
                "dur": s["t_end"] - s["t_start"], "short": syllables(s.get("text", "")) <= 2,
                "first": sa["pre_state"].get(sid) != "partner" or sb["pre_state"].get(sid) != "partner",
                "pre": f"{sa['pre_state'].get(sid)}/{sb['pre_state'].get(sid)}",
            })

    a_only = [r for r in rows if r["a_wrong"] and not r["b_wrong"]]
    b_only = [r for r in rows if r["b_wrong"] and not r["a_wrong"]]
    short = [r for r in rows if r["short"]]
    first = [r for r in rows if r["first"]]
    tag = f" {SYN}" if synthetic else ""
    md = [f"# 모드 간 오류 분석: `{A}` vs `{B}`{tag}", ""]
    if synthetic:
        md += [f"> ⚠ **{SYN}** 합성(TTS) 데이터 결과입니다. 발표 자료에 쓰지 마세요.", ""]
    for w in warnings:
        md += [f"> ⚠ {w}"]
    md += [f"시나리오: {', '.join(data)} · 판정이 다른 구간 {len(rows)}개", "",
           "## 요약", "",
           "| 항목 | 개수 |", "|---|---:|",
           f"| `{A}`만 틀림 | {len(a_only)} (놓침 {sum(r['label'] == 'y' for r in a_only)}, 잘못 표시 {sum(r['label'] == 'n' for r in a_only)}) |",
           f"| `{B}`만 틀림 | {len(b_only)} (놓침 {sum(r['label'] == 'y' for r in b_only)}, 잘못 표시 {sum(r['label'] == 'n' for r in b_only)}) |",
           f"| 후보 발화가 2음절 이하 | {len(short)} |",
           f"| 화자의 첫 응답(판정 직전 미등록, `{A}`/`{B}` 중 하나라도) | {len(first)} |",
           f"| 라벨 없음/건너뜀 | {sum(r['label'] not in ('y', 'n') for r in rows)} |", "",
           "## 판정이 다른 구간", "",
           f"| 시나리오 | 시각(s) | 라벨 | `{A}` | `{B}` | ASR 텍스트 | 착용자 직전 발화 | 간격 g | 화자(유사도) | 판정 직전 상태 {A}/{B} | LLM pair/conf/type | 길이 |",
           "|---|---|---|---|---|---|---|---:|---|---|---|---:|"]
    for r in rows:
        mark = lambda wrong: " ❌" if wrong else ""  # noqa: E731
        gap = "–" if r["gap"] is None else f"{r['gap']:+.2f}"
        md.append(f"| {r['scenario']} | {r['t']} | {r['label']} | {r['a']}{mark(r['a_wrong'])} | {r['b']}{mark(r['b_wrong'])} | "
                  f"{esc(r['text'])} | {esc(r['wearer'])} | {gap} | "
                  f"#{r['spk']} ({(r['sim'] or 0):.2f}) | {r['pre']} | {esc(r['llm'])} | {r['dur']:.1f}s |")

    # ---- 정책 개선 제안(최대 3개, 근거 개수와 함께). 코드에는 반영하지 않는다.
    props = []
    fn_llm_no = [r for r in a_only if r["label"] == "y" and r["llm_pair"] is False]
    if fn_llm_no:
        props.append(f"**첫 응답의 LLM '짝 아님'이 등록을 막음** — `{A}`만 놓친 응답 {len(fn_llm_no)}건이 LLM pair=false였고 "
                     f"그중 {sum(r['first'] for r in fn_llm_no)}건이 미등록 화자의 첫 응답. 응답 타이밍이 확실(T=1)한데 LLM이 no면 "
                     f"바로 접지 말고 '후보'로 두었다가 다음 교대에서 다시 판정(2회 확인)하는 방안 검토.")
    fp_b = [r for r in b_only if r["label"] == "n"]
    if fp_b:
        props.append(f"**LLM이 함정을 거른 경우는 유지** — `{B}`만 잘못 표시한 함정 {len(fp_b)}건은 `{A}`가 LLM으로 걸렀다. "
                     f"L을 빼는 단순화는 이 이득을 잃는다.")
    if short:
        props.append(f"**짧은 발화(2음절 이하) {len(short)}건에서 판정이 갈림** — '네/응'류는 의미 판정이 불안정하다. "
                     f"짧은 발화는 LLM을 생략하고 T·S(상속)만 쓰거나 L을 low 신뢰로 고정하는 규칙 검토.")
    fp_a = [r for r in a_only if r["label"] == "n" and r["llm_pair"] is True]
    if fp_a and len(props) < 3:
        props.append(f"**LLM 거짓 '짝' {len(fp_a)}건이 함정을 통과시킴** — 질문형 딴 얘기를 대답으로 오인하는 편향. "
                     f"pair=true라도 confidence가 high가 아니면 등록은 2회 교대 확인을 요구하는 방안 검토.")
    if not props:
        props.append("판정이 갈린 구간이 적어 근거 있는 제안이 없습니다. 실제 녹음(_take2)을 늘려 다시 실행하세요.")
    md += ["", "## 정책 개선 제안 (최대 3개 · 코드 미반영)", ""] + [f"{i}. {p}" for i, p in enumerate(props[:3], 1)]
    if len(rows) < 10:
        md += ["", f"> 근거 표본이 {len(rows)}건뿐이다. 제안은 가설로만 보고, 실제 녹음으로 다시 확인할 것."]
    out = results / (f"diff_{A}_vs_{B}" + (f"_{SYN}" if synthetic else "") + ".md")
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
