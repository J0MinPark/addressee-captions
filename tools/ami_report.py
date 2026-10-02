"""AMI 소거 실험 보고서: results/ami_ablation.md, ami_ablation.csv

    python tools/evaluate.py --ami                          # 평가용(_take2) AMI 시나리오 전부, 두 정의 모두
    python tools/evaluate.py --ami --to-me-definition single

소음 조건(clean/snr10/snr5) × '나에게 한 말' 정의(single / single+group) 마다 5개 모드 표.
자연 함정 = 타이밍 증거가 성립(T ≥ 0.5, 착용자 발화 직후)하는데 라벨이 n 인 구간 — 대본 녹음의 trap 시나리오를 대체.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from app.config import resolve_path
from app.policy import MODE_LABELS, MODES
from datasplit import check_eval_inputs
from evaluate import collect, llm_confusion, metrics, simulate, summarize

pct = lambda v: "–" if v is None else f"{v * 100:.0f}%"  # noqa: E731
DEF_LABEL = {"single": "single — 착용자 한 명에게 한 말만 y (그룹 발화는 n)",
             "single+group": "single+group — 그룹 전체에게 한 말도 y"}


def _meta(data_dir: Path, name: str) -> dict:
    p = data_dir / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def ami_report(cfg: dict, names: list[str], results: Path, defs: list[str]) -> Path:
    warnings = check_eval_inputs(names)
    data_dir = resolve_path(cfg, "data_dir")
    metas = {n: _meta(data_dir, n) for n in names}
    conds = [c for c in ("clean", "snr10", "snr5") if any(m.get("condition") == c for m in metas.values())]
    meetings = sorted({m.get("meeting") for m in metas.values() if m.get("meeting")})
    minutes = sorted({m.get("minutes") for m in metas.values() if m.get("minutes")})
    wearers = sorted({f"{m.get('meeting')}:{m.get('wearer')}" for m in metas.values()})

    # 라벨 분포·unmatched (labelstats)
    seg_lab, utt_lab, unmatched, nseg = Counter(), Counter(), 0, 0
    for n in names:
        p = results / f"{n}.labelstats.json"
        if p.exists():
            st = json.loads(p.read_text(encoding="utf-8"))
            for k, v in st["seg_labels"].items():
                seg_lab[k] += v
            if metas[n].get("condition") == "clean":
                for k, v in st["utterance_labels"].items():
                    utt_lab[k] += v
            unmatched += st["unmatched"]
            nseg += st["segments"]

    src = (f"AMI Meeting Corpus · 회의 {', '.join(meetings)} · 앞 {', '.join(f'{m:g}' for m in minutes)}분 · "
           f"착용자 {len(wearers)}명 · A=개별 헤드셋, B=Array1-01")
    md = ["# AMI 수신자 판별 소거 실험", "", f"**데이터 출처**: {src}",
          f"소음 조건: {', '.join(conds)} (snr = DEMAND PCAFETER 카페테리아 소음을 B 채널에 섞음)",
          f"프로필 `{cfg.get('_profile')}` · 정책 가중치 b={cfg['policy']['bias']}, w_t={cfg['policy']['w_t']}, "
          f"w_s={cfg['policy']['w_s']}, w_l={cfg['policy']['w_l']} (변경 없음) · own_margin_db={cfg['ownvoice']['own_margin_db']}", ""]
    for w in warnings:
        md.append(f"> ⚠ {w}")
    tot_u = sum(utt_lab.values()) or 1
    md += ["## 라벨 분포", "",
           "정답 발화(대화행위를 화자·addressee 기준으로 묶은 단위, clean 기준, 착용자 관점 합계): " +
           ", ".join(f"{k} {v} ({v / tot_u:.0%})" for k, v in sorted(utt_lab.items())),
           "시스템 구간 매칭 결과(비착용자 구간): " +
           ", ".join(f"{k.split(':')[1]} {v}" for k, v in sorted(seg_lab.items()) if k.startswith("other_seg")),
           f"unmatched(겹침 50% 미만, 평가 제외): {unmatched}/{nseg} ({unmatched / max(nseg, 1):.0%})", ""]

    csv_rows = []
    for d in defs:
        data = collect(names, results, d)
        md += [f"## 정의: {DEF_LABEL[d]}", ""]
        for cond in conds:
            sub = {n: v for n, v in data.items() if metas[n].get("condition") == cond}
            if not sub:
                continue
            per = {n: {m: metrics(segs, labels, simulate(cfg, segs, m)) for m in MODES}
                   for n, (meta, segs, labels) in sub.items()}
            rows = {m: summarize([per[n][m] for n in sub]) for m in MODES}
            title = f"AMI {', '.join(meetings)} · 앞 {', '.join(f'{x:g}' for x in minutes)}분 · {cond} · 정의 {d}"
            r0 = rows["full"]
            md += [f"### 표: {title}", "",
                   f"시나리오 {len(sub)}개 · 라벨 y {r0['n_pos']} / n {r0['n_neg']}", "",
                   "| 모드 | 정밀도 | 재현율 | F1 | 자막 오염도 ↓ | 등록 지연 | 자연 함정 오표시율 ↓ | 자연 함정 화자 오등록률 ↓ |",
                   "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for m in MODES:
                r = rows[m]
                dl = "–" if r["reg_delay_s"] is None else f"{r['reg_delay_s']:.1f}s"
                if r["unregistered"]:
                    dl += f" (미등록 {r['unregistered']}/{r['n_partner']})"
                name = f"**{MODE_LABELS[m]}**" if m == "full" else MODE_LABELS[m]
                md.append(f"| {name} (`{m}`) | {pct(r['precision'])} | {pct(r['recall'])} | {r['f1']:.2f} | "
                          f"{pct(r['contamination'])} | {dl} | {pct(r['trap_shown_rate'])} ({r['trap_shown']}) | "
                          f"{pct(r['trapspk_rate'])} ({r['trapspk']}) |")
                csv_rows.append({"definition": d, "condition": cond, "mode": m, "precision": r["precision"],
                                 "recall": r["recall"], "f1": r["f1"], "contamination": r["contamination"],
                                 "reg_delay_s": r["reg_delay_s"], "natural_trap_shown_rate": r["trap_shown_rate"],
                                 "natural_trap_n": r["trap_shown"].split("/")[1],
                                 "natural_trap_speaker_misreg_rate": r["trapspk_rate"],
                                 **{f"elig_{k}": v for k, v in r["elig"].items()}})
            # 모드별: 착용자 직후 구간의 표시 혼동행렬
            md += ["", "착용자 직후(T ≥ 0.5) 구간의 표시 결과 (라벨 × 표시/접힘):", "",
                   "| 모드 | y→표시 | y→접힘 | n→표시(자연 함정 오표시) | n→접힘 |", "|---|---:|---:|---:|---:|"]
            for m in MODES:
                e = rows[m]["elig"]
                md.append(f"| `{m}` | {e.get('y_shown', 0)} | {e.get('y_folded', 0)} | {e.get('n_shown', 0)} | {e.get('n_folded', 0)} |")
            # LLM 판정기 혼동행렬(LLM 출력은 모드와 무관: replay가 착용자 직후 구간 전부에 호출해 캐시)
            conf = defaultdict(int)
            for n, (meta, segs, labels) in sub.items():
                for k, v in llm_confusion(segs, labels).items():
                    conf[k] += v
            g = lambda k: conf.get(k, 0)  # noqa: E731
            yp, yn, yx, np_, nn, nx = g(("y", "pair")), g(("y", "nopair")), g(("y", "none")), \
                g(("n", "pair")), g(("n", "nopair")), g(("n", "none"))
            tot = yp + yn + np_ + nn
            md += ["", "LLM 판정기 혼동행렬 (착용자 직후 구간, LLM 출력은 모드와 무관 — `full`·`semantic`만 이 출력을 사용):", "",
                   "| 라벨 \\ LLM | 짝 | 짝 아님 | 결과 없음 |", "|---|---:|---:|---:|",
                   f"| y | {yp} | {yn} | {yx} |", f"| n | {np_} | {nn} | {nx} |", "",
                   f"정확도 {pct((yp + nn) / tot if tot else None)} (n={tot}) · 짝 재현율 "
                   f"{pct(yp / (yp + yn) if yp + yn else None)} · 짝 아님 재현율 {pct(nn / (np_ + nn) if np_ + nn else None)}", ""]
            csv_rows.append({"definition": d, "condition": cond, "mode": "llm_confusion",
                             "precision": f"y_pair={yp};y_nopair={yn};y_none={yx};n_pair={np_};n_nopair={nn};n_none={nx}"})
    out = results / "ami_ablation.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    keys = sorted({k for r in csv_rows for k in r}, key=lambda k: (k not in ("definition", "condition", "mode"), k))
    with open(results / "ami_ablation.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(csv_rows)
    print("\n".join(md))
    print(f"\n저장: {out}, {results / 'ami_ablation.csv'}")
    return out
