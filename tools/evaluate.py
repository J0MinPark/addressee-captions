"""소거 실험: 캐시(segments.jsonl)와 라벨(labels.csv)만으로 5개 모드의 정책을 다시 돌려 지표를 낸다.
모델을 다시 부르지 않는다(정책 엔진은 순수 로직).

    python tools/evaluate.py                 # results/ 의 평가용(_take2) 시나리오 전부
    python tools/evaluate.py demo_trap       # 지정(미지정 이름은 경고와 함께 허용, _take1 은 오류)

데이터 분할: 보정용(_take1)·calibrate.py가 쓴 녹음이 섞이면 오류로 멈춘다(tools/datasplit.py).
출력: results/ablation.md, ablation.csv  (합성 데이터는 ablation_[SYNTHETIC].md/.csv 로 따로)
  표 1 전체: 모드별 정밀도·재현율·F1·자막 오염도·등록 지연
  표 2 함정 시나리오(이름에 trap): 모드별 오등록률·함정 구간 오표시율
  표 3 LLM 판정기 혼동행렬(착용자 직후 구간, 라벨 y/n 대비)
"""
from __future__ import annotations

import argparse
import csv
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

from app.config import load_config, resolve_path  # noqa: E402
from app.policy import MODE_LABELS, MODES, PolicyEngine, SegFeat  # noqa: E402
from datasplit import SYN, check_eval_inputs, is_synthetic, role, scenario_name  # noqa: E402

TICK = 0.5


def load_segments(path: Path) -> tuple[dict, list[dict]]:
    meta, out = {}, []
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("_meta"):
            meta = r
        else:
            out.append(r)
    return meta, sorted(out, key=lambda r: (r["t_end"], r["seg_id"]))


TO_ME = ("single", "single+group")


def load_labels(path: Path, to_me: str = "single") -> dict[str, str]:
    """라벨 g(그룹 전체에게, AMI)는 정의에 따라 y 또는 n 으로 바꾼다.
    single: 착용자 한 명에게 한 말만 y / single+group: 그룹 전체에게 한 말도 y."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        raw = {r["seg_id"]: r["label"] for r in csv.DictReader(f)}
    g = "y" if to_me == "single+group" else "n"
    return {k: (g if v == "g" else v) for k, v in raw.items()}


def simulate(cfg: dict, segs: list[dict], mode: str) -> dict:
    """정책을 캐시로 재생.
    반환: roles/probs {seg_id}, reg {spk: 등록 시각}, pre_state {seg_id: 판정 직전 화자 상태}."""
    llm_available = any(s.get("llm_called") for s in segs)
    p = PolicyEngine(cfg, mode=mode, llm_available=llm_available)
    roles, probs, reg, pre = {}, {}, {}, {}
    last_tick = 0.0

    def handle(events, t):
        for e in events:
            if e["type"] == "caption_update" and e.get("id") in roles:   # --candidate-rejudge 의 나중 확정/접기
                roles[e["id"]], probs[e["id"]] = e["role"], e["prob"]
            if e["type"] == "speaker_state" and e["state"] == "partner" and e["speaker_id"] not in reg:
                reg[e["speaker_id"]] = t

    for s in segs:
        t = s["t_end"]
        while last_tick + TICK <= t:
            last_tick += TICK
            p.tick(last_tick)
        if s.get("is_wearer"):
            if s.get("skip") or s.get("turn_id") is None:
                continue
            p.on_wearer_end(s["t_start"], s["t_end"], now=t)
            p.set_wearer_text(len(p.turns) - 1, s.get("text") or "")
            continue
        if s.get("skip") or not s.get("text"):
            continue
        sid = s.get("speaker_id")
        pre[s["seg_id"]] = p.speakers[sid].state if sid in p.speakers else "unknown"
        feat = SegFeat(seg_id=s["seg_id"], t_start=s["t_start"], t_end=s["t_end"],
                       speaker_id=sid, sim=s.get("sim", 0.0), text=s["text"])
        dec, ev = p.on_segment(feat, now=t)
        handle(ev, t)
        roles[s["seg_id"]], probs[s["seg_id"]] = dec["role"], dec["prob"]
        if s.get("name_call"):
            handle(p.on_name_call(feat.speaker_id, "name", 1.0, now=t), t)
        if dec["pending_llm"]:
            res = s.get("llm") if s.get("llm_called") else None
            t_llm = t + ((res or {}).get("latency_ms") or cfg["llm"]["timeout_s"] * 1000) / 1000
            upd, ev = p.on_llm_result(s["seg_id"], res, now=t_llm)
            handle(ev, t_llm)
            if upd:
                roles[s["seg_id"]], probs[s["seg_id"]] = upd["role"], upd["prob"]
    return {"roles": roles, "probs": probs, "reg": reg, "pre_state": pre}


def labeled(segs, labels):
    for s in segs:
        if s.get("is_wearer") or s.get("skip") or not s.get("text"):
            continue
        lab = labels.get(s["seg_id"])
        if lab in ("y", "n"):
            yield s, lab == "y"


def metrics(segs: list[dict], labels: dict[str, str], sim: dict) -> dict:
    tp = fp = fn = tn = 0
    shown_chars = bad_chars = 0
    trap_n = trap_shown = 0
    trap_spk = set()
    elig = defaultdict(int)   # 착용자 직후(T>=0.5) 구간: (라벨, 표시 여부)
    spk_labels = defaultdict(list)
    first_y = {}
    for s, y in labeled(segs, labels):
        pos = sim["roles"].get(s["seg_id"]) == "partner"
        tp += pos and y
        fp += pos and not y
        fn += (not pos) and y
        tn += (not pos) and not y
        if pos:
            n = len(s["text"].replace(" ", ""))
            shown_chars += n
            bad_chars += 0 if y else n
        if not y and s.get("llm_eligible"):     # 함정 구간: 착용자 직후인데 착용자에게 한 말이 아님
            trap_n += 1
            trap_shown += pos
            if s.get("speaker_id") is not None:
                trap_spk.add(s["speaker_id"])
        if s.get("llm_eligible"):
            elig[("y" if y else "n", "shown" if pos else "folded")] += 1
        sid = s.get("speaker_id")
        if sid is not None:
            spk_labels[sid].append(y)
            if y and sid not in first_y:
                first_y[sid] = s["t_start"]
    true_partners = {k for k, v in spk_labels.items() if sum(v) >= 1 and sum(v) / len(v) >= 0.5}
    non_partners = set(spk_labels) - true_partners
    registered = set(sim["reg"])
    delays = [sim["reg"][k] - first_y[k] for k in true_partners if k in sim["reg"] and k in first_y]
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "shown_chars": shown_chars, "bad_chars": bad_chars,
            "n_nonpartner": len(non_partners), "misreg": len(registered & non_partners),
            "n_partner": len(true_partners), "unregistered": len(true_partners - registered),
            "delays": delays, "trap_n": trap_n, "trap_shown": trap_shown,
            "trapspk_n": len(trap_spk - true_partners), "trapspk_reg": len((trap_spk - true_partners) & registered),
            **{f"elig_{a}_{b}": v for (a, b), v in elig.items()}}


def llm_confusion(segs: list[dict], labels: dict[str, str]) -> dict:
    """착용자 직후 구간(LLM 호출 대상) 중 라벨 y/n 인 것: 라벨 × LLM 출력."""
    c = defaultdict(int)
    for s, y in labeled(segs, labels):
        if not s.get("llm_eligible"):
            continue
        res = s.get("llm")
        pred = "none" if res is None else ("pair" if res.get("pair") else "nopair")
        c[("y" if y else "n", pred)] += 1
    return dict(c)


def summarize(ms: list[dict]) -> dict:
    agg = defaultdict(float)
    delays = []
    for m in ms:
        for k, v in m.items():
            if k == "delays":
                delays += v
            else:
                agg[k] += v
    tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "precision": prec, "recall": rec, "f1": f1,
        "contamination": agg["bad_chars"] / agg["shown_chars"] if agg["shown_chars"] else 0.0,
        "misreg_rate": agg["misreg"] / agg["n_nonpartner"] if agg["n_nonpartner"] else None,
        "misreg": f"{int(agg['misreg'])}/{int(agg['n_nonpartner'])}",
        "trap_shown_rate": agg["trap_shown"] / agg["trap_n"] if agg["trap_n"] else None,
        "trap_shown": f"{int(agg['trap_shown'])}/{int(agg['trap_n'])}",
        "trapspk_rate": agg["trapspk_reg"] / agg["trapspk_n"] if agg["trapspk_n"] else None,
        "trapspk": f"{int(agg['trapspk_reg'])}/{int(agg['trapspk_n'])}",
        "elig": {k[5:]: int(v) for k, v in agg.items() if k.startswith("elig_")},
        "reg_delay_s": sum(delays) / len(delays) if delays else None,
        "unregistered": int(agg["unregistered"]), "n_partner": int(agg["n_partner"]),
        "n_pos": int(tp + fn), "n_neg": int(agg["fp"] + agg["tn"]),
    }


pct = lambda v: "–" if v is None else f"{v * 100:.0f}%"  # noqa: E731


def table_all(rows: dict[str, dict], tag: str) -> list[str]:
    out = [f"### 표 1. 전체{tag}", "",
           "| 모드 | 정밀도 | 재현율 | F1 | 자막 오염도 ↓ | 등록 지연 |", "|---|---:|---:|---:|---:|---:|"]
    for mode in MODES:
        r = rows[mode]
        d = "–" if r["reg_delay_s"] is None else f"{r['reg_delay_s']:.1f}s"
        if r["unregistered"]:
            d += f" (미등록 {r['unregistered']}/{r['n_partner']})"
        name = f"**{MODE_LABELS[mode]}**" if mode == "full" else MODE_LABELS[mode]
        out.append(f"| {name} (`{mode}`) | {pct(r['precision'])} | {pct(r['recall'])} | {r['f1']:.2f} | "
                   f"{pct(r['contamination'])} | {d} |")
    r = rows["full"]
    out += ["", f"라벨: 착용자에게 한 말(y) {r['n_pos']}개, 아님(n) {r['n_neg']}개"]
    return out


def table_trap(rows: dict[str, dict], tag: str, members: list[str]) -> list[str]:
    out = [f"### 표 2. 함정 시나리오만{tag}", "", f"시나리오: {', '.join(members)}", "",
           "| 모드 | 오등록률 ↓ | 함정 구간 오표시율 ↓ |", "|---|---:|---:|"]
    for mode in MODES:
        r = rows[mode]
        name = f"**{MODE_LABELS[mode]}**" if mode == "full" else MODE_LABELS[mode]
        out.append(f"| {name} (`{mode}`) | {pct(r['misreg_rate'])} ({r['misreg']}) | "
                   f"{pct(r['trap_shown_rate'])} ({r['trap_shown']}) |")
    out += ["", "오등록률 = 라벨상 대화 상대가 아닌 화자 중 partner로 등록된 비율. "
                "함정 구간 = 착용자 발화 직후(LLM 호출 창 안)에 시작했지만 라벨이 n인 구간, 오표시율 = 그중 큰 자막으로 표시된 비율."]
    return out


def table_confusion(c: dict, tag: str) -> list[str]:
    g = lambda k: c.get(k, 0)  # noqa: E731
    yp, yn, yx = g(("y", "pair")), g(("y", "nopair")), g(("y", "none"))
    np_, nn, nx = g(("n", "pair")), g(("n", "nopair")), g(("n", "none"))
    tot = yp + yn + np_ + nn
    acc = (yp + nn) / tot if tot else None
    out = [f"### 표 3. LLM 판정기 혼동행렬{tag}", "",
           "착용자 직후 구간(LLM 호출 대상) 중 라벨 y/n 인 것.", "",
           "| 라벨 \\ LLM 출력 | 짝(pair=true) | 짝 아님(false) | 결과 없음(타임아웃·실패) |", "|---|---:|---:|---:|",
           f"| 착용자에게 한 말 (y) | {yp} | {yn} | {yx} |",
           f"| 아님 (n) | {np_} | {nn} | {nx} |", "",
           f"정확도 {pct(acc)} (n={tot}) · 짝 재현율 {pct(yp / (yp + yn) if yp + yn else None)} · "
           f"짝 아님 재현율(함정 거르기) {pct(nn / (np_ + nn) if np_ + nn else None)}"]
    return out


def report(cfg, data: dict, synthetic: bool, warnings: list[str], results: Path) -> Path:
    tag = f" {SYN}" if synthetic else ""
    per = {n: {m: metrics(segs, labels, simulate(cfg, segs, m)) for m in MODES}
           for n, (meta, segs, labels) in data.items()}
    rows = {m: summarize([per[n][m] for n in data]) for m in MODES}
    traps = [n for n in data if "trap" in n.lower()]
    trap_rows = {m: summarize([per[n][m] for n in traps]) for m in MODES} if traps else None
    conf = defaultdict(int)
    for n, (meta, segs, labels) in data.items():
        for k, v in llm_confusion(segs, labels).items():
            conf[k] += v
    prov = {(m.get("llm_model"), m.get("llm_prompt_version")) for m, _, _ in data.values()}
    p = cfg["policy"]
    md = [f"# 소거 실험 결과{tag}", ""]
    if synthetic:
        md += [f"> ⚠ **{SYN}** 합성(TTS) 데이터 결과입니다. 발표 자료에 쓰지 마세요.", ""]
    md += [f"평가 시나리오({len(data)}): {', '.join(data)}",
           f"프로필 `{cfg.get('_profile')}` · b={p['bias']}, w_t={p['w_t']}, w_s={p['w_s']}, w_l={p['w_l']} · "
           f"표시 임계 {p['show_threshold']}",
           "LLM: " + ", ".join(f"{m or '없음/기록 없음'} (프롬프트 {v or '?'})" for m, v in sorted(prov, key=str)), ""]
    if len(prov) > 1:
        warnings.append("시나리오마다 LLM 모델/프롬프트 버전이 다릅니다(위 LLM 줄). 같은 조건으로 replay 를 다시 돌리세요.")
    for w in warnings:
        md += [f"> ⚠ {w}"]
    if warnings:
        md += [""]
    md += table_all(rows, tag) + [""]
    md += (table_trap(trap_rows, tag, traps) if trap_rows else [f"### 표 2. 함정 시나리오만{tag}", "",
                                                                "(이름에 trap 이 들어간 평가 시나리오 없음)"]) + [""]
    md += table_confusion(conf, tag) + [""]
    md += [f"### 시나리오별 F1{tag}", "", "| 시나리오 | " + " | ".join(MODES) + " |", "|---|" + "---:|" * len(MODES)]
    for n in data:
        md.append(f"| {n} | " + " | ".join(f"{summarize([per[n][m]])['f1']:.2f}" for m in MODES) + " |")
    stem = "ablation" + ("_" + SYN if synthetic else "")
    (results / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with open(results / f"{stem}.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["table", "mode", "precision", "recall", "f1", "contamination", "reg_delay_s", "misreg_rate",
                    "trap_shown_rate", "synthetic"])
        for m in MODES:
            r = rows[m]
            w.writerow(["all", m, r["precision"], r["recall"], r["f1"], r["contamination"], r["reg_delay_s"],
                        "", "", synthetic])
        if trap_rows:
            for m in MODES:
                r = trap_rows[m]
                w.writerow(["trap", m, "", "", "", "", "", r["misreg_rate"], r["trap_shown_rate"], synthetic])
        for (lab, pred), v in sorted(conf.items()):
            w.writerow(["llm_confusion", f"label={lab},llm={pred}", v, "", "", "", "", "", "", synthetic])
    print("\n".join(md))
    print(f"\n저장: {results / (stem + '.md')}, {results / (stem + '.csv')}\n")
    return results / f"{stem}.md"


def collect(names: list[str], results: Path, to_me: str = "single") -> dict:
    data = {}
    for n in names:
        sp, lp = results / f"{n}.segments.jsonl", results / f"{n}.labels.csv"
        if not sp.exists() or not lp.exists():
            print(f"[evaluate] 건너뜀 {n}: {sp.name if not sp.exists() else lp.name} 없음")
            continue
        meta, segs = load_segments(sp)
        data[n] = (meta, segs, load_labels(lp, to_me))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="*")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--to-me-definition", default=None, choices=list(TO_ME) + ["both"],
                    help="AMI 그룹 발화(g)를 '나에게 한 말'로 볼지: single | single+group | both")
    ap.add_argument("--ami", action="store_true", help="AMI 보고서(results/ami_ablation.md): 조건별·정의별 표")
    args = ap.parse_args()
    cfg = load_config(args.profile or ("ami" if args.ami else None))
    results = resolve_path(cfg, "results_dir")
    if args.ami:
        from ami_report import ami_report
        names = [scenario_name(n) for n in args.scenarios] if args.scenarios else sorted(
            scenario_name(p.name) for p in results.glob("ami_*.labels.csv") if role(scenario_name(p.name)) == "eval")
        d = args.to_me_definition or "both"
        ami_report(cfg, names, results, list(TO_ME) if d == "both" else [d])
        return
    args.to_me_definition = args.to_me_definition or "single"
    if args.scenarios:
        names = [scenario_name(n) for n in args.scenarios]
    else:
        names = sorted(scenario_name(p.name) for p in results.glob("*.labels.csv")
                       if role(scenario_name(p.name)) == "eval")
        if not names:
            raise SystemExit("평가용(_take2) 시나리오가 없습니다. 이름을 직접 지정하거나 NAME_take2 로 녹음하세요.")
    warnings = check_eval_inputs(names)
    if args.to_me_definition == "both":
        raise SystemExit("--to-me-definition both 는 --ami 보고서에서만 씁니다.")
    data = collect(names, results, args.to_me_definition)
    if not data:
        raise SystemExit("평가할 시나리오가 없습니다. replay → label 을 먼저 하세요.")
    real = {n: v for n, v in data.items() if not (v[0].get("synthetic") or is_synthetic(n))}
    syn = {n: v for n, v in data.items() if n not in real}
    if real:
        report(cfg, real, False, list(warnings), results)
    if syn:
        report(cfg, syn, True, list(warnings), results)


if __name__ == "__main__":
    main()
