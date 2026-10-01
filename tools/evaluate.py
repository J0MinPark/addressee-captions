"""소거 실험: 캐시(segments.jsonl)와 라벨(labels.csv)만으로 5개 모드의 정책을 다시 돌려 지표를 낸다.
모델을 다시 부르지 않는다(정책 엔진은 순수 로직).

    python tools/evaluate.py                 # results/*.labels.csv 전부
    python tools/evaluate.py demo trap1      # 지정한 시나리오만

출력: results/ablation.md (발표용 표), results/ablation.csv
지표
  P/R/F1      수신자 판별 (라벨 y = 양성, 예측 = 최종 role partner)
  오염도       표시된 글자 중 라벨 n 인 글자의 비율
  오등록률     라벨상 대화 상대가 아닌 화자 중 partner로 등록된 비율
  등록 지연    상대의 첫 응답(y) 시작 → 등록까지 초
  LLM 정확도   착용자 직후 구간 중 LLM 결과가 있고 라벨 y/n 인 것: pair == (y)
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

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from app.config import load_config, resolve_path  # noqa: E402
from app.policy import MODE_LABELS, MODES, PolicyEngine, SegFeat  # noqa: E402

TICK = 0.5


def load_segments(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if not r.get("_meta"):
            out.append(r)
    return sorted(out, key=lambda r: (r["t_end"], r["seg_id"]))


def load_labels(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {r["seg_id"]: r["label"] for r in csv.DictReader(f)}


def simulate(cfg: dict, segs: list[dict], mode: str) -> dict:
    """정책을 캐시로 재생. 반환: {"roles": {seg_id: role}, "reg": {spk: t}, "probs": {...}}."""
    llm_available = any(s.get("llm_called") for s in segs)
    p = PolicyEngine(cfg, mode=mode, llm_available=llm_available)
    roles, probs, reg = {}, {}, {}
    last_tick = 0.0

    def handle(events):
        for e in events:
            if e["type"] == "speaker_state" and e["state"] == "partner" and e["speaker_id"] not in reg:
                reg[e["speaker_id"]] = e.get("_t", None)

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
        feat = SegFeat(seg_id=s["seg_id"], t_start=s["t_start"], t_end=s["t_end"],
                       speaker_id=s.get("speaker_id"), sim=s.get("sim", 0.0), text=s["text"])
        dec, ev = p.on_segment(feat, now=t)
        for e in ev:
            e["_t"] = t
        handle(ev)
        roles[s["seg_id"]], probs[s["seg_id"]] = dec["role"], dec["prob"]
        if s.get("name_call"):
            ev = p.on_name_call(feat.speaker_id, "name", 1.0, now=t)
            handle(ev)
        if dec["pending_llm"]:
            res = s.get("llm") if s.get("llm_called") else None
            t_llm = t + ((res or {}).get("latency_ms") or cfg["llm"]["timeout_s"] * 1000) / 1000
            upd, ev = p.on_llm_result(s["seg_id"], res, now=t_llm)
            for e in ev:
                e["_t"] = t_llm
            handle(ev)
            if upd:
                roles[s["seg_id"]], probs[s["seg_id"]] = upd["role"], upd["prob"]
    return {"roles": roles, "probs": probs, "reg": reg}


def metrics(segs: list[dict], labels: dict[str, str], sim: dict) -> dict:
    tp = fp = fn = tn = 0
    shown_chars = bad_chars = 0
    spk_labels = defaultdict(list)
    first_y = {}
    llm_ok = llm_n = 0
    for s in segs:
        if s.get("is_wearer") or s.get("skip") or not s.get("text"):
            continue
        lab = labels.get(s["seg_id"])
        if lab not in ("y", "n"):
            continue
        pos = sim["roles"].get(s["seg_id"]) == "partner"
        y = lab == "y"
        tp += pos and y
        fp += pos and not y
        fn += (not pos) and y
        tn += (not pos) and not y
        if pos:
            n = len(s["text"].replace(" ", ""))
            shown_chars += n
            bad_chars += n if not y else 0
        sid = s.get("speaker_id")
        if sid is not None:
            spk_labels[sid].append(y)
            if y and sid not in first_y:
                first_y[sid] = s["t_start"]
        if s.get("llm") and s.get("llm_eligible"):
            llm_n += 1
            llm_ok += bool(s["llm"]["pair"]) == y
    true_partners = {k for k, v in spk_labels.items() if sum(v) >= 1 and sum(v) / len(v) >= 0.5}
    non_partners = set(spk_labels) - true_partners
    registered = set(sim["reg"])
    misreg = len(registered & non_partners)
    delays = [sim["reg"][k] - first_y[k] for k in true_partners if k in sim["reg"] and k in first_y
              and sim["reg"][k] is not None]
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "shown_chars": shown_chars, "bad_chars": bad_chars,
            "n_nonpartner": len(non_partners), "misreg": misreg, "n_partner": len(true_partners),
            "unregistered": len(true_partners - registered), "delays": delays, "llm_ok": llm_ok, "llm_n": llm_n}


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
        "misreg_rate": agg["misreg"] / agg["n_nonpartner"] if agg["n_nonpartner"] else 0.0,
        "misreg": f"{int(agg['misreg'])}/{int(agg['n_nonpartner'])}",
        "reg_delay_s": sum(delays) / len(delays) if delays else None,
        "unregistered": int(agg["unregistered"]), "n_partner": int(agg["n_partner"]),
        "llm_acc": agg["llm_ok"] / agg["llm_n"] if agg["llm_n"] else None, "llm_n": int(agg["llm_n"]),
        "n_pos": int(tp + fn), "n_neg": int(agg["fp"] + agg["tn"]),
    }


def md_table(title: str, rows: dict[str, dict], scenarios: list[str]) -> str:
    pct = lambda v: "–" if v is None else f"{v * 100:.0f}%"
    out = [f"### {title}", "", f"시나리오: {', '.join(scenarios) or '없음'}", "",
           "| 모드 | 정밀도 | 재현율 | F1 | 자막 오염도 ↓ | 오등록률 ↓ | 등록 지연 | LLM 정확도 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in MODES:
        r = rows.get(mode)
        if not r:
            continue
        d = "–" if r["reg_delay_s"] is None else f"{r['reg_delay_s']:.1f}s"
        if r["unregistered"]:
            d += f" (미등록 {r['unregistered']}/{r['n_partner']})"
        llm = f"{pct(r['llm_acc'])} (n={r['llm_n']})" if mode in ("full", "semantic") else "–"
        name = f"**{MODE_LABELS[mode]}**" if mode == "full" else MODE_LABELS[mode]
        out.append(f"| {name} (`{mode}`) | {pct(r['precision'])} | {pct(r['recall'])} | {r['f1']:.2f} | "
                   f"{pct(r['contamination'])} | {pct(r['misreg_rate'])} ({r['misreg']}) | {d} | {llm} |")
    if rows:
        any_r = next(iter(rows.values()))
        out += ["", f"라벨: 양성(y) {any_r['n_pos']}개, 음성(n) {any_r['n_neg']}개"]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="*")
    ap.add_argument("--profile", default=None)
    args = ap.parse_args()
    cfg = load_config(args.profile)
    results = resolve_path(cfg, "results_dir")
    names = args.scenarios or sorted(p.name[: -len(".labels.csv")] for p in results.glob("*.labels.csv"))
    names = [Path(n).name.replace(".labels.csv", "").replace(".segments.jsonl", "") for n in names]
    data = {}
    for n in names:
        sp, lp = results / f"{n}.segments.jsonl", results / f"{n}.labels.csv"
        if not sp.exists() or not lp.exists():
            print(f"[evaluate] 건너뜀 {n}: {sp.name if not sp.exists() else lp.name} 없음")
            continue
        data[n] = (load_segments(sp), load_labels(lp))
    if not data:
        raise SystemExit("평가할 시나리오가 없습니다. replay → label 을 먼저 하세요.")

    per = {n: {m: metrics(segs, labels, simulate(cfg, segs, m)) for m in MODES} for n, (segs, labels) in data.items()}
    groups = {"전체": list(data), "함정(trap) 시나리오": [n for n in data if "trap" in n.lower()]}
    md = ["# 소거 실험 결과", "",
          f"프로필 `{cfg.get('_profile')}` · 가중치 b={cfg['policy']['bias']}, w_t={cfg['policy']['w_t']}, "
          f"w_s={cfg['policy']['w_s']}, w_l={cfg['policy']['w_l']} · 표시 임계 {cfg['policy']['show_threshold']}", ""]
    rows_csv = []
    for gname, members in groups.items():
        if not members:
            continue
        rows = {m: summarize([per[n][m] for n in members]) for m in MODES}
        md += [md_table(gname, rows, members), ""]
        for m, r in rows.items():
            rows_csv.append({"group": gname, "mode": m, **{k: v for k, v in r.items()}})
    md += ["### 시나리오별 F1", "", "| 시나리오 | " + " | ".join(MODES) + " |", "|---|" + "---:|" * len(MODES)]
    for n in data:
        md.append(f"| {n} | " + " | ".join(f"{summarize([per[n][m]])['f1']:.2f}" for m in MODES) + " |")
    (results / "ablation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with open(results / "ablation.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        w.writerows(rows_csv)
    print("\n".join(md))
    print(f"\n저장: {results / 'ablation.md'}, {results / 'ablation.csv'}")


if __name__ == "__main__":
    main()
