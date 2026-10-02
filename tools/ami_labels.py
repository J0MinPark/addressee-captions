"""AMI 정답 라벨 자동 생성: replay 구간 ↔ 대화행위(NXT 단어 타임스탬프·화자·addressee).

    python tools/ami_labels.py ami_ES2008b_wA_clean_take2 [...]

발화 단위: 같은 화자의 연속 대화행위 중 addressee가 같고 사이 간격이 0.5초 이하인 것은 한 발화로 합친다.
구간 ↔ 발화: 시간 겹침이 가장 큰 발화. 겹침이 구간 길이의 50% 미만이면 unmatched → 라벨 's'(평가 제외, 개수 보고).
라벨(착용자 W 기준)
  w  W 본인의 발화
  y  addressee 가 W 한 명
  g  addressee 가 W를 포함한 2명 이상(그룹 전체)
  n  다른 사람에게, 또는 addressee 없음
결과: results/<시나리오>.labels.csv (label.py와 같은 형식) + results/<시나리오>.labelstats.json
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

MERGE_GAP = 0.5
MIN_OVERLAP = 0.5


def utterances(das: list[dict]) -> list[dict]:
    out = []
    for d in sorted(das, key=lambda x: (x["speaker"], x["start"])):
        last = out[-1] if out else None
        if (last and last["speaker"] == d["speaker"] and sorted(last["addressee"]) == sorted(d["addressee"])
                and d["start"] - last["end"] <= MERGE_GAP):
            last["end"] = max(last["end"], d["end"])
            last["text"] = (last["text"] + " " + d["text"]).strip()
            last["n_da"] += 1
        else:
            out.append(dict(d, n_da=1))
    return sorted(out, key=lambda x: x["start"])


def label_for(u: dict, wearer: str) -> str:
    if u["speaker"] == wearer:
        return "w"
    addr = set(u["addressee"])
    if addr == {wearer}:
        return "y"
    if wearer in addr and len(addr) >= 2:
        return "g"
    return "n"


def make_labels(name: str, data: Path, results: Path) -> dict:
    meta = json.loads((data / f"{name}.json").read_text(encoding="utf-8"))
    W, mid = meta["wearer"], meta["meeting"]
    das = json.loads((data / f"ami_{mid}.das.json").read_text(encoding="utf-8"))["das"]
    utts = utterances(das)
    segs = [json.loads(l) for l in (results / f"{name}.segments.jsonl").read_text(encoding="utf-8").splitlines()]
    segs = [s for s in segs if not s.get("_meta")]
    rows, stats = [], Counter()
    for s in segs:
        if s.get("skip") and not s.get("is_wearer"):
            continue   # 텍스트 없는 구간(평가에서도 제외됨)
        dur = max(s["t_end"] - s["t_start"], 1e-6)
        best, bu = 0.0, None
        for u in utts:
            if u["end"] < s["t_start"] or u["start"] > s["t_end"]:
                continue
            ov = min(s["t_end"], u["end"]) - max(s["t_start"], u["start"])
            if ov > best:
                best, bu = ov, u
        if bu is None or best / dur < MIN_OVERLAP:
            lab = "s"
            stats["unmatched"] += 1
        else:
            lab = label_for(bu, W)
        kind = "wearer_seg" if s.get("is_wearer") else "other_seg"
        stats[f"{kind}:{lab}"] += 1
        if s.get("is_wearer"):
            lab = "w" if lab != "s" else "s"   # 시스템이 착용자로 본 구간은 평가 대상이 아니다
        rows.append([s["seg_id"], lab, s["t_start"], s["t_end"], s.get("speaker_id"),
                     int(bool(s.get("is_wearer"))), s.get("text") or ""])
    with open(results / f"{name}.labels.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seg_id", "label", "t_start", "t_end", "speaker_id", "is_wearer", "text"])
        w.writerows(rows)
    # 정답(발화 단위) 분포: 착용자 W 관점
    gt = Counter(label_for(u, W) for u in utts)
    out = {"scenario": name, "meeting": mid, "wearer": W, "segments": len(rows),
           "unmatched": stats["unmatched"], "seg_labels": dict(stats), "utterance_labels": dict(gt),
           "utterances": len(utts)}
    (results / f"{name}.labelstats.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main():
    from app.config import load_config, resolve_path
    cfg = load_config("ami")
    data, results = resolve_path(cfg, "data_dir"), resolve_path(cfg, "results_dir")
    for n in sys.argv[1:]:
        st = make_labels(n, data, results)
        print(f"{n}: 구간 {st['segments']}개, unmatched {st['unmatched']}, 구간 라벨 {st['seg_labels']}")


if __name__ == "__main__":
    main()
