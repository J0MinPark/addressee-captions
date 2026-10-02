"""AI Hub 라벨 기반 대화 시뮬레이션(사전 등록 개정 1). 오디오 없이 판정 계층만 평가한다.

    source scripts/server_env.sh
    python tools/aihub_sim.py --split dev --dry-run      # 시나리오 통계 + 예상 시간만
    python tools/aihub_sim.py --split dev                # dev 24개 생성 + 판정(P1c, P1, P2, P3)
    (test 는 tools/final_test_aihub_sim.py 안에서만)

시나리오 = X 세션 10분(착용자 = 그 구간 발화가 가장 많은 화자) + Y 세션 10분(시작 오프셋 무작위), 실제 시각 기준으로 합침.
- 각 발화 = 한 구간(정답 분할), 텍스트 = 정제한 form. 라벨: w 착용자 / x X의 다른 참가자(→ 평가 라벨 y) / y Y 발화(→ n).
- 화자: 정답 화자 ID(같으면 유사도 1). 착용자는 정답 시간(본인 발화 검출 오류 없음).
- 파이프라인과 같은 순서(구간 끝 시각)로 정책 엔진을 한 번 돌려 gap·llm_eligible·LLM 입력(P1/P2: llm_context, P3: recent_turns)을
  만든다. 판정은 실제 Ollama(qwen3:4b, 한국어 프롬프트). P1c 결과는 기록의 llm 에, P1/P2/P3 는 llm_v2/<모델>__<변형>.json 에.
- 자연 함정(착용자 발화 종료 후 0~1.5초 안에 시작한 y 발화)이 5개 미만이면 Y 오프셋을 다시 뽑는다(최대 5회).
출력: $HEARME_DATA/aihub/sim/<split>/ (저장소 밖 — 라이선스). 저장소에는 수치 표만.
이름: aihubsim_<X세션>_w<착용자>_<Y세션>_<split>  (stats_boot.unit_of → (X세션, w착용자) = 부트스트랩·교차검증 단위)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 출력 위치(가짜 데이터 점검 때 저장소를 더럽히지 않게 바꿀 수 있다)
RES = Path(os.environ.get("AIHUB_RESULTS_DIR") or ROOT / "results")
SPLITS = Path(os.environ.get("AIHUB_SPLITS") or ROOT / "splits_aihub.json")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402

from app.config import load_config  # noqa: E402
from app.llm_judge import LLMJudge, recent_turns  # noqa: E402
from app.policy import PolicyEngine, SegFeat  # noqa: E402

SEED = 20261002
WINDOW = 600.0
LEAD = 1.0
MIN_TRAPS = 5
MAX_RESAMPLE = 5
MODEL = "qwen3:4b"


def sim_root(split: str) -> Path:
    p = Path(os.environ["HEARME_DATA"]) / "aihub" / "sim" / split
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣-]", "-", s)


def load_sessions() -> dict:
    p = Path(os.environ["HEARME_DATA"]) / "aihub" / "sessions.jsonl"
    if not p.exists():
        raise SystemExit("세션 기록 없음 → python tools/aihub_select.py")
    return {s["session"]: s for s in map(json.loads, p.read_text(encoding="utf-8").splitlines())}


def window(utts, t0, t1):
    return [u for u in utts if t0 <= u["start"] and u["end"] <= t1]


def compose(xs: dict, ys: dict, rng) -> tuple[list[dict], dict]:
    """X 10분 + Y 10분 → 발화 목록(시나리오 시각). Y 오프셋은 함정 기준을 만족할 때까지 최대 5회 다시 뽑는다."""
    st = xs["stats"]
    w0, wearer = st["win_start"], st["wearer"]
    xu = window(xs["utts"], w0, w0 + WINDOW)
    yd = ys["utts"][-1]["end"]
    tries = []
    for k in range(MAX_RESAMPLE):
        off = float(rng.uniform(ys["utts"][0]["start"], max(ys["utts"][0]["start"], yd - WINDOW)))
        yu = window(ys["utts"], off, off + WINDOW)
        utts = []
        for u in xu:
            lab = "w" if u["spk"] == wearer else "x"
            utts.append(dict(u, t0=u["start"] - w0 + LEAD, t1=u["end"] - w0 + LEAD, lab=lab, who=f"X:{u['spk']}"))
        for u in yu:
            utts.append(dict(u, t0=u["start"] - off + LEAD, t1=u["end"] - off + LEAD, lab="y", who=f"Y:{u['spk']}"))
        utts.sort(key=lambda u: (u["t0"], u["t1"]))
        wends = [u["t1"] for u in utts if u["lab"] == "w"]
        traps = sum(1 for u in utts if u["lab"] == "y" and any(0 <= u["t0"] - e <= 1.5 for e in wends))
        tries.append((traps, off, utts))
        if traps >= MIN_TRAPS:
            break
    traps, off, utts = max(tries, key=lambda t: t[0]) if tries[-1][0] < MIN_TRAPS else tries[-1]
    return utts, {"x_session": xs["session"], "y_session": ys["session"], "wearer": wearer, "x_win_start": w0,
                  "y_offset": round(off, 2), "resamples": len(tries) - 1, "natural_traps": traps,
                  "trap_ok": traps >= MIN_TRAPS}


def build_records(cfg: dict, utts: list[dict]) -> tuple[list[dict], dict]:
    """파이프라인과 같은 순서(구간 끝)로 정책 엔진을 돌려 구간 기록과 판정 입력을 만든다."""
    spk_ids, records, labels = {}, [], {}
    p = PolicyEngine(cfg, mode="full", llm_available=False)
    lc = cfg["llm"]
    turn_rec = {}
    nw = ns = 0
    for u in sorted(utts, key=lambda u: (u["t1"], u["t0"])):
        if u["lab"] == "w":
            nw += 1
            sid = f"w{nw:05d}"
            turn_id, _ = p.on_wearer_end(u["t0"], u["t1"], now=u["t1"])
            p.set_wearer_text(turn_id, u["text"])
            turn_rec[turn_id] = sid
            rec = {"seg_id": sid, "t_start": round(u["t0"], 3), "t_end": round(u["t1"], 3), "is_wearer": True,
                   "wearer_by": "truth", "turn_id": turn_id, "text": u["text"]}
            # 텍스트가 비어도 skip 하지 않는다: 파이프라인은 착용자 턴을 ASR과 무관하게 정책에 넣는다(턴 번호 일치)
            records.append(rec)
            labels[sid] = "w"
            continue
        ns += 1
        sid = f"s{ns:05d}"
        spk = spk_ids.setdefault(u["who"], len(spk_ids) + 1)
        rec = {"seg_id": sid, "t_start": round(u["t0"], 3), "t_end": round(u["t1"], 3), "is_wearer": False,
               "speaker_id": spk, "sim": 1.0, "new_speaker": False, "text": u["text"], "src": u["lab"]}
        labels[sid] = "y" if u["lab"] == "x" else "n"
        records.append(rec)
        if not u["text"]:
            rec["skip"] = True
            continue
        T, gap, turn_id = p.timing(u["t0"])
        rec["gap"] = None if gap is None else round(gap, 3)
        rec["turn_id"] = turn_id
        rec["wearer_seg"] = turn_rec.get(turn_id)
        p.on_segment(SegFeat(seg_id=sid, t_start=u["t0"], t_end=u["t1"], speaker_id=spk, sim=1.0, text=u["text"]),
                     now=u["t1"])
        eligible = gap is not None and cfg["policy"]["timing_early_s"] <= gap <= lc["call_window_s"]
        rec["llm_eligible"] = eligible
        if eligible:
            a_text, prev = p.llm_context(turn_id, lc["context_turns"])
            if a_text:
                rec["llm_input"] = {"prev": prev, "a": a_text, "b": u["text"]}
                a3, prev3 = recent_turns([r for r in records if r.get("t_start") is not None], rec)
                rec["llm_input_p3"] = {"prev": prev3, "a": a3 or a_text, "b": u["text"]}
    records.sort(key=lambda r: (r["t_end"], r["seg_id"]))
    return records, labels


def judge_all(cfg: dict, items: list[tuple], variants: list[str], workers: int = 1) -> dict:
    """items: [(시나리오, 구간 기록)] → {변형: {시나리오: {seg_id: 결과}}}"""
    out = {}
    for var in variants:
        c = load_config("server", overrides={"llm": {"variant": var, "models": [MODEL], "cache": True,
                                                    "timeout_s": 30, "warmup_timeout_s": 180,
                                                    "prompt_lang": "ko"}})
        j = LLMJudge(c, log=lambda *a: None)
        if not j.setup():
            raise SystemExit(f"[sim] Ollama 실패: {j.health_line()}")
        t0 = time.perf_counter()
        res: dict = {}

        def one(it):
            n, r = it
            inp = r["llm_input_p3"] if var == "P3" else r["llm_input"]
            return n, r["seg_id"], j.judge([tuple(x) for x in inp["prev"]], inp["a"], inp["b"])
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, (n, sid, r) in enumerate(ex.map(one, items)):
                res.setdefault(n, {})[sid] = r
                if (i + 1) % 200 == 0:
                    el = time.perf_counter() - t0
                    print(f"  [{var}] {i + 1}/{len(items)} · {el / (i + 1) * 1000:.0f}ms/건 · 남은 "
                          f"{el / (i + 1) * (len(items) - i - 1) / 60:.1f}분", flush=True)
        fails = sum(1 for n in res for v in res[n].values() if v is None)
        print(f"  [{var}] {len(items)}건 {(time.perf_counter() - t0) / 60:.1f}분 · 실패 {fails} · "
              f"logprobs {j.logprob_ok} · 프롬프트 {j.prompt_version}", flush=True)
        out[var] = res
    return out


def build_split(split: str, variants: list[str], dry_run: bool = False, workers: int = 1,
                _final_test_ok: bool = False) -> dict:
    if split == "test" and not _final_test_ok:
        raise SystemExit("[sim] 시험(test) 시나리오는 tools/final_test_aihub_sim.py 에서만 만든다")
    spl = json.loads(SPLITS.read_text(encoding="utf-8"))[split]
    other = json.loads(SPLITS.read_text(encoding="utf-8"))["test" if split == "dev" else "dev"]
    used = set(other["X"]) | set(other["Y"])
    assert not (set(spl["X"]) | set(spl["Y"])) & used, "dev/test 세션 겹침"
    assert not set(spl["X"]) & set(spl["Y"]), "같은 세션이 X와 Y"
    sessions = load_sessions()
    cfg = load_config("server", overrides={"namecall": {"enabled": False}, "sound": {"enabled": False}})
    rng = np.random.default_rng(SEED + (0 if split == "dev" else 1))
    out_dir = sim_root(split)
    scen, items, stats = {}, [], []
    for i, xid in enumerate(spl["X"]):
        yid = spl["Y"][i % len(spl["Y"])]
        utts, meta = compose(sessions[xid], sessions[yid], rng)
        name = f"aihubsim_{safe(xid)}_w{safe(meta['wearer'])}_{safe(yid)}_{split}"
        recs, labels = build_records(cfg, utts)
        meta.update(name=name, split=split, n_w=sum(v == "w" for v in labels.values()),
                    n_x=sum(v == "y" for v in labels.values()), n_y=sum(v == "n" for v in labels.values()),
                    n_eligible=sum(1 for r in recs if r.get("llm_input")),
                    n_empty=sum(1 for r in recs if r.get("skip")))
        stats.append(meta)
        scen[name] = (recs, labels, meta)
        items += [(name, r) for r in recs if r.get("llm_input")]
    n_calls = len(items) * len(variants)
    print(f"[sim] {split}: 시나리오 {len(scen)}개 · 판정 대상 {len(items)}건 × 변형 {len(variants)} = {n_calls}회 호출")
    if dry_run:
        return {"stats": stats, "items": len(items)}
    judged = judge_all(cfg, items, variants, workers)
    vdir = out_dir / "llm_v2"
    vdir.mkdir(exist_ok=True)
    for var, res in judged.items():
        if var != "P1c":
            (vdir / f"{MODEL.replace(':', '-')}__{var}.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    p1c = judged.get("P1c", {})
    for name, (recs, labels, meta) in scen.items():
        for r in recs:
            if r.get("llm_input") and "P1c" in judged:
                r["llm_called"] = True
                r["llm"] = p1c.get(name, {}).get(r["seg_id"])
        with open(out_dir / f"{name}.segments.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"_meta": True, "run": name, "profile": "aihub_sim", "llm_model": MODEL,
                                "llm_variant": "P1c", "language": "ko", "synthetic": False, "sim": meta},
                               ensure_ascii=False) + "\n")
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(out_dir / f"{name}.labels.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["seg_id", "label"])
            for k, v in labels.items():
                w.writerow([k, v])
    (out_dir / "scenarios.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"stats": stats, "items": len(items)}


def print_stats(stats: list[dict]) -> None:
    print("| 시나리오 | w | x | y | 판정 대상 | 자연 함정 | 재추첨 |\n|---|---:|---:|---:|---:|---:|---:|")
    for s in stats:
        print(f"| {s['name'][:60]} | {s['n_w']} | {s['n_x']} | {s['n_y']} | {s['n_eligible']} | {s['natural_traps']}"
              f"{'' if s['trap_ok'] else ' ⚠'} | {s['resamples']} |")
    t = lambda k: sum(s[k] for s in stats)  # noqa: E731
    print(f"| 합계 | {t('n_w')} | {t('n_x')} | {t('n_y')} | {t('n_eligible')} | {t('natural_traps')} | |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev"])
    ap.add_argument("--variants", nargs="+", default=["P1c", "P1", "P2", "P3"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    r = build_split(a.split, a.variants, a.dry_run, a.workers)
    print_stats(r["stats"])


if __name__ == "__main__":
    main()
