"""AI Hub 464(주요 영역별 회의 음성인식) 라벨: 구조 확인 · 전사 정제 · 세션 집계 · 후보 필터 · dev/test 분할.

    source scripts/server_env.sh
    python tools/aihub_select.py --inspect          # json 구조(키·예시)와 설명서 기준 차이, 정제 전후 예시 10개
    python tools/aihub_select.py                    # 묶음별 집계 + 후보 + splits_aihub.json

입력: $HEARME_DATA/aihub/download/<filekey>/ 아래 zip(또는 풀린 json) · $HEARME_DATA/aihub/manual/*.zip (웹에서 받은 것)
출력(저장소 밖): $HEARME_DATA/aihub/sessions.jsonl (세션별 정제 발화·통계) / 저장소: splits_aihub.json(세션 ID만), 콘솔 표
원천 음성은 쓰지 않는다(사전 등록 개정 1).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# 출력 위치(가짜 데이터 점검 때 저장소를 더럽히지 않게 바꿀 수 있다)
RES = Path(os.environ.get("AIHUB_RESULTS_DIR") or ROOT / "results")
SPLITS = Path(os.environ.get("AIHUB_SPLITS") or ROOT / "splits_aihub.json")
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

LABEL_KEYS = {61634: "TL1", 61635: "TL2", 61636: "TL3", 61637: "TL4", 61638: "TL5", 61639: "TL6", 61640: "TL7",
              563185: "VL1"}
MEDIA_PRI = ["독서토론", "회의", "온라인회의", "토론"]
WINDOW_S = 600.0
MIN_WEARER_UTTS = 15
MAX_OVERLAP_RATIO = 0.20
OVERLAP_MIN_S = 0.3
SEED = 20261002


def aihub_root() -> Path:
    d = os.environ.get("HEARME_DATA")
    if not d:
        raise SystemExit("HEARME_DATA 미설정 → source scripts/server_env.sh")
    return Path(d) / "aihub"


# ------------------------------------------------------------------ 읽기
def _zname(info: zipfile.ZipInfo) -> str:
    n = info.filename
    if not info.flag_bits & 0x800:      # 한글 파일명(cp949) 복원
        try:
            n = n.encode("cp437").decode("cp949")
        except Exception:
            pass
    return n


def bundle_of(path: str) -> str:
    m = re.search(r"\b([TV][LS]\d)\b", path.replace("/", " ").replace(".", " ").replace("_", " "))
    return m.group(1) if m else "?"


def iter_json(root: Path):
    """(묶음, 경로, dict) — zip 안의 json은 풀지 않고 읽는다."""
    srcs = sorted((root / "download").rglob("*.zip")) + sorted((root / "manual").rglob("*.zip"))
    seen = set()
    for z in srcs:
        b = bundle_of(z.name) if bundle_of(z.name) != "?" else bundle_of(str(z))
        if b == "?":
            fk = z.parent.name
            b = LABEL_KEYS.get(int(fk), "?") if fk.isdigit() else "?"
        if b.startswith(("TS", "VS")):
            continue
        with zipfile.ZipFile(z) as zf:
            for info in zf.infolist():
                n = _zname(info)
                if not n.lower().endswith(".json") or (b, Path(n).name) in seen:
                    continue
                seen.add((b, Path(n).name))
                raw = zf.read(info)
                yield b, n, json.loads(raw.decode("utf-8-sig"))
    for j in sorted((root / "download").rglob("*.json")) + sorted((root / "manual").rglob("*.json")):
        b = bundle_of(str(j))
        if (b, j.name) in seen:
            continue
        seen.add((b, j.name))
        yield b, str(j), json.loads(j.read_text(encoding="utf-8-sig"))


# ------------------------------------------------------------------ 정제
# 전사 규칙(실제 파일에서 확인 후 확정 · --inspect 로 정제 전후 예시 출력)
CLEAN_RULES = [
    ("(A)/(B) 이중 전사 → 발음 전사 대신 철자 전사(B) 선택", re.compile(r"\(([^()/]*)\)/\(([^()/]*)\)"), r"\2"),
    ("비식별화 표식 &이름& 등 → 지움", re.compile(r"&[^&\s]{0,20}&"), " "),
    ("잡음·비언어 태그 (SP:..) [..] {..} <..> → 지움", re.compile(r"\((?:SP|NO|FP|SN|BN|noise|laugh)[^)]*\)|\[[^\]]*\]|\{[^}]*\}|<[^>]*>", re.I), " "),
    ("간투사·잡음 기호 o/ n/ b/ l/ u/ → 지움", re.compile(r"(?<![가-힣A-Za-z])[onblu]/", re.I), " "),
    ("불확실 표식 * + / 단독 → 지움", re.compile(r"[*+]|(?<=\s)/(?=\s)|^/|/$"), " "),
    ("공백 정리", re.compile(r"\s+"), " "),
]


def clean_text(s: str) -> str:
    s = s or ""
    for _, rx, rep in CLEAN_RULES:
        s = rx.sub(rep, s)
    return s.strip()


# ------------------------------------------------------------------ 세션
def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def parse_session(b: str, path: str, js: dict) -> dict | None:
    meta = js.get("metadata") or js.get("meta") or {}
    spk = js.get("speaker") or js.get("speakers") or []
    utts = js.get("utterance") or js.get("utterances") or []
    if not utts:
        return None
    out = []
    for u in utts:
        t0, t1 = _f(u.get("start")), _f(u.get("end"))
        if not (t1 > t0 >= 0):
            continue
        form = u.get("form") or ""
        out.append({"id": str(u.get("id", len(out))), "start": t0, "end": t1,
                    "spk": str(u.get("speaker_id", u.get("speaker", "?"))), "role": u.get("speaker_role"),
                    "form": form, "text": clean_text(form), "env": u.get("environment") or ""})
    out.sort(key=lambda u: (u["start"], u["end"]))
    if not out:
        return None
    sid = Path(path).stem
    setting = js.get("setting") or {}
    return {"session": sid, "bundle": b, "path": path, "media": meta.get("media"), "type": meta.get("type"),
            "domain": meta.get("domain"), "topic": meta.get("topic"), "communication": meta.get("communication"),
            "speaker_num": int(meta.get("speaker_num") or len({u["spk"] for u in out})),
            "relation": setting.get("relation") or meta.get("relation"),
            "speakers": [{"id": str(s.get("id")), "role": s.get("role"), "sex": s.get("sex")} for s in spk],
            "utts": out}


def overlap_ratio(utts: list[dict]) -> float:
    """다른 화자 발화와 0.3초 이상 겹치는 발화 비율."""
    n = 0
    for i, u in enumerate(utts):
        hit = False
        for v in utts[max(0, i - 30): i + 30]:
            if v is u or v["spk"] == u["spk"]:
                continue
            if min(u["end"], v["end"]) - max(u["start"], v["start"]) >= OVERLAP_MIN_S:
                hit = True
                break
        n += hit
    return n / len(utts) if utts else 0.0


def best_window(utts: list[dict]) -> tuple[float, str, int]:
    """10분 창 중 한 화자의 발화 수가 가장 많은 (창 시작, 화자, 발화 수)."""
    best = (0.0, None, 0)
    starts = sorted({u["start"] for u in utts})
    for t0 in starts:
        c = Counter(u["spk"] for u in utts if t0 <= u["start"] and u["end"] <= t0 + WINDOW_S and u["text"])
        if c:
            s, k = c.most_common(1)[0]
            if k > best[2]:
                best = (t0, s, k)
    return best


def session_stats(s: dict) -> dict:
    utts = s["utts"]
    dur = utts[-1]["end"] - utts[0]["start"]
    gaps = [b["start"] - a["end"] for a, b in zip(utts, utts[1:]) if b["spk"] != a["spk"]]
    env = sum(1 for u in utts if u["env"]) / len(utts)
    w0, wspk, wn = best_window(utts)
    return {"duration": dur, "n_utts": len(utts), "overlap": overlap_ratio(utts), "env_ratio": env,
            "turn_gap_mean": float(np.mean(gaps)) if gaps else None, "win_start": w0, "wearer": wspk,
            "wearer_utts": wn, "long_enough": dur >= WINDOW_S}


def is_candidate(s: dict, st: dict) -> tuple[bool, str]:
    if not 3 <= s["speaker_num"] <= 4:
        return False, "speaker_num"
    if not st["long_enough"]:
        return False, "10분 미만"
    if st["overlap"] > MAX_OVERLAP_RATIO:
        return False, "겹침>20%"
    if st["wearer_utts"] < MIN_WEARER_UTTS:
        return False, "착용자 발화<15"
    return True, "ok"


def priority(s: dict, st: dict) -> tuple:
    m = s.get("media") or ""
    mp = next((i for i, k in enumerate(MEDIA_PRI) if k == m), len(MEDIA_PRI))
    rel = 0 if (s.get("relation") or "").replace(" ", "") == "사회자없이토론자" else 1
    return (mp, rel, round(st["env_ratio"], 3), s["session"])


# ------------------------------------------------------------------ 메인
def inspect(root: Path, n_files: int = 3) -> None:
    from itertools import islice
    shown = 0
    keys = Counter()
    ex = []
    for b, p, js in iter_json(root):
        keys.update(js.keys())
        for k in ("metadata", "speaker", "setting", "utterance"):
            v = js.get(k)
            if isinstance(v, dict):
                keys.update(f"{k}.{kk}" for kk in v)
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                keys.update(f"{k}[].{kk}" for kk in v[0])
        if shown < n_files:
            print(f"\n=== {b} {p}")
            short = {k: (v[:2] if isinstance(v, list) else v) for k, v in js.items()}
            print(json.dumps(short, ensure_ascii=False, indent=1)[:2500])
            shown += 1
        for u in (js.get("utterance") or []):
            f = u.get("form") or ""
            if clean_text(f) != f.strip() and len(ex) < 400:
                ex.append((f, clean_text(f), u.get("original_form")))
        if shown >= n_files and sum(keys.values()) > 0 and len(ex) >= 400:
            break
    print("\n키 빈도:", dict(keys.most_common(60)))
    want = {"metadata.media", "metadata.communication", "metadata.type", "metadata.domain", "metadata.topic",
            "metadata.speaker_num", "speaker[].id", "speaker[].role", "speaker[].sex", "setting.relation",
            "utterance[].id", "utterance[].start", "utterance[].end", "utterance[].speaker_id",
            "utterance[].speaker_role", "utterance[].form", "utterance[].original_form", "utterance[].environment"}
    miss = sorted(want - set(keys))
    print("설명서 기준 없는 키:", miss or "없음")
    print("\n정제 규칙:")
    for name, rx, rep in CLEAN_RULES:
        print(f"  - {name}: /{rx.pattern}/ → {rep!r}")
    rng = np.random.default_rng(SEED)
    print("\n정제 전후 예시 10개(바뀐 발화 중 무작위):")
    for i in rng.choice(len(ex), size=min(10, len(ex)), replace=False) if ex else []:
        f, c, o = ex[i]
        print(f"  전: {f}\n  후: {c}" + (f"\n  (original_form: {o})" if o else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--n-dev", type=int, default=24)
    ap.add_argument("--n-test", type=int, default=24)
    a = ap.parse_args()
    root = aihub_root()
    if a.inspect:
        inspect(root)
        return
    sessions, stats, why = [], {}, Counter()
    for b, p, js in iter_json(root):
        s = parse_session(b, p, js)
        if s is None:
            why["발화 없음"] += 1
            continue
        st = session_stats(s)
        ok, reason = is_candidate(s, st)
        why[reason] += 1
        s["stats"], s["candidate"] = st, ok
        sessions.append(s)
    if not sessions:
        raise SystemExit(f"[aihub] 라벨 json 없음: {root}/download, {root}/manual")
    # 묶음별 표
    by = defaultdict(list)
    for s in sessions:
        by[s["bundle"]].append(s)
    print(f"전체 세션 {len(sessions)}개 · 제외 사유 {dict(why)}\n")
    print("| 묶음 | 세션 | 후보 | 후보 총 길이(h) | 후보 media 분포 | 후보 평균 턴 간격(s) |\n|---|---:|---:|---:|---|---:|")
    for b in sorted(by):
        c = [s for s in by[b] if s["candidate"]]
        tg = [s["stats"]["turn_gap_mean"] for s in c if s["stats"]["turn_gap_mean"] is not None]
        md = ", ".join(f"{k} {v}" for k, v in Counter(s["media"] for s in c).most_common())
        print(f"| {b} | {len(by[b])} | {len(c)} | {sum(s['stats']['duration'] for s in c) / 3600:.1f} | {md or '–'} | "
              f"{np.mean(tg) if tg else float('nan'):.2f} |")
    cands = sorted([s for s in sessions if s["candidate"]], key=lambda s: priority(s, s["stats"]))
    need = 2 * (a.n_dev + a.n_test)
    print(f"\n후보 {len(cands)}개 (필요: X {a.n_dev + a.n_test} + Y {a.n_dev + a.n_test} = {need}, 우선순위 순)")
    if len(cands) < a.n_dev + a.n_test + 2:
        raise SystemExit("[aihub] 후보 부족")
    # 분할: 우선순위 상위 need개를 쓰고(부족하면 Y는 재사용), 세션 단위로 무작위 dev/test. X와 Y는 겹치지 않는다.
    pool = cands[:need] if len(cands) >= need else cands
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(pool))
    half = len(pool) // 2
    dev = [pool[i]["session"] for i in idx[:half]]
    test = [pool[i]["session"] for i in idx[half:]]
    def roles(ids, n):
        x = ids[:n]
        y = ids[n:] or ids[:n]
        return {"X": x, "Y": y}
    split = {"_doc": "AI Hub 464 라벨 기반 대화 시뮬레이션 분할(세션 ID만, 결과 보기 전 고정). 같은 세션은 X·Y 동시 사용 금지, dev·test 겹침 없음.",
             "seed": SEED, "source": "AI Hub 주요 영역별 회의 음성인식 데이터셋 활용",
             "filters": {"speaker_num": [3, 4], "max_overlap_ratio": MAX_OVERLAP_RATIO, "min_wearer_utts_10min": MIN_WEARER_UTTS,
                         "media_priority": MEDIA_PRI},
             "dev": roles(dev, a.n_dev), "test": roles(test, a.n_test)}
    for k in ("dev", "test"):
        assert not set(split[k]["X"]) & set(split[k]["Y"]) or len(pool) < need, k
    assert not (set(dev) & set(test))
    SPLITS.write_text(json.dumps(split, ensure_ascii=False, indent=1), encoding="utf-8")
    out = aihub_root() / "sessions.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for s in sessions:
            if s["session"] in set(dev) | set(test):
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"→ splits_aihub.json (dev X {len(split['dev']['X'])}/Y {len(split['dev']['Y'])}, "
          f"test X {len(split['test']['X'])}/Y {len(split['test']['Y'])}), 세션 기록 {out}")


if __name__ == "__main__":
    main()
