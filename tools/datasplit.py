"""보정용/평가용 데이터 분할 규칙.

- 이름에 `_take1` → 보정용(calibrate.py 전용), `_take2` → 평가용(evaluate.py 전용). 그 외는 '미지정'.
- calibrate.py 가 쓴 녹음은 results/calibration_used.json 에 기록되고, evaluate.py 는 그 녹음이
  평가 집합에 들어 있으면 오류로 멈춘다(같은 녹음으로 보정하고 평가하는 것 방지).
- 합성 데이터(data/NAME.json 또는 segments 메타의 "synthetic": true)는 결과물 파일 이름과 표 제목에 [SYNTHETIC].
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "results" / "calibration_used.json"
SYN = "[SYNTHETIC]"


def scenario_name(arg: str) -> str:
    n = Path(arg).name
    for suf in (".segments.jsonl", ".labels.csv", ".emb.npz", "_A.wav", "_B.wav", ".wav", ".json"):
        if n.endswith(suf):
            n = n[: -len(suf)]
    return n


def role(name: str) -> str:
    n = name.lower()
    if "_take1" in n:
        return "calib"
    if "_take2" in n:
        return "eval"
    return "untagged"


def recording_key(name: str) -> str:
    """같은 녹음 판별용 키(시나리오 이름 = data/NAME_A.wav, _B.wav 묶음)."""
    return scenario_name(name)


def is_synthetic(name: str) -> bool:
    name = scenario_name(name)
    for p in (ROOT / "data" / f"{name}.json",):
        if p.exists():
            try:
                if json.loads(p.read_text(encoding="utf-8")).get("synthetic"):
                    return True
            except Exception:
                pass
    seg = ROOT / "results" / f"{name}.segments.jsonl"
    if seg.exists():
        try:
            first = json.loads(seg.read_text(encoding="utf-8").splitlines()[0])
            if first.get("_meta") and first.get("synthetic"):
                return True
        except Exception:
            pass
    return False


def load_registry() -> set[str]:
    if not REGISTRY.exists():
        return set()
    try:
        return set(json.loads(REGISTRY.read_text(encoding="utf-8")).get("recordings", []))
    except Exception:
        return set()


def register_calibration(names: list[str]) -> None:
    used = load_registry() | {recording_key(n) for n in names}
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps({"recordings": sorted(used)}, ensure_ascii=False, indent=1), encoding="utf-8")


def check_calibration_inputs(names: list[str], allow_untagged: bool = False) -> None:
    bad = [n for n in names if role(n) == "eval"]
    if bad:
        raise SystemExit(f"[분할 오류] 평가용(_take2) 녹음은 보정에 쓸 수 없습니다: {', '.join(bad)}")
    untagged = [n for n in names if role(n) == "untagged"]
    if untagged and not allow_untagged:
        raise SystemExit(f"[분할 오류] 보정용 표시(_take1)가 없는 녹음: {', '.join(untagged)}\n"
                         f"  → 파일 이름을 NAME_take1 로 하거나, 개인 등록 녹음 등이면 --allow-untagged")


def check_eval_inputs(names: list[str]) -> list[str]:
    """평가 집합 검사. 반환: 경고 메시지들."""
    bad = [n for n in names if role(n) == "calib"]
    if bad:
        raise SystemExit(f"[분할 오류] 보정용(_take1) 녹음은 평가에 쓸 수 없습니다: {', '.join(bad)}")
    used = load_registry()
    overlap = [n for n in names if recording_key(n) in used]
    if overlap:
        raise SystemExit(f"[분할 오류] 보정에 이미 쓴 녹음이 평가 집합에 있습니다: {', '.join(overlap)} "
                         f"(기록: {REGISTRY})")
    return [f"분할 표시 없음(_take2 아님): {n}" for n in names if role(n) == "untagged"]
