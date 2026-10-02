"""AMI dev/test/calib 분할 강제.

- splits.json(커밋됨)이 유일한 기준. 회의 ID는 시나리오 이름 ami_<회의>_w<L>_<조건>_<take> 에서 읽는다.
- guard(names, "tune")  : 튜닝·학습·선택 스크립트. test(그리고 test 그룹) 회의가 하나라도 있으면 오류로 멈춤.
- guard(names, "calib") : 보정. calib 회의만 허용.
- guard(names, "test")  : 최종 시험. test 회의만, 그리고 results/final_test.lock 이 없을 때만(한 번만).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLITS = json.loads((ROOT / "splits.json").read_text(encoding="utf-8"))
LOCK = ROOT / "results" / "final_test.lock"


def meeting_of(name: str) -> str | None:
    m = re.match(r"ami_([A-Z]{2}\d{4}[a-z])", Path(name).name)
    return m.group(1) if m else None


def group_of(meeting: str) -> str:
    return meeting[:-1]


def split_of(meeting: str) -> str:
    for k in ("calib", "dev", "test"):
        if meeting in SPLITS[k]:
            return k
    for k, gs in SPLITS["groups"].items():
        if group_of(meeting) in gs:
            return f"{k}-group"   # 같은 그룹이지만 선택되지 않은 회의(사용 금지)
    return "unassigned"


def meetings(split: str) -> list[str]:
    return list(SPLITS[split])


def guard(names: list[str], purpose: str) -> None:
    ms = {n: meeting_of(n) for n in names}
    bad = {n: m for n, m in ms.items() if m is None}
    if bad:
        raise SystemExit(f"[splits] AMI 시나리오 이름이 아님: {list(bad)}")
    sp = {n: split_of(m) for n, m in ms.items()}
    if purpose == "tune":
        leak = [n for n, s in sp.items() if s.startswith("test")]
        if leak:
            raise SystemExit(f"[splits] 튜닝/학습/선택에 시험(test) 회의를 쓸 수 없습니다: {sorted(set(ms[n] for n in leak))}")
        wrong = [n for n, s in sp.items() if s != "dev"]
        if wrong:
            raise SystemExit(f"[splits] 튜닝은 dev 회의만: {sorted(set(ms[n] for n in wrong))}")
    elif purpose == "calib":
        wrong = [n for n, s in sp.items() if s != "calib"]
        if wrong:
            raise SystemExit(f"[splits] 보정은 calib 회의만: {sorted(set(ms[n] for n in wrong))}")
    elif purpose == "test":
        wrong = [n for n, s in sp.items() if s != "test"]
        if wrong:
            raise SystemExit(f"[splits] 시험은 test 회의만: {sorted(set(ms[n] for n in wrong))}")
    else:
        raise ValueError(purpose)


def acquire_test_lock(config_name: str) -> None:
    """최종 시험은 한 번만. 이미 돌렸으면 오류."""
    if LOCK.exists():
        raise SystemExit(f"[splits] 시험 세트는 이미 한 번 평가됐습니다: {LOCK.read_text(encoding='utf-8').strip()}\n"
                         f"  (다시 돌리면 시험 세트로 선택하는 셈이 된다. 정말 필요하면 lock 파일을 지우고 그 사실을 보고할 것)")
    import time
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} config={config_name}\n", encoding="utf-8")


if __name__ == "__main__":
    for k in ("calib", "dev", "test"):
        print(k, SPLITS[k], "groups:", SPLITS["groups"][k])
    # 그룹이 겹치지 않는지 자체 검사
    gs = {k: {group_of(m) for m in SPLITS[k]} for k in ("calib", "dev", "test")}
    assert not (gs["dev"] & gs["test"]) and not (gs["calib"] & (gs["dev"] | gs["test"])), gs
    print("그룹 겹침 없음 ✓")
