"""시연 구성 적용: results/demo_selection.json(AI Hub 라벨 시뮬레이션 dev, 사전 등록 개정 1 규칙) → app/demo_config.yaml.

    python tools/apply_selection.py           # 적용 + 확인
    python tools/apply_selection.py --check   # 확인만
    python tools/apply_selection.py --revert  # demo_config.yaml 삭제 → 시연 프로필은 v1(P1c · 손 가중치)

시연 프로필(server, gpu_4060, cpu_light)에만 적용된다. AMI dev 선택(results/selection.json → app/selected_config.yaml,
P1-qwen3:4b-learned)은 보고용이며 ami 프로필에만 적용되고, 이 스크립트는 그 파일을 바꾸지 않는다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import yaml  # noqa: E402

DEMO_YAML = ROOT / "app" / "demo_config.yaml"
DEMO_PROFILES = ("server", "gpu_4060", "cpu_light")


def apply(sel: dict) -> dict:
    model = sel["model"]
    others = [m for m in ("qwen3:4b", "qwen3:1.7b", "qwen2.5:3b") if m != model]
    block = {
        "llm": {"variant": sel["variant"], "models": [model] + others},
        "policy": {"default_mode": sel.get("mode", "full"), "fusion": sel["fusion"],
                   "candidate_rejudge": bool(sel["flags"].get("candidate_rejudge", False)),
                   "short_skip_llm": bool(sel["flags"].get("short_skip_llm", False)),
                   "config_name": sel["config_name"]},
    }
    txt = ("# 자동 생성: tools/apply_selection.py ← results/demo_selection.json\n"
           "# AI Hub 라벨 시뮬레이션 dev에서 사전 등록 개정 1 규칙으로 고른 시연 구성. 시연 프로필에만 적용된다.\n"
           "# 되돌리기(v1): python tools/apply_selection.py --revert  또는 서버 --no-selected\n"
           + yaml.safe_dump(block, allow_unicode=True, sort_keys=False))
    DEMO_YAML.write_text(txt, encoding="utf-8")
    return block


def check() -> bool:
    from app.config import load_config
    from app.pipeline import config_name
    rows, keys = [], set()
    for prof in DEMO_PROFILES + ("ami",):
        c = load_config(prof)
        p, l = c["policy"], c["llm"]
        key = (l["variant"], json.dumps(p.get("fusion"), sort_keys=True), p.get("candidate_rejudge"),
               p.get("short_skip_llm"), p.get("default_mode"))
        if prof != "ami":
            keys.add(key)
        rows.append(f"  {prof:<10} 구성={config_name(c):<44} 변형={l['variant']:<4} 언어={l.get('prompt_lang')} "
                    f"모델={l['models'][0]} 모드={p.get('default_mode')}")
    print("\n".join(rows))
    same = len(keys) == 1
    print("→ 시연 프로필 " + ("모두 같은 구성 OK" if same else "프로필마다 다름!") +
          f" · 출처: {'app/demo_config.yaml' if DEMO_YAML.exists() else 'v1 기본값(demo_config.yaml 없음)'}"
          " · ami 프로필은 AMI 선택(보고용)")
    return same


def main():
    if "--revert" in sys.argv:
        if DEMO_YAML.exists():
            DEMO_YAML.unlink()
        print("되돌림: 시연 프로필 = v1")
    elif "--check" not in sys.argv:
        sel = json.loads((ROOT / "results" / "demo_selection.json").read_text(encoding="utf-8"))
        apply(sel)
        print(f"적용: {DEMO_YAML} ← {sel['config_name']}")
    sys.exit(0 if check() else 1)


if __name__ == "__main__":
    main()
