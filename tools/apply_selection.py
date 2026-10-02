"""results/selection.json(개발 세트에서 고른 구성)을 app/selected_config.yaml 로 적용하고,
한국어 시연 프로필과 AMI 평가 프로필이 같은 구성인지 확인한다.

    python tools/apply_selection.py           # 적용 + 확인
    python tools/apply_selection.py --check   # 확인만
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

SEL_YAML = ROOT / "app" / "selected_config.yaml"


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
    txt = ("# 자동 생성: tools/apply_selection.py ← results/selection.json (개발 세트에서 고른 최종 구성)\n"
           "# 모든 프로필(gpu_4060·cpu_light 한국어 시연, ami 평가)에 같이 적용된다. 언어(prompt_lang)만 프로필이 정한다.\n"
           + yaml.safe_dump(block, allow_unicode=True, sort_keys=False))
    SEL_YAML.write_text(txt, encoding="utf-8")
    return block


def check() -> bool:
    from app.config import load_config
    from app.pipeline import config_name
    rows, keys = [], set()
    for prof in (None, "gpu_4060", "cpu_light", "ami"):
        c = load_config(prof)
        p, l = c["policy"], c["llm"]
        key = (l["variant"], json.dumps(p.get("fusion"), sort_keys=True), p.get("candidate_rejudge"),
               p.get("short_skip_llm"), p.get("default_mode"))
        keys.add(key)
        rows.append(f"  {prof or 'default':<10} 구성={config_name(c):<40} 언어={l.get('prompt_lang')} "
                    f"모델={l['models'][0]} 모드={p.get('default_mode')}")
    print("\n".join(rows))
    same = len(keys) == 1
    print("→ 판정기 변형·융합 가중치·플래그·모드가 " + ("모든 프로필에서 같음 OK" if same else "프로필마다 다름!"))
    print("  (cpu_light 는 같은 구성에서 LLM 모델 우선순위만 가벼운 쪽 — 의도된 차이)" if same else "")
    return same


def main():
    if "--check" not in sys.argv:
        sel = json.loads((ROOT / "results" / "selection.json").read_text(encoding="utf-8"))
        apply(sel)
        print(f"적용: {SEL_YAML} ← {sel['config_name']}")
    check()


if __name__ == "__main__":
    main()
