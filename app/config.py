"""config.yaml 로딩. 프로필을 기본값 위에 deep merge 한다."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(profile: Optional[str] = None, path: Optional[str] = None,
                overrides: Optional[dict] = None) -> dict:
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    profiles = raw.pop("profiles", {}) or {}
    cfg = raw
    profile = profile or os.environ.get("HEARME_PROFILE")
    # 선택된 구성(자동 생성 파일). HEARME_NO_SELECTED=1(--no-selected)이면 둘 다 쓰지 않는다(v1 구성).
    # - selected_config.yaml: AMI dev에서 고른 구성. AMI 평가 프로필(ami)에만 적용(보고·재현용).
    # - demo_config.yaml: AI Hub dev 규칙(results/preregistration.md)으로 고른 시연 구성. 시연 프로필에만 적용.
    #   없으면 시연 프로필은 v1 구성(P1c · 손 가중치)이다.
    sel_name = "selected_config.yaml" if profile == "ami" else "demo_config.yaml"
    sel = Path(__file__).resolve().parent / sel_name
    if sel.exists() and os.environ.get("HEARME_NO_SELECTED") != "1":
        with open(sel, "r", encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    if profile:
        if profile not in profiles:
            raise SystemExit(f"[config] 알 수 없는 프로필: {profile} (가능: {', '.join(profiles)})")
        cfg = deep_merge(cfg, profiles[profile])
    if overrides:
        cfg = deep_merge(cfg, overrides)
    if os.environ.get("OLLAMA_URL"):   # 서버: 우리 Ollama 인스턴스(예: http://127.0.0.1:11435)
        cfg["llm"]["url"] = os.environ["OLLAMA_URL"]
    cfg["_profile"] = profile or "default"
    return cfg


def data_root() -> Optional[Path]:
    """HEARME_DATA: 저장소 밖 데이터 루트(AI Hub·DEMAND·시나리오). 없으면 None(노트북: 저장소 data/)."""
    v = os.environ.get("HEARME_DATA")
    if not v:
        return None
    p = Path(v).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_path(cfg: dict, key: str) -> Path:
    """paths.* 를 프로젝트 루트 기준 절대경로로."""
    env = {"models_dir": "HEARME_MODELS_DIR", "data_dir": "HEARME_DATA_DIR"}.get(key)
    if env and os.environ.get(env):
        p = Path(os.environ[env])
        p.mkdir(parents=True, exist_ok=True)
        return p
    if key == "data_dir" and data_root() is not None:   # HEARME_DATA/data
        p = data_root() / "data"
        p.mkdir(parents=True, exist_ok=True)
        return p
    p = Path(cfg["paths"][key])
    if not p.is_absolute():
        p = ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p
