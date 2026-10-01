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
    if profile:
        if profile not in profiles:
            raise SystemExit(f"[config] 알 수 없는 프로필: {profile} (가능: {', '.join(profiles)})")
        cfg = deep_merge(cfg, profiles[profile])
    if overrides:
        cfg = deep_merge(cfg, overrides)
    cfg["_profile"] = profile or "default"
    return cfg


def resolve_path(cfg: dict, key: str) -> Path:
    """paths.* 를 프로젝트 루트 기준 절대경로로."""
    if key == "models_dir" and os.environ.get("HEARME_MODELS_DIR"):
        p = Path(os.environ["HEARME_MODELS_DIR"])
        p.mkdir(parents=True, exist_ok=True)
        return p
    p = Path(cfg["paths"][key])
    if not p.is_absolute():
        p = ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p
