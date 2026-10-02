"""(회의, 착용자) 단위 부트스트랩. 단위별 카운트를 재표본해 합산한 뒤 지표를 계산한다(벡터화).

units = {(회의, 착용자): [evaluate.metrics() 결과, ...]}  (조건 3개를 합산하면 한 단위에 여러 개)
"""
from __future__ import annotations

import numpy as np

FIELDS = ("tp", "fp", "fn", "tn", "trap_n", "trap_shown", "trapspk_n", "trapspk_reg")
METRICS = ("f05", "precision", "recall", "f1", "trap_shown_rate", "trapspk_rate")


def unit_of(name: str) -> tuple[str, str]:
    p = name.split("_")       # ami_<회의>_w<L>_<조건>_<take>
    return p[1], p[2]


def _matrix(units: dict, keys: list) -> np.ndarray:
    return np.array([[sum(m.get(f, 0) for m in units[k]) for f in FIELDS] for k in keys], dtype=float)


def _metrics(c: np.ndarray) -> dict:
    """c: (..., len(FIELDS)) 합산 카운트 → 지표 배열."""
    tp, fp, fn, tn, tn_, ts, sn, sr = (c[..., i] for i in range(len(FIELDS)))
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
        f05 = np.where(0.25 * prec + rec > 0, 1.25 * prec * rec / (0.25 * prec + rec), 0.0)
        trap = np.where(tn_ > 0, ts / tn_, np.nan)
        spk = np.where(sn > 0, sr / sn, np.nan)
    return {"f05": f05, "precision": prec, "recall": rec, "f1": f1, "trap_shown_rate": trap, "trapspk_rate": spk}


def _draws(n_units: int, B: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_units, (B, n_units))
    w = np.zeros((B, n_units))
    np.add.at(w, (np.arange(B)[:, None], idx), 1)
    return w


def bootstrap(units: dict, B: int = 1000, seed: int = 0) -> dict:
    """→ {지표: (점추정, 2.5%, 97.5%)}"""
    keys = sorted(units)
    M = _matrix(units, keys)
    pt = _metrics(M.sum(0))
    bs = _metrics(_draws(len(keys), B, seed) @ M)
    out = {}
    for m in METRICS:
        v = bs[m][~np.isnan(bs[m])]
        p = float(pt[m]) if not np.isnan(pt[m]) else None
        out[m] = (p, *(np.percentile(v, [2.5, 97.5]) if len(v) else (None, None)))
    return out


def paired(units_a: dict, units_b: dict, metric: str = "f05", B: int = 1000, seed: int = 0) -> tuple:
    """같은 재표본으로 (a − b): (점추정, 2.5%, 97.5%, a>b 인 재표본 비율)"""
    keys = sorted(set(units_a) & set(units_b))
    Ma, Mb = _matrix(units_a, keys), _matrix(units_b, keys)
    W = _draws(len(keys), B, seed)
    d = _metrics(W @ Ma)[metric] - _metrics(W @ Mb)[metric]
    d = d[~np.isnan(d)]
    d0 = float(_metrics(Ma.sum(0))[metric] - _metrics(Mb.sum(0))[metric])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return d0, float(lo), float(hi), float((d > 0).mean())


def fmt(ci: tuple, pct: bool = True) -> str:
    p, lo, hi = ci
    if p is None:
        return "–"
    if pct:
        return f"{p * 100:.1f}% [{lo * 100:.1f}, {hi * 100:.1f}]" if lo is not None else f"{p * 100:.1f}%"
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}]" if lo is not None else f"{p:.3f}"
