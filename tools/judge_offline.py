"""판정기 v2 오프라인 비교(개발 세트 전용): 재생 기록의 판정 입력을 변형·모델별로 다시 매긴다.

    python tools/judge_offline.py                                # dev, 모델 qwen3:4b qwen3:8b × 변형 P1 P2 P3
    python tools/judge_offline.py --models qwen3:4b --variants P2

입력: results/<dev 시나리오>.segments.jsonl 의 llm_input(P1·P2) / llm_input_p3(P3) — 착용자 직후 구간 전부.
출력: results/llm_v2/<모델>__<변형>.json  {시나리오: {seg_id: 결과}}
      results/judge_dev.md  (AUC, 최적 임계값 정확도, 지연, VRAM, Whisper와 동시 탑재)
      results/llm_v2/pr_curves.png
P1c(= v1, 재생 때 실시간으로 받은 결과)도 같은 표에 기준선으로 넣는다.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from app import winsetup  # noqa: E402

winsetup.setup()

import numpy as np  # noqa: E402

from app.config import load_config, resolve_path  # noqa: E402
from evaluate import load_labels, load_segments  # noqa: E402
from splits import guard, meetings  # noqa: E402


def dev_scenarios(results: Path, split: str = "dev") -> list[str]:
    ms = set(meetings(split))
    names = sorted(p.name[: -len(".segments.jsonl")] for p in results.glob("ami_*.segments.jsonl")
                   if p.name.split("_")[1] in ms and "_take2" in p.name)
    return names


def items_of(results: Path, names: list[str]) -> list[dict]:
    """판정 대상: LLM 호출 대상(착용자 직후)이었던 구간 전부(라벨 유무 무관 — 정책 시뮬레이션에 필요)."""
    out = []
    for n in names:
        _, segs = load_segments(results / f"{n}.segments.jsonl")
        labels = load_labels(results / f"{n}.labels.csv", "single")
        for s in segs:
            if s.get("llm_input"):
                out.append({"scenario": n, "seg_id": s["seg_id"], "label": labels.get(s["seg_id"]),
                            "in": s["llm_input"], "in3": s.get("llm_input_p3") or s["llm_input"], "v1": s.get("llm")})
    return out


def auc(scores: np.ndarray, y: np.ndarray) -> float | None:
    pos, neg = scores[y == 1], scores[y == 0]
    if not len(pos) or not len(neg):
        return None
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    # 동점 평균 순위
    for v in np.unique(allv):
        idx = allv == v
        ranks[idx] = ranks[idx].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pr_curve(scores, y):
    order = np.argsort(-scores)
    s, yy = scores[order], y[order]
    tp = np.cumsum(yy)
    fp = np.cumsum(1 - yy)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(yy.sum(), 1)
    return prec, rec, s


def best_threshold_acc(scores, y):
    best = (0.5, 0.0)
    for t in np.unique(np.concatenate([scores, [0.5]])):
        acc = float(((scores >= t) == (y == 1)).mean())
        if acc > best[1]:
            best = (float(t), acc)
    return best


def vram_used_mb() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().split(",")
        return int(out[0]), int(out[1])
    except Exception:
        return None


def ps_model(session, url, name):
    try:
        for m in session.get(f"{url}/api/ps", timeout=3).json().get("models", []):
            if m.get("name") == name:
                return m
    except Exception:
        pass
    return None


def run(split: str, models: list[str], variants: list[str], names: list[str] | None = None,
        out_md: str = "judge_dev.md", _final_test_ok: bool = False) -> dict:
    from app.llm_judge import LLMJudge
    cfg = load_config("ami")
    results = resolve_path(cfg, "results_dir")
    names = names or dev_scenarios(results, split)
    guard(names, "tune" if split == "dev" else "test")
    if split == "test" and not _final_test_ok:
        raise SystemExit("[splits] 시험 세트 판정은 tools/final_test.py 에서만")
    items = items_of(results, names)
    lab = [it for it in items if it["label"] in ("y", "n")]
    print(f"[judge] {split}: 시나리오 {len(names)}개 · 판정 대상 {len(items)}개(라벨 y/n {len(lab)}개)")
    outdir = results / "llm_v2" / split
    outdir.mkdir(parents=True, exist_ok=True)

    # Whisper 를 같은 GPU 에 올려 둔 채로 측정(동시 탑재 확인)
    from app.asr import WhisperASR
    base_mb = vram_used_mb()
    asr = WhisperASR(cfg, log=lambda *a: None)
    whisper_mb = vram_used_mb()
    rows, curves = [], {}
    # 기준선: v1(P1c) 실시간 결과
    if all(it["v1"] is not None or it["v1"] is None for it in items):
        sc = np.array([it["v1"]["prob"] if it["v1"] else 0.5 for it in lab])
        y = np.array([1 if it["label"] == "y" else 0 for it in lab])
        a = auc(sc, y)
        thr, acc = best_threshold_acc(sc, y)
        rows.append({"name": "qwen3:4b · P1c (v1, confidence 매핑)", "key": "v1", "auc": a, "thr": thr, "acc": acc,
                     "acc05": float(((sc >= 0.5) == (y == 1)).mean()) if len(y) else None,
                     "lat": None, "vram": None, "dev": "GPU", "fit": None, "n": len(lab)})
        curves["qwen3:4b·P1c"] = pr_curve(sc, y)
    for model in models:
        for var in variants:
            key = f"{model.replace(':', '-')}__{var}"
            c = load_config("ami", overrides={"llm": {"variant": var, "models": [model], "cache": True,
                                                      "timeout_s": 20, "warmup_timeout_s": 180}})
            j = LLMJudge(c, log=lambda *a: None)
            if not j.setup():
                print(f"  ✗ {model} {var}: {j.health_line()}")
                continue
            m = ps_model(j.session, j.url, j.model) or {}
            gpu_mb = vram_used_mb()
            res, lat, t0 = {}, [], time.perf_counter()
            for i, it in enumerate(items):
                inp = it["in3"] if var == "P3" else it["in"]
                r = j.judge([tuple(x) for x in inp["prev"]], inp["a"], inp["b"])
                if r is not None and not r.get("cached"):
                    lat.append(r["latency_ms"])
                res.setdefault(it["scenario"], {})[it["seg_id"]] = r
                if (i + 1) % 100 == 0:
                    print(f"    {model} {var}: {i + 1}/{len(items)}", flush=True)
            (outdir / f"{key}.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
            sc = np.array([(res[it["scenario"]][it["seg_id"]] or {}).get("prob", 0.5) for it in lab])
            y = np.array([1 if it["label"] == "y" else 0 for it in lab])
            a = auc(sc, y)
            thr, acc = best_threshold_acc(sc, y)
            fit = None
            if whisper_mb and gpu_mb:
                fit = f"{gpu_mb[0]}/{gpu_mb[1]}MB · " + ("동시 탑재 OK" if gpu_mb[0] < gpu_mb[1] - 300 else "VRAM 한계")
            rows.append({"name": f"{model} · {var}", "key": key, "auc": a, "thr": thr, "acc": acc,
                         "acc05": float(((sc >= 0.5) == (y == 1)).mean()) if len(y) else None,
                         "lat": float(np.median(lat)) if lat else None, "lat95": float(np.percentile(lat, 95)) if lat else None,
                         "vram": round((m.get("size_vram") or 0) / 1e9, 2), "dev": j.status.get("device"),
                         "fit": fit, "n": len(lab), "logprob": j.logprob_ok, "secs": time.perf_counter() - t0})
            curves[f"{model}·{var}"] = pr_curve(sc, y)
            print(f"  ✓ {model} {var}: AUC {a:.3f} · 최적 임계 {thr:.2f} 정확도 {acc:.0%} · 지연 중앙 {rows[-1]['lat'] or 0:.0f}ms "
                  f"· VRAM {rows[-1]['vram']}GB ({rows[-1]['dev']}) · {fit}", flush=True)
    del asr
    # PR 곡선
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 5))
        for k, (p, r, _) in curves.items():
            plt.plot(r, p, label=k)
        plt.xlabel("recall (pair)")
        plt.ylabel("precision (pair)")
        plt.title(f"Judge PR curves · AMI {split} (single definition)")
        plt.legend(fontsize=7)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / "pr_curves.png", dpi=120)
    except Exception as e:
        print(f"[judge] PR 그림 실패: {e}")
    md = [f"# 판정기 단독 비교 · AMI {split}", "",
          f"착용자 직후 구간 중 라벨 y/n(single 정의) {len(lab)}개 · 시나리오 {len(names)}개",
          f"Whisper large-v3-turbo 를 같은 GPU에 올린 상태에서 측정(Whisper 전 {base_mb}, 후 {whisper_mb} MB 사용/전체).", "",
          "| 판정기 | AUC | 임계 0.5 정확도 | 최적 임계값 | 최적 정확도 | 지연 중앙/p95 | 모델 VRAM | 적재 | 동시 탑재(Whisper+LLM) |",
          "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in rows:
        lat = "–" if r["lat"] is None else f"{r['lat']:.0f}/{r['lat95']:.0f}ms"
        a_s = "–" if r["auc"] is None else f"{r['auc']:.3f}"
        a05 = "–" if r["acc05"] is None else f"{r['acc05']:.0%}"
        vr = "–" if r["vram"] is None else f"{r['vram']}GB"
        md.append(f"| {r['name']} | {a_s} | {a05} | {r['thr']:.2f} | {r['acc']:.0%} | {lat} | {vr} | {r['dev']} | "
                  f"{r['fit'] or '–'} |")
    lp = [r.get("logprob") for r in rows if r.get("logprob") is not None]
    md += ["", f"연속 점수: Ollama logprobs 사용 {'가능(true/false 토큰 확률)' if lp and all(lp) else '불가 → confidence 매핑으로 대체'}.",
           f"![PR](llm_v2/{split}/pr_curves.png)"]
    (results / out_md).write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return {"rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen3:4b", "qwen3:8b"])
    ap.add_argument("--variants", nargs="+", default=["P1", "P2", "P3"])
    a = ap.parse_args()
    run("dev", a.models, a.variants)


if __name__ == "__main__":
    main()
