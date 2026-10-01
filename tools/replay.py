"""녹음 파일로 파이프라인 실행 → results/NAME.segments.jsonl (+ NAME.emb.npz 특징 캐시)

    python tools/replay.py data/demo                 # 최대 속도, 화면 없음 (평가용)
    python tools/replay.py data/demo data/trap1      # 여러 개
    python tools/replay.py data/demo --realtime      # 실시간 속도 + 대시보드/폰 (라이브 시연 백업)

segments.jsonl 에는 구간마다 시간, 화자 ID/유사도, ASR 텍스트, 착용자 직전 발화, LLM 결과가 들어간다.
LLM은 소거 모드와 무관하게 착용자 직후 구간 전부에 호출해 캐시한다 → tools/evaluate.py 가 모델 없이 정책만 다시 돌린다.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import winsetup  # noqa: E402

winsetup.setup()

from app.audio_source import ReplaySource  # noqa: E402
from app.config import load_config, resolve_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefixes", nargs="+", help="data/NAME (data/NAME_A.wav, _B.wav)")
    ap.add_argument("--realtime", action="store_true", help="실시간 재생 + 서버(대시보드/폰)")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--no-llm-cache", action="store_true", help="LLM 판정 캐시 끄기(매번 새로 호출)")
    ap.add_argument("--loop", action="store_true", help="--realtime 과 함께: 무한 반복")
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()
    over = {"llm": {"always_call": True}}
    if args.no_llm:
        over["llm"]["enabled"] = False
    if args.no_llm_cache:
        over["llm"]["cache"] = False
    cfg = load_config(args.profile, overrides=over)
    results = resolve_path(cfg, "results_dir")

    from app.pipeline import Pipeline, load_models
    models = load_models(cfg)

    if args.realtime:
        from app.server import serve
        prefix = args.prefixes[0]
        src = ReplaySource(prefix, cfg, realtime=True, loop=args.loop)
        out = results / f"{src.name}.segments.jsonl"

        def done(pipe):
            n = pipe.write_segments(out)
            print(f"\n[replay] 재생 끝. {n}개 구간 → {out}  (서버는 계속 켜져 있음, Ctrl+C로 종료)")
            pipe.emit({"type": "end"})

        print(f"[replay] {prefix} 실시간 재생 ({src.duration:.1f}s)")
        serve(cfg, src, models=models, mode=args.mode, port=args.port, on_finished=done,
              run_name=f"replay_{src.name}_{time.strftime('%H%M%S')}", record_segments=True)
        return

    for prefix in args.prefixes:
        src = ReplaySource(prefix, cfg, realtime=False)
        t0 = time.perf_counter()
        calls0 = getattr(models.judge, "calls", 0)
        hits0 = getattr(models.judge, "cache_hits", 0)
        pipe = Pipeline(cfg, src, models, mode=args.mode, record_segments=True,
                        run_name=f"replay_{src.name}_{time.strftime('%H%M%S')}")
        n_ev = {"caption": 0, "caption_update": 0, "alert": 0, "partner_added": 0}

        def count(ev, n_ev=n_ev):
            if ev["type"] in n_ev:
                n_ev[ev["type"]] += 1
            if ev["type"] in ("alert", "partner_added"):
                print(f"   [{ev.get('t', 0):7.2f}s] {ev['type']}: {ev.get('label')}")
        pipe.add_listener(count)
        pipe.start()
        ok = pipe.wait_finished(timeout=max(600, src.duration * 5))
        pipe.stop()
        out = results / f"{src.name}.segments.jsonl"
        n = pipe.write_segments(out)
        pipe.close()
        dt = time.perf_counter() - t0
        j = models.judge
        if j is not None:
            lat = list(j.lat_ms)[-max(j.calls - calls0, 0):] if j.calls > calls0 else []
            print(f"[replay] LLM: {j.health_line()} · 이번 호출 {j.calls - calls0}회, 캐시 적중 {j.cache_hits - hits0}회, "
                  f"지연 평균 {sum(lat) / len(lat):.0f}ms" if lat else
                  f"[replay] LLM: {j.health_line()} · 이번 호출 {j.calls - calls0}회, 캐시 적중 {j.cache_hits - hits0}회")
        print(f"[replay] {src.name}: {src.duration:.1f}s 오디오 → {dt:.1f}s 처리 (x{src.duration / max(dt, 1e-6):.1f}), "
              f"{n}개 구간 → {out}  이벤트 {n_ev}" + ("" if ok else "  [경고: 시간 초과]"))
    print("다음: python tools/label.py " + " ".join(args.prefixes))


if __name__ == "__main__":
    main()
