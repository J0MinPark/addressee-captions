"""필요한 모든 모델을 미리 받는다(행사장 네트워크를 믿지 않는다).

    python scripts/download_models.py              # 전부 (gpu + cpu 폴백 모델 포함)
    python scripts/download_models.py --check      # 받은 뒤 오프라인으로 로드 + 워밍업 시간 출력
    python scripts/download_models.py --yamnet     # YAMNet(tensorflow-hub)도 (선택)

받은 뒤에는 models/.downloaded 파일이 생기고, 이후 실행은 HF 오프라인 모드로 동작한다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def step(name, fn):
    t0 = time.perf_counter()
    print(f"→ {name} ...", flush=True)
    try:
        fn()
        print(f"  ✓ {name} ({time.perf_counter() - t0:.1f}s)")
        return True
    except Exception as e:
        print(f"  ✗ {name} 실패: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="다운로드 없이 로드/워밍업만 확인")
    ap.add_argument("--profile", default="gpu_4060")
    ap.add_argument("--yamnet", action="store_true")
    ap.add_argument("--skip-ollama", action="store_true")
    args = ap.parse_args()

    if not args.check:
        os.environ["HEARME_ONLINE"] = "1"
    from app import winsetup
    winsetup.setup()
    from app.config import load_config, resolve_path
    cfg = load_config(args.profile)
    models = resolve_path(cfg, "models_dir")

    if args.check:
        from app.pipeline import load_models
        m = load_models(cfg)
        ok = all(v is not None for v in (m.vad_b, m.embedder, m.asr, m.sound))
        print("\n워밍업 시간(s):", json.dumps(m.timings, ensure_ascii=False))
        print("사용 모델:", json.dumps(m.describe(), ensure_ascii=False))
        sys.exit(0 if ok else 1)

    results = {}

    def silero():
        from silero_vad import load_silero_vad
        load_silero_vad()   # pip 패키지에 포함(다운로드 없음)
    results["silero"] = step("Silero VAD", silero)

    def ecapa():
        from app.speaker import EcapaEmbedder
        import numpy as np
        e = EcapaEmbedder(cfg)
        e(np.random.randn(16000).astype("float32") * 0.01)
    results["ecapa"] = step("SpeechBrain ECAPA (spkrec-ecapa-voxceleb)", ecapa)

    def whisper(name):
        def f():
            from faster_whisper.utils import download_model
            download_model(name, cache_dir=str(models / "whisper"))
        return f
    names = []
    for prof in ("gpu_4060", "cpu_light"):
        c = load_config(prof)
        names += [c["asr"]["model"], c["asr"]["fallback_model"]]
    for n in dict.fromkeys(names):
        results[f"whisper:{n}"] = step(f"faster-whisper {n}", whisper(n))

    def ast():
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        ASTFeatureExtractor.from_pretrained(cfg["sound"]["ast_model"])
        ASTForAudioClassification.from_pretrained(cfg["sound"]["ast_model"])
    results["ast"] = step("AST AudioSet", ast)

    if args.yamnet:
        def yam():
            import tensorflow_hub as hub
            hub.load("https://tfhub.dev/google/yamnet/1")
        results["yamnet"] = step("YAMNet (선택)", yam)

    if not args.skip_ollama:
        import requests
        url = cfg["llm"]["url"].rstrip("/")
        want = []
        for prof in ("gpu_4060", "cpu_light"):
            want += load_config(prof)["llm"]["models"][:1]
        for name in dict.fromkeys(want):
            def pull(name=name):
                r = requests.post(f"{url}/api/pull", json={"model": name, "stream": True}, stream=True, timeout=30)
                r.raise_for_status()
                last = ""
                for line in r.iter_lines():
                    if not line:
                        continue
                    st = json.loads(line).get("status", "")
                    if st != last:
                        print("   ", st, flush=True)
                        last = st
                    if "error" in json.loads(line):
                        raise RuntimeError(json.loads(line)["error"])
            ok = step(f"Ollama {name}", pull)
            results[f"ollama:{name}"] = ok
            if not ok:
                print(f"    Ollama가 실행 중인지 확인하고 직접 받으세요:  ollama pull {name}")

    core = [k for k in results if not k.startswith("ollama") and k != "yamnet"]
    if all(results[k] for k in core):
        (models / ".downloaded").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n완료. 이후 실행은 오프라인으로 동작합니다. ({models / '.downloaded'})")
    else:
        print("\n일부 실패:", [k for k in core if not results[k]])
    print("다음: python scripts/download_models.py --check")


if __name__ == "__main__":
    main()
