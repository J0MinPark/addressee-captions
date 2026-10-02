"""서버 환경 점검(PASS/FAIL 표) + 환경 기록.

    source scripts/server_env.sh && python tools/env_check.py               # 점검 표
    python tools/env_check.py --manifest     # + results/env_manifest_server.json, requirements-server.lock.txt

항목: GPU 고정(우리 프로세스가 GPU 2에만), CUDA(torch가 Blackwell 커널 실행), Whisper·AST가 GPU 2에 올라감,
Ollama(OLLAMA_URL, 러너가 GPU 2에만), 모델 다이제스트, logprobs(p_pair), DEMAND 소음, 단위 테스트.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import winsetup  # noqa: E402

winsetup.setup()

rows: list[tuple[str, str, str]] = []
manifest: dict = {}


def add(name: str, ok, detail: str) -> None:
    rows.append((name, "PASS" if ok is True else ("WARN" if ok is None else "FAIL"), detail))


def nvml_procs() -> dict[int, list[dict]]:
    """GPU 번호(nvidia-smi) → 그 GPU의 프로세스 목록."""
    import pynvml
    pynvml.nvmlInit()
    out = {}
    for i in range(pynvml.nvmlDeviceGetCount()):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        ps = []
        for fn in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
            try:
                ps += [{"pid": int(p.pid), "mem_mb": round((p.usedGpuMemory or 0) / 2**20)} for p in getattr(pynvml, fn)(h)]
            except Exception:
                pass
        out[i] = ps
    return out


def where(pid: int, procs: dict) -> list[int]:
    return sorted(i for i, ps in procs.items() if any(p["pid"] == pid for p in ps))


def ollama_pids() -> list[int]:
    """pid 파일의 우리 ollama serve 와 그 자식(러너)."""
    port = os.environ.get("HEARME_OLLAMA_PORT", "11435")
    pidf = ROOT / "logs" / "run" / f"ollama_{port}.pid"
    if not pidf.exists():
        return []
    pid = int(pidf.read_text().strip())
    kids = subprocess.run(["ps", "-o", "pid=", "--ppid", str(pid)], capture_output=True, text=True).stdout.split()
    return [pid] + [int(k) for k in kids]


def hf_rev(path: Path) -> str | None:
    ref = path / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="store_true", help="환경 기록 파일도 쓴다")
    ap.add_argument("--no-tests", action="store_true")
    args = ap.parse_args()
    from app.config import load_config
    from app.gpumon import gpu_index
    cfg = load_config("server")
    gidx = gpu_index()

    # 1. GPU 고정
    import torch
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    order = os.environ.get("CUDA_DEVICE_ORDER")
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    import pynvml
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(gidx) if gidx is not None else None
    want_bus = pynvml.nvmlDeviceGetPciInfo(h).busId if h else None
    want_uuid = pynvml.nvmlDeviceGetUUID(h) if h else None
    want_bus, want_uuid = [v.decode() if isinstance(v, bytes) else v for v in (want_bus, want_uuid)]
    t_uuid = str(getattr(torch.cuda.get_device_properties(0), "uuid", "")) if n else ""
    add("GPU 고정(UUID)", bool(t_uuid) and want_uuid is not None and want_uuid.endswith(t_uuid),
        f"torch cuda:0 = GPU-{t_uuid[:8]}… / nvidia-smi GPU {gidx} = {str(want_uuid)[:12]}… ({want_bus})")

    # 2. CUDA
    ok_cuda, det = False, ""
    try:
        cap = torch.cuda.get_device_capability(0)
        x = torch.randn(256, 256, device="cuda")
        y = (x @ x).sum().item()
        ok_cuda = bool(y == y)
        det = f"torch {torch.__version__} CUDA {torch.version.cuda} sm_{cap[0]}{cap[1]} 행렬곱 OK"
    except Exception as e:
        det = f"{type(e).__name__}: {e}"
    add("CUDA(torch 커널)", ok_cuda, det)

    # 3. Whisper, AST (실제로 GPU 2에 올라가는지: NVML 프로세스 목록)
    me = os.getpid()
    asr = snd = None
    try:
        from app.asr import WhisperASR
        asr = WhisperASR(cfg, lambda *a: None)
        import numpy as np
        asr.transcribe(np.zeros(16000, np.float32))
    except Exception as e:
        add("Whisper", False, f"{type(e).__name__}: {e}")
    try:
        from app.sound_events import make_sound_classifier
        snd = make_sound_classifier(cfg, lambda *a: None)
    except Exception as e:
        add("AST", False, f"{type(e).__name__}: {e}")
    time.sleep(0.5)
    procs = nvml_procs()
    on = where(me, procs)
    if asr is not None:
        add("Whisper", "cuda" in str(getattr(asr, "desc", "")) and on == [gidx],
            f"{getattr(asr, 'desc', None)} · 이 프로세스가 있는 GPU {on}")
    if snd is not None:
        add("AST", getattr(snd, "name", "").endswith("cuda") and on == [gidx],
            f"{getattr(snd, 'name', None)} · 이 프로세스가 있는 GPU {on}")

    # 4. Ollama
    from app.llm_judge import LLMJudge
    url = cfg["llm"]["url"]
    jcfg = load_config("server", overrides={"llm": {"variant": "P1"}})
    j = LLMJudge(jcfg, lambda *a: None)
    ver = None
    try:
        ver = j.session.get(f"{url}/api/version", timeout=2).json().get("version")
    except Exception:
        pass
    ok = j.setup()
    time.sleep(0.5)
    procs = nvml_procs()
    opids = ollama_pids()
    gpus = sorted({g for p in opids for g in where(p, procs)})
    add("Ollama", ok and url.endswith(":11435") and gpus == [gidx],
        f"{url} v{ver} · {j.health_line()} · 러너 pid {opids[1:] or '-'} → GPU {gpus}")
    tags = {}
    try:
        tags = {m["name"]: m.get("digest") for m in j.session.get(f"{url}/api/tags", timeout=2).json()["models"]}
    except Exception:
        pass

    # 5. logprobs (P1 연속 점수)
    p = None
    try:
        res = j.judge([], "혹시 지금 몇 시예요?", "세 시 반이요.")
        p = res.get("p_pair") if res else None
    except Exception:
        pass
    add("logprobs(p_pair)", j.logprob_ok is True and p is not None,
        f"P1 p_pair={p if p is None else round(p, 3)} logprob_ok={j.logprob_ok}")

    # 6. 모델 다이제스트
    mdir = ROOT / "models"
    digests = {
        "ollama": tags,
        "hf": {d.name: hf_rev(d) for d in sorted((mdir / "hf" / "hub").glob("models--*"))},
        "whisper": {d.name: hf_rev(d) for d in sorted((mdir / "whisper").glob("models--*"))},
    }
    need = ["qwen3:4b" in tags, any("ast" in k for k in digests["hf"]), any("large-v3-turbo" in k for k in digests["whisper"])]
    add("모델 다이제스트", all(need) and all(digests["hf"].values()) and all(digests["whisper"].values()),
        f"ollama {len(tags)}개 · hf {len(digests['hf'])}개 · whisper {len(digests['whisper'])}개 (qwen3:4b {str(tags.get('qwen3:4b'))[:12]})")

    # 7. DEMAND
    from app.config import data_root
    dr = data_root()
    noise = (dr / "demand" / "PCAFETER" / "ch01.wav") if dr else None
    if noise and noise.exists():
        import soundfile as sf
        info = sf.info(str(noise))
        add("DEMAND PCAFETER", info.samplerate == 16000 and info.duration > 60,
            f"{noise} · {info.samplerate}Hz · {info.duration:.0f}s")
        zp = dr / "demand" / "PCAFETER_16k.zip"
        manifest["demand"] = {"path": str(noise), "zip_sha256": hashlib.sha256(zp.read_bytes()).hexdigest()
                              if zp.exists() else None, "source": "https://zenodo.org/records/1227121"}
    else:
        add("DEMAND PCAFETER", False, f"없음: {noise} (HEARME_DATA={os.environ.get('HEARME_DATA')})")

    # 8. 단위 테스트
    if not args.no_tests:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT, capture_output=True, text=True)
        last = (r.stdout.strip().splitlines() or ["?"])[-1]
        add("단위 테스트", r.returncode == 0, last)

    w = max(len(a) for a, _, _ in rows)
    print(f"\n| {'항목':<{w}} | 결과 | 상세 |\n|---|---|---|")
    for a, b, c in rows:
        print(f"| {a:<{w}} | {b} | {c} |")
    fails = [a for a, b, _ in rows if b == "FAIL"]
    print("\n→ " + ("모두 PASS" if not fails else f"FAIL: {', '.join(fails)}"))

    if args.manifest:
        drv = pynvml.nvmlSystemGetDriverVersion()
        manifest.update({
            "_doc": "서버 환경 기록(tools/env_check.py --manifest). AMI 결과는 이전 PC, AI Hub 결과는 전부 이 서버.",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "host": platform.node(), "os": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "driver": drv if isinstance(drv, str) else drv.decode(),
            "gpu": {"index": gidx, "name": torch.cuda.get_device_name(0) if n else None, "pci_bus_id": want_bus,
                    "capability": ".".join(map(str, torch.cuda.get_device_capability(0))) if n else None},
            "ollama": {"version": ver, "url": url, "env": {"OLLAMA_VULKAN": "0", "OLLAMA_MAX_LOADED_MODELS": "1",
                                                          "OLLAMA_KEEP_ALIVE": "30m", "OLLAMA_CONTEXT_LENGTH": "4096"}},
            "packages": {k: _ver(k) for k in ("faster-whisper", "ctranslate2", "speechbrain", "transformers",
                                              "silero-vad", "numpy", "scipy", "fastapi", "websockets")},
            "models": digests,
            "env_check": [{"item": a, "result": b, "detail": c} for a, b, c in rows],
        })
        (ROOT / "results" / "env_manifest_server.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                                                  encoding="utf-8")
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
        (ROOT / "requirements-server.lock.txt").write_text(
            "# 서버(RTX PRO 6000 Blackwell, Linux) 실제 설치 목록 — pip freeze. torch 2.7.1+cu128 필요(Blackwell sm_120).\n"
            "# 설치: pip install -r requirements-server.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128\n"
            + freeze, encoding="utf-8")
        print("기록: results/env_manifest_server.json, requirements-server.lock.txt")
    sys.exit(1 if fails else 0)


def _ver(pkg: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:
        return None


if __name__ == "__main__":
    main()
