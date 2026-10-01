"""플랫폼 초기화. 다른 무거운 import보다 먼저 호출한다.

- Windows: pip의 nvidia-cublas-cu12 / nvidia-cudnn-cu12 DLL 경로를 등록한다
  (faster-whisper/ctranslate2 GPU가 cuBLAS 12, cuDNN 9를 찾을 수 있게).
- 콘솔 출력을 UTF-8로 맞춘다(한글, QR 코드).
- HF 캐시를 프로젝트 models/ 아래로 고정하고, 모델이 받아져 있으면 오프라인 모드.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_DONE = False


def _nvidia_bin_dirs():
    dirs = []
    for sp in list(sys.path):
        base = Path(sp) / "nvidia"
        if not base.is_dir():
            continue
        for sub in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc"):
            for name in ("bin", "lib"):
                d = base / sub / name
                if d.is_dir():
                    dirs.append(d)
    return dirs


def setup(offline_if_cached: bool = True) -> None:
    global _DONE
    if _DONE:
        return
    _DONE = True
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    root = Path(__file__).resolve().parent.parent
    models = Path(os.environ.get("HEARME_MODELS_DIR") or (root / "models"))
    models.mkdir(exist_ok=True)
    os.environ.setdefault("HF_HOME", str(models / "hf"))
    os.environ.setdefault("TORCH_HOME", str(models / "torch"))
    os.environ.setdefault("TFHUB_CACHE_DIR", str(models / "tfhub"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    if offline_if_cached and (models / ".downloaded").exists() and os.environ.get("HEARME_ONLINE") != "1":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    if os.name == "nt":
        for d in _nvidia_bin_dirs():
            try:
                os.add_dll_directory(str(d))
            except Exception:
                pass
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
    else:
        # Linux: ctranslate2는 LD_LIBRARY_PATH가 필요할 수 있다. 미리 로드해 둔다.
        import ctypes
        for d in _nvidia_bin_dirs():
            for so in sorted(d.glob("lib*.so*")):
                if any(k in so.name for k in ("cublas", "cudnn")):
                    try:
                        ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass
