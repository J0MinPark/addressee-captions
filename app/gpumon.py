"""GPU 사용률(대시보드 metrics). 서버는 GPU 하나(HEARME_GPU, nvidia-smi 번호)만 쓰므로 그 GPU만 읽는다.

NVML은 CUDA_VISIBLE_DEVICES를 무시하고 PCI 순서 번호를 쓴다(nvidia-smi와 같다).
nvidia-ml-py가 없거나 GPU가 없으면 None(노트북 CPU 프로필).
"""
from __future__ import annotations

import os
import threading
import time
from typing import Optional


def gpu_index() -> Optional[int]:
    v = os.environ.get("HEARME_GPU") or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    v = v.split(",")[0].strip()
    return int(v) if v.isdigit() else None


class GPUMonitor:
    def __init__(self, index: Optional[int] = None, min_interval_s: float = 1.0):
        self.index = gpu_index() if index is None else index
        self.min_interval = min_interval_s
        self._h = None
        self._last = 0.0
        self._cache: Optional[dict] = None
        self._lock = threading.Lock()
        if self.index is None:
            return
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nv = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(self.index)
            self.bus_id = pynvml.nvmlDeviceGetPciInfo(self._h).busId
            if isinstance(self.bus_id, bytes):
                self.bus_id = self.bus_id.decode()
        except Exception:
            self._h = None

    @property
    def available(self) -> bool:
        return self._h is not None

    def read(self) -> Optional[dict]:
        if self._h is None:
            return None
        with self._lock:
            now = time.monotonic()
            if self._cache is not None and now - self._last < self.min_interval:
                return self._cache
            try:
                u = self._nv.nvmlDeviceGetUtilizationRates(self._h)
                m = self._nv.nvmlDeviceGetMemoryInfo(self._h)
                self._cache = {"index": self.index, "bus_id": self.bus_id, "util_pct": int(u.gpu),
                               "mem_used_mb": round(m.used / 2**20), "mem_total_mb": round(m.total / 2**20)}
            except Exception:
                self._cache = None
            self._last = now
            return self._cache

    def processes(self) -> list[dict]:
        """이 GPU에서 도는 프로세스(pid, 사용 메모리). env_check가 우리 프로세스가 올라갔는지 확인할 때 쓴다."""
        if self._h is None:
            return []
        out = []
        for fn in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
            try:
                for p in getattr(self._nv, fn)(self._h):
                    out.append({"pid": int(p.pid), "mem_mb": round((p.usedGpuMemory or 0) / 2**20)})
            except Exception:
                pass
        return out
