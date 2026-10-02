"""B 채널 특징 캐시(재생 평가용). 같은 회의·같은 소음 조건의 B 채널은 착용자가 달라도 똑같으므로
VAD 확률(청크 순서), ASR 결과·화자 임베딩(구간 오디오 해시)을 공유해 재계산을 피한다.
정책·판정 결과는 캐시하지 않는다(착용자마다 다름). 디스크에 pickle로 남겨 프로세스 간에도 공유한다.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Optional

import numpy as np


def audio_key(x: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(x, dtype=np.float32).tobytes()).hexdigest()[:24]


class FeatureCache:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.d = {"vad": {}, "asr": {}, "emb": {}}
        self.hits = {"vad": 0, "asr": 0, "emb": 0}
        self.miss = {"vad": 0, "asr": 0, "emb": 0}
        if self.path and self.path.exists():
            try:
                with open(self.path, "rb") as f:
                    self.d.update(pickle.load(f))
            except Exception:
                pass

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(self.d, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(self.path)

    def stats(self) -> str:
        return " · ".join(f"{k} 적중 {self.hits[k]}/{self.hits[k] + self.miss[k]}" for k in self.d)


class CachedVAD:
    """청크 순서대로 확률을 기록/재생. 같은 B 스트림이면 두 번째부터 모델을 부르지 않는다."""

    def __init__(self, inner, cache: FeatureCache, stream_key: str):
        self.inner, self.cache = inner, cache
        self.probs = cache.d["vad"].setdefault(stream_key, [])
        self.i = 0
        self.name = getattr(inner, "name", "vad")
        try:
            inner.reset()
        except Exception:
            pass

    def __call__(self, chunk: np.ndarray) -> float:
        if self.i < len(self.probs):
            p = self.probs[self.i]
            self.cache.hits["vad"] += 1
        else:
            p = float(self.inner(chunk))
            self.probs.append(p)
            self.cache.miss["vad"] += 1
        self.i += 1
        return p

    def reset(self):
        self.i = 0


class CachedASR:
    def __init__(self, inner, cache: FeatureCache):
        self.inner, self.cache = inner, cache
        self.desc = getattr(inner, "desc", "?")

    @property
    def name_check(self):
        return getattr(self.inner, "name_check", None)

    @name_check.setter
    def name_check(self, fn):
        if hasattr(self.inner, "name_check"):
            self.inner.name_check = fn

    def transcribe(self, audio: np.ndarray):
        k = audio_key(audio)
        hit = self.cache.d["asr"].get(k)
        if hit is not None:
            self.cache.hits["asr"] += 1
            return hit
        self.cache.miss["asr"] += 1
        out = self.inner.transcribe(audio)
        self.cache.d["asr"][k] = out
        return out


class CachedEmbedder:
    def __init__(self, inner, cache: FeatureCache):
        self.inner, self.cache = inner, cache
        self.name = getattr(inner, "name", "?")

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        k = audio_key(audio)
        hit = self.cache.d["emb"].get(k)
        if hit is not None:
            self.cache.hits["emb"] += 1
            return hit
        self.cache.miss["emb"] += 1
        e = self.inner(audio)
        self.cache.d["emb"][k] = e
        return e
