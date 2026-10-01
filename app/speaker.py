"""화자 구분: ECAPA 임베딩 + 온라인 군집(코사인 유사도, 중심 EMA)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

SR = 16000


def _norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    return v / (np.linalg.norm(v) + 1e-9)


class EcapaEmbedder:
    name = "ecapa"

    def __init__(self, cfg: dict):
        import torch
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:  # speechbrain < 1.0
            from speechbrain.pretrained import EncoderClassifier
        from app.config import resolve_path
        self.torch = torch
        sc = cfg["speaker"]
        savedir = resolve_path(cfg, "models_dir") / "spkrec-ecapa-voxceleb"
        source = str(savedir) if (savedir / "hyperparams.yaml").exists() else sc["model"]
        kwargs = dict(source=source, savedir=str(savedir), run_opts={"device": sc.get("device", "cpu")})
        try:
            from speechbrain.utils.fetching import LocalStrategy
            kwargs["local_strategy"] = LocalStrategy.COPY   # Windows: 심볼릭 링크 대신 복사
        except Exception:
            pass
        try:
            self.model = EncoderClassifier.from_hparams(**kwargs)
        except TypeError:
            kwargs.pop("local_strategy", None)
            self.model = EncoderClassifier.from_hparams(**kwargs)
        self.model.eval()
        self.max_s = sc.get("max_embed_s", 8.0)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        x = audio[: int(self.max_s * SR)]
        with self.torch.no_grad():
            e = self.model.encode_batch(self.torch.from_numpy(x).float().unsqueeze(0))
        return _norm(e.squeeze().cpu().numpy())


class SpectralEmbedder:
    """ECAPA 로딩 실패 시 폴백: 로그 멜 비슷한 대역 에너지 평균. 정확도는 낮지만 죽지 않는다."""
    name = "spectral"

    def __init__(self, n_bands: int = 64):
        self.n_bands = n_bands

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        n = 512
        if len(audio) < n:
            audio = np.pad(audio, (0, n - len(audio)))
        frames = np.lib.stride_tricks.sliding_window_view(audio, n)[::160]
        spec = np.abs(np.fft.rfft(frames * np.hanning(n), axis=1)) ** 2
        edges = np.unique(np.geomspace(2, spec.shape[1] - 1, self.n_bands + 1).astype(int))
        bands = np.stack([spec[:, a:b].sum(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1)
        logb = np.log(bands + 1e-8)
        feat = logb.mean(axis=0)
        feat = feat - feat.mean()
        return _norm(feat)


def make_embedder(cfg: dict, log=print):
    if not cfg["speaker"].get("enabled", True):
        return SpectralEmbedder()
    try:
        return EcapaEmbedder(cfg)
    except Exception as e:
        log(f"[speaker] ECAPA 로딩 실패 → 스펙트럼 임베딩 폴백(정확도 낮음): {e}")
        return SpectralEmbedder()


class SpeakerRegistry:
    """온라인 화자 군집. 순수 numpy.

    - 길이 >= min_embed_s: 최대 유사도 >= threshold면 배정+EMA 갱신, 아니면 새 화자.
    - short_s <= 길이 < min_embed_s: 기존 화자와 맞으면 배정만(갱신·신규 없음), 아니면 None.
    - 길이 < short_s: None (정책 엔진이 직전 상대 상속 여부를 판단).
    """

    def __init__(self, cfg: dict):
        s = cfg["speaker"]
        self.threshold = s["spk_threshold"]
        self.ema = s["ema"]
        self.min_embed_s = s["min_embed_s"]
        self.short_s = s["short_s"]
        self.max_speakers = s.get("max_speakers", 30)
        self.reset()

    def reset(self):
        self.centroids: dict[int, np.ndarray] = {}
        self.counts: dict[int, int] = {}
        self.next_id = 1
        self.wearer: Optional[np.ndarray] = None

    def needs_embedding(self, duration: float) -> bool:
        return duration >= self.short_s

    def best(self, emb: np.ndarray) -> tuple[Optional[int], float]:
        if not self.centroids:
            return None, 0.0
        ids = list(self.centroids)
        sims = np.array([float(np.dot(self.centroids[i], emb)) for i in ids])
        k = int(np.argmax(sims))
        return ids[k], float(sims[k])

    def assign(self, emb: Optional[np.ndarray], duration: float) -> tuple[Optional[int], float, bool]:
        """반환: (speaker_id 또는 None, 유사도, 새 화자 여부)."""
        if emb is None or duration < self.short_s:
            return None, 0.0, False
        emb = _norm(emb)
        sid, sim = self.best(emb)
        if duration < self.min_embed_s:
            return (sid, sim, False) if (sid is not None and sim >= self.threshold) else (None, sim, False)
        if sid is not None and (sim >= self.threshold or len(self.centroids) >= self.max_speakers):
            c = self.ema * self.centroids[sid] + (1 - self.ema) * emb
            self.centroids[sid] = _norm(c)
            self.counts[sid] += 1
            return sid, sim, False
        nid = self.next_id
        self.next_id += 1
        self.centroids[nid] = emb
        self.counts[nid] = 1
        return nid, 1.0, True

    def wearer_sim(self, emb: np.ndarray) -> float:
        if self.wearer is None:
            return 0.0
        return float(np.dot(self.wearer, _norm(emb)))
