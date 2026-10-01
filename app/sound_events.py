"""위험 소리 감지: AST(AudioSet) 분류 + 연속/쿨다운 디바운스.

분류는 별도 스레드에서 '최신 요청만' 처리한다(느리면 건너뛴다. 절대 쌓이지 않음).
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Optional

import numpy as np


class ASTClassifier:
    name = "ast"

    def __init__(self, cfg: dict):
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        s = cfg["sound"]
        self.torch = torch
        dev = s.get("device", "cuda")
        if dev.startswith("cuda") and not torch.cuda.is_available():
            dev = "cpu"
        self.device = dev
        self.fe = ASTFeatureExtractor.from_pretrained(s["ast_model"])
        self.model = ASTForAudioClassification.from_pretrained(s["ast_model"]).to(dev).eval()
        if dev.startswith("cuda"):
            self.model = self.model.half()
        id2label = self.model.config.id2label
        self.targets = {}
        for i, lab in id2label.items():
            if lab in s["classes"]:
                self.targets[int(i)] = lab
        missing = set(s["classes"]) - set(self.targets.values())
        if missing:
            print(f"[sound] AST 라벨에 없음(무시): {sorted(missing)}")
        self.name = f"ast/{dev}"

    def __call__(self, audio: np.ndarray) -> dict[str, float]:
        feats = self.fe(audio, sampling_rate=16000, return_tensors="pt")
        x = feats["input_values"].to(self.device)
        if self.device.startswith("cuda"):
            x = x.half()
        with self.torch.no_grad():
            logits = self.model(input_values=x).logits[0].float()
        probs = self.torch.sigmoid(logits).cpu().numpy()
        return {lab: float(probs[i]) for i, lab in self.targets.items()}


class YAMNetClassifier:
    name = "yamnet"

    def __init__(self, cfg: dict):
        import csv
        import tensorflow as tf  # noqa: F401
        import tensorflow_hub as hub
        self.model = hub.load("https://tfhub.dev/google/yamnet/1")
        path = self.model.class_map_path().numpy().decode()
        with open(path, encoding="utf-8") as f:
            names = [r["display_name"] for r in csv.DictReader(f)]
        cls = cfg["sound"]["classes"]
        self.targets = {i: n for i, n in enumerate(names) if n in cls}

    def __call__(self, audio: np.ndarray) -> dict[str, float]:
        scores, _, _ = self.model(audio.astype(np.float32))
        s = scores.numpy().max(axis=0)
        return {lab: float(s[i]) for i, lab in self.targets.items()}


def make_sound_classifier(cfg: dict, log=print):
    s = cfg["sound"]
    if not s.get("enabled", True):
        return None
    backend = s.get("backend", "ast")
    order = {"ast": ["ast"], "yamnet": ["yamnet", "ast"], "auto": ["ast", "yamnet"]}.get(backend, ["ast"])
    for b in order:
        try:
            clf = ASTClassifier(cfg) if b == "ast" else YAMNetClassifier(cfg)
            clf(np.zeros(16000, np.float32))  # 워밍업
            return clf
        except Exception as e:
            log(f"[sound] {b} 로딩 실패: {e}")
    log("[sound] 위험 소리 감지 비활성 (모델 없음)")
    return None


class AlertDebouncer:
    """대상 클래스 점수 >= threshold 가 consecutive 회 연속이면 알림. 종류별 쿨다운. 순수 로직."""

    def __init__(self, cfg: dict):
        s = cfg["sound"]
        self.thr = s["score_threshold"]
        self.need = s["consecutive"]
        self.cooldown = s["cooldown_s"]
        self.class_kind = dict(s["classes"])
        self.streak: dict[str, int] = {}
        self.last_alert: dict[str, float] = {}

    def update(self, t: float, scores: dict[str, float]) -> list[dict]:
        best: dict[str, tuple[str, float]] = {}
        for lab, sc in scores.items():
            kind = self.class_kind.get(lab)
            if kind and sc >= self.thr and sc > best.get(kind, ("", -1))[1]:
                best[kind] = (lab, sc)
        alerts = []
        for kind in set(self.class_kind.values()):
            if kind in best:
                self.streak[kind] = self.streak.get(kind, 0) + 1
            else:
                self.streak[kind] = 0
                continue
            if self.streak[kind] >= self.need and t - self.last_alert.get(kind, -1e9) >= self.cooldown:
                self.last_alert[kind] = t
                lab, sc = best[kind]
                alerts.append({"type": "alert", "kind": kind, "label": lab, "score": round(sc, 3), "t": t})
        return alerts


class SoundWorker:
    """최신 요청 하나만 보관하는 분류 스레드."""

    def __init__(self, clf, callback: Callable[[float, dict], None], log=print):
        self.clf = clf
        self.callback = callback
        self.log = log
        self.slot: Optional[tuple[float, np.ndarray]] = None
        self.cv = threading.Condition()
        self.stop_flag = False
        self.skipped = 0
        self.proc_ms: deque[float] = deque(maxlen=50)
        self.busy = False
        self.th = threading.Thread(target=self._run, name="sound", daemon=True)
        self.th.start()

    def submit(self, t: float, audio: np.ndarray) -> None:
        with self.cv:
            if self.slot is not None:
                self.skipped += 1
            self.slot = (t, audio)
            self.cv.notify()

    def idle(self) -> bool:
        return self.slot is None and not self.busy

    def _run(self):
        while True:
            with self.cv:
                while self.slot is None and not self.stop_flag:
                    self.cv.wait(0.5)
                if self.stop_flag:
                    return
                t, audio = self.slot
                self.slot = None
                self.busy = True
            t0 = time.perf_counter()
            try:
                scores = self.clf(audio)
            except Exception as e:
                self.log(f"[sound] 분류 오류: {e}")
                scores = {}
            self.proc_ms.append((time.perf_counter() - t0) * 1000)
            self.busy = False
            try:
                self.callback(t, scores)
            except Exception as e:
                self.log(f"[sound] 콜백 오류: {e}")

    def stop(self):
        with self.cv:
            self.stop_flag = True
            self.cv.notify_all()
