"""받아쓰기: faster-whisper 작업 스레드 + 환각 필터.

판정 경로를 막지 않도록 별도 스레드/큐에서 돈다. GPU 로딩/추론 실패 시 CPU small 로 폴백.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


def filter_segments(segs: list[dict], cfg_asr: dict, hotwords: Optional[list[str]] = None) -> str:
    """segs: [{"text", "no_speech_prob", "avg_logprob"}] -> 통과한 텍스트."""
    out = []
    for s in segs:
        if s.get("no_speech_prob", 0) > cfg_asr["no_speech_prob_max"]:
            continue
        if s.get("avg_logprob", 0) < cfg_asr["avg_logprob_min"]:
            continue
        txt = (s.get("text") or "").strip()
        if not txt or is_blacklisted(txt, cfg_asr.get("blacklist", [])):
            continue
        out.append(txt)
    text = " ".join(out).strip()
    if cfg_asr.get("drop_hotword_echo", True) and hotwords and is_hotword_echo(text, hotwords):
        return ""
    return text


def is_hotword_echo(text: str, hotwords: list[str]) -> bool:
    """"민수 민수", "민수 씨 민수 씨 …" 처럼 핫워드만 반복된 출력(무음/잡음에서 흔한 환각)."""
    import re
    t = re.sub(r"[\s,.!?~…]+", "", text)
    base = min((h.replace(" ", "") for h in hotwords if h.strip()), key=len, default="")
    if not base or t.count(base) < 2:
        return False
    rest = t
    for h in sorted({h.replace(" ", "") for h in hotwords}, key=len, reverse=True):
        rest = rest.replace(h, "")
    rest = re.sub(r"(씨|님|야|아|이)", "", rest)
    return len(rest) <= 1


def is_blacklisted(text: str, blacklist: list[str]) -> bool:
    t = text.replace(" ", "")
    return any(b.replace(" ", "") in t for b in blacklist)


class WhisperASR:
    def __init__(self, cfg: dict, log=print):
        from app.config import resolve_path
        self.cfg = cfg
        self.a = cfg["asr"]
        self.log = log
        self.root = str(resolve_path(cfg, "models_dir") / "whisper")
        self.model = None
        self.desc = "none"
        name = cfg["wearer"]["name"]
        self.name_check: Optional[Callable[[str], bool]] = None   # pipeline이 호명 감지기로 설정
        self.hotword_list = [name] + list(cfg["wearer"].get("name_variants", []))
        self.hotwords = " ".join(dict.fromkeys([name] + [v for v in cfg["wearer"].get("name_variants", [])]))
        tried = [(self.a["model"], self.a["device"], self.a["compute_type"]),
                 (self.a["fallback_model"], self.a["fallback_device"], self.a["fallback_compute_type"])]
        for model, device, ctype in dict.fromkeys(tried):
            try:
                self._load(model, device, ctype)
                self.transcribe(np.zeros(16000, np.float32) + 1e-4 * np.random.randn(16000).astype(np.float32))
                self.desc = f"{model}/{device}/{ctype}"
                return
            except Exception as e:
                log(f"[asr] {model} ({device},{ctype}) 실패: {e}")
                self.model = None
        log("[asr] 모든 Whisper 로딩 실패 → 받아쓰기 없이 동작")

    def _load(self, model: str, device: str, ctype: str):
        from faster_whisper import WhisperModel
        kw = dict(device=device, compute_type=ctype, download_root=self.root)
        if device == "cpu":
            kw["cpu_threads"] = int(self.a.get("cpu_threads", 4))
        try:
            self.model = WhisperModel(model, local_files_only=True, **kw)
        except Exception:
            self.model = WhisperModel(model, **kw)

    def _run(self, audio: np.ndarray, hot: bool) -> tuple[str, list[dict]]:
        kw = dict(language=self.a["language"], beam_size=self.a["beam_size"], vad_filter=False,
                  condition_on_previous_text=False, without_timestamps=True)
        x = audio.astype(np.float32)
        if hot:
            try:
                segs, _ = self.model.transcribe(x, hotwords=self.hotwords, **kw)
            except TypeError:  # 구버전: hotwords 미지원
                segs, _ = self.model.transcribe(x, initial_prompt=self.hotwords, **kw)
        else:
            segs, _ = self.model.transcribe(x, **kw)
        raw = [{"text": s.text, "no_speech_prob": s.no_speech_prob, "avg_logprob": s.avg_logprob} for s in segs]
        return filter_segments(raw, self.a, self.hotword_list), raw

    def transcribe(self, audio: np.ndarray) -> tuple[str, list[dict]]:
        """hotwords=착용자 이름으로 받아쓴다. 결과에 이름이 나오면 핫워드 없이 한 번 더 받아써서
        이름이 여전히 있을 때만 믿는다(핫워드가 "지훈아"를 "민수"로 바꾸는 편향 방지)."""
        if self.model is None:
            return "", []
        text, raw = self._run(audio, hot=True)
        if self.a.get("verify_hotword", True) and self.name_check and text and self.name_check(text):
            text2, raw2 = self._run(audio, hot=False)
            if not self.name_check(text2):
                return text2, raw2
        return text, raw


@dataclass
class ASRJob:
    job_id: str
    audio: np.ndarray
    callback: Callable[[str, str, dict], None]   # (job_id, text, info)
    priority: bool = False                        # 착용자 구간: 버리지 않음
    enq: float = field(default_factory=time.monotonic)


class ASRWorker:
    """FIFO 작업 스레드. 큐가 max_queue를 넘으면 가장 오래된 비착용자 작업을 버린다(빈 텍스트로 콜백)."""

    def __init__(self, engine, max_queue: int = 12, log=print):
        self.engine = engine
        self.max_queue = max_queue
        self.log = log
        self.q: deque[ASRJob] = deque()
        self.cv = threading.Condition()
        self.stop_flag = False
        self.busy = False
        self.done = 0
        self.dropped = 0
        self.proc_ms: deque[float] = deque(maxlen=100)
        self.th = threading.Thread(target=self._run, name="asr", daemon=True)
        self.th.start()

    def submit(self, job: ASRJob) -> None:
        drop = None
        with self.cv:
            self.q.append(job)
            if len(self.q) > self.max_queue:
                for j in self.q:
                    if not j.priority:
                        drop = j
                        break
                if drop is not None:
                    self.q.remove(drop)
                    self.dropped += 1
            self.cv.notify()
        if drop is not None:
            drop.callback(drop.job_id, "", {"dropped": True, "asr_ms": 0})

    def qsize(self) -> int:
        return len(self.q) + (1 if self.busy else 0)

    def idle(self) -> bool:
        return not self.q and not self.busy

    def _run(self):
        while True:
            with self.cv:
                while not self.q and not self.stop_flag:
                    self.cv.wait(0.5)
                if self.stop_flag:
                    return
                job = self.q.popleft()
                self.busy = True
            t0 = time.perf_counter()
            try:
                text, raw = self.engine.transcribe(job.audio)
                info = {"raw": raw}
            except Exception as e:
                self.log(f"[asr] 오류: {e}")
                text, info = "", {"error": str(e)}
            ms = (time.perf_counter() - t0) * 1000
            self.proc_ms.append(ms)
            info["asr_ms"] = round(ms, 1)
            info["queue_wait_ms"] = round((time.monotonic() - job.enq) * 1000 - ms, 1)
            self.busy = False
            self.done += 1
            try:
                job.callback(job.job_id, text, info)
            except Exception as e:
                self.log(f"[asr] 콜백 오류: {e}")

    def stop(self):
        with self.cv:
            self.stop_flag = True
            self.cv.notify_all()
