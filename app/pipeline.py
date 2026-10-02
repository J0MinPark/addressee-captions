"""모듈 연결: 오디오 → 본인 발화/구간 분할 → 화자 → ASR → 정책(+LLM) → 이벤트.

스레드 구조
  audio   : Block 읽기, 링버퍼, VAD, 본인 발화, 구간 분할, 위험 소리 요청 (가볍게 유지)
  control : 모든 정책/화자 상태를 '한 스레드'에서 순서대로 처리 (락 없이 결정적)
  asr     : faster-whisper 작업 큐
  sound   : AST 최신 요청만
  llm     : Ollama 호출 풀
모든 이벤트는 logs/run_<timestamp>.jsonl 에 기록되고 등록된 리스너(server WebSocket)로 방송된다.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from app.asr import ASRJob, ASRWorker
from app.audio_source import AudioSource, Block, RingBuffer
from app.config import resolve_path
from app.namecall import NameCallDetector
from app.ownvoice import OwnVoiceDetector, rms_db
from app.policy import MODES, PolicyEngine, SegFeat
from app.segmenter import Segment, Segmenter, StreamingVAD
from app.sound_events import AlertDebouncer, SoundWorker
from app.speaker import SpeakerRegistry


# --------------------------------------------------------------------- 모델
@dataclass
class Models:
    vad_a: Any = None
    vad_b: Any = None
    embedder: Any = None
    asr: Any = None
    sound: Any = None
    judge: Any = None
    timings: dict = field(default_factory=dict)

    def describe(self) -> dict:
        return {
            "vad": getattr(self.vad_b, "name", None),
            "speaker": getattr(self.embedder, "name", None),
            "asr": getattr(self.asr, "desc", None),
            "sound": getattr(self.sound, "name", None),
            "llm": getattr(self.judge, "model", None),
        }


def load_models(cfg: dict, log=print, skip: tuple = ()) -> Models:
    """모든 모델을 미리 로드하고 더미 입력으로 워밍업한다. 실패는 폴백으로 흡수."""
    from app.segmenter import make_vad
    m = Models()

    def timed(name, fn):
        t0 = time.perf_counter()
        try:
            out = fn()
        except Exception as e:  # 어떤 경우에도 죽지 않는다
            log(f"[load] {name} 실패: {e}")
            out = None
        m.timings[name] = round(time.perf_counter() - t0, 2)
        return out

    def vad():
        v = make_vad(cfg, log)
        v(np.zeros(512, np.float32))
        v.reset()
        return v

    m.vad_a = timed("vad_a", vad)
    m.vad_b = timed("vad_b", vad)
    if "speaker" not in skip:
        def emb():
            from app.speaker import make_embedder
            e = make_embedder(cfg, log)
            e(np.random.randn(16000).astype(np.float32) * 0.01)
            return e
        m.embedder = timed("speaker", emb)
    if "asr" not in skip:
        def asr():
            from app.asr import WhisperASR
            return WhisperASR(cfg, log)
        m.asr = timed("asr", asr)
    if "sound" not in skip:
        def snd():
            from app.sound_events import make_sound_classifier
            return make_sound_classifier(cfg, log)
        m.sound = timed("sound", snd)
    if "llm" not in skip:
        def llm():
            from app.llm_judge import LLMJudge
            j = LLMJudge(cfg, log)
            j.setup()
            return j
        m.judge = timed("llm", llm)
    log("[load] 워밍업 시간(s): " + ", ".join(f"{k}={v}" for k, v in m.timings.items()))
    log("[load] 사용 모델: " + ", ".join(f"{k}={v}" for k, v in m.describe().items()))
    if m.judge is not None:
        log_llm_banner(m.judge, log)
    log(f"[구성] {config_name(cfg)}")
    return m


def config_name(cfg: dict) -> str:
    """판정 구성 이름: <판정기 변형>-<LLM 모델>-<융합>[+플래그]. 시작 로그·결과 메타에 찍힌다."""
    p, l = cfg["policy"], cfg["llm"]
    if p.get("config_name"):
        return p["config_name"]
    fusion = "learned" if (p.get("fusion") or {}).get("type") == "logistic" else "hand"
    flags = "".join(f"+{n}" for n, k in (("rejudge", "candidate_rejudge"), ("shortskip", "short_skip_llm")) if p.get(k))
    return f"{l.get('variant', 'P1c')}-{l['models'][0]}-{fusion}{flags}"


def log_llm_banner(judge, log=print) -> None:
    """LLM 상태를 시작 로그에 크게. FAIL이면 의미 판정이 꺼진 채로 돈다는 뜻이다."""
    line = judge.health_line()
    bad = judge.status["state"] != "ok"
    bar = ("!" if bad else "=") * 64
    log(bar)
    log(f"{'!!' if bad else '  '}  {line}")
    if bad and judge.status["state"] != "off":
        log("!!  → 의미 판정 꺼짐: 타이밍 규칙으로만 등록됩니다. Ollama를 켜면 10초 안에 자동 복구.")
    log(bar)


# ----------------------------------------------------------------- 파이프라인
class Pipeline:
    def __init__(self, cfg: dict, source: AudioSource, models: Models, mode: Optional[str] = None,
                 log=print, log_events: bool = True, record_segments: bool = False, run_name: str = "",
                 feature_cache=None, cache_key: Optional[str] = None):
        self.cfg = cfg
        self.src = source
        self.m = models
        self.log = log
        self.sr = cfg["audio"]["sample_rate"]
        self.single_mic = bool(getattr(source, "single_mic", False) or cfg["audio"].get("single_mic"))
        ring_s = cfg["audio"]["ring_seconds"]
        self.ring_a = RingBuffer(ring_s, self.sr)
        self.ring_b = RingBuffer(ring_s, self.sr)
        for v in (models.vad_a, models.vad_b):   # 재생을 연달아 돌릴 때 이전 스트림의 VAD 상태가 남지 않게
            try:
                v.reset()
            except Exception:
                pass
        self.svad_a = StreamingVAD(models.vad_a, self.sr) if models.vad_a else None
        self.fcache = feature_cache
        asr_engine, self.embedder, vad_b = models.asr, models.embedder, models.vad_b
        if feature_cache is not None and cache_key:   # 같은 B 채널을 공유하는 재생(AMI 착용자 4명)
            from app.featcache import CachedASR, CachedEmbedder, CachedVAD
            vad_b = CachedVAD(models.vad_b, feature_cache, cache_key)
            asr_engine = CachedASR(models.asr, feature_cache) if models.asr else None
            self.embedder = CachedEmbedder(models.embedder, feature_cache) if models.embedder else None
        self.svad_b = StreamingVAD(vad_b, self.sr)
        self.own = OwnVoiceDetector(cfg)
        self.b_hist: deque = deque(maxlen=48)   # 최근 B VAD 청크 (~1.5초)
        self.tail_guard = cfg["ownvoice"].get("tail_guard_s", 0.1)
        self.pad = cfg["vad"].get("pad_s", 0.1)
        self.segmenter = Segmenter(cfg, prefix="s")
        self.registry = SpeakerRegistry(cfg)
        judge_ok = bool(models.judge and models.judge.available)
        self.policy = PolicyEngine(cfg, mode=mode, llm_available=judge_ok)
        nc = cfg["namecall"]
        self.namecall = NameCallDetector(cfg["wearer"]["name_variants"], nc["max_edit_distance"],
                                         nc.get("short_max_edit_distance", nc["max_edit_distance"]),
                                         nc.get("short_jamo_len", 0)) if nc.get("enabled", True) else None
        self.debouncer = AlertDebouncer(cfg)
        if self.namecall is not None and hasattr(asr_engine, "name_check"):
            asr_engine.name_check = lambda text: self.namecall.detect(text) is not None
        self.asr = ASRWorker(asr_engine or _NullASR(), cfg["asr"]["max_queue"], log)
        self.sound = SoundWorker(models.sound, self._on_sound, log) if models.sound else None
        self.ctl: "queue.Queue[tuple]" = queue.Queue()
        self.listeners: list[Callable[[dict], None]] = []
        self.captions: OrderedDict[str, dict] = OrderedDict()
        self.seg_info: dict[str, dict] = {}
        self.records: dict[str, dict] = {}          # segments.jsonl 용
        self.record_segments = record_segments
        self.embs: dict[str, np.ndarray] = {}      # 특징 캐시(보정용 임베딩)
        self.llm_deadline: dict[str, float] = {}
        self.llm_deferred: list[tuple[str, int]] = []
        self.lat_ms: deque[float] = deque(maxlen=200)
        self.turn_rec: dict[int, str] = {}
        self.n_wearer = 0
        self.running = False
        self.audio_done = threading.Event()
        self.stopped = threading.Event()
        self.next_sound_t = cfg["sound"]["window_s"]
        self.next_tick_t = 0.0
        self.last_metrics = 0.0
        self.t_stream = 0.0
        self.start_wall = time.time()
        self.log_file = None
        self.run_name = run_name or time.strftime("%Y%m%d_%H%M%S")
        if log_events:
            p = resolve_path(cfg, "logs_dir") / f"run_{self.run_name}.jsonl"
            self.log_file = open(p, "a", encoding="utf-8")
            self.log_path = p
        self.always_llm = bool(cfg["llm"].get("always_call")) or record_segments

    # ------------------------------------------------------------ 이벤트
    def add_listener(self, fn: Callable[[dict], None]) -> None:
        self.listeners.append(fn)

    def emit(self, ev: dict) -> None:
        ev.setdefault("ts", round(time.time(), 3))
        ev.setdefault("t", round(self.t_stream, 3))
        if ev["type"] == "caption":
            self.captions[ev["id"]] = dict(ev)
            while len(self.captions) > self.cfg["server"]["history"]:
                self.captions.popitem(last=False)
        elif ev["type"] == "caption_update" and ev["id"] in self.captions:
            self.captions[ev["id"]].update({k: v for k, v in ev.items() if k not in ("type", "ts")})
        if self.log_file:
            try:
                self.log_file.write(json.dumps(ev, ensure_ascii=False, default=_json_default) + "\n")
            except Exception:
                pass
        for fn in list(self.listeners):
            try:
                fn(ev)
            except Exception as e:
                self.log(f"[emit] 리스너 오류: {e}")

    def snapshot(self) -> dict:
        return {"type": "snapshot", "mode": self.policy.mode, "speakers": self.policy.speakers_snapshot(),
                "captions": list(self.captions.values()), "models": self.m.describe(),
                "wearer": self.cfg["wearer"]["name"], "profile": self.cfg.get("_profile"),
                "llm_status": self._llm_status()}

    def _llm_status(self) -> dict:
        j = self.m.judge
        if j is None or not hasattr(j, "status"):
            return {"state": "off", "model": None, "device": "-", "ms": None, "reason": "LLM 없음"}
        return dict(j.status)

    # ------------------------------------------------------------ 명령
    def command(self, cmd: dict) -> None:
        """WebSocket/HTTP 에서 오는 명령. control 스레드에서 처리된다."""
        self.ctl.put(("cmd", cmd))

    # ------------------------------------------------------------ 실행
    def start(self) -> None:
        self.running = True
        self.src.start()
        if self.single_mic:
            self._enroll()
        self.th_ctl = threading.Thread(target=self._control_loop, name="control", daemon=True)
        self.th_audio = threading.Thread(target=self._audio_loop, name="audio", daemon=True)
        self.th_ctl.start()
        self.th_audio.start()
        self.emit({"type": "mode", "mode": self.policy.mode})
        j = self.m.judge
        if j is not None and hasattr(j, "start_monitor"):
            j.start_monitor(self.cfg["llm"].get("monitor_interval_s", 10),
                            lambda st: self.ctl.put(("llm_status", st)))
        self.emit({"type": "llm_status", **self._llm_status()})

    def _on_llm_status(self, st: dict) -> None:
        """LLM 상태 변화: 정책의 LLM 사용 여부를 맞추고 대시보드에 알린다."""
        self.policy.llm_available = st["state"] == "ok"
        self.log(f"[llm] 상태 변경 → {self.m.judge.health_line()}")
        self.emit({"type": "llm_status", **st})

    def stop(self) -> None:
        self.running = False
        if self.m.judge is not None and hasattr(self.m.judge, "stop_monitor"):
            self.m.judge.stop_monitor()
        try:
            self.src.stop()
        except Exception:
            pass
        self.asr.stop()
        if self.sound:
            self.sound.stop()
        self.ctl.put(("stop",))
        self.stopped.set()

    def wait_finished(self, timeout: float = 3600) -> bool:
        """재생 소스가 끝나고 모든 작업 큐가 빌 때까지 기다린다."""
        t_end = time.monotonic() + timeout
        self.audio_done.wait(timeout)
        while time.monotonic() < t_end:
            busy = (not self.asr.idle()) or self.ctl.qsize() > 0 or self.llm_deadline or self.llm_deferred \
                or (self.m.judge is not None and self.m.judge.inflight > 0) \
                or (self.sound is not None and not self.sound.idle())
            if not busy:
                time.sleep(0.3)
                if self.asr.idle() and self.ctl.qsize() == 0 and not self.llm_deadline:
                    return True
            time.sleep(0.05)
        return False

    # ------------------------------------------------------- 단일 마이크 등록
    def _enroll(self) -> None:
        a = self.cfg["audio"]
        emb = self.m.embedder
        if emb is None:
            self.log("[enroll] 화자 임베딩 없음 → 단일 마이크 본인 판정 불가")
            return
        wav = a.get("enroll_wav")
        if wav and Path(wav).exists():
            from app.audio_source import read_wav
            x = read_wav(wav, self.sr)
        else:
            n = int(a["enroll_seconds"] * self.sr)
            self.log(f"[enroll] 착용자 목소리 등록: {a['enroll_seconds']}초 동안 평소처럼 말해 주세요...")
            buf = []
            got = 0
            while got < n:
                blk = self.src.read(2.0)
                if blk is None:
                    break
                buf.append(blk.a)
                got += len(blk.a)
            x = np.concatenate(buf) if buf else np.zeros(0, np.float32)
        # 말소리 청크만 사용
        vad = self.m.vad_a or self.m.vad_b
        voiced = [x[i:i + 512] for i in range(0, len(x) - 512, 512) if vad(x[i:i + 512]) >= 0.5]
        try:
            vad.reset()
        except Exception:
            pass
        if len(voiced) < 30:
            self.log("[enroll] 말소리가 너무 적음(1초 미만) → 등록 실패, 모든 발화를 타인으로 취급")
            return
        self.registry.wearer = emb(np.concatenate(voiced))
        self.log(f"[enroll] 완료 (말소리 {len(voiced) * 512 / self.sr:.1f}초)")

    # ------------------------------------------------------------ audio
    def _audio_loop(self) -> None:
        cfg = self.cfg
        interval = cfg["sound"]["interval_s"]
        win = cfg["sound"]["window_s"]
        try:
            while self.running:
                blk = self.src.read(1.0)
                if blk is None:
                    if self.src.finished:
                        break
                    continue
                self._process_block(blk)
                if not getattr(self.src, "realtime", True):
                    self._backpressure()
                if self.sound and blk.t >= self.next_sound_t:
                    self.sound.submit(blk.t, self.ring_b.get(blk.t - win, blk.t))
                    self.next_sound_t += interval
                    if self.next_sound_t < blk.t:  # 뒤처지면 따라잡기
                        self.next_sound_t = blk.t + interval
                if blk.t >= self.next_tick_t:
                    self.ctl.put(("tick", blk.t))
                    self.next_tick_t = blk.t + 0.5
        except Exception as e:
            import traceback
            self.log(f"[audio] 오류: {e}\n{traceback.format_exc()}")
        # 끝: 열린 구간 정리
        t = self.ring_b.t_now
        own = self.own.flush()
        if own and not self.single_mic:
            self._push_wearer(own)
        seg = self.segmenter.flush()
        if seg:
            seg.wearer_overlap = self.own.overlap_ratio(seg.t_start, seg.t_end, t)
            self.ctl.put(("seg", seg))
        self.ctl.put(("tick", t))
        self.audio_done.set()

    def _backpressure(self) -> None:
        """최대 속도 재생: 버리지 말고 기다린다(평가가 결정적이도록). 실시간에서는 쓰지 않는다."""
        # 대기 중인 LLM 판정도 기다린다: 실시간에서는 판정(~0.3초)이 다음 발화보다 먼저 끝나므로, 재생에서도
        # 판정 결과(등록)가 뒤 구간보다 늦게 처리되지 않게 한다(이 순서가 바뀌면 결과가 실행마다 달라진다).
        while self.running and (self.asr.qsize() >= 2 or self.ctl.qsize() > 20 or self.llm_deadline
                                or (self.sound is not None and not self.sound.idle())):
            time.sleep(0.002)

    def _process_block(self, blk: Block) -> None:
        dur = len(blk.b) / self.sr
        self.ring_a.push(blk.a)
        self.ring_b.push(blk.b)
        self.t_stream = blk.t + dur
        own_closed = None
        if not self.single_mic and self.svad_a is not None:
            self.svad_a.push(blk.t, blk.a)
            was_active = self.own.active
            own = self.own.update(blk.t, dur, rms_db(blk.a), rms_db(blk.b), self.svad_a.last_prob)
            if self.own.active and not was_active:
                # 착용자가 말을 시작: 열린 주변 구간을 착용자 시작 시점에서 자른다
                self._push_seg(self.segmenter.cut(self.own.start))
            if was_active and not self.own.active:
                own_closed = self.own.last_own
            if own:
                self._push_wearer(own)
        chunks = self.svad_b.push(blk.t, blk.b)
        if own_closed is not None:
            # 착용자 발화가 끝났다: hangover 동안 가려졌던 B 프레임을 실제 종료 시점부터 다시 넣는다
            for tc, p, d in self.b_hist:
                if tc >= own_closed + self.tail_guard:
                    self._push_seg(self.segmenter.update(tc, p, d))
        self.b_hist.extend(chunks)
        if self.single_mic or not self.own.active:
            for tc, p, d in chunks:
                self._push_seg(self.segmenter.update(tc, p, d))

    def _push_wearer(self, iv: tuple[float, float]) -> None:
        ring = self.ring_b if self.single_mic else self.ring_a
        audio = ring.get(iv[0] - self.pad, iv[1] + self.pad)
        self.ctl.put(("wearer", iv, time.monotonic(), audio))

    def _push_seg(self, seg: Optional[Segment]) -> None:
        if seg is None:
            return
        seg.audio = self.ring_b.get(seg.t_start - self.pad, seg.t_end + self.pad)
        seg.wearer_overlap = 0.0 if self.single_mic else \
            self.own.overlap_ratio(seg.t_start, seg.t_end, self.t_stream)
        self.ctl.put(("seg", seg))

    # ------------------------------------------------------------ control
    def _now(self, t_hint: float) -> float:
        """정책에 넘길 '현재' 스트림 시간. 실시간이면 실제 오디오 시각, 최대 속도 재생이면 구간 끝(지연 0 가정)."""
        if getattr(self.src, "realtime", True):
            return max(self.t_stream, t_hint)
        return t_hint

    def _control_loop(self) -> None:
        while True:
            try:
                msg = self.ctl.get(timeout=0.2)
            except queue.Empty:
                msg = None
            try:
                if msg is not None:
                    if msg[0] == "stop":
                        break
                    getattr(self, "_on_" + msg[0])(*msg[1:])
                self._housekeeping()
            except Exception as e:
                import traceback
                self.log(f"[control] {msg[0] if msg else 'tick'} 처리 오류: {e}\n{traceback.format_exc()}")

    def _housekeeping(self) -> None:
        now_wall = time.monotonic()
        for sid, dl in list(self.llm_deadline.items()):
            if now_wall > dl:   # requests 타임아웃이 안 먹혀도 정책은 진행
                del self.llm_deadline[sid]
                self._apply_llm(sid, None)
        if now_wall - self.last_metrics >= self.cfg["server"]["metrics_interval_s"]:
            self.last_metrics = now_wall
            self.emit(self._metrics())

    def _metrics(self) -> dict:
        lat = list(self.lat_ms)
        j = self.m.judge
        llm = list(j.lat_ms) if j else []
        rss = None
        try:
            import psutil
            rss = round(psutil.Process().memory_info().rss / 1e6, 1)
        except Exception:
            pass
        return {
            "type": "metrics",
            "latency_mean_ms": round(float(np.mean(lat)), 1) if lat else None,
            "latency_p95_ms": round(float(np.percentile(lat, 95)), 1) if lat else None,
            "llm_mean_ms": round(float(np.mean(llm)), 1) if llm else None,
            "llm_p95_ms": round(float(np.percentile(llm, 95)), 1) if llm else None,
            "llm_timeouts": j.timeouts if j else 0,
            "asr_queue": self.asr.qsize(),
            "asr_dropped": self.asr.dropped,
            "asr_mean_ms": round(float(np.mean(self.asr.proc_ms)), 1) if self.asr.proc_ms else None,
            "control_queue": self.ctl.qsize(),
            "llm_inflight": j.inflight if j else 0,
            "llm_calls": getattr(j, "calls", 0) if j else 0,
            "llm_cache_hits": getattr(j, "cache_hits", 0) if j else 0,
            "llm_status": self._llm_status(),
            "sound_skipped": self.sound.skipped if self.sound else 0,
            "rss_mb": rss,
            "stream_t": round(self.t_stream, 1),
        }

    def _trim(self) -> None:
        """장시간 실행: 기록용이 아니면 오래된 구간 기록을 버린다(메모리 일정)."""
        if not self.record_segments and len(self.records) > 1000:
            for k in list(self.records)[:300]:
                del self.records[k]
            for k in list(self.turn_rec)[:-200]:
                del self.turn_rec[k]

    def _on_tick(self, t: float) -> None:
        self._trim()
        for ev in self.policy.tick(self._now(t)):
            self.emit(ev)

    def _on_wearer(self, iv: tuple[float, float], closed_wall: float, audio: Optional[np.ndarray] = None,
                   by: str = "margin") -> None:
        t0, t1 = iv
        turn_id, evs = self.policy.on_wearer_end(t0, t1, now=self._now(t1))
        for ev in evs:
            self.emit(ev)
        self.n_wearer += 1
        wid = f"w{self.n_wearer:05d}"
        self.turn_rec[turn_id] = wid
        if audio is None:
            ring = self.ring_b if self.single_mic else self.ring_a
            audio = ring.get(t0 - self.pad, t1 + self.pad)
        self.seg_info[wid] = {"t_start": t0, "t_end": t1, "closed_wall": closed_wall, "turn_id": turn_id}
        self.records[wid] = {"seg_id": wid, "t_start": round(t0, 3), "t_end": round(t1, 3), "is_wearer": True,
                             "wearer_by": by, "turn_id": turn_id, "text": None}
        self.asr.submit(ASRJob(wid, audio, lambda jid, text, info: self.ctl.put(("asr", jid, text, info)),
                               priority=True))

    def _on_seg(self, seg: Segment) -> None:
        ov = getattr(seg, "wearer_overlap", 0.0)
        if not self.single_mic and ov >= self.cfg["ownvoice"]["overlap_ratio"]:
            # 착용자 구간: 다른 사람 판정에서 제외(받아쓰기는 채널 A 본인 구간이 담당)
            self.records[seg.seg_id] = {"seg_id": seg.seg_id, "t_start": round(seg.t_start, 3),
                                        "t_end": round(seg.t_end, 3), "is_wearer": True,
                                        "wearer_by": "overlap", "overlap": round(ov, 3), "skip": True}
            return
        audio = seg.audio if seg.audio is not None else self.ring_b.get(seg.t_start - self.pad, seg.t_end + self.pad)
        emb = None
        if self.embedder is not None and self.registry.needs_embedding(seg.duration):
            try:
                emb = self.embedder(audio)
            except Exception as e:
                self.log(f"[speaker] 임베딩 오류: {e}")
        if self.single_mic and emb is not None and self.registry.wearer is not None:
            ws = self.registry.wearer_sim(emb)
            if ws >= self.cfg["ownvoice"]["single_mic_sim"]:
                self._on_wearer((seg.t_start, seg.t_end), seg.closed_wall, audio, by=f"ecapa:{ws:.2f}")
                return
        sid, sim, is_new = self.registry.assign(emb, seg.duration)
        if self.record_segments and emb is not None:
            self.embs[seg.seg_id] = emb
        self.seg_info[seg.seg_id] = {"t_start": seg.t_start, "t_end": seg.t_end, "closed_wall": seg.closed_wall,
                                     "speaker_id": sid, "sim": sim}
        self.records[seg.seg_id] = {"seg_id": seg.seg_id, "t_start": round(seg.t_start, 3),
                                    "t_end": round(seg.t_end, 3), "is_wearer": False,
                                    "speaker_id": sid, "sim": round(sim, 4), "new_speaker": is_new,
                                    "forced_cut": seg.forced, "overlap": round(ov, 3)}
        self.asr.submit(ASRJob(seg.seg_id, audio, lambda jid, text, info: self.ctl.put(("asr", jid, text, info))))

    def _on_asr(self, jid: str, text: str, info: dict) -> None:
        si = self.seg_info.get(jid)
        if si is None:
            return
        rec = self.records.get(jid, {})
        rec["text"] = text
        rec["asr_ms"] = info.get("asr_ms")
        if info.get("dropped"):
            rec["dropped"] = True
        lat = (time.monotonic() - si["closed_wall"]) * 1000
        if jid.startswith("w"):
            self.policy.set_wearer_text(si["turn_id"], text)
            if text:
                self.emit({"type": "caption", "id": jid, "t_start": round(si["t_start"], 3),
                           "t_end": round(si["t_end"], 3), "speaker_id": None, "role": "wearer", "text": text,
                           "prob": None, "evidence": {}, "pending_llm": False, "latency_ms": round(lat, 1)})
            # 착용자 텍스트를 기다리던 LLM 판정 진행
            waiting = [d for d in self.llm_deferred if d[1] == si["turn_id"]]
            self.llm_deferred = [d for d in self.llm_deferred if d[1] != si["turn_id"]]
            for seg_id, turn_id in waiting:
                self._request_llm(seg_id, turn_id)
            self.seg_info.pop(jid, None)
            return
        if not text:   # 환각 필터/무음/버려짐 → 표시하지 않음
            rec["skip"] = True
            self.seg_info.pop(jid, None)
            return
        now = self._now(si["t_end"])
        feat = SegFeat(seg_id=jid, t_start=si["t_start"], t_end=si["t_end"], speaker_id=si["speaker_id"],
                       sim=si["sim"], text=text)
        T, gap, turn_id = self.policy.timing(si["t_start"])
        rec["gap"] = None if gap is None else round(gap, 3)
        rec["turn_id"] = turn_id
        rec["wearer_seg"] = self.turn_rec.get(turn_id)
        decision, evs = self.policy.on_segment(feat, now)
        rec["speaker_resolved"] = feat.speaker_id
        decision["type"] = "caption"
        decision["latency_ms"] = round(lat, 1)
        self.lat_ms.append(lat)
        self.emit(decision)
        for ev in evs:
            self.emit(ev)
        # 호명: 판정과 무관하게 알림 + 후보
        if self.namecall:
            hit = self.namecall.detect(text)
            if hit:
                rec["name_call"] = hit
                for ev in self.policy.on_name_call(feat.speaker_id, f"호명: {hit['match']}", hit["score"], now):
                    self.emit(ev)
        # 인접쌍 판정
        lc = self.cfg["llm"]
        eligible = gap is not None and self.cfg["policy"]["timing_early_s"] <= gap <= lc["call_window_s"]
        rec["llm_eligible"] = eligible
        if eligible and self.m.judge is not None and self.m.judge.available and \
                (decision["pending_llm"] or self.always_llm):
            self._request_llm(jid, turn_id)
        self.seg_info.pop(jid, None)

    def _request_llm(self, seg_id: str, turn_id: int) -> None:
        a_text, prev = self.policy.llm_context(turn_id, self.cfg["llm"]["context_turns"])
        if a_text is None:   # 착용자 ASR이 아직 → 기다린다
            self.llm_deferred.append((seg_id, turn_id))
            self.llm_deadline[seg_id] = time.monotonic() + self.cfg["llm"]["timeout_s"] + 5.0
            return
        b_text = self.records.get(seg_id, {}).get("text", "")
        if not a_text or not b_text:
            self.llm_deadline.pop(seg_id, None)
            self._apply_llm(seg_id, None)
            return
        rec = self.records.get(seg_id, {})
        rec["llm_input"] = {"prev": prev, "a": a_text, "b": b_text}
        # P3 문맥(화자 표시 최근 4턴)도 함께 기록 — 오프라인 판정기 비교(tools/judge_offline.py)와 같은 함수
        from app.llm_judge import recent_turns
        a3, prev3 = recent_turns([r for r in self.records.values() if r.get("t_start") is not None], rec)
        rec["llm_input_p3"] = {"prev": prev3, "a": a3 or a_text, "b": b_text}
        if getattr(self.m.judge, "variant", "P1c") == "P3":
            prev, a_text = prev3, a3 or a_text
        rec["llm_called"] = True
        self.llm_deadline[seg_id] = time.monotonic() + self.cfg["llm"]["timeout_s"] + 1.5
        self.m.judge.judge_async(prev, a_text, b_text, lambda res: self.ctl.put(("llm", seg_id, res)))

    def _on_llm(self, seg_id: str, res: Optional[dict]) -> None:
        if seg_id not in self.llm_deadline:
            return   # 이미 타임아웃 처리됨
        del self.llm_deadline[seg_id]
        self._apply_llm(seg_id, res)

    def _apply_llm(self, seg_id: str, res: Optional[dict]) -> None:
        rec = self.records.get(seg_id)
        if rec is not None:
            rec["llm"] = res
        self.llm_deferred = [d for d in self.llm_deferred if d[0] != seg_id]
        info = self.records.get(seg_id, {})
        now = self._now(info.get("t_end", self.t_stream))
        upd, evs = self.policy.on_llm_result(seg_id, res, now)
        if upd:
            upd["type"] = "caption_update"
            if res and res.get("latency_ms") is not None:
                upd["llm_ms"] = res["latency_ms"]
            self.emit(upd)
        for ev in evs:
            self.emit(ev)

    def _on_sound(self, t: float, scores: dict) -> None:
        self.ctl.put(("sound_result", t, scores))

    def _on_sound_result(self, t: float, scores: dict) -> None:
        for ev in self.debouncer.update(t, scores):
            self.emit(ev)

    def _on_cmd(self, cmd: dict) -> None:
        c = cmd.get("cmd")
        now = self.t_stream
        if c == "set_mode" and cmd.get("mode") in MODES:
            for ev in self.policy.set_mode(cmd["mode"]):
                self.emit(ev)
        elif c == "toggle_partner" and cmd.get("speaker_id") is not None:
            sid = int(cmd["speaker_id"])
            on = not self.policy.is_partner(sid) if cmd.get("on") is None else bool(cmd["on"])
            for ev in self.policy.set_partner(sid, on, now):
                self.emit(ev)
        elif c == "reset":
            mode = self.policy.mode
            self.policy.reset()
            self.policy.mode = mode
            wearer = self.registry.wearer
            self.registry.reset()
            self.registry.wearer = wearer
            self.captions.clear()
            self.emit({"type": "reset"})
            self.emit(self.snapshot())

    # ------------------------------------------------------------ 출력
    def write_segments(self, path: Path) -> int:
        recs = sorted(self.records.values(), key=lambda r: (r["t_end"], r["seg_id"]))
        with open(path, "w", encoding="utf-8") as f:
            meta = {"_meta": True, "run": self.run_name, "profile": self.cfg.get("_profile"),
                    "models": self.m.describe(), "single_mic": self.single_mic,
                    "wearer": self.cfg["wearer"]["name"],
                    "llm_model": getattr(self.m.judge, "model", None),
                    "llm_prompt_version": getattr(self.m.judge, "prompt_version", None) or _prompt_version(),
                    "llm_variant": getattr(self.m.judge, "variant", None),
                    "config_name": config_name(self.cfg),
                    "language": self.cfg["asr"].get("language"),
                    "llm_cache": bool(self.cfg["llm"].get("cache", True)),
                    "synthetic": bool(getattr(self.src, "meta", {}).get("synthetic", False))}
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")
        if self.embs:
            np.savez_compressed(str(path).replace(".segments.jsonl", ".emb.npz"), **self.embs)
        return len(recs)

    def close(self) -> None:
        if self.log_file:
            self.log_file.close()
            self.log_file = None


def _prompt_version() -> str:
    try:
        from app.llm_judge import PROMPT_VERSION
        return PROMPT_VERSION
    except Exception:
        return "?"


class _NullASR:
    desc = "none"

    def transcribe(self, audio):
        return "", []


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    return str(o)
