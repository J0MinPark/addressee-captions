"""인접쌍 판정: 로컬 Ollama LLM. 타임아웃 2.5초, 실패 시 None, 같은 입력은 캐시.

Qwen3 사고 모드는 끈다: /api/chat 에 "think": false. 지원하지 않는 Ollama면 프롬프트 끝에 "/no_think".
LLM 상태는 숨기지 않는다: 시작 헬스체크(실제 판정 1회 + 지연), 10초마다 /api/ps 점검,
status 를 시작 로그·대시보드 칩/배너·metrics.llm_status 로 내보낸다.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

SYSTEM_PROMPT = (
    "너는 대화 분석기다. A가 방금 말했고, 그 직후 B가 말했다. B의 말이 A의 말에 대한 응답으로서 "
    "인접쌍(질문→대답, 인사→인사, 요청→수락/거절, 제안→응답, 평가→반응)을 이루는지 판단하라. "
    "B가 A가 아닌 다른 사람에게 말하거나 A의 말과 무관한 주제를 말하면 pair는 false다. JSON만 출력하라."
)

PAIR_TYPES = ["질문-대답", "인사-인사", "요청-수락거절", "제안-응답", "평가-반응", "없음"]

# 필드 순서가 중요하다: 작은 모델은 먼저 type(행위 분류)을 고르게 하면 pair 판단이 정확해진다
# (18개 점검 문장에서 pair 먼저 11/18 → type 먼저 13/18, qwen3:4b 기준).
SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": PAIR_TYPES},
        "pair": {"type": "boolean"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["type", "pair", "confidence"],
}

# few-shot 6개: 짝 3개, 짝 아님 3개 (짝 아님에는 "착용자 직후 옆 사람이 다른 사람에게 하는 말" 포함)
FEW_SHOT = [
    ({"prev": [], "a": "혹시 지금 몇 시예요?", "b": "세 시 반이요."},
     {"type": "질문-대답", "pair": True, "confidence": "high"}),
    ({"prev": [], "a": "안녕하세요, 오랜만이에요.", "b": "어 안녕하세요! 잘 지내셨어요?"},
     {"type": "인사-인사", "pair": True, "confidence": "high"}),
    ({"prev": [("B", "이거 어떻게 하는 거예요?")], "a": "그 파일 좀 저한테 보내 주실 수 있어요?",
      "b": "네, 지금 바로 보내 드릴게요."},
     {"type": "요청-수락거절", "pair": True, "confidence": "high"}),
    ({"prev": [], "a": "점심 뭐 드실래요?", "b": "야 지훈아, 너 어제 그 경기 봤어? 대박이더라."},
     {"type": "없음", "pair": False, "confidence": "high"}),
    ({"prev": [], "a": "이 발표 자료 어떤 것 같아요?", "b": "아 맞다, 나 주차비 정산 안 했네."},
     {"type": "없음", "pair": False, "confidence": "high"}),
    ({"prev": [], "a": "여기 앉아도 돼요?", "b": "엄마, 나 오늘 좀 늦을 것 같아. 저녁 먼저 먹어."},
     {"type": "없음", "pair": False, "confidence": "high"}),
]


# 영어 버전(AMI 평가용). type 값(enum)은 한국어 그대로 둔다(정책·평가 코드와 공유).
SYSTEM_PROMPT_EN = (
    "You are a conversation analyzer. A has just spoken, and B spoke right after. Decide whether B's utterance "
    "forms an adjacency pair as a response to A's utterance (question→answer, greeting→greeting, "
    "request→accept/decline, proposal→response, assessment→reaction). If B is talking to someone other than A, "
    "or about a topic unrelated to what A said, pair is false. Output JSON only. "
    "type must be one of: 질문-대답 (question-answer), 인사-인사 (greeting), 요청-수락거절 (request-accept/decline), "
    "제안-응답 (proposal-response), 평가-반응 (assessment-reaction), 없음 (none)."
)

FEW_SHOT_EN = [
    ({"prev": [], "a": "What time is the next meeting?", "b": "I think it's at three."},
     {"type": "질문-대답", "pair": True, "confidence": "high"}),
    ({"prev": [], "a": "Shall we go with the rubber case then?", "b": "Yeah, that sounds good to me."},
     {"type": "제안-응답", "pair": True, "confidence": "high"}),
    ({"prev": [("B", "So that's the budget.")], "a": "Could you send me those slides after the meeting?",
      "b": "Sure, I'll email them to you."},
     {"type": "요청-수락거절", "pair": True, "confidence": "high"}),
    ({"prev": [], "a": "Do you think the remote needs a display?",
      "b": "Mark, can you pass me that pen over there?"},
     {"type": "없음", "pair": False, "confidence": "high"}),
    ({"prev": [], "a": "I really like the yellow colour.", "b": "Wait, is the projector still on?"},
     {"type": "없음", "pair": False, "confidence": "high"}),
    ({"prev": [], "a": "How much would the speech recognition add to the cost?",
      "b": "Sarah, did you get the email from the marketing people?"},
     {"type": "없음", "pair": False, "confidence": "high"}),
]

PROMPTS = {"ko": (SYSTEM_PROMPT, FEW_SHOT), "en": (SYSTEM_PROMPT_EN, FEW_SHOT_EN)}
PROBES = {"ko": ([], "이거 얼마예요?", "만 이천 원입니다."),          # 헬스체크용 문장 쌍(정답: 짝)
          "en": ([], "How much does this cost?", "It's twelve euros.")}


def _prompt_version(lang: str = "ko") -> str:
    """프롬프트·few-shot·스키마·언어가 바뀌면 바뀌는 8자리 해시. 캐시 키와 결과 기록에 들어간다."""
    sys_p, shots = PROMPTS[lang]
    blob = json.dumps([lang, sys_p, shots, SCHEMA], ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]


PROMPT_VERSION = _prompt_version("ko")
PROBE = PROBES["ko"]


def format_user(prev: list[tuple[str, str]], a: str, b: str, lang: str = "ko") -> str:
    lines = []
    if lang == "en":
        if prev:
            lines.append("Previous conversation:")
            lines += [f"{who}: {txt}" for who, txt in prev]
            lines.append("")
        lines.append(f"A (just said): {a}")
        lines.append(f"B (right after): {b}")
        return "\n".join(lines)
    if prev:
        lines.append("이전 대화:")
        lines += [f"{who}: {txt}" for who, txt in prev]
        lines.append("")
    lines.append(f"A(방금): {a}")
    lines.append(f"B(직후): {b}")
    return "\n".join(lines)


def to_prob(pair: bool, confidence: str, table: dict) -> float:
    key = f"{'yes' if pair else 'no'}_{confidence if confidence in ('high', 'medium', 'low') else 'low'}"
    return float(table.get(key, 0.5))


def parse_output(content: str) -> Optional[dict]:
    if not content:
        return None
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    m = re.search(r"\{.*\}", content, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    pair = obj.get("pair")
    if isinstance(pair, str):
        pair = pair.strip().lower() in ("true", "yes", "1")
    if not isinstance(pair, bool):
        return None
    conf = str(obj.get("confidence", "low")).lower()
    if conf not in ("high", "medium", "low"):
        conf = "low"
    typ = obj.get("type", "없음")
    if typ not in PAIR_TYPES:
        typ = "없음"
    return {"pair": pair, "confidence": conf, "type": typ}


def model_device(m: dict) -> str:
    """/api/ps 항목 → GPU | GPU+CPU | CPU."""
    size, vram = m.get("size", 0) or 0, m.get("size_vram", 0) or 0
    if vram <= 0:
        return "CPU"
    return "GPU" if vram >= 0.95 * size else "GPU+CPU"


class LLMJudge:
    """Ollama 인접쌍 판정기 + 상태 감시.

    status = {"state": ok|fail|off|init, "model", "device": GPU|GPU+CPU|CPU|-, "ms", "reason"}
    state 가 ok 가 아니면 available=False → 파이프라인은 LLM을 부르지 않고 정책은 타이밍 규칙으로 간다.
    """

    def __init__(self, cfg: dict, log=print):
        self.c = cfg["llm"]
        self.table = cfg["policy"]["llm_prob"]
        self.log = log
        self.url = self.c["url"].rstrip("/")
        self.model: Optional[str] = None
        self.use_cache = bool(self.c.get("cache", True))
        self.lang = self.c.get("prompt_lang", "ko") if self.c.get("prompt_lang", "ko") in PROMPTS else "ko"
        self.prompt_version = _prompt_version(self.lang)
        self.think_supported = True
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="llm")
        self.lat_ms: deque[float] = deque(maxlen=100)
        self.calls = self.timeouts = self.failures = self.cache_hits = 0
        self.consec_timeouts = 0
        self.inflight = 0
        self.status = {"state": "init", "model": None, "device": "-", "ms": None, "reason": "시작 전"}
        self._mon_stop = threading.Event()
        import requests
        self.session = requests.Session()
        self.session.trust_env = False  # 프록시 무시(로컬)

    # ------------------------------------------------------------ 상태
    @property
    def available(self) -> bool:
        return self.model is not None and self.status["state"] == "ok"

    def _set(self, state: str, reason: str = "", **kw) -> None:
        st = {"state": state, "model": self.model, "device": self.status.get("device", "-"),
              "ms": self.status.get("ms"), "reason": reason}
        if state in ("fail", "off") and "device" not in kw:
            st["device"] = "-"
        st.update(kw)
        self.status = st

    def health_line(self) -> str:
        st = self.status
        if st["state"] == "ok":
            return f"llm={st['model']} ok {st['ms']:.0f}ms ({st['device']})"
        if st["state"] == "off":
            return "llm=OFF (--no-llm 또는 config)"
        return f"llm=FAIL {st['reason']}"

    def _tags(self) -> list[str]:
        r = self.session.get(f"{self.url}/api/tags", timeout=2.0)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def _ps(self) -> list[dict]:
        r = self.session.get(f"{self.url}/api/ps", timeout=2.0)
        r.raise_for_status()
        return r.json().get("models", []) or []

    def unload_others(self) -> list[str]:
        """선택한 모델이 아닌, 로드된 모델을 내린다(VRAM 확보)."""
        out = []
        try:
            for m in self._ps():
                name = m.get("name") or m.get("model")
                if name and name != self.model:
                    self.session.post(f"{self.url}/api/generate", json={"model": name, "keep_alive": 0}, timeout=5)
                    out.append(name)
        except Exception as e:
            self.log(f"[llm] 다른 모델 내리기 실패: {e}")
        if out:
            self.log(f"[llm] 다른 모델 내림: {', '.join(out)}")
        return out

    def _probe(self, timeout: float) -> bool:
        """실제 판정 1회로 지연을 잰다. 성공하면 status=ok."""
        t0 = time.perf_counter()
        try:
            res = self._call(*PROBES[self.lang], timeout=timeout)
        except Exception as e:
            self._set("fail", f"판정 호출 실패({type(e).__name__})")
            return False
        ms = (time.perf_counter() - t0) * 1000
        if res is None:
            self._set("fail", "판정 출력 파싱 실패")
            return False
        dev = "-"
        try:
            dev = next((model_device(m) for m in self._ps() if (m.get("name") or m.get("model")) == self.model), "-")
        except Exception:
            pass
        if dev == "CPU":
            self._set("fail", "CPU로 밀려남(VRAM 부족)", device=dev, ms=ms)
            return False
        self.consec_timeouts = 0
        self._set("ok", "" if res.get("pair") else "점검 문장을 '짝 아님'으로 판정(연결은 정상)", device=dev, ms=ms)
        return True

    def setup(self, warmup_timeout: Optional[float] = None) -> bool:
        """설치된 모델 중 config 순서대로 첫 번째를 고르고, 다른 모델을 내리고, 판정 1회로 점검한다."""
        if not self.c.get("enabled", True):
            self.model = None
            self._set("off", "비활성")
            return False
        try:
            names = self._tags()
        except Exception as e:
            self.model = None
            self._set("fail", f"Ollama 연결 실패({type(e).__name__})")
            return False
        self.model = None
        for want in self.c["models"]:
            hit = [n for n in names if n == want or n == f"{want}:latest"]
            if hit:
                self.model = hit[0]
                break
        if not self.model:
            self._set("fail", f"모델 없음 — ollama pull {self.c['models'][0]}")
            return False
        self.unload_others()
        return self._probe(warmup_timeout or self.c.get("warmup_timeout_s", 60))

    def check(self) -> dict:
        """주기 점검: 연결, 로드된 모델 이름(config와 같은지), GPU 여부, 연속 타임아웃. 끊겼으면 복구 시도."""
        if not self.c.get("enabled", True):
            return self.status
        rt = self.c.get("recover_timeout_s", 8)
        if self.model is None:
            self.setup(warmup_timeout=rt)
            return self.status
        try:
            loaded = self._ps()
        except Exception:
            self._set("fail", "Ollama 연결 끊김")
            self.model = None   # 다시 켜지면 setup 부터
            return self.status
        others = [(m.get("name") or m.get("model")) for m in loaded
                  if (m.get("name") or m.get("model")) != self.model]
        mine = next((m for m in loaded if (m.get("name") or m.get("model")) == self.model), None)
        if others:
            self._set("fail", f"모델 불일치(로드됨: {', '.join(others)})")
            self.unload_others()
            self._probe(rt)
            return self.status
        if mine is None:              # keep_alive 만료 등으로 내려감 → 다시 올린다
            self._probe(rt)
            return self.status
        dev = model_device(mine)
        if dev == "CPU":
            self._set("fail", "CPU로 밀려남(VRAM 부족)", device=dev)
        elif self.consec_timeouts >= 2:
            self._set("fail", f"타임아웃 {self.consec_timeouts}회 연속", device=dev)
            self._probe(self.c["timeout_s"] * 2)
        elif self.status["state"] != "ok":
            self._probe(rt)
        else:
            recent = list(self.lat_ms)[-10:]
            ms = sum(recent) / len(recent) if recent else self.status.get("ms")
            self._set("ok", self.status.get("reason", ""), device=dev, ms=ms)
        return self.status

    def start_monitor(self, interval: float, on_change: Callable[[dict], None]) -> None:
        """정상일 땐 interval(10초)마다, 실패 중엔 monitor_fail_interval_s(2초)마다 점검 → 켜지면 10초 안에 복구."""
        fail_iv = self.c.get("monitor_fail_interval_s", 2)

        def loop():
            last = None
            while not self._mon_stop.wait(interval if self.status["state"] in ("ok", "off") else fail_iv):
                try:
                    was_ok = self.status["state"] == "ok"
                    st = self.check()
                    if st["state"] == "ok" and not was_ok and (st.get("ms") or 0) > 1500:
                        self._probe(self.c["timeout_s"] * 2)   # 콜드 로딩 시간 말고 실제 판정 지연을 표시
                        st = self.status
                except Exception as e:
                    self._set("fail", f"점검 오류({type(e).__name__})")
                    st = self.status
                key = (st["state"], st["reason"], st.get("device"), st.get("model"))
                if key != last:
                    last = key
                    on_change(dict(st))
        threading.Thread(target=loop, name="llm-monitor", daemon=True).start()

    def stop_monitor(self) -> None:
        self._mon_stop.set()

    # ------------------------------------------------------------ 호출
    def _messages(self, prev, a, b) -> list[dict]:
        sys_p, shots = PROMPTS[self.lang]
        msgs = [{"role": "system", "content": sys_p}]
        for ex_in, ex_out in shots:
            msgs.append({"role": "user", "content": format_user(ex_in["prev"], ex_in["a"], ex_in["b"], self.lang)})
            msgs.append({"role": "assistant", "content": json.dumps(ex_out, ensure_ascii=False)})
        user = format_user(prev, a, b, self.lang)
        if not self.think_supported:
            user += " /no_think"
        msgs.append({"role": "user", "content": user})
        return msgs

    def _call(self, prev, a, b, timeout: float) -> Optional[dict]:
        body = {
            "model": self.model, "stream": False, "format": SCHEMA, "keep_alive": self.c["keep_alive"],
            "options": {"temperature": self.c["temperature"], "num_predict": self.c["num_predict"]},
            "messages": self._messages(prev, a, b),
        }
        if self.think_supported:
            body["think"] = False
        r = self.session.post(f"{self.url}/api/chat", json=body, timeout=timeout)
        if r.status_code == 400 and self.think_supported and "think" in r.text.lower():
            self.think_supported = False
            self.log("[llm] 이 Ollama는 think 옵션 미지원 → /no_think 프롬프트 사용")
            return self._call(prev, a, b, timeout)
        r.raise_for_status()
        out = parse_output(r.json().get("message", {}).get("content", ""))
        if out is not None:
            out["prob"] = to_prob(out["pair"], out["confidence"], self.table)
        return out

    def judge(self, prev: list[tuple[str, str]], a: str, b: str) -> Optional[dict]:
        """동기 호출. 타임아웃/파싱 실패 → None. 캐시 키 = (모델, 프롬프트 버전, 입력)."""
        if not self.available or not a or not b:
            return None
        model = self.model
        key = (model, self.prompt_version, tuple(tuple(x) for x in prev), a, b)
        if self.use_cache:
            with self.lock:
                if key in self.cache:
                    self.cache.move_to_end(key)
                    self.cache_hits += 1
                    return dict(self.cache[key], cached=True)
        self.calls += 1
        t0 = time.perf_counter()
        try:
            res = self._call(prev, a, b, timeout=self.c["timeout_s"])
            self.consec_timeouts = 0
        except Exception as e:
            name = type(e).__name__
            if "Timeout" in name:
                self.timeouts += 1
                self.consec_timeouts += 1
                if self.consec_timeouts >= 2 and self.status["state"] == "ok":
                    self._set("fail", f"타임아웃 {self.consec_timeouts}회 연속")
            else:
                self.failures += 1
                self.log(f"[llm] 호출 실패: {name}: {e}")
            res = None
        ms = (time.perf_counter() - t0) * 1000
        self.lat_ms.append(ms)
        if res is None:
            return None
        res.update(latency_ms=round(ms, 1), model=model, prompt_version=self.prompt_version)
        if self.use_cache:
            with self.lock:
                self.cache[key] = res
                while len(self.cache) > self.c.get("cache_size", 512):
                    self.cache.popitem(last=False)
        return dict(res)

    def judge_async(self, prev, a, b, callback: Callable[[Optional[dict]], None]) -> None:
        self.inflight += 1

        def run():
            try:
                res = self.judge(prev, a, b)
            except Exception:
                res = None
            self.inflight -= 1
            callback(res)

        self.pool.submit(run)
