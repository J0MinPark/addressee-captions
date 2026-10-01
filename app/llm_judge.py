"""인접쌍 판정: 로컬 Ollama LLM. 타임아웃 2.5초, 실패 시 None, 같은 입력은 캐시.

Qwen3 사고 모드는 끈다: /api/chat 에 "think": false. 지원하지 않는 Ollama면 프롬프트 끝에 "/no_think".
"""
from __future__ import annotations

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

SCHEMA = {
    "type": "object",
    "properties": {
        "pair": {"type": "boolean"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "type": {"type": "string", "enum": PAIR_TYPES},
    },
    "required": ["pair", "confidence", "type"],
}

# few-shot 6개: 짝 3개, 짝 아님 3개 (짝 아님에는 "착용자 직후 옆 사람이 다른 사람에게 하는 말" 포함)
FEW_SHOT = [
    ({"prev": [], "a": "혹시 지금 몇 시예요?", "b": "세 시 반이요."},
     {"pair": True, "confidence": "high", "type": "질문-대답"}),
    ({"prev": [], "a": "안녕하세요, 오랜만이에요.", "b": "어 안녕하세요! 잘 지내셨어요?"},
     {"pair": True, "confidence": "high", "type": "인사-인사"}),
    ({"prev": [("B", "이거 어떻게 하는 거예요?")], "a": "그 파일 좀 저한테 보내 주실 수 있어요?",
      "b": "네, 지금 바로 보내 드릴게요."},
     {"pair": True, "confidence": "high", "type": "요청-수락거절"}),
    ({"prev": [], "a": "점심 뭐 드실래요?", "b": "야 지훈아, 너 어제 그 경기 봤어? 대박이더라."},
     {"pair": False, "confidence": "high", "type": "없음"}),
    ({"prev": [], "a": "이 발표 자료 어떤 것 같아요?", "b": "아 맞다, 나 주차비 정산 안 했네."},
     {"pair": False, "confidence": "high", "type": "없음"}),
    ({"prev": [], "a": "여기 앉아도 돼요?", "b": "엄마, 나 오늘 좀 늦을 것 같아. 저녁 먼저 먹어."},
     {"pair": False, "confidence": "high", "type": "없음"}),
]


def format_user(prev: list[tuple[str, str]], a: str, b: str) -> str:
    lines = []
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


class LLMJudge:
    def __init__(self, cfg: dict, log=print):
        self.c = cfg["llm"]
        self.table = cfg["policy"]["llm_prob"]
        self.log = log
        self.url = self.c["url"].rstrip("/")
        self.model: Optional[str] = None
        self.think_supported = True
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="llm")
        self.lat_ms: deque[float] = deque(maxlen=100)
        self.calls = self.timeouts = self.failures = 0
        self.inflight = 0
        import requests
        self.session = requests.Session()
        self.session.trust_env = False  # 프록시 무시(로컬)

    @property
    def available(self) -> bool:
        return self.model is not None

    def setup(self) -> bool:
        """설치된 모델 중 config 순서대로 첫 번째를 고르고 워밍업한다."""
        if not self.c.get("enabled", True):
            self.log("[llm] 비활성(config)")
            return False
        try:
            r = self.session.get(f"{self.url}/api/tags", timeout=2.0)
            names = [m["name"] for m in r.json().get("models", [])]
        except Exception as e:
            self.log(f"[llm] Ollama 연결 실패({self.url}) → LLM 없이 동작: {e}")
            return False
        for want in self.c["models"]:
            hit = [n for n in names if n == want or n == f"{want}:latest" or n.split(":")[0] == want and ":" not in want]
            if hit:
                self.model = hit[0]
                break
        if not self.model:
            self.log(f"[llm] 설치된 모델 없음 (원하는 것: {self.c['models']}, 있는 것: {names}). "
                     f"`ollama pull {self.c['models'][0]}` 필요")
            return False
        t0 = time.perf_counter()
        res = self._call([], "안녕하세요.", "네, 안녕하세요!", timeout=self.c.get("warmup_timeout_s", 60))
        self.log(f"[llm] {self.model} 워밍업 {(time.perf_counter() - t0):.1f}s → {res}")
        if res is None:
            self.log("[llm] 워밍업 실패 → LLM 없이 동작")
            self.model = None
            return False
        return True

    def _messages(self, prev, a, b) -> list[dict]:
        sys = SYSTEM_PROMPT
        msgs = [{"role": "system", "content": sys}]
        for ex_in, ex_out in FEW_SHOT:
            msgs.append({"role": "user", "content": format_user(ex_in["prev"], ex_in["a"], ex_in["b"])})
            msgs.append({"role": "assistant", "content": json.dumps(ex_out, ensure_ascii=False)})
        user = format_user(prev, a, b)
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
        """동기 호출. 타임아웃/파싱 실패 → None."""
        if not self.available or not a or not b:
            return None
        key = (tuple(prev), a, b)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return dict(self.cache[key], cached=True)
        self.calls += 1
        t0 = time.perf_counter()
        try:
            res = self._call(prev, a, b, timeout=self.c["timeout_s"])
        except Exception as e:
            name = type(e).__name__
            if "Timeout" in name:
                self.timeouts += 1
            else:
                self.failures += 1
                self.log(f"[llm] 호출 실패: {name}: {e}")
            res = None
        ms = (time.perf_counter() - t0) * 1000
        self.lat_ms.append(ms)
        if res is None:
            return None
        res["latency_ms"] = round(ms, 1)
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
