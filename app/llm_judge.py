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


# ---------------------------------------------------------------- v2: 연속 점수(p_pair) 판정기
# 변형: P1c = v1(위 프롬프트, confidence 매핑) · P1 = 인접쌍 정의 · P2 = 반응 정의 확장 · P3 = P2 + 화자 표시 최근 4턴
# P1/P2/P3 는 {"pair": bool} 만 출력하게 하고, Ollama logprobs 에서 true/false 토큰 확률로 p_pair 를 계산한다.
SCHEMA_PAIR = {"type": "object", "properties": {"pair": {"type": "boolean"}}, "required": ["pair"]}
VARIANTS = ("P1c", "P1", "P2", "P3")

_P1_KO = ("너는 대화 분석기다. A가 방금 말했고, 그 직후 B가 말했다. B의 말이 A의 말에 대한 응답으로서 인접쌍(질문→대답, "
          "인사→인사, 요청→수락/거절, 제안→응답, 평가→반응)을 이루는지 판단하라. B가 A가 아닌 다른 사람에게 말하거나 "
          "A의 말과 무관한 주제를 말하면 pair는 false다. JSON {\"pair\": true|false}만 출력하라.")
_P1_EN = ("You are a conversation analyzer. A has just spoken, and B spoke right after. Decide whether B's utterance forms "
          "an adjacency pair as a response to A's utterance (question→answer, greeting→greeting, request→accept/decline, "
          "proposal→response, assessment→reaction). If B is talking to someone other than A, or about a topic unrelated "
          "to what A said, pair is false. Output only JSON {\"pair\": true|false}.")
_P2_KO = ("너는 대화 분석기다. A가 방금 말했고, 그 직후 B가 말했다. B의 말이 A의 말에 대한 반응(대답, 동의·반대, 이어받기, "
          "맞장구, 되묻기)인가, 아니면 다른 사람이나 다른 화제를 향한 말인가? A의 말에 대한 반응이면 pair는 true, "
          "다른 사람이나 다른 화제를 향한 말이면 false다. JSON {\"pair\": true|false}만 출력하라.")
_P2_EN = ("You are a conversation analyzer. A has just spoken, and B spoke right after. Is B's utterance a reaction to A's "
          "utterance (an answer, agreement or disagreement, taking up the point, a backchannel, or a clarification "
          "question), or is it directed at someone else or at a different topic? If it is a reaction to A, pair is true; "
          "if it is directed at someone else or a different topic, pair is false. Output only JSON {\"pair\": true|false}.")
_P3_KO_ADD = (" 이전 대화의 각 줄 앞에는 화자가 표시된다: A = 방금 말한 사람(착용자), B = 판정 대상 화자, "
              "X1·X2… = 그 밖의 사람. 이전 대화를 참고해 B가 누구에게, 무엇에 대해 말하는지 판단하라.")
_P3_EN_ADD = (" Each line of the previous conversation is labelled with its speaker: A = the person who just spoke "
              "(the wearer), B = the speaker being judged, X1, X2… = other people. Use the previous conversation to "
              "decide whom B is addressing and about what.")

_T, _F = {"pair": True}, {"pair": False}
_SHOT_P1_KO = [(ex_in, {"pair": ex_out["pair"]}) for ex_in, ex_out in FEW_SHOT]
_SHOT_P1_EN = [(ex_in, {"pair": ex_out["pair"]}) for ex_in, ex_out in FEW_SHOT_EN]
# P2/P3: 짝 3(대답·맞장구·되묻기) + 짝 아님 3(질문 직후 제3자에게 하는 다른 질문, 다른 화제, 다른 사람에게)
_SHOT_P2_KO = [
    ({"prev": [], "a": "이 디자인 너무 복잡하지 않아요?", "b": "음, 좀 그런 것 같아요."}, _T),
    ({"prev": [], "a": "그래서 버튼을 두 개로 줄이면 될 것 같아요.", "b": "두 개요? 어떤 거 두 개요?"}, _T),
    ({"prev": [], "a": "배터리는 충전식으로 하죠.", "b": "그건 단가가 너무 올라가서 반대예요."}, _T),
    ({"prev": [], "a": "회의 몇 시에 끝나요?", "b": "지수 씨, 혹시 펜 하나 있어요?"}, _F),
    ({"prev": [], "a": "색은 노란색이 좋겠어요.", "b": "아 맞다, 프로젝터 꺼야 되나?"}, _F),
    ({"prev": [], "a": "이 부분은 제가 정리할게요.", "b": "엄마, 나 회의 중이라 이따 전화할게."}, _F),
]
_SHOT_P2_EN = [
    ({"prev": [], "a": "Isn't this design a bit too complicated?", "b": "Mm, yeah, I think so."}, _T),
    ({"prev": [], "a": "So we could cut it down to two buttons.", "b": "Two? Which two do you mean?"}, _T),
    ({"prev": [], "a": "Let's make the battery rechargeable.", "b": "I'm against that, it pushes the unit cost up too much."}, _T),
    ({"prev": [], "a": "What time does the meeting end?", "b": "Sarah, do you have a spare pen?"}, _F),
    ({"prev": [], "a": "I think yellow would be the best colour.", "b": "Oh right, should I switch off the projector?"}, _F),
    ({"prev": [], "a": "I'll put this part together.", "b": "Mum, I'm in a meeting, I'll call you later."}, _F),
]
_SHOT_P3_KO = [
    ({"prev": [("X1", "자 다음은 버튼 얘기죠."), ("A", "네.")], "a": "이 디자인 너무 복잡하지 않아요?", "b": "음, 좀 그런 것 같아요."}, _T),
    ({"prev": [("B", "버튼 수를 줄여야 해요.")], "a": "그래서 버튼을 두 개로 줄이면 될 것 같아요.", "b": "두 개요? 어떤 거 두 개요?"}, _T),
    ({"prev": [("X1", "충전 방식 정해야죠.")], "a": "배터리는 충전식으로 하죠.", "b": "그건 단가가 너무 올라가서 반대예요."}, _T),
    ({"prev": [("X1", "펜이 안 나오네.")], "a": "회의 몇 시에 끝나요?", "b": "지수 씨, 혹시 펜 하나 있어요?"}, _F),
    ({"prev": [("B", "화면이 좀 어둡네요.")], "a": "색은 노란색이 좋겠어요.", "b": "아 맞다, 프로젝터 꺼야 되나?"}, _F),
    ({"prev": [], "a": "이 부분은 제가 정리할게요.", "b": "엄마, 나 회의 중이라 이따 전화할게."}, _F),
]
_SHOT_P3_EN = [
    ({"prev": [("X1", "Right, next up is the buttons."), ("A", "Yeah.")], "a": "Isn't this design a bit too complicated?",
      "b": "Mm, yeah, I think so."}, _T),
    ({"prev": [("B", "We need fewer buttons.")], "a": "So we could cut it down to two buttons.",
      "b": "Two? Which two do you mean?"}, _T),
    ({"prev": [("X1", "We still have to decide on the power.")], "a": "Let's make the battery rechargeable.",
      "b": "I'm against that, it pushes the unit cost up too much."}, _T),
    ({"prev": [("X1", "My pen's run out.")], "a": "What time does the meeting end?", "b": "Sarah, do you have a spare pen?"}, _F),
    ({"prev": [("B", "The screen's a bit dark.")], "a": "I think yellow would be the best colour.",
      "b": "Oh right, should I switch off the projector?"}, _F),
    ({"prev": [], "a": "I'll put this part together.", "b": "Mum, I'm in a meeting, I'll call you later."}, _F),
]
PROMPTS_V2 = {
    ("ko", "P1"): (_P1_KO, _SHOT_P1_KO), ("en", "P1"): (_P1_EN, _SHOT_P1_EN),
    ("ko", "P2"): (_P2_KO, _SHOT_P2_KO), ("en", "P2"): (_P2_EN, _SHOT_P2_EN),
    ("ko", "P3"): (_P2_KO + _P3_KO_ADD, _SHOT_P3_KO), ("en", "P3"): (_P2_EN + _P3_EN_ADD, _SHOT_P3_EN),
}


def prompt_for(lang: str, variant: str) -> tuple[str, list, dict]:
    if variant == "P1c":
        sys_p, shots = PROMPTS[lang]
        return sys_p, shots, SCHEMA
    sys_p, shots = PROMPTS_V2[(lang, variant)]
    return sys_p, shots, SCHEMA_PAIR


def _prompt_version(lang: str = "ko", variant: str = "P1c") -> str:
    """프롬프트·few-shot·스키마·언어·변형이 바뀌면 바뀌는 8자리 해시. 캐시 키와 결과 기록에 들어간다."""
    sys_p, shots, schema = prompt_for(lang, variant)
    parts = [lang, sys_p, shots, schema] + ([variant] if variant != "P1c" else [])
    blob = json.dumps(parts, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]


def p_pair_from_logprobs(logprobs: list) -> Optional[float]:
    """{"pair": <bool>} 의 bool 위치 토큰 상위 후보에서 P(true) / (P(true)+P(false))."""
    import math
    for tk in logprobs or []:
        if tk.get("token", "").strip() in ("true", "false"):
            lt = max([x["logprob"] for x in tk.get("top_logprobs", []) if x["token"].strip() == "true"] or [-30.0])
            lf = max([x["logprob"] for x in tk.get("top_logprobs", []) if x["token"].strip() == "false"] or [-30.0])
            return 1.0 / (1.0 + math.exp(lf - lt))
    return None


def recent_turns(recs: list[dict], cand: dict, n: int = 4, merge_gap: float = 1.5) -> tuple[Optional[str], list]:
    """P3 문맥: 후보 구간 직전의 화자 표시 대화. recs = 구간 기록(파이프라인 records/segments.jsonl 동일 형식).
    반환 (A 텍스트 = 후보 직전의 연속 착용자 발화, 그 이전 n턴 [(A|B|X<id>, 텍스트)])."""
    t0 = cand["t_start"]
    sid = cand.get("speaker_id")
    turns = []
    for r in sorted(recs, key=lambda r: r["t_start"]):
        if r is cand or r.get("seg_id") == cand.get("seg_id") or r.get("skip") or not r.get("text"):
            continue
        if r["t_start"] >= t0:
            continue
        who = "A" if r.get("is_wearer") else ("B" if sid is not None and r.get("speaker_id") == sid
                                              else f"X{r.get('speaker_id') if r.get('speaker_id') is not None else '?'}")
        if turns and turns[-1][0] == who and r["t_start"] - turns[-1][2] <= merge_gap:
            turns[-1] = (who, turns[-1][1] + " " + r["text"], r["t_end"])
        else:
            turns.append((who, r["text"], r["t_end"]))
    if not turns or turns[-1][0] != "A":
        return None, []
    a_text = turns[-1][1]
    prev = [(w, t if len(t) <= 80 else "…" + t[-80:]) for w, t, _ in turns[:-1][-n:]]
    return a_text, prev


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
        self.variant = self.c.get("variant", "P1c") if self.c.get("variant", "P1c") in VARIANTS else "P1c"
        self.prompt_version = _prompt_version(self.lang, self.variant)
        self.logprob_ok: Optional[bool] = None   # Ollama가 logprobs를 돌려주는지(첫 호출에서 확인)
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
        sys_p, shots, _ = prompt_for(self.lang, self.variant)
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
            "model": self.model, "stream": False, "format": prompt_for(self.lang, self.variant)[2],
            "keep_alive": self.c["keep_alive"],
            "options": {"temperature": self.c["temperature"], "num_predict": self.c["num_predict"]},
            "messages": self._messages(prev, a, b),
        }
        if self.variant != "P1c":
            body["logprobs"] = True
            body["top_logprobs"] = 10
        if self.think_supported:
            body["think"] = False
        r = self.session.post(f"{self.url}/api/chat", json=body, timeout=timeout)
        if r.status_code == 400 and self.think_supported and "think" in r.text.lower():
            self.think_supported = False
            self.log("[llm] 이 Ollama는 think 옵션 미지원 → /no_think 프롬프트 사용")
            return self._call(prev, a, b, timeout)
        r.raise_for_status()
        js = r.json()
        out = parse_output(js.get("message", {}).get("content", ""))
        if out is None:
            return None
        if self.variant == "P1c":
            out["prob"] = to_prob(out["pair"], out["confidence"], self.table)
            return out
        p = p_pair_from_logprobs(js.get("logprobs"))
        if p is None:   # logprobs 미지원 → 기존 confidence 매핑으로 대체(보고됨)
            if self.logprob_ok is None:
                self.log("[llm] 이 Ollama는 logprobs를 돌려주지 않음 → p_pair 대신 confidence 매핑 사용")
            self.logprob_ok = False
            p = 0.9 if out["pair"] else 0.1
        else:
            self.logprob_ok = True
        out.update(pair=p >= 0.5, prob=float(p), p_pair=float(p), type="반응" if p >= 0.5 else "없음",
                   confidence="high" if abs(p - 0.5) > 0.4 else "medium" if abs(p - 0.5) > 0.2 else "low")
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
