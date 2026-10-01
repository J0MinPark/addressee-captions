"""정책 엔진: "이 발화는 착용자에게 하는 말인가?"

순수 로직. 입출력/스레드/모델 없음. 시간은 항상 인자(스트림 시간, 초)로 받는다.
같은 입력 순서면 같은 결과가 나오므로 재생 평가(tools/evaluate.py)가 이걸 그대로 다시 돌린다.

증거
  T 타이밍  g = 구간 시작 - 착용자 마지막 발화 종료
  S 화자    현재 partner 이고 유사도 >= spk_threshold
  L 의미    LLM 인접쌍 확률 (없으면 기여 0)
  z = b + w_t*T + w_s*S + w_l*(2L-1),  prob = sigmoid(z)

화자 상태: unknown -> candidate -> partner -> expired(-> unknown)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

MODES = ("all", "timing", "timing_speaker", "full", "semantic")
MODE_LABELS = {
    "all": "전부 표시",
    "timing": "타이밍",
    "timing_speaker": "타이밍+화자",
    "full": "전체 융합",
    "semantic": "의미만",
}
PAIR_LABELS = {
    "질문-대답": "질문→대답",
    "인사-인사": "인사→인사",
    "요청-수락거절": "요청→수락/거절",
    "제안-응답": "제안→응답",
    "평가-반응": "평가→반응",
}


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class SegFeat:
    seg_id: str
    t_start: float
    t_end: float
    speaker_id: Optional[int] = None   # None = 임베딩 없음/미상
    sim: float = 0.0
    text: str = ""

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


@dataclass
class SpeakerInfo:
    speaker_id: int
    state: str = "unknown"
    sim: float = 0.0
    last_exchange: float = -1e9
    state_since: float = 0.0
    timing_turns: int = 0
    last_counted_turn: int = -1
    last_end: float = -1e9
    manual: bool = False
    registered_at: Optional[float] = None


@dataclass
class WearerTurn:
    turn_id: int
    t_start: float
    t_end: float
    text: Optional[str] = None


@dataclass
class _Pending:
    feat: SegFeat
    T: float
    S: float
    gap: Optional[float]
    turn_id: Optional[int]
    llm_expected: bool
    L: Optional[float] = None
    pair: Optional[bool] = None
    pair_type: Optional[str] = None
    prob: float = 0.0
    role: str = "other"
    final: bool = False


class PolicyEngine:
    def __init__(self, cfg: dict, mode: Optional[str] = None, llm_available: bool = True):
        """cfg: 전체 config dict (policy, speaker 절을 읽는다)."""
        p = cfg["policy"]
        self.p = dict(p)
        self.spk_threshold = cfg["speaker"]["spk_threshold"]
        self.short_s = cfg["speaker"]["min_embed_s"]   # 이보다 짧고 화자 미상이면 상속 규칙 적용
        self.inherit_gap = cfg["speaker"]["inherit_gap_s"]
        self.call_window = cfg.get("llm", {}).get("call_window_s", p["timing_half_max_s"])
        self.mode = mode or p.get("default_mode", "full")
        assert self.mode in MODES, self.mode
        self.llm_available = llm_available
        self.reset()

    # ------------------------------------------------------------------ 상태
    def reset(self) -> None:
        self.speakers: dict[int, SpeakerInfo] = {}
        self.turns: list[WearerTurn] = []
        self.history: list[tuple[str, object]] = []   # ("wearer", WearerTurn) | ("other", text)
        self.pending: dict[str, _Pending] = {}
        self.prev_seg: Optional[tuple[Optional[int], float, float]] = None  # (spk, t_end, sim)
        self.partner_count = 0

    def set_mode(self, mode: str) -> list[dict]:
        if mode not in MODES:
            return []
        self.mode = mode
        return [{"type": "mode", "mode": mode, "label": MODE_LABELS[mode]}]

    def weights(self, mode: Optional[str] = None) -> tuple[float, float, float, float]:
        mode = mode or self.mode
        b, wt, ws, wl = self.p["bias"], self.p["w_t"], self.p["w_s"], self.p["w_l"]
        if mode == "timing":
            ws = wl = 0.0
        elif mode == "timing_speaker":
            wl = 0.0
        elif mode == "semantic":
            wt = ws = 0.0
        return b, wt, ws, wl

    def mode_uses_llm(self, mode: Optional[str] = None) -> bool:
        return (mode or self.mode) in ("full", "semantic")

    def _speaker(self, sid: int, now: float) -> tuple[SpeakerInfo, list[dict]]:
        if sid in self.speakers:
            return self.speakers[sid], []
        s = SpeakerInfo(speaker_id=sid, state_since=now)
        self.speakers[sid] = s
        return s, [self._spk_event(s)]

    @staticmethod
    def _spk_event(s: SpeakerInfo) -> dict:
        return {"type": "speaker_state", "speaker_id": s.speaker_id, "state": s.state,
                "sim": round(float(s.sim), 3), "manual": s.manual}

    def _set_state(self, s: SpeakerInfo, state: str, now: float) -> list[dict]:
        if s.state == state:
            return []
        s.state = state
        s.state_since = now
        ev = [self._spk_event(s)]
        if state == "partner":
            s.registered_at = now
            s.last_exchange = now
            self.partner_count += 1
            ev.append({"type": "partner_added", "speaker_id": s.speaker_id,
                       "label": f"대화 상대 #{s.speaker_id} 추가", "t": now})
        if state in ("unknown", "expired"):
            s.timing_turns = 0
        return ev

    def is_partner(self, sid: Optional[int]) -> bool:
        return sid is not None and sid in self.speakers and self.speakers[sid].state == "partner"

    # ------------------------------------------------------------- 착용자
    def on_wearer_end(self, t_start: float, t_end: float, now: Optional[float] = None) -> tuple[int, list[dict]]:
        """착용자 발화 종료(텍스트는 나중에 set_wearer_text)."""
        now = t_end if now is None else now
        turn = WearerTurn(turn_id=len(self.turns), t_start=t_start, t_end=t_end)
        self.turns.append(turn)
        self.history.append(("wearer", turn))
        # 착용자가 상대의 말에 바로 답했다 -> 그 상대와의 교대
        for s in self.speakers.values():
            if s.state == "partner" and self.p["timing_early_s"] <= t_start - s.last_end <= self.p["timing_half_max_s"]:
                s.last_exchange = max(s.last_exchange, t_end)
        return turn.turn_id, []

    def set_wearer_text(self, turn_id: int, text: str) -> None:
        if 0 <= turn_id < len(self.turns):
            self.turns[turn_id].text = text or ""

    def wearer_turn_for(self, t_start: float) -> Optional[WearerTurn]:
        """이 구간 직전의 착용자 발화: 구간 시작 전에 시작한 마지막 착용자 발화."""
        best = None
        for turn in reversed(self.turns):
            if turn.t_start < t_start:
                best = turn
                break
        return best

    def timing(self, t_start: float) -> tuple[float, Optional[float], Optional[int]]:
        turn = self.wearer_turn_for(t_start)
        if turn is None:
            return 0.0, None, None
        g = t_start - turn.t_end
        p = self.p
        if 0 <= g <= p["timing_full_max_s"]:
            T = 1.0
        elif p["timing_early_s"] <= g < 0 or p["timing_full_max_s"] < g <= p["timing_half_max_s"]:
            T = 0.5
        else:
            T = 0.0
        return T, g, turn.turn_id

    def llm_context(self, turn_id: int, n_prev: int = 2) -> tuple[Optional[str], list[tuple[str, str]]]:
        """(착용자 마지막 말, 그 이전 대화 n턴 [(화자, 텍스트)])."""
        if turn_id is None or not (0 <= turn_id < len(self.turns)):
            return None, []
        turn = self.turns[turn_id]
        idx = next((i for i, (k, v) in enumerate(self.history) if k == "wearer" and v is turn), None)
        prev = []
        if idx is not None:
            for k, v in reversed(self.history[:idx]):
                if len(prev) >= n_prev:
                    break
                if k == "wearer":
                    if v.text:
                        prev.append(("A", v.text))
                elif v:
                    prev.append(("B", v))
            prev.reverse()
        return turn.text, prev

    # ------------------------------------------------------------- 점수
    def score(self, T: float, S: float, L: Optional[float], mode: Optional[str] = None) -> float:
        b, wt, ws, wl = self.weights(mode)
        z = b + wt * T + ws * S + (wl * (2 * L - 1) if L is not None else 0.0)
        return sigmoid(z)

    def _role(self, prob: float, sid: Optional[int]) -> str:
        if prob >= self.p["show_threshold"]:
            return "partner"
        return "unknown" if sid is None else "other"

    def chip(self, pd: _Pending) -> str:
        parts = []
        if pd.gap is not None and pd.T > 0:
            parts.append(f"응답 {max(pd.gap, 0):.1f}초" if pd.gap >= 0 else f"겹침 {-pd.gap:.1f}초")
        if pd.L is not None:
            parts.append(PAIR_LABELS.get(pd.pair_type, pd.pair_type) if pd.pair else "짝 아님")
        elif pd.llm_expected and not pd.final:
            parts.append("판정 중")
        sid = pd.feat.speaker_id
        parts.append(f"화자 #{sid} ({pd.feat.sim:.2f})" if sid is not None else "화자 미상")
        return " · ".join(parts)

    def _decision(self, pd: _Pending, latency_ms: Optional[float] = None) -> dict:
        f = pd.feat
        return {
            "id": f.seg_id, "t_start": round(f.t_start, 3), "t_end": round(f.t_end, 3),
            "speaker_id": f.speaker_id, "role": pd.role, "text": f.text,
            "prob": round(pd.prob, 3),
            "evidence": {"T": pd.T, "S": pd.S, "L": None if pd.L is None else round(pd.L, 3),
                         "gap": None if pd.gap is None else round(pd.gap, 3),
                         "sim": round(float(f.sim), 3), "pair_type": pd.pair_type},
            "chip": self.chip(pd),
            "pending_llm": pd.llm_expected and not pd.final,
            "mode": self.mode,
        }

    # ---------------------------------------------------------- 비착용자
    def on_segment(self, feat: SegFeat, now: float) -> tuple[dict, list[dict]]:
        """ASR 텍스트가 붙은 비착용자 구간. 반환: (caption 결정, 상태 이벤트들)."""
        events: list[dict] = []
        # 짧은 구간: 직전 비착용자 구간이 대화 상대였고 간격이 짧으면 화자 상속
        if feat.speaker_id is None and feat.duration < self.short_s and self.prev_seg is not None:
            psid, pend, psim = self.prev_seg
            if self.is_partner(psid) and 0 <= feat.t_start - pend <= self.inherit_gap:
                feat.speaker_id, feat.sim = psid, psim
        sid = feat.speaker_id
        spk = None
        if sid is not None:
            spk, ev = self._speaker(sid, now)
            events += ev
            spk.sim = feat.sim
        T, gap, turn_id = self.timing(feat.t_start)
        S = 1.0 if (spk is not None and spk.state == "partner" and feat.sim >= self.spk_threshold) else 0.0
        eligible = gap is not None and self.p["timing_early_s"] <= gap <= self.call_window
        llm_expected = eligible and self.mode_uses_llm() and self.llm_available
        pd = _Pending(feat=feat, T=T, S=S, gap=gap, turn_id=turn_id, llm_expected=llm_expected)

        if spk is not None and T >= 0.5 and spk.state in ("unknown", "expired"):
            events += self._set_state(spk, "candidate", now)

        if self.mode == "all":
            pd.prob, pd.role, pd.final = 1.0, "partner", True
            if spk is not None and spk.state != "partner":
                events += self._set_state(spk, "partner", now)
        else:
            pd.prob = self.score(T, S, None)
            pd.role = self._role(pd.prob, sid)
            if not llm_expected:
                events += self._finalize(pd, now)
        if spk is not None:
            spk.last_end = feat.t_end
            if spk.state == "partner" and T >= 0.5:
                spk.last_exchange = max(spk.last_exchange, now)
        self.pending[feat.seg_id] = pd
        if len(self.pending) > 500:
            for k in list(self.pending)[:100]:
                del self.pending[k]
        self.prev_seg = (feat.speaker_id, feat.t_end, feat.sim)
        if pd.role == "partner" and feat.text:
            self.history.append(("other", feat.text))
        return self._decision(pd), events

    def on_llm_result(self, seg_id: str, result: Optional[dict], now: float) -> tuple[Optional[dict], list[dict]]:
        """result: {"prob": L, "pair": bool, "type": str} 또는 None(타임아웃/실패)."""
        pd = self.pending.get(seg_id)
        if pd is None or pd.final:
            return None, []
        was_partner = pd.role == "partner"
        if result is not None:
            pd.L = float(result["prob"])
            pd.pair = bool(result.get("pair"))
            pd.pair_type = result.get("type")
            pd.prob = self.score(pd.T, pd.S, pd.L)
            pd.role = self._role(pd.prob, pd.feat.speaker_id)
        events = self._finalize(pd, now)
        if pd.role == "partner" and not was_partner and pd.feat.text:
            self.history.append(("other", pd.feat.text))
        d = self._decision(pd)
        upd = {k: d[k] for k in ("id", "role", "prob", "evidence", "chip", "pending_llm", "speaker_id")}
        return upd, events

    def _finalize(self, pd: _Pending, now: float) -> list[dict]:
        """최종 판정(LLM 결과 도착/불필요/실패) 후 화자 등록 처리."""
        pd.final = True
        sid = pd.feat.speaker_id
        if sid is None or sid not in self.speakers:
            return []
        spk = self.speakers[sid]
        if spk.state == "partner":
            if pd.T >= 0.5 and (pd.L is None or pd.pair):
                spk.last_exchange = max(spk.last_exchange, now)
            return []
        events = []
        if pd.llm_expected and pd.L is None:
            # LLM 실패: 타이밍 규칙만으로 연속 교대 수를 센다
            if pd.T >= 0.5 and pd.turn_id is not None and pd.turn_id != spk.last_counted_turn:
                spk.timing_turns += 1
                spk.last_counted_turn = pd.turn_id
                if spk.timing_turns >= self.p["timing_only_turns"]:
                    events += self._set_state(spk, "partner", now)
            elif pd.T < 0.5:
                spk.timing_turns = 0
            return events
        if pd.L is not None and not pd.pair:
            spk.timing_turns = 0
        if pd.T >= 0.5 and pd.prob >= self.p["register_threshold"]:
            events += self._set_state(spk, "partner", now)
        elif pd.T >= 0.5 and not pd.llm_expected and pd.L is None and self.mode != "semantic":
            # LLM을 쓰지 않는 모드에서 점수가 모자란 응답(T=0.5): 연속 교대 규칙
            if pd.turn_id is not None and pd.turn_id != spk.last_counted_turn:
                spk.timing_turns += 1
                spk.last_counted_turn = pd.turn_id
                if spk.timing_turns >= self.p["timing_only_turns"]:
                    events += self._set_state(spk, "partner", now)
        return events

    # ------------------------------------------------------------- 기타
    def on_name_call(self, speaker_id: Optional[int], label: str, score: float, now: float) -> list[dict]:
        events = [{"type": "alert", "kind": "name", "label": label, "score": round(float(score), 3),
                   "speaker_id": speaker_id, "t": now}]
        if speaker_id is not None:
            spk, ev = self._speaker(speaker_id, now)
            events += ev
            if spk.state in ("unknown", "expired"):
                events += self._set_state(spk, "candidate", now)
        return events

    def set_partner(self, speaker_id: int, on: bool, now: float) -> list[dict]:
        """대시보드 수동 등록/해제."""
        spk, events = self._speaker(speaker_id, now)
        spk.manual = bool(on)
        if on:
            events += self._set_state(spk, "partner", now)
            spk.last_exchange = now
        else:
            events += self._set_state(spk, "unknown", now)
        events.append(self._spk_event(spk))
        return events

    def tick(self, now: float) -> list[dict]:
        """만료 처리. 주기적으로(예: 1초마다) 호출."""
        events = []
        for s in self.speakers.values():
            if s.state == "partner" and not s.manual and now - s.last_exchange > self.p["partner_expire_s"]:
                events += self._set_state(s, "expired", now)
                s.state = "unknown"   # 다음부터는 unknown 으로 취급
            elif s.state == "candidate" and now - s.state_since > self.p["candidate_expire_s"]:
                events += self._set_state(s, "unknown", now)
        return events

    def speakers_snapshot(self) -> list[dict]:
        return [self._spk_event(s) for s in sorted(self.speakers.values(), key=lambda x: x.speaker_id)]
