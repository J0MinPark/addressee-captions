"""호명 감지: 한글을 자모로 분해해 이름 변형과 편집거리로 비교한다. 외부 라이브러리 없음."""
from __future__ import annotations

import re
from typing import Optional

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ",
        "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
_STRIP = re.compile(r"[\s\.,!?~…·\"'“”‘’\-_()\[\]]+")


def decompose_syllable(ch: str) -> str:
    code = ord(ch) - 0xAC00
    if 0 <= code < 11172:
        return CHO[code // 588] + JUNG[(code % 588) // 28] + JONG[code % 28]
    return ch.lower()


def to_jamo(text: str) -> str:
    return "".join(decompose_syllable(c) for c in text)


def normalize(text: str) -> str:
    return _STRIP.sub("", text or "")


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class NameCallDetector:
    def __init__(self, variants: list[str], max_dist: int = 2, short_max_dist: int = 1,
                 short_jamo_len: int = 5):
        self.variants = []
        for v in variants:
            nv = normalize(v)
            if nv:
                self.variants.append((v, nv, to_jamo(nv)))
        # 긴 변형을 먼저(“민수야”가 “민수”보다 먼저 보고됨)
        self.variants.sort(key=lambda x: -len(x[1]))
        self.max_dist = max_dist
        # 자모 5개 이하(예: "민수")는 거리 2면 "만세"까지 잡히므로 더 엄격하게
        self.short_max_dist = short_max_dist
        self.short_jamo_len = short_jamo_len

    def _limit(self, jv: str) -> int:
        return self.short_max_dist if len(jv) <= self.short_jamo_len else self.max_dist

    def detect(self, text: str) -> Optional[dict]:
        """호명이면 {"variant", "match", "distance", "score"}; 아니면 None.
        텍스트를 음절 경계 기준 같은 길이(음절 수) 창으로 잘라 자모 편집거리를 잰다."""
        t = normalize(text)
        if not t or not self.variants:
            return None
        best = None
        for raw, nv, jv in self.variants:
            n = len(nv)
            if len(t) < n:
                windows = [t]
            else:
                windows = [t[i:i + n] for i in range(len(t) - n + 1)]
            for w in windows:
                d = edit_distance(to_jamo(w), jv)
                if d <= self._limit(jv) and (best is None or d < best["distance"]):
                    best = {"variant": raw, "match": w, "distance": d,
                            "score": round(1.0 - d / max(len(jv), 1), 3)}
                    if d == 0:
                        return best
        return best
