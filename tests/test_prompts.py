"""프롬프트 버전 고정: 영어 프롬프트(AMI 평가에 쓴 것)는 바꾸지 않는다. 한국어 P1은 2026-10 버그 수정으로 바뀌었다."""
from app import llm_judge as L

EN_FROZEN = {"P1c": "e389a86c", "P1": "b627183d", "P2": "0cc65e8f", "P3": "2faa83e5"}


def test_english_prompts_unchanged():
    assert {v: L._prompt_version("en", v) for v in L.VARIANTS} == EN_FROZEN


def test_korean_p1_fixed_and_cache_key_changes():
    assert L._SHOT_P1_KO == []                                  # 수정: few-shot 없음
    assert L._prompt_version("ko", "P1") != "243b6e2a"          # 수정 전 해시 → 캐시 키도 바뀐다
    assert L._prompt_version("ko", "P1c") == "29f29731"         # v1(시연 기본)은 그대로
    assert L._prompt_version("ko", "P2") == "e8bd838f"
