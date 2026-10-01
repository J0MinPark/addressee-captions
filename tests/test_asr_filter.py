from app.asr import filter_segments, is_hotword_echo
from app.config import load_config

H = ["민수", "민수야", "민수씨", "민수 씨", "민수님"]


def test_hotword_echo():
    assert is_hotword_echo("민수 씨 민수 씨 민수 씨", H)
    assert is_hotword_echo("민수 민수", H)
    assert not is_hotword_echo("민수야", H)                 # 한 번 부른 건 진짜 호명일 수 있음
    assert not is_hotword_echo("민수야 이거 좀 봐", H)


def test_filter_thresholds_and_blacklist():
    a = load_config()["asr"]
    segs = [
        {"text": " 안녕하세요", "no_speech_prob": 0.1, "avg_logprob": -0.3},
        {"text": " 잡음", "no_speech_prob": 0.9, "avg_logprob": -0.3},
        {"text": " 웅얼", "no_speech_prob": 0.1, "avg_logprob": -1.5},
        {"text": " 시청해 주셔서 감사합니다", "no_speech_prob": 0.1, "avg_logprob": -0.2},
    ]
    assert filter_segments(segs, a, H) == "안녕하세요"
