from app.namecall import NameCallDetector, edit_distance, to_jamo

V = ["민수", "민수야", "민수 씨", "민수님"]


def det():
    return NameCallDetector(V, max_dist=2, short_max_dist=1, short_jamo_len=5)


def test_jamo():
    assert to_jamo("민수") == "ㅁㅣㄴㅅㅜ"
    assert to_jamo("값") == "ㄱㅏㅄ"
    assert to_jamo("A1") == "a1"


def test_edit_distance():
    assert edit_distance("abc", "abc") == 0
    assert edit_distance("abc", "abd") == 1
    assert edit_distance("", "ab") == 2


def test_exact_and_variants():
    d = det()
    assert d.detect("민수야 이거 봐")["distance"] == 0
    assert d.detect("저기요 민수 씨!")["distance"] == 0
    assert d.detect("어 민수님 오셨어요")


def test_asr_typos():
    d = det()
    assert d.detect("민숙아 이리 와")            # 민수야 ↔ 민숙아 (자모 거리 1)
    assert d.detect("인수야 밥 먹자")            # ㅁ 탈락 ASR 오류 (거리 1)


def test_negatives():
    d = det()
    assert d.detect("오늘 날씨 좋네요") is None
    assert d.detect("만세") is None                # 짧은 이름은 더 엄격(거리 2 불허)
    assert d.detect("") is None
