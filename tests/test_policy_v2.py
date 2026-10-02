from app.config import load_config
from app.policy import FUSION_FEATURES, PolicyEngine, SegFeat, is_question, syllable_count

NO = {"prob": 0.1, "pair": False, "type": "없음"}
YES = {"prob": 0.9, "pair": True, "type": "반응"}


def make(**pol):
    cfg = load_config(overrides={"policy": pol})
    return PolicyEngine(cfg, mode="full", llm_available=True)


def seg(i, t0, t1, spk, text="말을 합니다"):
    return SegFeat(seg_id=f"s{i}", t_start=t0, t_end=t1, speaker_id=spk, sim=0.8, text=text)


def test_syllables_and_question():
    assert syllable_count("네") == 1 and syllable_count("Yeah.") == 1 and syllable_count("Okay") == 2
    assert syllable_count("그렇구나") == 4
    assert is_question("What do you think") and is_question("이거 맞아요?") and not is_question("Okay.")


def test_short_skip_llm():
    p = make(short_skip_llm=True)
    p.on_wearer_end(0, 2)
    dec, _ = p.on_segment(seg(1, 2.3, 2.6, 1, text="네"), now=2.6)
    assert dec["pending_llm"] is False            # 2음절 이하 → LLM 생략, 즉시 확정
    dec, _ = p.on_segment(seg(2, 2.7, 4.0, 2, text="그건 좀 비싸요"), now=4.0)


def test_candidate_rejudge_confirm_on_next_exchange():
    p = make(candidate_rejudge=True)
    p.on_wearer_end(0, 2)
    p.on_segment(seg(1, 2.2, 3.5, 1), now=3.5)
    upd, ev = p.on_llm_result("s1", NO, now=4.0)
    assert upd["role"] == "partner" and "보류" in upd["chip"]   # 바로 접지 않음
    assert p.speakers[1].state == "candidate"
    p.on_wearer_end(5, 6)
    p.on_segment(seg(2, 6.3, 7.5, 1), now=7.5)
    _, ev = p.on_llm_result("s2", YES, now=8.0)
    ups = [e for e in ev if e["type"] == "caption_update" and e["id"] == "s1"]
    assert ups and ups[0]["role"] == "partner"
    assert p.speakers[1].state == "partner"


def test_candidate_rejudge_fold_when_no_followup():
    p = make(candidate_rejudge=True)
    p.on_wearer_end(0, 2)
    p.on_segment(seg(1, 2.2, 3.5, 1), now=3.5)
    p.on_llm_result("s1", NO, now=4.0)
    ev = p.tick(40.0)
    ups = [e for e in ev if e["type"] == "caption_update"]
    assert ups and ups[0]["role"] == "other"


def test_learned_fusion_path():
    coef = {k: 0.0 for k in FUSION_FEATURES}
    coef["p_pair"] = 6.0
    p = make(fusion={"type": "logistic", "coef": coef, "intercept": -3.0, "threshold": 0.5})
    p.on_wearer_end(0, 2)
    p.on_segment(seg(1, 2.2, 3.5, 1), now=3.5)
    upd, _ = p.on_llm_result("s1", YES, now=4.0)
    assert upd["role"] == "partner" and abs(upd["prob"] - 0.917) < 0.01
    p.set_mode("timing")                                  # 다른 모드는 손 가중치 그대로
    assert not p.learned()
