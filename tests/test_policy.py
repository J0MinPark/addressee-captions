from app.config import load_config
from app.policy import PolicyEngine, SegFeat

YES_HIGH = {"prob": 0.95, "pair": True, "type": "질문-대답"}
NO_HIGH = {"prob": 0.05, "pair": False, "type": "없음"}


def make(mode="full", llm=True):
    return PolicyEngine(load_config(), mode=mode, llm_available=llm)


def seg(i, t0, t1, spk, sim=0.8, text="말"):
    return SegFeat(seg_id=f"s{i}", t_start=t0, t_end=t1, speaker_id=spk, sim=sim, text=text)


def states(events):
    return [(e["speaker_id"], e["state"]) for e in events if e["type"] == "speaker_state"]


def test_a_partner_answers_then_registered():
    p = make()
    p.on_wearer_end(0.0, 2.0)
    p.set_wearer_text(0, "지금 몇 시예요?")
    dec, ev = p.on_segment(seg(1, 2.3, 3.5, 1, text="세 시 반이요"), now=3.5)
    assert dec["pending_llm"] is True
    assert dec["role"] == "partner"            # 음향 증거만으로 임시 표시
    assert (1, "candidate") in states(ev)
    upd, ev2 = p.on_llm_result("s1", YES_HIGH, now=4.0)
    assert upd["role"] == "partner" and upd["pending_llm"] is False
    assert upd["prob"] > 0.9
    assert (1, "partner") in states(ev2)
    assert any(e["type"] == "partner_added" for e in ev2)
    assert "응답 0.3초" in upd["chip"] and "질문→대답" in upd["chip"] and "화자 #1" in upd["chip"]


def test_b_trap_timing_ok_but_llm_no():
    p = make()
    p.on_wearer_end(0.0, 2.0)
    p.set_wearer_text(0, "점심 뭐 드실래요?")
    dec, _ = p.on_segment(seg(1, 2.2, 4.0, 2, text="야 지훈아 어제 경기 봤어?"), now=4.0)
    assert dec["pending_llm"]
    upd, ev = p.on_llm_result("s1", NO_HIGH, now=4.5)
    assert upd["role"] == "other"
    assert (2, "partner") not in states(ev)
    assert p.speakers[2].state == "candidate"


def test_b2_trap_registered_in_timing_mode():
    p = make(mode="timing")
    p.on_wearer_end(0.0, 2.0)
    dec, ev = p.on_segment(seg(1, 2.2, 4.0, 2), now=4.0)
    assert dec["pending_llm"] is False
    assert dec["role"] == "partner"
    assert (2, "partner") in states(ev)


def test_c_llm_timeout_two_turns_then_registered():
    p = make()
    p.on_wearer_end(0.0, 2.0)
    p.on_segment(seg(1, 2.3, 3.5, 3), now=3.5)
    _, ev = p.on_llm_result("s1", None, now=6.0)
    assert (3, "partner") not in states(ev)
    assert p.speakers[3].state == "candidate"
    p.on_wearer_end(5.0, 6.5)
    p.on_segment(seg(2, 6.8, 8.0, 3), now=8.0)
    _, ev = p.on_llm_result("s2", None, now=10.5)
    assert (3, "partner") in states(ev)


def test_d_registered_partner_monologue_keeps_showing():
    p = make()
    p.on_wearer_end(0.0, 2.0)
    p.on_segment(seg(1, 2.3, 3.5, 1), now=3.5)
    p.on_llm_result("s1", YES_HIGH, now=4.0)
    # 착용자가 말하지 않는 동안 상대가 계속 말함 (T=0, LLM 호출 없음)
    for i, t in enumerate([6.0, 10.0, 15.0, 22.0], start=2):
        dec, _ = p.on_segment(seg(i, t, t + 3.0, 1, sim=0.7), now=t + 3.0)
        assert dec["role"] == "partner", dec
        assert dec["pending_llm"] is False
        assert dec["evidence"]["T"] == 0 and dec["evidence"]["S"] == 1.0


def test_e_expire_after_60s_without_exchange():
    p = make()
    p.on_wearer_end(0.0, 2.0)
    p.on_segment(seg(1, 2.3, 3.5, 1), now=3.5)
    p.on_llm_result("s1", YES_HIGH, now=4.0)
    assert p.speakers[1].state == "partner"
    assert p.tick(50.0) == []
    ev = p.tick(64.5)
    assert (1, "expired") in states(ev)
    assert p.speakers[1].state == "unknown"
    dec, _ = p.on_segment(seg(2, 80.0, 82.0, 1), now=82.0)
    assert dec["role"] != "partner"


def test_f_name_call_alert_and_candidate():
    p = make()
    ev = p.on_name_call(5, "호명: 민수야", 1.0, now=10.0)
    assert ev[0]["type"] == "alert" and ev[0]["kind"] == "name"
    assert (5, "candidate") in states(ev)


def test_short_segment_inherits_partner():
    p = make()
    p.on_wearer_end(0.0, 2.0)
    p.on_segment(seg(1, 2.3, 3.5, 1), now=3.5)
    p.on_llm_result("s1", YES_HIGH, now=4.0)
    dec, _ = p.on_segment(SegFeat("s2", 4.0, 4.5, None, 0.0, "네"), now=4.5)
    assert dec["speaker_id"] == 1 and dec["role"] == "partner"


def test_manual_toggle_and_modes():
    p = make()
    ev = p.set_partner(7, True, now=1.0)
    assert (7, "partner") in states(ev)
    assert p.tick(500.0) == []          # 수동 등록은 만료되지 않음
    p.set_partner(7, False, now=2.0)
    assert p.speakers[7].state == "unknown"
    p.set_mode("all")
    dec, _ = p.on_segment(seg(1, 10, 12, 9), now=12)
    assert dec["role"] == "partner" and dec["prob"] == 1.0
    p.set_mode("semantic")
    dec, _ = p.on_segment(seg(2, 30, 32, 9), now=32)   # 착용자 직후가 아님
    assert dec["role"] != "partner" or p.is_partner(9) is False or dec["evidence"]["S"] == 0


def test_score_values():
    p = make()
    assert abs(p.score(1, 0, None) - 0.6225) < 1e-3
    assert p.score(1, 0, 0.05) < 0.3
    assert p.score(0, 1, None) > 0.8
