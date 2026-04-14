from cvc_policy.recorder import EventRecorder, fmt


def test_emit_appends_event_with_step_and_stream():
    rec = EventRecorder()
    rec.set_step(7)
    rec.emit(type="action", agent=0, stream="py", payload={"role": "miner"})
    assert rec.events == [
        {
            "step": 7,
            "agent": 0,
            "stream": "py",
            "type": "action",
            "payload": {"role": "miner"},
        }
    ]


def test_emit_without_step_defaults_to_zero():
    rec = EventRecorder()
    rec.emit(type="note", agent=None, stream="py", payload={"text": "hi"})
    assert rec.events[0]["step"] == 0


def test_fmt_action_event():
    ev = {
        "step": 3,
        "agent": 0,
        "stream": "py",
        "type": "action",
        "payload": {"role": "miner", "summary": "mine_carbon"},
    }
    assert fmt(ev) == "[py] a0 step=3 action role=miner summary=mine_carbon"


def test_fmt_team_event_has_no_agent_prefix():
    ev = {
        "step": 10,
        "agent": None,
        "stream": "py",
        "type": "note",
        "payload": {"text": "season changed"},
    }
    assert fmt(ev) == "[py] step=10 note text='season changed'"


def test_fmt_patch_applied_shows_applied_fields():
    ev = {
        "step": 500,
        "agent": 2,
        "stream": "llm",
        "type": "patch_applied",
        "payload": {
            "applied": {"resource_bias": "carbon"},
            "rationale": "low supply",
        },
    }
    s = fmt(ev)
    assert s.startswith("[llm] a2 step=500 patch_applied")
    assert "resource_bias=carbon" in s
