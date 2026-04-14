from cvc_policy.recorder import EventRecorder


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
