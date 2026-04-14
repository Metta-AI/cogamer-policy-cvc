from __future__ import annotations

from cvc_policy.cogamer_policy import CvCPolicy
from tests.conftest import _fake_policy_env_info


def test_cvc_policy_has_recorder_by_default():
    p = CvCPolicy(_fake_policy_env_info())
    assert p._recorder is not None


def test_cvc_policy_record_dir_kwarg_creates_recorder(tmp_path):
    p = CvCPolicy(_fake_policy_env_info(), record_dir=str(tmp_path))
    assert p._recorder is not None
    assert p._recorder._record_dir == str(tmp_path)


def test_cvc_policy_log_py_enables_stderr(capsys):
    p = CvCPolicy(_fake_policy_env_info(), log_py=True)
    p._recorder.emit(type="note", agent=None, stream="py", payload={"text": "hi"})
    assert "[py]" in capsys.readouterr().err


def test_cvc_policy_log_llm_enables_llm_stream(capsys):
    p = CvCPolicy(_fake_policy_env_info(), log_llm=True)
    p._recorder.emit(type="note", agent=None, stream="llm", payload={"text": "x"})
    assert "[llm]" in capsys.readouterr().err


def test_cvc_policy_log_all_enables_both(capsys):
    p = CvCPolicy(_fake_policy_env_info(), log="py+llm")
    p._recorder.emit(type="note", agent=None, stream="py", payload={})
    p._recorder.emit(type="note", agent=None, stream="llm", payload={})
    err = capsys.readouterr().err
    assert "[py]" in err and "[llm]" in err
