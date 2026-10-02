"""接入驱动：mock 起草 → 必答项确认 → 写出配置；起草失败、LLM 出错都落到"接入失败"，可以重试。"""
import json

import pytest

from data.sample import SAMPLE_CSV

OTHER_CSV = b"a,b,y\n1,2,0\n3,4,1\n"


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.setenv("DATAMINER_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    return tmp_path / "ws"


def _task(csv=None):
    from server import tasks
    return tasks.create(csv or SAMPLE_CSV.read_bytes(), "hotel_bookings_sample.csv", "预测是否取消", "mock")


def _types(tid):
    from runtime.events import EventLog
    from server import tasks
    return [e["type"] for e in EventLog(str(tasks.events_db(tid)), tid).query()]


def _questions(tid):
    from server import tasks
    state = json.loads((tasks.task_dir(tid) / "intake.json").read_text())
    return next(e["payload"] for e in reversed(state["events"]) if e["type"] == "questions")


def test_mock_draft_asks_required_items_then_writes(ws):
    from server import intake_runner, tasks
    tid = _task()["id"]
    intake_runner.start(tid)
    m = tasks.load(tid)
    assert m["status"] == "intake" and m["busy"] is False
    qs = _questions(tid)
    assert {q["key"] for q in qs} == {"label", "observation_time_expr", "windows", "metric"}
    intake_runner.answer(tid, {q["key"]: q["recommended"] for q in qs})
    assert tasks.load(tid)["status"] == "config_ready"
    assert (tasks.task_dir(tid) / "config" / "knowhow" / "hotel_bookings_sample" / "data_dictionary.yaml").exists()
    types = _types(tid)
    assert types[0] == "intake.profile" and types[-1] == "intake.written" and "intake.answers" in types


def test_draft_that_never_validates_ends_in_intake_failed(ws):
    """mock 起草的是 hotel 配置：换一份别的数据，字段对不上，校验一直不过。"""
    from server import intake_runner, tasks
    tid = _task(OTHER_CSV)["id"]
    intake_runner.start(tid)
    m = tasks.load(tid)
    assert m["status"] == "intake_failed" and m["busy"] is False and "不在表头中" in m["error"]
    assert "intake.validation_failed" in _types(tid)


def test_retry_after_failure_runs_the_round_again(ws):
    from server import intake_runner, tasks
    tid = _task(OTHER_CSV)["id"]
    intake_runner.start(tid)
    n = len(_types(tid))
    intake_runner.retry(tid)
    assert tasks.load(tid)["status"] == "intake_failed" and len(_types(tid)) > n


def test_llm_error_during_drafting_is_reported(ws, monkeypatch):
    from server import intake_runner, tasks

    class Down:
        def complete(self, *a, **k):
            raise ConnectionError("network down")

    monkeypatch.setattr(intake_runner, "_llm", lambda provider: Down())
    tid = _task()["id"]
    intake_runner.start(tid)
    m = tasks.load(tid)
    assert m["status"] == "intake_failed" and "ConnectionError" in m["error"] and m["busy"] is False


def test_exception_outside_drafting_still_clears_busy(ws, monkeypatch):
    """step() 之外抛异常（读 intake.json、answer 本身）也要落到"接入失败"，不能停在"起草中"。"""
    from server import intake_runner, tasks
    tid = _task()["id"]
    intake_runner.start(tid)
    tasks.update(tid, busy=True)

    def broken(_tid):
        raise KeyError("label")

    monkeypatch.setattr(intake_runner, "_load", broken)
    intake_runner.answer(tid, {})
    m = tasks.load(tid)
    assert m["status"] == "intake_failed" and m["busy"] is False and "KeyError" in m["error"]
