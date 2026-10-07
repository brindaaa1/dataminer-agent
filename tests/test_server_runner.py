"""建模子进程：进程内跑完、子进程中止后续跑、进程意外消失时标为失败。"""
import json
import subprocess
import time

import pytest

from data.sample import SAMPLE_CSV


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.setenv("DATAMINER_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    return tmp_path / "ws"


def _ready(**settings) -> str:
    """mock 接入走到"配置已确认"。"""
    from server import intake_runner, tasks
    m = tasks.create(SAMPLE_CSV.read_bytes(), "hotel_bookings_sample.csv", "预测是否取消", "mock",
                     {"max_rounds": 1, "full_trials": 3, "low_trials": 3, **settings})
    intake_runner.start(m["id"])
    state = json.loads((tasks.task_dir(m["id"]) / "intake.json").read_text())
    qs = next(e["payload"] for e in reversed(state["events"]) if e["type"] == "questions")
    intake_runner.answer(m["id"], {q["key"]: q["recommended"] for q in qs})
    assert tasks.load(m["id"])["status"] == "config_ready"
    return m["id"]


def _wait(tid, cond, timeout=300):
    from server import runner, tasks
    t0 = time.time()
    while time.time() - t0 < timeout:
        m = runner.refresh(tasks.load(tid))
        if cond(m):
            return m
        time.sleep(1)
    raise AssertionError(f"超时：{tasks.load(tid)}")


def test_worker_runs_to_done_in_process(ws):
    from server import tasks, worker
    tid = _ready()
    tasks.update(tid, status="running")
    worker.run(tid)
    m = tasks.load(tid)
    assert m["status"] == "done" and m["pid"] is None and m["error"] is None
    assert (tasks.task_dir(tid) / "artifacts" / "reports" / f"model_card_{tid}.md").exists()


def test_stop_then_resume_finishes(ws):
    from runtime.events import EventLog
    from server import runner, tasks
    tid = _ready(max_rounds=3)
    runner.start(tid)
    log = EventLog(str(tasks.events_db(tid)), tid)
    _wait(tid, lambda m: any(e["type"] == "state_transition" for e in log.query()))
    runner.stop(tid)
    m = tasks.load(tid)
    assert m["status"] == "stopped" and not runner.alive(tid, m["pid"])
    runner.start(tid)                                       # Orchestrator 从检查点接着跑
    m = _wait(tid, lambda m: m["status"] != "running")
    assert m["status"] == "done", m


def test_running_task_whose_process_is_gone_is_marked_failed(ws):
    from server import runner, tasks
    tid = _ready()
    p = subprocess.Popen(["true"])
    p.wait()
    tasks.update(tid, status="running", pid=p.pid)          # 例如 web 服务重启前启动的进程已经不在了
    m = runner.refresh(tasks.load(tid))
    assert m["status"] == "failed" and "worker.log" in m["error"]
