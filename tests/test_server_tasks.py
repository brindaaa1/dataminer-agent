"""工作台任务存储：目录布局、元数据读写、按任务生成的 cfg、复制任务。"""
import json

import pytest

from data.sample import SAMPLE_CSV


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.setenv("DATAMINER_WORKSPACE", str(tmp_path / "ws"))
    return tmp_path / "ws"


def _new(**kw):
    from server import tasks
    return tasks.create(SAMPLE_CSV.read_bytes(), "hotel_bookings_sample.csv", "预测是否取消", "mock",
                        {"max_rounds": 2, "full_trials": 3, "low_trials": 3, **kw})


def test_create_lays_out_task_dir(ws):
    from server import tasks
    m = _new()
    d = tasks.task_dir(m["id"])
    assert (d / "input" / "data.csv").read_bytes() == SAMPLE_CSV.read_bytes()
    assert (d / "input" / "description.md").read_text() == "预测是否取消"
    assert m["status"] == "intake" and m["busy"] is False and tasks.load(m["id"]) == m
    assert tasks.dataset_id(m) == "hotel_bookings_sample"


def test_update_and_list_newest_first(ws):
    from server import tasks
    a, b = _new(), _new()
    tasks.update(a["id"], status="config_ready")
    assert tasks.load(a["id"])["status"] == "config_ready"
    assert [m["id"] for m in tasks.list_tasks()] == [b["id"], a["id"]]


def test_task_cfg_points_everything_into_the_task(ws):
    from server import tasks
    m = _new()
    cfg = tasks.task_cfg(m)
    d = tasks.task_dir(m["id"])
    assert cfg["paths"]["artifacts_root"] == str(d / "artifacts")
    assert cfg["memory"]["db"] == str(ws / "memory.db")                 # 经验库跨任务共享
    assert cfg["run"]["max_rounds"] == 2 and cfg["fidelity"]["full"]["n_trials"] == 3 and cfg["fidelity"]["low"]["n_trials"] == 3
    assert tasks.events_db(m["id"]) == d / "artifacts" / "state.db"


def test_clone_copies_inputs_and_config_with_new_paths(ws):
    from server import tasks
    m = _new()
    d = tasks.task_dir(m["id"])
    (d / "config").mkdir()
    (d / "intake.json").write_text(json.dumps({"written": {"csv": str(d / "input" / "data.csv"),
                                                           "knowhow_root": str(d / "config" / "knowhow")}}))
    tasks.update(m["id"], status="done")
    c = tasks.clone(m["id"])
    cd = tasks.task_dir(c["id"])
    assert c["status"] == "config_ready" and c["source"] == m["id"]
    assert (cd / "input" / "data.csv").exists()
    w = json.loads((cd / "intake.json").read_text())["written"]
    assert w["csv"] == str(cd / "input" / "data.csv") and w["knowhow_root"] == str(cd / "config" / "knowhow")
