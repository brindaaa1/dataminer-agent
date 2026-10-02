"""工作台 API：mock 走完整流程、状态不对时 409、SSE 续传、示例任务、复制任务。
TestClient 会在请求返回前执行完 BackgroundTasks，所以接入阶段在测试里是同步的。"""
import io
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

from data.sample import SAMPLE_CSV


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATAMINER_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    from server.app import app
    return TestClient(app)


def _create(c, **form) -> str:
    r = c.post("/api/tasks", files={"csv": ("hotel_bookings_sample.csv", SAMPLE_CSV.read_bytes(), "text/csv")},
               data={"description": "预测是否取消", "provider": "mock", "max_rounds": "1", "full_trials": "3",
                     "low_trials": "3", **form})
    assert r.status_code == 202
    return r.json()["id"]


def _accept_all(c, tid) -> dict:
    s = c.get(f"/api/tasks/{tid}").json()
    replies = {q["key"]: q["recommended"] for q in s["intake"]["questions"]}
    assert c.post(f"/api/tasks/{tid}/answers", json={"replies": replies}).status_code == 202
    return c.get(f"/api/tasks/{tid}").json()


def test_full_flow_with_mock(client):
    c = client
    tid = _create(c)
    s = c.get(f"/api/tasks/{tid}").json()
    assert s["stage"] == "intake" and len(s["intake"]["questions"]) == 4 and s["data"]["n_rows"] == 7997
    s = _accept_all(c, tid)
    assert s["status"] == "config_ready" and "data_dictionary.yaml" in s["intake"]["yaml"]
    assert c.post(f"/api/tasks/{tid}/run").status_code == 200
    t0 = time.time()
    while (s := c.get(f"/api/tasks/{tid}").json())["status"] == "running":
        assert time.time() - t0 < 300
        time.sleep(1)
    assert s["status"] == "done" and s["stage"] == "report" and s["run"]["narration"]
    assert c.get(f"/api/tasks/{tid}/bundle.zip").status_code == 409            # 确认之前不能下载
    assert c.post(f"/api/tasks/{tid}/accept", json={"note": "看过了"}).status_code == 200
    s = c.get(f"/api/tasks/{tid}").json()
    assert s["status"] == "accepted" and s["readonly"] and s["note"] == "看过了"
    z = zipfile.ZipFile(io.BytesIO(c.get(f"/api/tasks/{tid}/bundle.zip").content))
    assert {"model_card.md", "events.json", "task_spec.yaml", "lessons.json"} <= set(z.namelist())
    assert any(e["type"] == "accepted" for e in c.get(f"/api/tasks/{tid}/events").json())
    assert c.post(f"/api/tasks/{tid}/accept", json={}).status_code == 409        # 已确认：只读


def test_actions_in_wrong_state_are_rejected(client):
    c = client
    tid = _create(c)
    assert c.post(f"/api/tasks/{tid}/run").status_code == 409                    # 配置还没确认
    assert c.post(f"/api/tasks/{tid}/accept", json={}).status_code == 409
    assert c.post("/api/tasks/ex-hotel_deepseek3/run").status_code == 409        # 示例只读
    assert c.get("/api/tasks/nope").status_code == 404


def test_answers_while_drafting_are_rejected(client):
    from server import tasks
    tid = _create(client)
    tasks.update(tid, busy=True)
    assert client.post(f"/api/tasks/{tid}/answers", json={"replies": {}}).status_code == 409


def test_stream_replays_after_last_event_id_then_goes_idle(client):
    c = client
    tid = _create(c)
    _accept_all(c, tid)                                                          # config_ready：不活跃，推完就结束
    ids = [e["id"] for e in c.get(f"/api/tasks/{tid}/events").json()]
    body = c.get(f"/api/tasks/{tid}/stream", headers={"Last-Event-ID": str(ids[2])}).text
    assert [int(ln[4:]) for ln in body.splitlines() if ln.startswith("id: ")] == ids[3:]
    assert body.rstrip().endswith("event: idle\ndata: {}")


def test_examples_are_listed_and_browsable(client):
    c = client
    assert any(t["id"] == "ex-hotel_deepseek3" and t["example"] for t in c.get("/api/tasks").json())
    s = c.get("/api/tasks/ex-hotel_deepseek3").json()
    assert s["readonly"] and s["run"]["rounds"]
    assert "风险提示" in c.get("/api/tasks/ex-hotel_deepseek3/artifacts/model_card.md").text
    assert c.get("/api/tasks/ex-hotel_deepseek3/bundle.zip").status_code == 200


def test_clone_starts_from_confirmed_config(client):
    c = client
    tid = _create(c)
    _accept_all(c, tid)
    new = c.post(f"/api/tasks/{tid}/clone").json()["id"]
    s = c.get(f"/api/tasks/{new}").json()
    assert s["status"] == "config_ready" and s["intake"]["yaml"] and s["intake"]["draft"]


def test_drafting_interrupted_by_server_restart_becomes_retryable(client):
    """web 服务在起草中途重启：busy 还是 True 但已经没有线程在跑 → 标为接入失败，可以重试。"""
    from server import tasks
    tid = _create(client)
    tasks.update(tid, status="intake", busy=True)          # 模拟重启：标记还在，进程里没有对应的起草线程
    s = client.get(f"/api/tasks/{tid}").json()
    assert s["status"] == "intake_failed" and not s["busy"] and "中断" in s["error"]
    assert client.post(f"/api/tasks/{tid}/retry").status_code == 202
    assert client.get(f"/api/tasks/{tid}").json()["intake"]["questions"]
