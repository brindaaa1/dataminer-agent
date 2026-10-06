"""工作台的评测视图：把 eval/suite/results 下的版本包装成只读任务（spec 2026-10-05 §8）。"""
import json

import pytest

from memory.schema import ExperimentRecord
from memory.store import Store
from runtime.events import EventLog


def _rec(tid, exp_id, action="RACE", verdict="PROMISING"):
    return ExperimentRecord(exp_id=exp_id, task_id=tid, parent_exp_id=None, hypothesis="h", action_type=action, diff={},
                            config={"model": "lgbm", "fidelity": "low"}, config_hash=exp_id, fidelity="low",
                            metrics={"oot_dev_auc": .7, "gap": .01}, diagnosis_codes=[], cost={"cpu_minutes": 0, "wall_minutes": 0},
                            verdict=verdict)


@pytest.fixture
def evroot(tmp_path, monkeypatch):
    """一个版本 v9：hotel 种子 0（有产物）、种子 1（产物目录被删了）、情景 leak@hotel（不通过）。"""
    res, runs = tmp_path / "results", tmp_path / "runs"
    monkeypatch.setenv("DATAMINER_EVAL_RESULTS", str(res))
    monkeypatch.setenv("DATAMINER_EVAL_RUNS", str(runs))
    tid = "v9_hotel_s0"
    art = runs / "v9" / "hotel" / "s0"
    art.mkdir(parents=True)
    log = EventLog(str(art / "state.db"), tid)
    log.append("state_transition", {"to": "PLAN", "sec": 1.0})
    log.append("state_transition", {"to": "DECIDE", "sec": 1.0})
    log.append("decision_rejected", {"errors": ["禁忌表：与 e01 相同"]})
    log.append("recorded", {"verdict": "PROMISING", "codes": [], "action": "RACE"}, exp_id=f"{tid}_e01")
    Store(str(art / "state.db")).add(_rec(tid, f"{tid}_e01"))
    (art / "reports").mkdir()
    (art / "reports" / f"model_card_{tid}.md").write_text("# 模型卡")
    proc = {"wasted_rounds": {"value": 0.0}, "rejected_decisions": {"value": 1, "by_kind": {"duplicate": 1}}}
    v = {"label": "v9", "tier": "fast", "finished": 100.0, "wall_sec": 10, "cost_usd": 0.01,
         "settings": {"max_rounds": 8, "llm": "deepseek", "scenario": {"max_rounds": 5}},
         "datasets": {"hotel": {"baselines": {"b0": {"oot_dev_auc": .82, "holdout_auc": .777}, "b1": {"holdout_auc": .78}},
                                "runs": [{"seed": 0, "status": "ok", "holdout": .79, "oot_dev": .83, "task_id": tid, "process": proc,
                                          "cost_usd": 0.005, "error": None},
                                         {"seed": 1, "status": "failed", "holdout": None, "task_id": "v9_hotel_s1", "process": {},
                                          "cost_usd": 0.001, "error": "运行结束于 FAILED"}]}},
         "scenarios": [{"scenario": "leak", "base": "hotel", "passed": False, "reason": "采纳了泄漏列", "task_id": "v9_scn_leak_hotel",
                        "status": "ok", "cost_usd": 0.002}]}
    (res / "v9").mkdir(parents=True)
    (res / "v9" / "version.json").write_text(json.dumps(v))
    return {"res": res, "runs": runs, "tid": tid}


def test_index_lists_seeds_and_scenarios_grouped_by_version(evroot):
    from server import eval_tasks
    s = eval_tasks.summaries()
    assert [x["id"] for x in s] == ["ev-v9_hotel_s0", "ev-v9_hotel_s1", "ev-v9_scn_leak_hotel"]
    assert [x["status"] for x in s] == ["done", "failed", "done"]
    assert s[0]["eval"] == {"label": "v9", "tier": "fast", "conclusion": None} and s[0]["title"] == "hotel · 种子 0"
    assert s[2]["title"] == "情景 leak @ hotel"
    e = eval_tasks.index()["ev-v9_scn_leak_hotel"]
    assert e["art"] == evroot["runs"] / "v9" / "hotel" / "scn_leak" and e["max_rounds"] == 5


def test_load_has_events_records_and_files(evroot):
    from server import eval_tasks
    src = eval_tasks.load("ev-v9_hotel_s0")
    assert src["meta"]["eval"] == "v9" and src["meta"]["settings"] == {"max_rounds": 8}
    assert [e["type"] for e in src["events"]][-1] == "recorded" and src["records"][0]["exp_id"] == "v9_hotel_s0_e01"
    assert src["files"]["model_card.md"] == "# 模型卡" and "events.json" in src["files"]


def test_missing_artifacts_gives_empty_run(evroot):
    """Review Focus 1：产物目录删了、version.json 还在：照常列出，打开是空的运行。"""
    from server import eval_tasks
    src = eval_tasks.load("ev-v9_hotel_s1")
    assert src["events"] == [] and src["records"] == [] and src["meta"]["error"] == "运行结束于 FAILED"


def test_conclusion_comes_from_regression_json(evroot):
    from server import eval_tasks
    (evroot["res"] / "v9" / "regression.json").write_text(json.dumps({"conclusion": "无退步", "old": "v8"}))
    assert eval_tasks.summaries()[0]["eval"]["conclusion"] == "无退步"
    v, reg = eval_tasks.version_of("ev-v9_hotel_s0")
    assert v["label"] == "v9" and reg["old"] == "v8"


def test_evidence_maps_to_rounds():
    """Review Focus 4：证据有事件 id（整数）也有实验 id（字符串）；实验 id 按它的 recorded 事件定位；找不到的 round 为 None。"""
    from server.eval_view import process_view, segments
    rec = {"verdict": "PROMISING", "codes": [], "action": "RACE"}
    ev = [{"id": i, "ts": float(i), "type": t, "payload": p, "exp_id": x} for i, t, p, x in (
          (1, "state_transition", {"to": "PLAN", "sec": 1.0}, None), (2, "recorded", rec, "e00"),
          (3, "state_transition", {"to": "DECIDE", "sec": 1.0}, None), (4, "decision_rejected", {"errors": ["禁忌表"]}, None),
          (5, "recorded", rec, "e01"), (6, "state_transition", {"to": "FINAL_GATE", "sec": 1.0}, None), (7, "final_gate", {"exp_id": "e01"}, None))]
    assert segments(ev) == {1: "setup", 2: "setup", 3: "setup", 4: "r1", 5: "r1", 6: "r1", 7: "finale"}
    rows = {r["key"]: r for r in process_view(ev, [], 8)}
    assert rows["rejected_decisions"]["evidence"] == [{"ref": "4", "round": "r1"}]
    assert rows["rejected_decisions"]["label"] == "被拒决策" and rows["rejected_decisions"]["detail"]["by_kind"] == {"duplicate": 1}
    from server.eval_view import _evidence
    race = [{"id": 0, "ts": 0.0, "type": "evaluated", "payload": {}, "exp_id": "race_lgbm"}]     # 赛跑实验只有 evaluated，没有 recorded
    assert _evidence(["race_lgbm"], race + ev, segments(race + ev)) == [{"ref": "race_lgbm", "round": "setup"}]
    assert _evidence(["e01", "e99", 2], ev, segments(ev)) == [{"ref": "e01", "round": "r1"}, {"ref": "e99", "round": None},
                                                              {"ref": "2", "round": "setup"}]


def test_version_view_and_scenarios(evroot):
    from server import eval_tasks
    from server.eval_view import scenarios_view, version_view
    (evroot["res"] / "v9" / "regression.json").write_text(json.dumps({
        "new": "v9", "old": "v8", "tier": "fast", "noise_calibrated": True, "conclusion": "有退步需看（1 项）",
        "table": [{"dataset": "hotel", "metric": "holdout", "new": .79, "old": .80, "delta": -.01, "n": 1, "verdict": "worse",
                   "worst_task": "v9_hotel_s0"}],
        "flags": [{"kind": "failed_run", "detail": "运行结束于 FAILED", "task_id": "v9_hotel_s1"}, {"kind": "cost_up", "detail": "x"}]}))
    v, reg = eval_tasks.version_of("ev-v9_hotel_s0")
    vv = version_view(v, reg)
    assert vv["against"] == "v8" and vv["table"][0]["task"] == "ev-v9_hotel_s0"
    assert [f["task"] for f in vv["flags"]] == ["ev-v9_hotel_s1", None]
    assert vv["runs"][0] == {"task": "ev-v9_hotel_s0", "dataset": "hotel", "seed": 0, "status": "ok", "holdout": .79,
                             "delta_b1": pytest.approx(.01)}
    assert scenarios_view(v) == [{"scenario": "leak", "base": "hotel", "passed": False, "reason": "采纳了泄漏列",
                                  "task": "ev-v9_scn_leak_hotel"}]


def test_version_view_without_regression(evroot):
    """Review Focus 3：第一个版本没有 regression.json。"""
    from server import eval_tasks
    from server.eval_view import version_view
    vv = version_view(*eval_tasks.version_of("ev-v9_hotel_s0"))
    assert vv["against"] is None and vv["conclusion"] == "没有回归对比（没有可比的同档上一版本）" and vv["table"] == []
    assert len(vv["runs"]) == 2


@pytest.fixture
def client(evroot, tmp_path, monkeypatch):
    monkeypatch.setenv("DATAMINER_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    from fastapi.testclient import TestClient
    from server.app import app
    return TestClient(app)


def test_api_lists_and_serves_eval_tasks_read_only(client):
    c = client
    ts = c.get("/api/tasks").json()
    ev = [t for t in ts if t["eval"]]
    assert [t["id"] for t in ev][0] == "ev-v9_hotel_s0" and all(t["eval"] is None for t in ts if not t["id"].startswith("ev-"))
    s = c.get("/api/tasks/ev-v9_hotel_s0").json()
    assert s["readonly"] and s["eval_label"] == "v9" and s["status"] == "done" and s["run"] is not None
    assert c.post("/api/tasks/ev-v9_hotel_s0/resume").status_code == 409
    assert c.post("/api/tasks/ev-v9_hotel_s0/accept", json={"note": ""}).status_code == 409
    e = c.get("/api/tasks/ev-v9_hotel_s0/eval").json()
    assert e["version"]["label"] == "v9" and e["scenarios"][0]["scenario"] == "leak"
    assert any(r["key"] == "rejected_decisions" and r["value"] == 1 for r in e["process"])
    assert c.get("/api/tasks/ev-v9_hotel_s0/bundle.zip").status_code == 200
    assert c.get("/api/tasks/ev-nope/eval").status_code == 404


def test_eval_view_for_example_has_process_only(client):
    """Review Focus 2：示例任务（和用户任务）只有「本次运行」。"""
    ex = next(t["id"] for t in client.get("/api/tasks").json() if t["example"])
    e = client.get(f"/api/tasks/{ex}/eval").json()
    assert e["version"] is None and e["scenarios"] is None and {r["key"] for r in e["process"]} >= {"rounds", "wasted_rounds"}
