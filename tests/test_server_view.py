"""快照：示例任务（DeepSeek 真实运行）的轮次、播报、结论卡；接入阶段的问题显示与草稿视图。"""
import json

import pytest

from data.sample import SAMPLE_CSV


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.setenv("DATAMINER_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    return tmp_path / "ws"


def _example():
    from server import examples
    return examples.load("hotel_deepseek3")


def test_example_snapshot_rounds_and_narration():
    from server.view import snapshot
    s = snapshot(_example())
    assert s["id"] == "ex-hotel_deepseek3" and s["stage"] == "report" and s["readonly"]
    r = s["run"]
    assert [x["key"] for x in r["rounds"]] == ["setup", "r1", "r2", "r3", "r4", "r5"]
    r1 = r["rounds"][1]
    assert r1["verdict"] == "REJECT" and r1["codes"] == ["OVERFIT_GAP"] and r1["hypothesis"] and r1["alternatives"]
    assert r1["metrics"]["gap"] == pytest.approx(0.0406, abs=1e-4)
    texts = [n["text"] for n in r["narration"]]
    assert texts[0].startswith("模型赛跑：lgbm 时间外 AUC 0.841（有希望）")
    assert texts[1] == "第 1 轮 · 全量复验：拒绝 · 时间外 AUC 0.833，gap 0.041（OVERFIT_GAP）"
    assert texts[5].startswith("第 5 轮 · 停止：已存在被 ACCEPT 的全保真最终模型")
    assert "最终检验：holdout AUC 0.731，比 OOT-dev 低 0.086，疑似对 OOT-dev 过拟合" in texts
    assert texts[-1] == "本次沉淀 5 条经验"
    assert [p["label"] for p in r["chart"]] == ["赛跑", "R1", "R2", "R3", "R4"]


def test_example_final_card_reads_risks_from_model_card():
    from server.view import snapshot
    s = snapshot(_example())
    f = s["run"]["final"]
    assert f["model"] == "lr_scorecard" and f["overfit"] and f["n_experiments"] == 6 and f["n_accepted"] == 1
    assert any("疑似对 OOT-dev 过拟合" in x and "**" not in x for x in f["risks"])
    assert {"model_card.md", "experiment_tree.mmd", "events.json"} <= {a["name"] for a in s["artifacts"]}


def test_question_display_is_human_readable():
    from server.view import display
    assert display("label", json.dumps({"col": "is_canceled", "positive": "1", "negative": "0", "definition": "订单被取消"})) \
        == "is_canceled = 1：订单被取消"
    assert display("windows", json.dumps({"train_valid": ["2015-07", "2016-12"], "oot_dev": ["2017-01", "2017-04"],
                                          "holdout": ["2017-05", "2017-08"]})) \
        == "训练/验证 2015-07~2016-12 · OOT-dev 2017-01~2017-04 · holdout 2017-05~2017-08"
    assert display("metric", json.dumps({"primary": "auc", "guards": {"ks": 0.02}})) == "主指标 AUC，护栏 KS 跌幅 ≤ 0.02"
    assert display("assigned_room_type", "隔离（at_checkin）") == "隔离（at_checkin）"


def test_live_intake_snapshot(ws):
    from server import intake_runner, sources, tasks
    from server.view import snapshot
    tid = tasks.create(SAMPLE_CSV.read_bytes(), "hotel_bookings_sample.csv", "预测是否取消", "mock")["id"]
    intake_runner.start(tid)
    s = snapshot(sources.live(tid))
    assert s["stage"] == "intake" and s["data"]["n_rows"] == 7997 and s["data"]["n_cols"] == 32
    qs = s["intake"]["questions"]
    assert len(qs) == 4 and all(q["required"] for q in qs)
    assert next(q for q in qs if q["key"] == "label")["display"].startswith("is_canceled = 1：")
    groups = {f["name"]: f["group"] for f in s["intake"]["draft"]["fields"]}
    assert groups["reservation_status"] == "leak" and groups["lead_time"] == "ok"
    assert s["intake"]["draft"]["confirmed"] == [] and s["run"] is None
    intake_runner.answer(tid, {q["key"]: q["recommended"] for q in qs})
    s = snapshot(sources.live(tid))
    assert s["stage"] == "confirm" and s["intake"]["questions"] == []
    assert sorted(s["intake"]["draft"]["confirmed"]) == ["label", "metric", "observation_time_expr", "windows"]
    assert {"data_dictionary.yaml", "blacklist.yaml", "task_spec.yaml"} <= {a["name"] for a in s["artifacts"]}


def test_resumed_race_does_not_list_the_same_experiment_twice():
    """中止在模型赛跑中途、续跑时 PLAN 重跑：同一实验会有两条 evaluated 事件，展示时只保留最后一条。"""
    from server.view import snapshot
    ev = lambda i, t, p, x=None: {"id": i, "ts": i, "type": t, "exp_id": x, "payload": p}
    m = {"valid_auc": 0.85, "oot_dev_auc": 0.835}
    events = [ev(1, "state_transition", {"from": "INTAKE", "to": "PROFILE"}),
              ev(2, "evaluated", {"verdict": "PROMISING", "codes": [], "model": "lgbm", "fidelity": "low", "metrics": m}, "t_race_lgbm"),
              ev(3, "evaluated", {"verdict": "PROMISING", "codes": [], "model": "lgbm", "fidelity": "low", "metrics": m}, "t_race_lgbm"),
              ev(4, "evaluated", {"verdict": "REJECT", "codes": [], "model": "lr_scorecard", "fidelity": "low",
                                  "metrics": {"oot_dev_auc": 0.814}}, "t_race_lr_scorecard")]
    src = {"meta": {"id": "t", "title": "t", "status": "running", "provider": "mock", "settings": {}}, "events": events,
           "records": [], "trials": {}, "intake": None, "data": {"csv_name": "x.csv", "description": ""}, "files": {}}
    setup = snapshot(src)["run"]["rounds"][0]
    assert [e["exp_id"] for e in setup["evaluations"]] == ["t_race_lgbm", "t_race_lr_scorecard"]
    assert snapshot(src)["run"]["narration"][0]["text"].count("lgbm") == 1


def test_race_narration_mentions_tie():
    """赛跑打平时播报里说明"效果相当、按偏好推荐"，否则用户会以为选的是分数最高的。"""
    from server.view import snapshot
    src = _example()
    first = next(e["id"] for e in src["events"] if e["type"] == "state_transition" and e["payload"].get("to") == "DECIDE")
    tie = {"id": first - 0.5, "ts": 0.0, "type": "race_tie", "exp_id": None,
           "payload": {"recommended": "lgbm", "tied": ["lgbm", "catboost"], "best": "catboost", "mde": 0.0099}}
    s = snapshot({**src, "events": sorted(src["events"] + [tie], key=lambda e: e["id"])})
    assert "lgbm、catboost 效果相当（差距 < MDE 0.0099），按偏好推荐 lgbm" in s["run"]["narration"][0]["text"]


def test_rule_decision_and_reference_drop_in_narration():
    """第一轮由代码按打平规则升全量：播报和轮次标出"按规则"，不能看起来像 LLM 的决策；
    最终检验写明默认参照模型自己掉了多少（疑似过拟合按相对口径判定）。"""
    from server.view import snapshot
    src = _example()
    first = next(e["id"] for e in src["events"] if e["type"] == "decision")
    fg = next(e["id"] for e in src["events"] if e["type"] == "final_gate")
    ev = [{**e, "payload": {**e["payload"], "by": "rule"}} if e["id"] == first else
          {**e, "payload": {**e["payload"], "reference_drop": 0.02}} if e["id"] == fg else e for e in src["events"]]
    s = snapshot({**src, "events": ev})
    r = s["run"]
    assert r["rounds"][1]["by_rule"] is True and r["rounds"][2]["by_rule"] is False
    texts = [n["text"] for n in r["narration"]]
    assert texts[1].startswith("第 1 轮 · 全量复验（按规则）：")
    assert "最终检验：holdout AUC 0.731，比 OOT-dev 低 0.086（默认参照模型低 0.020，多掉 0.066），疑似对 OOT-dev 过拟合" in texts


def test_final_narration_wording_when_holdout_is_higher():
    """holdout 比 OOT-dev 还高时写"高"，不写"低 -0.003"（v10b lending_club）。"""
    from server.view import _closing
    sm = {"stop_reason": None, "final": {"holdout_auc": 0.655, "drop": -0.003, "reference_drop": -0.006, "overfit_to_oot_dev": False}}
    t = _closing([], sm, "auc", "AUC")[0]["text"]
    assert t == "最终检验：holdout AUC 0.655，比 OOT-dev 高 0.003（默认参照模型高 0.006，多掉 0.003）"
