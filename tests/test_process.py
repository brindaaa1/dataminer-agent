"""过程指标（spec §4）：每项至少一个正例、一个反例。事件和记录都是构造的，形状与 EventLog.query()、ExperimentRecord 一致。"""
import pytest

from eval.process import process_metrics


def rec(exp_id, action, verdict, fid="low", parent=None, diff=None, codes=(), feats=("x",), h=None, model="lgbm"):
    return {"exp_id": exp_id, "parent_exp_id": parent, "action_type": action, "verdict": verdict, "fidelity": fid,
            "diff": diff or {}, "diagnosis_codes": list(codes), "config": {"model": model, "features": list(feats)},
            "config_hash": h if h is not None else exp_id, "metrics": {}}


def ev(i, t, p=None, x=None, sec=0.0):
    return {"id": i, "ts": float(i), "type": t, "exp_id": x, "payload": p or {}}


def run():
    """赛跑 r → 升全量 e01（最终模型）→ 加特征 e02（低保真，比对照高，不确定）→ e03 加特征一个没留 →
    e04 调参被拒 → e05 和 e04 同样的调参（重复）→ 升 e02 为 e06 并被采纳。"""
    records = [rec("r", "RACE", "PROMISING"),
               rec("e01", "PROMOTE_FIDELITY", "ACCEPT", "full", "r", {"exp_id": "r"}),
               rec("e02", "EXPAND_FEATURES", "INCONCLUSIVE", "low", "e01", {"template_id": "t1", "expand": {"kept": ["f1"]}}, feats=("x", "f1")),
               rec("e03", "EXPAND_FEATURES", "NONE", "none", "e01", {"template_id": "t2"}, codes=["EXPAND_EMPTY"], h=""),
               rec("e04", "TUNE", "REJECT", "low", "e01", {"space": {"lr": 1}}, h="hT"),
               rec("e05", "TUNE", "REJECT", "low", "e01", {"space": {"lr": 1}}, h="hT"),
               rec("e06", "PROMOTE_FIDELITY", "ACCEPT", "full", "e02", {"exp_id": "e02"}, feats=("x", "f1"))]
    events = [ev(1, "oot_power", {"mde": 0.012, "underpowered": True}),
              ev(2, "evaluated", {"verdict": "INCONCLUSIVE", "compare": {"delta": 0.004, "alpha": 0.04, "k": 2, "significant": False}}, "e02"),
              ev(3, "decision_rejected", {"errors": ["该配置已经试过（禁忌表）：与 e04 完全相同"]}),
              ev(4, "decision_rejected", {"errors": ["模板 t9 不可用；可用模板：['t1']"]}),
              ev(5, "evaluated", {"verdict": "ACCEPT", "compare": {"delta": 0.006, "alpha": 0.03, "k": 3, "significant": True}}, "e06"),
              ev(6, "final_gate", {"exp_id": "e06", "metric": "auc", "holdout_auc": 0.76}),
              ev(7, "await_human", {"reason": "spec_incomplete"}), ev(8, "spec_confirmed", {"key": "label_def"}),
              ev(9, "stop", {"reason": "MAX_ROUNDS"})]
    events += [ev(20 + i, "recorded", {"verdict": r["verdict"], "worth": r["exp_id"] == "e02"}, r["exp_id"]) for i, r in enumerate(records[1:])]
    events += [ev(40, "state_transition", {"from": "PLAN", "to": "DECIDE", "sec": 30.0})]
    return events, records


def test_wasted_rounds_count_failures_and_duplicates():
    m = process_metrics(*run(), max_rounds=8)
    assert m["wasted_rounds"]["evidence"] == ["e03", "e05"] and m["wasted_rounds"]["value"] == pytest.approx(2 / 6, abs=1e-3)
    assert m["rounds"]["value"] == 6 and m["rounds"]["max"] == 8 and m["rounds"]["stop_reason"] == "MAX_ROUNDS"


def test_rejected_decisions_are_classified():
    m = process_metrics(*run(), max_rounds=8)
    assert m["rejected_decisions"]["value"] == 2 and m["rejected_decisions"]["by_kind"] == {"duplicate": 1, "template": 1}


def test_upgrade_hit_and_success_and_feature_funnel():
    m = process_metrics(*run(), max_rounds=8)
    assert m["upgrade_hit_rate"]["value"] == 1.0 and m["upgrade_hit_rate"]["worth"] == ["e02"]
    assert m["upgrade_success_rate"]["value"] == 1.0
    f = m["fe_funnel"]
    assert f["value"] is True and (f["attempts"], f["kept"], f["promising"], f["promoted"], f["accepted"]) == (2, 1, 1, 1, 1)
    assert m["final_lineage"]["evidence"] == ["r", "e01", "e02", "e06"] and "加特征 t1" in m["final_lineage"]["value"]


def test_recovery_flags_repeating_after_failure():
    m = process_metrics(*run(), max_rounds=8)
    assert m["recovery"]["evidence"] == ["e04"]                       # e04 被拒后 e05 原样重复
    assert m["recovery"]["value"] == pytest.approx(2 / 3, abs=1e-3)   # e03→e04 换了、e04→e05 没换、e05→e06 换了


def test_human_oot_cost_and_decide_failed():
    events, records = run()
    m = process_metrics(events, records, max_rounds=8)
    assert m["human"]["value"] == 1 and m["human"]["confirmed"] == 1
    assert m["oot"]["value"] == 3 and m["oot"]["mde"] == 0.012 and m["oot"]["underpowered"] is True
    assert m["cost"]["wall_sec"] == 30.0 and m["decide_failed"]["value"] is False
    m2 = process_metrics(events + [ev(50, "decide_failed", {"errors": ["x"]})], records, max_rounds=8)
    assert m2["decide_failed"]["value"] is True and m2["decide_failed"]["rounds_lost"] == 2


def test_run_without_experiments_has_no_division_by_zero():
    """第一轮就决策失败：只有赛跑，没有其他实验。比例为 None。"""
    m = process_metrics([ev(1, "decide_failed", {"errors": ["x"]})], [rec("r", "RACE", "PROMISING")], max_rounds=8)
    assert m["wasted_rounds"]["value"] is None and m["upgrade_hit_rate"]["value"] is None
    assert m["recovery"]["value"] is None and m["fe_funnel"]["value"] is False and m["final_lineage"]["value"] is None


def test_last_round_candidate_is_unreachable_not_a_miss():
    """终审 Important 3：最后一轮才出现的"值得升"候选，之后没有决策了，不可能被升；算成没升会拉低命中率，
    情景运行只有 5 轮时很常见，还会混进 A/A 的噪声带。单独列为 unreachable，不进命中率的分母。"""
    events, records = run()
    records = records + [rec("e07", "EXPAND_FEATURES", "PROMISING", "low", "e06", {"template_id": "t3", "expand": {"kept": ["f3"]}})]
    events = events + [ev(60, "recorded", {"verdict": "PROMISING", "worth": True}, "e07")]
    m = process_metrics(events, records, max_rounds=8)
    assert m["upgrade_hit_rate"]["worth"] == ["e02"] and m["upgrade_hit_rate"]["unreachable"] == ["e07"]
    assert m["upgrade_hit_rate"]["value"] == 1.0


def test_worth_promoting_follows_agent_worth():
    """"值得升"以 agent 记下的 worth 为准（低保真比最优高 ≥ MDE）：差距不够的不算，和 agent 的升全量规则同一口径。"""
    events, records = run()
    events = [{**e, "payload": {**e["payload"], "worth": False}} if e["type"] == "recorded" and e["exp_id"] == "e02" else e
              for e in events]
    m = process_metrics(events, records, max_rounds=8)
    assert m["upgrade_hit_rate"]["worth"] == [] and m["fe_funnel"]["promising"] == 0


def test_upgrade_success_excludes_the_first_final_model():
    """第一次升全量（还没有最终模型时）几乎总被采纳，算进去会把成功率抬高；只看已有最终模型之后的升全量。"""
    m = process_metrics(*run(), max_rounds=8)
    assert m["upgrade_success_rate"]["evidence"] == ["e06"] and m["upgrade_success_rate"]["value"] == 1.0
    events, records = run()
    records = records[:2]                                     # 只有赛跑 + 第一次升全量
    assert process_metrics(events, records, max_rounds=8)["upgrade_success_rate"]["value"] is None
