"""决策校验与诊断码：v5 评测 home_credit 暴露的重复实验、升全量提示。"""
import pytest

from agent.actions import config_hash
from agent.decide import Validator
from agent.schemas import Decision, TaskSpec
from data.knowhow import load_config
from evaluation.guardrails import decision_codes
from memory.schema import ExperimentRecord
from memory.store import Store

A = {"model": "random_forest", "features": ["x1", "x2"], "space": None, "fidelity": "low", "n_trials": None}
B = {**A, "features": ["x1", "x2", "f1"], "feature_sets": ["fs1"]}


def rec(exp_id, action, cfg, verdict, parent=None, diff=None):
    return ExperimentRecord(exp_id=exp_id, task_id="t", parent_exp_id=parent, hypothesis="h", action_type=action, diff=diff or {},
                            config=cfg, config_hash=config_hash(cfg), fidelity=cfg["fidelity"], metrics={"oot_dev_auc": .7, "gap": .01},
                            diagnosis_codes=[], cost={"cpu_minutes": 0, "wall_minutes": 0}, verdict=verdict)


@pytest.fixture
def env(tmp_path):
    """赛跑 race → 升全量 e01（最终模型）→ 在 e01 上用模板 t1 加特征 e02（低保真）→ 升全量 e03（不确定）。"""
    store = Store(str(tmp_path / "s.db"))
    for r in (rec("race", "RACE", A, "PROMISING"), rec("e01", "PROMOTE_FIDELITY", {**A, "fidelity": "full"}, "ACCEPT", "race", {"exp_id": "race"}),
              rec("e02", "EXPAND_FEATURES", B, "PROMISING", "e01", {"template_id": "t1", "expand": {"feature_set_id": "fs1", "kept": ["f1"]}}),
              rec("e03", "PROMOTE_FIDELITY", {**B, "fidelity": "full"}, "INCONCLUSIVE", "e02", {"exp_id": "e02"})):
        store.add(r)
    p = {"task_id": "t", "pending_leaks": {}, "current_node": "e01", "quarantined": [], "investigate_counts": {}, "best_exp_id": "e01",
         "round_no": 3, "templates": ["t1", "t2"], "last_codes": []}
    spec = TaskSpec(dataset="d", label_def="x", observation_time_col="t", oot_windows={"oot_dev": .1}, autonomy="L0",
                    budget={"tokens": 1e9, "cpu_minutes": 1e9, "wall_minutes": 1e9})
    cfg = load_config()
    return store, p, Validator(cfg, spec, store, p, {"tokens": 1e9, "cpu_minutes": 1e9, "wall_minutes": 1e9}, 1000), cfg


def dec(action, params):
    return Decision.model_validate(dict(action=action, params=params, hypothesis="h", expected_gain="g", est_cost={}, rationale="r" * 40))


def test_same_template_on_same_node_is_rejected_before_running(env):
    """v5 s2：e04 在同一节点上又用了同一个模板，生成的特征、配置与 e02 完全相同，白跑一轮（训练配置要生成后才知道，禁忌表拦不住）。"""
    _, _, v, _ = env
    errs = " ".join(v.check(dec("EXPAND_FEATURES", {"template_id": "t1"})))
    assert "e02" in errs and "t1" in errs
    assert v.check(dec("EXPAND_FEATURES", {"template_id": "t2"})) == []


def test_unknown_or_unsupported_template_is_rejected(env):
    """v5 s0：LLM 选了代码不支持的模板（多级聚合），执行时 NotImplementedError，白费一轮。只能从可用模板里选。"""
    _, _, v, _ = env
    assert "t9" in " ".join(v.check(dec("EXPAND_FEATURES", {"template_id": "t9"})))


def test_promoting_a_duplicate_of_a_promoted_config_says_it_was_already_verified(env):
    """v5 s2：e04 与 e02 配置相同，升全量 = e03。旧提示让它『用 PROMOTE_FIDELITY 复验』，LLM 照做又撞，连撞 4 次决策失败。"""
    store, p, v, _ = env
    store.add(rec("e04", "EXPAND_FEATURES", B, "PROMISING", "e01", {"template_id": "t1"}))
    errs = " ".join(v.check(dec("PROMOTE_FIDELITY", {"exp_id": "e04"})))
    assert "e03" in errs and "已经在全量数据上复验过" in errs and "用 PROMOTE_FIDELITY" not in errs


def test_promising_low_fidelity_result_nudges_promotion(env):
    """v5 s0/s1：有了最终模型后，低保真加特征比对照好（PROMISING 或不确定但更高），agent 没去升全量而是继续调参。
    这类结果没升过时出 PROMOTE_CANDIDATE，策略先验指向 PROMOTE_FIDELITY；升过就消失。"""
    store, p, _, _ = env
    recs = store.all("t")
    p["promote_hint"] = ["e02"]
    assert "PROMOTE_CANDIDATE" not in decision_codes(p, recs)                       # e02 已经升过（e03）
    store.add(rec("e05", "EXPAND_FEATURES", {**B, "features": ["x1", "x2", "f2"]}, "INCONCLUSIVE", "e01", {"template_id": "t2"}))
    p["promote_hint"] = ["e02", "e05"]
    assert "PROMOTE_CANDIDATE" in decision_codes(p, store.all("t"))
    p["best_exp_id"] = None
    assert "PROMOTE_CANDIDATE" not in decision_codes(p, store.all("t"))             # 还没有最终模型时由 NO_FINAL_MODEL 负责


def test_retry_feedback_accumulates_all_errors(env):
    """v6 s1/s2：模板 A 被拒换 B，B 被拒又换回 A——每次重试只看到最近一条错误，连拒 4 次决策失败。重试时带上本次决策所有被拒原因。"""
    from agent.decide import decide
    from agent.llm import ScriptedLLM
    _, _, v, cfg = env
    a = dict(action="EXPAND_FEATURES", params={"template_id": "t1"}, hypothesis="h", expected_gain="g", est_cost={}, rationale="r" * 40)
    b = dict(a, params={"template_id": "t9"})
    ok = dict(a, params={"template_id": "t2"})
    llm = ScriptedLLM([a, b, ok])
    d, _ = decide(llm, {"mode": "DECIDE"}, v, cfg)
    assert d.params["template_id"] == "t2"
    assert "e02" in llm.calls[2] and "t9" in llm.calls[2]                     # 第三次同时看到两次被拒的原因（e02 只出现在 t1 的错误里）
