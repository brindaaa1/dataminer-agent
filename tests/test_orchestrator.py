"""第 4 项测试：状态机（mock LLM）、DECIDE 校验/重试、禁忌表、LLM 判定被忽略、泄漏处置、resume。"""
import json
from pathlib import Path

import pytest

import tools.run_experiment as tre
from agent.llm import PolicyMockLLM, ScriptedLLM
from agent.orchestrator import Orchestrator
from agent.states import State, legal
from conftest import make_toy, toy_spec
from evaluation.final_gate import FinalGate, FinalGateAlreadyRun


def build(cfg, llm, task="t1", crash_hook=None, **spec_kw):
    return Orchestrator(toy_spec(cfg, **spec_kw), cfg, llm, FinalGate("toy", cfg, task), task, crash_hook)


def rows(o):
    return [(r.exp_id, r.action_type, r.verdict, round(r.metrics.get("oot_dev_auc", -1), 6)) for r in o.store.all(o.task_id)]


def test_full_loop_with_mock_llm(toy_cfg):
    o = build(toy_cfg, PolicyMockLLM())
    assert o.run() == State.DONE
    trans = [e["payload"] for e in o.events.query("state_transition")]
    assert all(legal(State(t["from"]), State(t["to"])) for t in trans)
    visited = [t["from"] for t in trans]
    for s in ("INTAKE", "PROFILE", "PLAN", "DECIDE", "EXECUTE", "EVALUATE", "RECORD", "STOP_CHECK", "FINAL_GATE", "REPORT", "CONSOLIDATE"):
        assert s in visited
    assert o.p["best_exp_id"] and o.p["final"]["holdout_auc"] > 0.6
    summ = json.load(open(f"{toy_cfg['paths']['reports_root']}/summary_t1.json"))
    assert summ["final_gate"]["exp_id"] == o.p["best_exp_id"] and summ["stop_reason"]
    # 每个动作都对应一条 ExperimentRecord；race 也有
    assert {"RACE", "PROMOTE_FIDELITY"} <= {r.action_type for r in o.store.all("t1")}


def test_decide_retries_with_feedback_then_succeeds(toy_cfg):
    ok = dict(action="STOP", params={"reason": "x"}, hypothesis="h", expected_gain="g", est_cost={}, rationale="r" * 30)
    bad_param = dict(ok, action="TUNE", params={"space": {"nope": {"low": 0, "high": 1}}})
    llm = ScriptedLLM([{"directions": []}, "这不是 JSON", bad_param, ok])
    o = build(toy_cfg, llm)
    o.run()
    assert o.state == State.DONE
    rej = o.events.query("decision_rejected")
    assert len(rej) == 2 and "不在" in rej[1]["payload"]["errors"][0]
    assert any("修正" in c for c in llm.calls)                          # 错误信息被回灌给 LLM


def test_decide_exhausts_retries_l1_awaits_l0_fails(toy_cfg):
    junk = lambda u: "garbage" if '"mode": "DECIDE"' in u else {"directions": []}
    o = build(toy_cfg, ScriptedLLM(junk))
    assert o.run() == State.AWAIT_HUMAN and o.p["await"]["reason"] == "decide_failed"
    cfg2 = make_toy(Path(toy_cfg["paths"]["artifacts_root"]).parent / "l0")
    o2 = build(cfg2, ScriptedLLM(junk), task="t2", autonomy="L0")
    assert o2.run() == State.FAILED


def test_llm_verdict_ignored_and_taboo(toy_cfg):
    tune = lambda n: dict(action="TUNE", params={"space": {"learning_rate": {"low": 0.02, "high": 0.1, "log": True}}, "n_trials": n},
                          hypothesis="h", expected_gain="g", est_cost={"cpu_minutes": 0.01}, rationale="r" * 30,
                          verdict="ACCEPT", judgement="ACCEPT")                    # LLM 夹带判定
    promote = lambda o: dict(action="PROMOTE_FIDELITY", params={"exp_id": o}, hypothesis="h", expected_gain="g", est_cost={}, rationale="r" * 30)
    stop = dict(action="STOP", params={"reason": "x"}, hypothesis="h", expected_gain="g", est_cost={}, rationale="r" * 30)
    state = {"n": 0}

    def script(u):
        ctx = json.loads(u.split("```json")[-1].split("```")[0])
        if ctx.get("mode") == "PLAN":
            return {"directions": []}
        state["n"] += 1
        if state["n"] == 1:
            return promote(ctx["race"][0]["exp_id"] if ctx["race"][0]["model"] == "lgbm" else ctx["race"][1]["exp_id"])
        if state["n"] in (2, 3):
            return tune(7)                      # 第 3 次和第 2 次完全相同 → 第 3 次被禁忌表拦截并重试
        return stop
    o = build(toy_cfg, ScriptedLLM(script))
    o.run()
    assert any("禁忌表" in " ".join(e["payload"]["errors"]) for e in o.events.query("decision_rejected"))
    tuned = [r for r in o.store.all("t1") if r.action_type == "TUNE"]
    assert len(tuned) == 1 and tuned[0].verdict in ("ACCEPT", "REJECT", "INCONCLUSIVE")   # 判定来自 evaluator，不是 LLM 的 ACCEPT 也可能恰好一致，但来源是 evaluator
    assert tuned[0].verdict == o.events.query("recorded", exp_id=tuned[0].exp_id)[0]["payload"]["verdict"]


def test_spec_incomplete_asks_human_never_guesses(toy_cfg):
    llm = ScriptedLLM([])
    spec = toy_spec(toy_cfg).model_copy(update={"label_def": None, "oot_windows": None})
    o = Orchestrator(spec, toy_cfg, llm, FinalGate("toy", toy_cfg, "t1"), "t1")
    assert o.run() == State.AWAIT_HUMAN
    assert set(o.p["await"]["missing"]) == {"label_def", "oot_windows"} and llm.calls == []   # 没有调用 LLM 去脑补
    o.resume_with({"spec": {"label_def": "toy", "oot_windows": {"oot_dev": ["2013-01", "2013-06"]}}})
    assert o.state == State.INTAKE


def leaky_env(tmp_path):
    return make_toy(tmp_path, leak=True)


def _leak_llm():
    ok = lambda a, p: dict(action=a, params=p, hypothesis="h", expected_gain="g", est_cost={}, rationale="r" * 30)

    def script(u):
        ctx = json.loads(u.split("```json")[-1].split("```")[0])
        if ctx.get("mode") == "PLAN":
            return {"directions": []}
        if ctx["unhandled_leaks"]:
            return ok("ESCALATE_HUMAN", {"reason": "leak", "options": ["quarantine"]}) if "TUNE 已被拒" else None
        return ok("STOP", {"reason": "done"})
    return script


def test_leak_suspect_forces_escalate_l1_and_resume_quarantines(tmp_path):
    cfg = leaky_env(tmp_path)
    tune = dict(action="TUNE", params={"space": {"learning_rate": {"low": 0.02, "high": 0.1}}}, hypothesis="h",
                expected_gain="g", est_cost={}, rationale="r" * 30)
    esc = dict(action="ESCALATE_HUMAN", params={"reason": "leak", "options": []}, hypothesis="h", expected_gain="g", est_cost={}, rationale="r" * 30)
    stop = dict(action="STOP", params={"reason": "x"}, hypothesis="h", expected_gain="g", est_cost={}, rationale="r" * 30)
    llm = ScriptedLLM([{"directions": []}, tune, esc, stop])
    o = build(cfg, llm)
    assert o.run() == State.AWAIT_HUMAN
    assert "leaky" in o.p["pending_leaks"]                                              # 单特征 AUC 触发
    assert any("LEAK_SUSPECT" in " ".join(e["payload"]["errors"]) for e in o.events.query("decision_rejected"))  # 其他动作被拦截
    o.resume_with({"quarantine": ["leaky"], "spec": {"constraints": {"banned_models": ["lgbm"]}}})
    assert o.run() == State.DONE
    assert o.events.query("USER_INSTRUCTION")
    inv = [r for r in o.store.all("t1") if r.invalidated]
    assert inv and all(r.config["model"] == "lgbm" for r in inv)                        # 失效但不删除
    assert "leaky" in o.p["quarantined"]


def test_leak_suspect_l0_auto_quarantines(tmp_path):
    cfg = leaky_env(tmp_path)
    o = build(cfg, PolicyMockLLM(), autonomy="L0")
    assert o.run() == State.DONE
    assert o.events.query("auto_quarantine") and "leaky" in o.p["quarantined"]
    best = o._result(o.p["best_exp_id"])
    assert "leaky" not in best["config"]["features"]


def test_final_gate_runs_once(toy_cfg):
    o = build(toy_cfg, PolicyMockLLM())
    o.run()
    fg = FinalGate("toy", toy_cfg, "t1")
    again = fg.run_once(o._result(o.p["best_exp_id"]))
    assert again == o.p["final"]                                                        # 幂等返回，不重新读
    other = next(r for r in o.store.all("t1") if r.exp_id != o.p["best_exp_id"] and r.artifact_refs)
    with pytest.raises(FinalGateAlreadyRun):
        fg.run_once(o._result(other.exp_id))


def test_stop_on_budget(toy_cfg):
    toy_cfg["run"]["budget"] = {"tokens": 1e9, "cpu_minutes": 1e-6, "wall_minutes": 1e9}
    o = build(toy_cfg, PolicyMockLLM())
    assert o.run() == State.DONE
    assert o.p["best_exp_id"] is None and o.p["final"] is None and "预算不足" in o.p["stop_reason"]   # 没有 ACCEPT 就不碰 holdout


# ---------- resume：在每个状态中途打断，恢复后结果一致且不重复计算 ----------
class Crash(Exception):
    pass


def _count_runs(monkeypatch):
    calls = []
    real = tre.run_study
    monkeypatch.setattr(tre, "run_study", lambda exp_id, *a, **k: (calls.append(exp_id), real(exp_id, *a, **k))[1])
    return calls


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    root = tmp_path_factory.mktemp("base")
    cfg = make_toy(root)
    o = build(cfg, PolicyMockLLM())
    o.run()
    return rows(o), o.p["best_exp_id"], o.p["final"]["holdout_auc"], o.evaluator.budget.used, len(o.store.all("t1"))


@pytest.mark.parametrize("crash_state", ["INTAKE", "PROFILE", "PLAN", "DECIDE", "EXECUTE", "EVALUATE", "RECORD", "STOP_CHECK", "FINAL_GATE", "REPORT"])
def test_resume_after_crash_in_each_state(tmp_path, monkeypatch, baseline, crash_state):
    cfg = make_toy(tmp_path)
    calls = _count_runs(monkeypatch)
    fired = {"done": False}

    def hook(state):
        if state == crash_state and not fired["done"]:
            fired["done"] = True
            raise Crash

    o = build(cfg, PolicyMockLLM(), crash_hook=hook)
    with pytest.raises(Crash):
        o.run()
    o2 = build(cfg, PolicyMockLLM())                        # 新进程：只从检查点恢复
    assert o2.run() == State.DONE
    b_rows, b_best, b_hold, b_oot, b_n = baseline
    assert rows(o2) == b_rows and o2.p["best_exp_id"] == b_best
    assert o2.p["final"]["holdout_auc"] == pytest.approx(b_hold)
    assert o2.evaluator.budget.used == b_oot                # OOT-dev 计数没有被重放翻倍
    assert len(calls) == len(set(calls)) == len(b_rows) - sum(1 for r in b_rows if r[1] in ("BACKTRACK", "INVESTIGATE"))  # 每个实验只算一次


def test_context_exposes_tunable_params(toy_cfg):
    seen = {}

    def script(u):
        ctx = json.loads(u.split("```json")[-1].split("```")[0])
        if ctx.get("mode") == "DECIDE":
            seen.setdefault("ctx", ctx)
            return dict(action="STOP", params={"reason": "x"}, hypothesis="h", expected_gain="g", est_cost={}, rationale="r" * 30)
        return {"directions": []}
    o = build(toy_cfg, ScriptedLLM(script))
    o.run()
    t = seen["ctx"]["tunable"]
    assert "learning_rate" in t and "max_depth" not in t and t["learning_rate"]["low"] < t["learning_rate"]["high"]
    assert seen["ctx"]["fidelity_options"]["low"]["n_trials"] == 3


def test_no_final_model_nudges_promotion(toy_cfg):
    """还没有全保真度最优模型、又有 PROMISING 的低保真实验时：上下文写明规则，策略先验指向 PROMOTE_FIDELITY，
    不写理由就换别的动作会被校验拒绝；升保真度后有了最终模型，这个诊断码消失。"""
    base = dict(hypothesis="h", expected_gain="g", est_cost={})
    tune_short = dict(base, action="TUNE", params={"space": {"learning_rate": {"low": 0.02, "high": 0.1, "log": True}}}, rationale="调参")
    ctxs = []

    def script(user):
        if '"mode": "DECIDE"' not in user:
            return {"directions": []} if '"mode": "PLAN"' in user else {"lessons": []}
        ctx = json.loads(user.split("```json")[1].split("```")[0])
        ctxs.append(ctx)
        if len(ctxs) == 1:
            return tune_short                                             # 偏离先验且理由太短 → 拒绝
        if not ctx["final_model"]["exists"]:
            return dict(base, action="PROMOTE_FIDELITY", params={"exp_id": ctx["final_model"]["promotable"][0]["exp_id"]}, rationale="r" * 30)
        return dict(base, action="STOP", params={"reason": "done"}, rationale="r" * 30)

    o = build(toy_cfg, ScriptedLLM(script))
    assert o.run() == State.DONE
    first = ctxs[0]
    assert first["final_model"]["exists"] is False
    cands = {c["exp_id"]: c for c in first["final_model"]["promotable"]}
    races = {r.exp_id: r for r in o.store.all("t1") if r.action_type == "RACE"}
    assert set(cands) == {e for e, r in races.items() if r.verdict in ("PROMISING", "INCONCLUSIVE")}   # 不确定的赛跑模型也可升
    assert all(c["gap"] is not None and c["oot_dev_auc"] for c in cands.values())
    assert "NO_FINAL_MODEL" in first["diagnosis"]["codes"] and "PROMOTE_FIDELITY" in first["diagnosis"]["policy_prior"]["NO_FINAL_MODEL"]
    rej = o.events.query("decision_rejected")
    assert rej and "偏离策略先验" in rej[0]["payload"]["errors"][0] and "PROMOTE_FIDELITY" in rej[0]["payload"]["errors"][0]
    assert o.p["best_exp_id"] and o.p["final"]                            # 升保真度后被接受，有最终模型
    assert "NO_FINAL_MODEL" not in ctxs[-1]["diagnosis"]["codes"] and ctxs[-1]["final_model"]["exists"]


STOP = {"action": "STOP", "params": {"reason": "x"}, "hypothesis": "h", "expected_gain": "g",
        "est_cost": {"tokens": 0, "cpu_minutes": 0}, "rationale": "stop"}


def _raced(cfg, plan, task_id, **spec_kw):
    from agent.llm import last_json_block

    def llm(user):
        ctx = last_json_block(user)
        return plan(ctx) if ctx.get("mode") == "PLAN" else STOP
    o = Orchestrator(toy_spec(cfg, autonomy="L0", **spec_kw), cfg, ScriptedLLM(llm), None, task_id)
    o.run()
    return o, {r.config["model"] for r in o.store.all(task_id) if r.action_type == "RACE"}


def test_race_only_runs_llm_candidates(toy_cfg):
    from modeling.zoo import ZOO

    def plan(ctx):
        assert set(ctx["models"]) == set(ZOO) and ctx["data_profile"]["n_categorical"] == 1
        return {"directions": [], "candidate_models": [{"model": "lr_scorecard", "reason": "r"},
                                                       {"model": "random_forest", "reason": "r"}]}
    o, raced = _raced(toy_cfg, plan, "c1")
    assert raced == {"lr_scorecard", "random_forest"}
    assert o.events.query("plan_candidates")


def test_candidates_intersect_with_constraints(toy_cfg):
    plan = lambda ctx: {"directions": [], "candidate_models": [{"model": "lgbm", "reason": "r"},
                                                               {"model": "random_forest", "reason": "r"}]}
    _, raced = _raced(toy_cfg, plan, "c2", constraints={"banned_models": ["lgbm"]})
    assert raced == {"random_forest"}


def test_no_candidates_races_all_allowed(toy_cfg):
    from modeling.zoo import ZOO
    _, raced = _raced(toy_cfg, lambda ctx: {"directions": []}, "c3")
    assert raced == set(ZOO)


def test_decide_context_keeps_templates_and_model_profiles(toy_cfg):
    seen = {}

    def plan(ctx):
        return {"directions": [], "candidate_models": [{"model": "lr_scorecard", "reason": "r"}]}

    def llm(user):
        from agent.llm import last_json_block
        ctx = last_json_block(user)
        if ctx.get("mode") == "PLAN":
            return plan(ctx)
        seen.setdefault("ctx", ctx)
        return STOP
    o = Orchestrator(toy_spec(toy_cfg, autonomy="L0"), toy_cfg, ScriptedLLM(llm), None, "tpl")
    o.run()
    assert "templates" in seen["ctx"] and isinstance(seen["ctx"]["models_available"], dict)


def test_guard_not_applied_between_race_peers(toy_cfg):
    from agent.schemas import MetricSpec
    plan = lambda ctx: {"directions": [], "candidate_models": [{"model": "lgbm", "reason": "r"},
                                                               {"model": "lr_scorecard", "reason": "r"}]}
    o, _ = _raced(toy_cfg, plan, "g1", metric=MetricSpec(guards={"pr_auc": -1.0}))   # 容差为负：任何比较都会触发护栏
    races = [r for r in o.store.all("g1") if r.action_type == "RACE"]
    assert len(races) == 2 and not any("GUARD_FAIL" in r.diagnosis_codes for r in races)


def test_duplicate_candidates_race_once(toy_cfg):
    plan = lambda ctx: {"directions": [], "candidate_models": [{"model": "lr_scorecard", "reason": "r"},
                                                               {"model": "lr_scorecard", "reason": "r"}]}
    o, _ = _raced(toy_cfg, plan, "d1")
    assert o.p["race"] == ["d1_race_lr_scorecard"] and len(o.events.query("evaluated")) == 1


def test_metric_locked_from_checkpoint_on_resume(toy_cfg):
    from agent.schemas import MetricSpec
    o = Orchestrator(toy_spec(toy_cfg, autonomy="L0"), toy_cfg, PolicyMockLLM(), None, "lk")
    o.run(max_steps=1)                                                       # 写入检查点，口径为默认 auc
    Orchestrator(toy_spec(toy_cfg, autonomy="L0", metric=MetricSpec(primary="ks")), toy_cfg, PolicyMockLLM(), None, "lk")
    assert toy_cfg["metric"]["primary"] == "auc"
