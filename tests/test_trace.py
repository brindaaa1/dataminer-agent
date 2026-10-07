"""事件 → 按轮次 trace：纯函数单测 + 真实状态机跑出的事件（llm_call / evaluated）。"""
import pytest

from agent.llm import PolicyMockLLM, ScriptedLLM, TracedLLM
from agent.orchestrator import Orchestrator
from agent.states import State
from conftest import toy_spec
from evaluation.final_gate import FinalGate
from runtime.trace import build_trace, round_rows, segment_evaluations, summarize


def ev(i, type_, payload=None, exp_id=None):
    return {"id": i, "ts": 100.0 + i, "type": type_, "exp_id": exp_id, "payload": payload or {}}


def tr(i, a, b):
    return ev(i, "state_transition", {"from": a, "to": b})


def llm(i, stage, tokens=10, **kw):
    return ev(i, "llm_call", {"stage": stage, "tokens": tokens, "latency_s": 0.5, "prompt": "p", "response": "r", **kw})


DEC = {"action": "TUNE", "params": {"n_trials": 5}, "hypothesis": "调学习率"}


def sample_events():
    es = [tr(1, "INTAKE", "PROFILE"), tr(2, "PROFILE", "PLAN"), llm(3, "PLAN"),
          ev(4, "evaluated", {"verdict": "INCONCLUSIVE", "codes": [], "metrics": {"oot_dev_auc": 0.70}}, "t_race_lgbm"),
          ev(5, "evaluated", {"verdict": "INCONCLUSIVE", "codes": [], "metrics": {"oot_dev_auc": 0.72}}, "t_race_lr"),
          tr(6, "PLAN", "DECIDE"),
          # 第 1 轮：一次被拒，重试成功
          llm(7, "DECIDE"), ev(8, "decision_rejected", {"attempt": 0, "errors": ["禁忌表"]}),
          llm(9, "DECIDE"), ev(10, "decision", {"attempt": 1, "decision": DEC}),
          tr(11, "DECIDE", "EXECUTE"), tr(12, "EXECUTE", "EVALUATE"),
          ev(13, "evaluated", {"verdict": "ACCEPT", "codes": [], "metrics": {"oot_dev_auc": 0.75},
                               "compare": {"delta": 0.03, "significant": True}}, "t_e01"),
          tr(14, "EVALUATE", "RECORD"), ev(15, "recorded", {"verdict": "ACCEPT", "codes": [], "action": "TUNE"}, "t_e01"),
          tr(16, "RECORD", "STOP_CHECK"), tr(17, "STOP_CHECK", "DECIDE"),
          # 第 2 轮：请求人工 → 回复 → 回到 DECIDE（不写 state_transition，仍属本轮）→ STOP
          llm(18, "DECIDE"), ev(19, "decision", {"attempt": 0, "decision": {**DEC, "action": "ESCALATE_HUMAN"}}),
          ev(20, "await_human", {"reason": "escalate_human"}), tr(21, "DECIDE", "AWAIT_HUMAN"),
          ev(22, "USER_INSTRUCTION", {"approve": ["x"]}), ev(23, "resumed", {"to": "DECIDE"}),
          llm(24, "DECIDE"), ev(25, "decision", {"attempt": 0, "decision": {**DEC, "action": "STOP"}}),
          tr(26, "DECIDE", "FINAL_GATE"),
          ev(26.5, "final_gate", {"exp_id": "t_e01", "holdout_auc": 0.74}, "t_e01"), tr(27, "FINAL_GATE", "REPORT"), llm(28, "REPORT", tokens=5), ev(29, "report_built", {}),
          tr(30, "REPORT", "CONSOLIDATE"), llm(31, "CONSOLIDATE", error="TimeoutError: x", tokens=0),
          ev(32, "consolidated", {}), tr(33, "CONSOLIDATE", "DONE")]
    return es


def test_segments_by_round():
    t = build_trace(sample_events())
    assert [(s["kind"], s["round"]) for s in t] == [("setup", 0), ("round", 1), ("round", 2), ("finale", None)]
    setup, r1, r2, fin = t
    assert [c["stage"] for c in setup["llm_calls"]] == ["PLAN"] and setup["exp_ids"] == ["t_race_lgbm", "t_race_lr"]
    assert setup["states"] == ["INTAKE", "PROFILE", "PLAN"]
    assert len(r1["llm_calls"]) == 2 and len(r1["rejected"]) == 1 and r1["decision"]["action"] == "TUNE"
    assert r1["recorded"]["exp_id"] == "t_e01" and r1["evaluations"][0]["compare"]["delta"] == 0.03
    assert r1["states"] == ["DECIDE", "EXECUTE", "EVALUATE", "RECORD", "STOP_CHECK"] and r1["tokens"] == 20
    assert r1["llm_latency_s"] == 1.0 and (r1["ts_start"], r1["ts_end"]) == (107.0, 117.0)
    # 人工暂停与恢复留在同一轮；最后一次决策为准
    assert [a["type"] for a in r2["awaited"]] == ["await_human", "USER_INSTRUCTION", "resumed"]
    assert r2["decision"]["action"] == "STOP" and len(r2["llm_calls"]) == 2 and r2["recorded"] is None
    assert [c["stage"] for c in fin["llm_calls"]] == ["REPORT", "CONSOLIDATE"]


def test_summarize_and_rows():
    t = build_trace(sample_events())
    s = summarize(t)
    assert s == {"n_rounds": 2, "n_llm_calls": 7, "n_llm_errors": 1, "tokens": 55, "n_rejected": 1,
                 "verdicts": {"ACCEPT": 1}, "stop_reason": "LLM_STOP: ", "final_state": "DONE",   # 第 2 轮 STOP 决策
                 "final": {"exp_id": "t_e01", "holdout_auc": 0.74}}
    assert t[-1]["final"]["holdout_auc"] == 0.74
    rows = round_rows(t)
    assert [r["段"] for r in rows] == ["准备", "第 1 轮", "第 2 轮", "收尾"]
    assert rows[0]["动作"] == "RACE" and rows[0]["OOT-dev AUC"] == 0.72          # 赛跑取最好的一个
    assert rows[1]["实验"] == "t_e01" and rows[1]["结论"] == "ACCEPT" and rows[1]["ΔAUC"] == 0.03 and rows[1]["被拒"] == 1
    assert rows[2]["动作"] == "STOP" and rows[2]["结论"] is None
    assert rows[3]["动作"] is None and rows[3]["LLM 调用"] == 2


def test_edge_cases():
    assert build_trace([]) == [] and summarize([])["n_rounds"] == 0 and round_rows([]) == []
    # 顺序被打乱也按 id 排；中途崩溃（没有到 DECIDE）只有准备段
    es = sample_events()[:5]
    t = build_trace(list(reversed(es)))
    assert len(t) == 1 and [e["id"] for e in t[0]["events"]] == [1, 2, 3, 4, 5]
    assert summarize(t)["final_state"] == "PLAN"
    # 没有 stop 事件、最后一次决策是 STOP：用它的理由
    es2 = sample_events()
    es2[24]["payload"]["decision"]["params"] = {"reason": "收益低"}
    assert summarize(build_trace(es2))["stop_reason"] == "LLM_STOP: 收益低"
    # 停止原因来自 stop 事件；输入不被修改
    es = sample_events() + [ev(40, "stop", {"reason": "MAX_ROUNDS"})]
    before = repr(es)
    assert summarize(build_trace(es))["stop_reason"] == "MAX_ROUNDS" and repr(es) == before


def test_records_fallback_for_old_logs():
    """早期日志没有 evaluated 事件：用实验记录补指标；有 evaluated 事件时不用记录。"""
    es = [e for e in sample_events() if e["type"] != "evaluated"]
    recs = [{"exp_id": "t_race_lgbm", "action_type": "RACE", "verdict": "INCONCLUSIVE", "diagnosis_codes": [],
             "metrics": {"oot_dev_auc": 0.70}, "config": {"model": "lgbm"}, "fidelity": "low"},
            {"exp_id": "t_race_lr", "action_type": "RACE", "verdict": "REJECT", "diagnosis_codes": ["PLATEAU"],
             "metrics": {"oot_dev_auc": 0.69}, "config": {"model": "lr"}, "fidelity": "low"},
            {"exp_id": "t_e01", "action_type": "TUNE", "verdict": "ACCEPT", "diagnosis_codes": [],
             "metrics": {"oot_dev_auc": 0.75, "valid_auc": 0.76}, "config": {"model": "lgbm"}, "fidelity": "full"}]
    t = build_trace(es)
    assert round_rows(t)[1]["OOT-dev AUC"] is None                       # 不给记录：没有指标
    rows = round_rows(t, recs)
    assert rows[0]["动作"] == "RACE" and rows[0]["OOT-dev AUC"] == 0.70 and rows[0]["实验"] == "t_race_lgbm, t_race_lr"
    assert rows[1]["OOT-dev AUC"] == 0.75 and rows[1]["ΔAUC"] is None
    assert segment_evaluations(t[1], recs)[0]["source"] == "record" and segment_evaluations(t[2], recs) == []
    t2 = build_trace(sample_events())                                    # 有 evaluated 事件时以事件为准
    assert round_rows(t2, recs)[1]["ΔAUC"] == 0.03 and "source" not in segment_evaluations(t2[1], recs)[0]


def test_traced_llm_records_success_and_error():
    got = []
    t = TracedLLM(ScriptedLLM(["hello"]), lambda ty, pl: got.append((ty, pl)), lambda: "DECIDE")
    assert t.complete("sys", "user prompt") == ("hello", 0)
    assert t.calls == ["user prompt"]                                   # 其他属性透传
    ty, pl = got[0]
    assert ty == "llm_call" and pl["stage"] == "DECIDE" and pl["prompt"] == "user prompt" and pl["response"] == "hello"
    assert pl["system_chars"] == 3 and "error" not in pl
    with pytest.raises(IndexError):                                     # 脚本用完 → 异常照常抛出，同时留痕
        t.complete("sys", "again")
    assert "IndexError" in got[1][1]["error"] and "response" not in got[1][1]


def test_orchestrator_emits_llm_call_and_evaluated(toy_cfg):
    o = Orchestrator(toy_spec(toy_cfg), toy_cfg, PolicyMockLLM(), FinalGate("toy", toy_cfg, "t1"), "t1")
    assert o.run() == State.DONE
    evs = o.events.query()
    calls = [e["payload"] for e in evs if e["type"] == "llm_call"]
    assert {"PLAN", "DECIDE", "REPORT", "CONSOLIDATE"} <= {c["stage"] for c in calls}
    assert all(c["response"] and c["model"] == "PolicyMockLLM" for c in calls)
    # 每条带结论的实验记录都有一条 evaluated 事件，结论一致（含模型赛跑）
    evald = {e["exp_id"]: e["payload"] for e in evs if e["type"] == "evaluated"}
    scored = [r for r in o.store.all("t1") if r.verdict != "NONE"]
    assert scored and all(evald[r.exp_id]["verdict"] == r.verdict for r in scored)
    assert all(evald[r.exp_id]["metrics"]["oot_dev_auc"] == pytest.approx(r.metrics["oot_dev_auc"]) for r in scored)

    t = build_trace(evs)
    s = summarize(t)
    to_decide = [e for e in evs if e["type"] == "state_transition" and e["payload"]["to"] == "DECIDE"]
    assert s["final_state"] == "DONE" and s["n_rounds"] == len(to_decide) and s["n_llm_calls"] == len(calls)
    rounds = [x for x in t if x["kind"] == "round"]
    recorded = [x["recorded"]["exp_id"] for x in rounds if x["recorded"]]
    assert len(recorded) == o.p["round_no"]                             # 最后一轮 LLM 选 STOP：有决策、无记录
    assert s["final"] == o.p["final"] and t[-1]["kind"] == "finale"      # holdout 验收结果进了收尾段
    assert recorded == [e["exp_id"] for e in evs if e["type"] == "recorded"]
    assert all(x["decision"] for x in rounds)


def test_load_trials_reads_complete_and_pruned(tmp_path):
    import optuna
    from runtime.trace import load_trials
    db = tmp_path / "optuna.db"
    st_ = optuna.create_study(study_name="t_e01", storage=f"sqlite:///{db}", direction="maximize")
    st_.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=3)
    t = st_.ask()
    st_.tell(t, state=optuna.trial.TrialState.PRUNED)
    out = load_trials(str(db), ["t_e01", "t_missing"])
    assert set(out) == {"t_e01"}
    assert [r["state"] for r in out["t_e01"]] == ["COMPLETE"] * 3 + ["PRUNED"]
    assert out["t_e01"][0]["n"] == 0 and "x" in out["t_e01"][0]["params"]


def test_orchestrator_logs_locked_metric(toy_cfg):
    o = Orchestrator(toy_spec(toy_cfg, autonomy="L0"), toy_cfg, PolicyMockLLM(), None, "sl")
    o.run(max_steps=2)
    assert o.events.query("spec_locked")[0]["payload"]["metric"]["primary"] == "auc"
