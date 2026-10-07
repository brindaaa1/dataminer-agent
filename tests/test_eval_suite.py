"""评测：用量、花费、过程指标、汇总表；runner 冒烟；Langfuse 写入。"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DS3 = json.loads((ROOT / "tests" / "fixtures" / "hotel_deepseek3.json").read_text())


def ev(i, t, p, x=None):
    return {"id": i, "ts": float(i), "type": t, "exp_id": x, "payload": p}


def test_usage_totals_sum_breakdown_and_fall_back_for_old_events():
    from eval.suite.metrics import usage_totals
    es = [ev(1, "llm_call", {"tokens": 150, "usage": {"input_cache_hit": 80, "input_cache_miss": 20, "output": 50, "reasoning": 30}}),
          ev(2, "llm_call", {"tokens": 40}),                                    # 旧事件：只有总数
          ev(3, "llm_call", {"tokens": 0, "error": "boom"})]
    assert usage_totals(es) == {"input_cache_hit": 80, "input_cache_miss": 60, "output": 50, "reasoning": 30, "calls": 3}


def test_cost_usd():
    from eval.suite.metrics import cost_usd
    u = {"input_cache_hit": 1_000_000, "input_cache_miss": 1_000_000, "output": 1_000_000, "reasoning": 0, "calls": 1}
    assert cost_usd(u, {"input_cache_miss": 0.14, "input_cache_hit": 0.0028, "output": 0.28}) == pytest.approx(0.4228)
    assert cost_usd(u, None) is None


def test_process_stats_on_recorded_run():
    from eval.suite.metrics import process_stats
    s = process_stats(DS3["events"])
    assert s["n_rounds"] == 5 and s["stop_reason"].startswith("LLM_STOP") and s["final_exp"] == "hotel_deepseek3_e04"
    assert s["holdout"] == pytest.approx(0.7305, abs=1e-4) and s["overfit"] and not s["no_final_model"]


def test_run_without_final_model_counts_as_loss():
    from eval.suite.metrics import process_stats, summary_md
    es = [ev(1, "state_transition", {"from": "INTAKE", "to": "PROFILE"}),
          ev(2, "final_gate_skipped", {"why": "没有通过 evaluator 的全保真度实验"})]
    s = process_stats(es)
    assert s["no_final_model"] and s["holdout"] is None
    row = {"dataset": "hotel_bookings", "seed": 0, "status": "ok", "error": None, "final_model": None,
           "usage": {"input_cache_hit": 0, "input_cache_miss": 10, "output": 5, "reasoning": 0, "calls": 1}, "cost_usd": 0.0, **s}
    md = summary_md("v0", {"max_rounds": 5}, [row], {"hotel_bookings": {"b0": {"holdout_auc": 0.78, "oot_dev_auc": 0.83, "wall_sec": 1.0},
                                                                         "b1": {"holdout_auc": 0.77, "oot_dev_auc": 0.82, "wall_sec": 30.0}}})
    assert "无最终模型" in md and "0/1" in md                                # 赢 B0 的种子数：0/1


def test_summary_md_delta_and_wins():
    from eval.suite.metrics import summary_md
    base = {"status": "ok", "error": None, "final_model": "lgbm", "n_rounds": 5, "n_rejected": 0, "n_llm_errors": 0,
            "stop_reason": "MAX_ROUNDS", "final_exp": "x", "oot_dev": 0.83, "overfit": False, "no_final_model": False,
            "wall_sec": 30.0, "usage": {"input_cache_hit": 0, "input_cache_miss": 1000, "output": 500, "reasoning": 200, "calls": 6},
            "cost_usd": 0.001}
    rows = [{**base, "dataset": "d", "seed": 0, "holdout": 0.80}, {**base, "dataset": "d", "seed": 1, "holdout": 0.76}]
    md = summary_md("v0", {"max_rounds": 5}, rows, {"d": {"b0": {"holdout_auc": 0.78, "oot_dev_auc": 0.8, "wall_sec": 1.0},
                                                         "b1": {"holdout_auc": 0.79, "oot_dev_auc": 0.8, "wall_sec": 30.0}}})
    assert "+0.0200" in md and "-0.0200" in md and "1/2" in md and "0.0020" in md      # 两行 Δ、胜率、美元合计


def test_mock_run_on_hotel_sample_produces_row(eval_env):
    from agent.llm import PolicyMockLLM
    row = eval_env.run_agent("hotel_sample", "t", 0, "mock", llm=PolicyMockLLM())
    assert row["status"] == "ok" and row["dataset"] == "hotel_sample" and row["seed"] == 0
    assert row["n_rounds"] == 1 and row["usage"]["calls"] >= 2 and row["cost_usd"] is None    # mock 没有价格


def test_failed_run_is_recorded_and_suite_continues(eval_env):
    class Down:
        def complete(self, *a, **k):
            raise ConnectionError("network down")
    row = eval_env.run_agent("hotel_sample", "t", 1, "mock", llm=Down())
    assert row["status"] == "failed" and "ConnectionError" in row["error"]


def test_rerun_same_label_starts_fresh(eval_env):
    """每个种子一个产物目录：重跑某个种子只清它自己的，不动其他种子。"""
    d0, d2 = eval_env.art_dir("t", "hotel_sample", 0), eval_env.art_dir("t", "hotel_sample", 2)
    for d in (d0, d2):
        d.mkdir(parents=True)
        (d / "stale.txt").write_text("旧结果")
    eval_env.prepare_run("t", "hotel_sample", 2)
    assert not (d2 / "stale.txt").exists() and (d0 / "stale.txt").exists()


def test_merge_rows_replaces_only_rerun_seeds():
    """v2 评测：补跑一个种子时，不能把同一数据集其他种子的结果丢掉。"""
    from eval.suite.runner import merge_rows
    old = [{"dataset": "a", "seed": s, "v": "old"} for s in (0, 1, 2)] + [{"dataset": "b", "seed": 0, "v": "old"}]
    new = [{"dataset": "a", "seed": 2, "v": "new"}]
    got = {(r["dataset"], r["seed"]): r["v"] for r in merge_rows(old, new)}
    assert got == {("a", 0): "old", ("a", 1): "old", ("a", 2): "new", ("b", 0): "old"}


def test_baselines_are_cached_with_a_fixed_budget(eval_env, monkeypatch):
    """B1 与 agent 代码无关，却每个版本都重跑（hotel 一次 7 分钟以上），预算还随 agent 用时变，"比 B1"跟着抖。
    改为固定预算、按数据集 + 切分种子 + 预算缓存，换版本直接复用。"""
    calls = []
    monkeypatch.setattr(eval_env.b0_default_lgbm, "run", lambda ds, cfg, name=None: calls.append("b0") or {"holdout_auc": 0.8})
    monkeypatch.setattr(eval_env.b1_automl, "run", lambda ds, cfg, budget, name=None: calls.append(budget) or {"holdout_auc": 0.81})
    a = eval_env.run_baselines("hotel_sample", "v1")
    b = eval_env.run_baselines("hotel_sample", "v2")
    assert calls == ["b0", eval_env.B1_BUDGET_SEC] and a == b == {"b0": {"holdout_auc": 0.8}, "b1": {"holdout_auc": 0.81}}
    monkeypatch.setattr(eval_env, "B1_BUDGET_SEC", 5.0)                     # 预算变了：缓存不复用
    eval_env.run_baselines("hotel_sample", "v3")
    assert calls[-1] == 5.0


class FakeLF:
    def __init__(self):
        self.calls = []
        from types import SimpleNamespace
        self.api = SimpleNamespace(dataset_run_items=SimpleNamespace(create=lambda **kw: self.calls.append(("run_item", kw))))

    def create_dataset(self, **kw):
        self.calls.append(("dataset", kw))

    def create_dataset_item(self, **kw):
        self.calls.append(("item", kw))

    def create_score(self, **kw):
        self.calls.append(("score", kw))

    def flush(self):
        self.calls.append(("flush", {}))


def test_langfuse_report_links_traces_and_scores():
    from eval.suite.langfuse_report import report
    lf = FakeLF()
    row = {"dataset": "hotel_sample", "seed": 0, "status": "ok", "trace_id": "tr1", "holdout": 0.80, "cost_usd": 0.001,
           "usage": {"output": 500}}
    report("v0", [row, {**row, "seed": 1, "trace_id": None}], {"hotel_sample": {"b0": {"holdout_auc": 0.78}}}, client=lf)
    kinds = [k for k, _ in lf.calls]
    assert kinds.count("dataset") == 1 and kinds.count("item") == 2 and kinds.count("run_item") == 1   # 没有 trace 的不挂
    item = next(kw for k, kw in lf.calls if k == "item")
    assert item["id"] == "hotel_sample-s0" and item["dataset_name"] == "dataminer-eval/hotel_sample"
    run = next(kw for k, kw in lf.calls if k == "run_item")
    assert run == {"run_name": "v0", "dataset_item_id": "hotel_sample-s0", "trace_id": "tr1"}
    scores = {kw["name"]: kw["value"] for k, kw in lf.calls if k == "score"}
    assert scores["delta_vs_b0"] == pytest.approx(0.02) and scores["output_tokens"] == 500 and kinds[-1] == "flush"


def test_langfuse_report_unreachable_only_warns(capsys):
    """v8：结果都写完了，最后写 Langfuse 时连不上，整次评测以退出码 1 结束。只告警。"""
    from eval.suite.langfuse_report import report

    class Down:
        def create_dataset(self, **kw):
            raise ConnectionError("Connection refused")
    report("v0", [{"dataset": "d", "seed": 0}], {}, client=Down())
    assert "Langfuse" in capsys.readouterr().out


def test_langfuse_report_without_client_is_noop(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    from eval.suite.langfuse_report import report
    report("v0", [], {})


def test_baselines_build_their_own_splits(eval_env, monkeypatch):
    """v2 评测 lending_club：基线目录里没有切分（切分是 agent 运行时在各种子目录里生成的），B0 读不到 train.parquet 崩溃。"""
    monkeypatch.setattr(eval_env.b1_automl, "run", lambda ds, cfg, budget, name=None: {"holdout_auc": 0.7})
    b = eval_env.run_baselines("hotel_sample", "t")
    assert 0.5 < b["b0"]["holdout_auc"] < 1


def test_decide_failed_reason_reaches_the_row(eval_env, monkeypatch):
    """决策失败事件把原因放在 errors 列表里，评测行要带上它，不能只写"运行结束于 FAILED"。"""
    from agent.llm import ScriptedLLM
    junk = lambda u: "garbage" if '"mode": "DECIDE"' in u else {"directions": []}
    orig = eval_env.build_cfg
    monkeypatch.setattr(eval_env, "build_cfg", lambda *a, **k: (c := orig(*a, **k), c["race"].update(rule_first_promotion=False))[0])   # 还没有最终模型时决策失败
    row = eval_env.run_agent("hotel_sample", "t", 3, "mock", llm=ScriptedLLM(junk))
    assert row["status"] == "failed" and "不是合法的 Decision JSON" in row["error"]


def test_run_pauses_on_question_and_resumes_with_answer(eval_env, monkeypatch):
    """评测里 agent 要问人：不判失败，停在 awaiting_human 并带上问题和推荐答案；带着回答续跑，从检查点接着走。"""
    from agent.llm import PolicyMockLLM, ScriptedLLM
    from agent.schemas import TaskSpec
    orig = TaskSpec.from_knowhow
    monkeypatch.setattr(eval_env.TaskSpec, "from_knowhow", lambda *a, **k: orig(*a, **k).model_copy(update={"observation_time_col": None}))
    mock = PolicyMockLLM()
    prop = {"proposals": {"observation_time_col": {"question": "预测时点？", "recommended": "到店日期前的预订时刻",
                                                   "reason": "r", "value": {"col": "lead_time"}}}}
    llm = ScriptedLLM(lambda u: prop if '"mode": "CLARIFY"' in u else mock.complete("", u)[0])
    row = eval_env.run_agent("hotel_sample", "t", 0, "mock", llm=llm)
    assert row["status"] == "awaiting_human" and row["questions"][0]["recommended"] == "到店日期前的预订时刻"
    row = eval_env.run_agent("hotel_sample", "t", 0, "mock", llm=llm, answers={"observation_time_col": "确认"})
    assert row["status"] == "ok" and row["n_rounds"] == 1


def test_wall_time_sums_state_durations_excluding_human_wait():
    """v6 评测 home_credit：续跑后耗时只算续跑那一段（几秒），B1 的时间预算跟着只给了 30 秒，对 B1 不公平。
    用各状态耗时之和：跨续跑累加，不含等人回复的时间。"""
    from eval.suite.metrics import process_stats
    events = [ev(1, "state_transition", {"from": "INTAKE", "to": "CLARIFY", "sec": 2.0}),
              ev(2, "state_transition", {"from": "CLARIFY", "to": "AWAIT_HUMAN", "sec": 0.5}),
              {**ev(3, "state_transition", {"from": "PROFILE", "to": "PLAN", "sec": 60.0}), "ts": 5000.0},   # 人隔了一个多小时才回
              {**ev(4, "state_transition", {"from": "PLAN", "to": "DONE", "sec": 30.0}), "ts": 5030.0}]
    assert process_stats(events)["wall_sec"] == 92.5


def test_row_carries_task_id_and_process_metrics(eval_env):
    from agent.llm import PolicyMockLLM
    row = eval_env.run_agent("hotel_sample", "t", 4, "mock", llm=PolicyMockLLM())
    assert row["task_id"] == "t_hotel_sample_s4"
    assert set(row["process"]) >= {"wasted_rounds", "rejected_decisions", "upgrade_hit_rate", "fe_funnel", "cost"}
    assert row["process"]["rounds"]["max"] == 1                      # eval_env 把轮数上限设成 1


def test_drive_auto_answers_until_done_and_gives_up_after_limit(eval_env, monkeypatch):
    """情景用例里模拟人工回答：停下就自动答；推荐答案缺失时"确认"不被采纳，答 3 次后停下，不死循环（Review Focus 1）。"""
    from agent.llm import PolicyMockLLM, ScriptedLLM
    from agent.schemas import TaskSpec
    orig = TaskSpec.from_knowhow
    monkeypatch.setattr(eval_env.TaskSpec, "from_knowhow", lambda *a, **k: orig(*a, **k).model_copy(update={"observation_time_col": None}))
    mock = PolicyMockLLM()
    empty = {"proposals": {}}                                         # 起草一直失败 → recommended 为空
    llm = ScriptedLLM(lambda u: empty if '"mode": "CLARIFY"' in u else mock.complete("", u)[0])
    cfg = eval_env.build_cfg("hotel_sample", "t", 5)
    from agent.orchestrator import Orchestrator
    from evaluation.final_gate import FinalGate
    o = Orchestrator(TaskSpec.from_knowhow("hotel_bookings", cfg, autonomy="L0"), cfg, llm, FinalGate("hotel_bookings", cfg, "x5"), "x5")
    assert eval_env.drive(o, auto_answers={"observation_time_col": "确认"}).value == "AWAIT_HUMAN"
    assert len(o.events.query("USER_INSTRUCTION")) == 3


def test_tiers_define_fast_and_full():
    from eval.suite.tiers import load_tier
    f, a = load_tier("fast"), load_tier("full")
    assert f["datasets"] == ["hotel_bookings", "lending_club"] and f["seeds"] == [0, 1]
    assert a["datasets"] == ["hotel_bookings", "lending_club", "home_credit"] and a["seeds"] == [0, 1, 2]
    assert {s["name"] for s in f["scenarios"]} == {"leak", "missing_spec", "small_oot", "no_signal"}


def test_version_json_and_latest_same_tier(tmp_path):
    import json
    from eval.suite.version import build_version, latest_version
    row = {"dataset": "d", "seed": 0, "status": "ok", "holdout": 0.7, "task_id": "t", "process": {}, "wall_sec": 10, "cost_usd": 0.01,
           "usage": {"calls": 3}, "questions": None}
    sc = [{"scenario": "leak", "base": "d", "passed": True, "wall_sec": 5, "cost_usd": 0.002}]
    v = build_version("v7", "fast", {"max_rounds": 8}, [row], {"d": {"b1": {"holdout_auc": 0.69}}}, sc, started=1.0)
    assert v["datasets"]["d"]["runs"][0]["task_id"] == "t" and "usage" not in v["datasets"]["d"]["runs"][0]
    assert v["wall_sec"] == 15 and v["cost_usd"] == pytest.approx(0.012) and set(v["git"]) == {"sha", "dirty"}
    for lab, tier, fin in (("v5", "fast", 1), ("v6", "fast", 2), ("v6b", "full", 3), ("v7", "fast", 4)):
        (tmp_path / lab).mkdir()
        (tmp_path / lab / "version.json").write_text(json.dumps({"label": lab, "tier": tier, "finished": fin, "status": "done"}))
    assert latest_version(tmp_path, "fast", before="v7")["label"] == "v6"
    assert latest_version(tmp_path, "nope", before="v7") is None
    assert v["status"] == "done" and build_version("v7", "fast", {}, [], {}, [], 1.0, status="running")["status"] == "running"


def test_summary_has_process_and_scenario_sections():
    from eval.suite.metrics import process_md, scenarios_md
    p = {"wasted_rounds": {"value": 0.25}, "rejected_decisions": {"value": 2}, "decide_failed": {"value": False},
         "upgrade_hit_rate": {"value": 1.0}, "fe_funnel": {"value": True}, "final_lineage": {"value": "赛跑 lgbm → 升全量"}}
    md = process_md([{"dataset": "d", "seed": 0, "status": "ok", "process": p}])
    assert "过程指标" in md and "25%" in md and "赛跑 lgbm → 升全量" in md
    s = scenarios_md([{"scenario": "leak", "base": "d", "passed": False, "reason": "泄漏列进了模型", "task_id": "t"}])
    assert "情景用例" in s and "不通过" in s and "泄漏列进了模型" in s


def test_version_json_has_per_dataset_summary():
    """版本层面（spec §4）：每个数据集对各种子取平均、计比例，并点出最差的那次运行，作为排查入口。"""
    from eval.suite.version import build_version
    r = lambda seed, h, fe, w, c: {"dataset": "d", "seed": seed, "status": "ok", "holdout": h, "task_id": f"t{seed}", "wall_sec": w,
                                   "cost_usd": c, "process": {"rounds": {"value": 4}, "fe_funnel": {"value": fe}}}
    rows = [r(0, 0.70, True, 100, 0.01), r(1, 0.66, False, 50, 0.02), {**r(2, None, False, 10, 0.0), "status": "failed"}]
    s = build_version("v", "fast", {}, rows, {"d": {"b1": {"holdout_auc": 0.67}}}, [], 1.0)["datasets"]["d"]["summary"]
    assert s["n"] == 3 and s["n_ok"] == 2 and s["holdout_mean"] == pytest.approx(0.68) and s["delta_b1_mean"] == pytest.approx(0.01)
    assert s["fe_in_final"] == "1/3" and s["worst_task"] == "t2" and s["wall_sec"] == 160 and s["cost_usd"] == pytest.approx(0.03)


def test_parallel_jobs_split_the_threads(eval_env, monkeypatch):
    """v11 第一次并行：4 个进程各开 8 线程抢 8 个核，负载 24，赛跑慢到把 CPU 预算耗光，hotel s0 因超预算失败。
    并行时每个进程的训练线程数 = 原线程数 // 并行数。"""
    base = eval_env.build_cfg("hotel_sample", "t", 0)["inner_loop"]["n_threads"]
    monkeypatch.setenv("DATAMINER_EVAL_JOBS", "4")
    assert eval_env.build_cfg("hotel_sample", "t", 0)["inner_loop"]["n_threads"] == max(1, base // 4)


def test_memory_from_another_seed_of_a_previous_version(eval_env, monkeypatch):
    """memory 评测：每个种子开始前，拷一份指定版本里另一个种子跑完后的 memory 库（热启动 + 经验），再和那个版本的冷启动配对比。"""
    for s in (0, 1):
        d = eval_env.RUNS / "v13" / "hotel_sample" / f"s{s}"
        d.mkdir(parents=True)
        (d / f"memory_s{s}.db").write_text(f"from s{s}")
    monkeypatch.setenv("DATAMINER_EVAL_MEMORY_FROM", "v13")
    cfg = eval_env.build_cfg("hotel_sample", "m1", 1)
    from pathlib import Path
    assert Path(cfg["memory"]["db"]).read_text() == "from s0"                   # 种子 1 用种子 0 的经验
    assert Path(eval_env.build_cfg("hotel_sample", "m1", 0)["memory"]["db"]).read_text() == "from s1"
    base = eval_env.build_cfg("hotel_sample", "m1", 0, "baselines")                # 基线不拷
    assert not Path(base["memory"]["db"]).exists()
