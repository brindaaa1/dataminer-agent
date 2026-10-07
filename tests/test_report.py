"""第 7 项测试：Mermaid 实验树、摘要数字校验、模型卡内容（数据由代码填充）。"""
import re

from agent.llm import PolicyMockLLM, ScriptedLLM
from agent.orchestrator import Orchestrator
from conftest import make_toy, toy_spec
from evaluation.final_gate import FinalGate
from memory.schema import Cost, ExperimentRecord
from report.model_card import build_report, mermaid_tree, safe_narrative


def rec(i, parent, verdict, auc, inv=False, codes=()):
    return ExperimentRecord(exp_id=i, task_id="t", parent_exp_id=parent, hypothesis="h", action_type="TUNE", diff={}, config={}, config_hash="x",
                            fidelity="full", metrics={"oot_dev_auc": auc}, diagnosis_codes=list(codes), cost=Cost(), verdict=verdict, invalidated=inv)


def test_mermaid_tree_structure_and_delta():
    t = mermaid_tree([rec("a-1", None, "ACCEPT", 0.70), rec("b.2", "a-1", "REJECT", 0.69, codes=["PLATEAU"]), rec("c3", "a-1", "ACCEPT", 0.71, inv=True)], "auc")
    assert t.startswith("flowchart TD") and "ROOT --> n_a_1" in t and "n_a_1 --> n_b_2" in t   # 非法字符被规整
    assert "(-0.0100)" in t and "(+0.0100)" in t and "PLATEAU" in t
    assert ":::accept_inv" in t                                                                # 失效节点单独样式
    assert re.search(r"\[\"[^\"]*\"\]:::reject", t)


def test_narrative_guard_rejects_invented_numbers():
    facts = {"n_experiments": 7, "oot_dev_auc": 0.6735, "best_exp": "t1_e03"}
    assert safe_narrative("共 7 个实验，OOT-dev AUC 为 0.6735。", facts)
    assert safe_narrative("AUC 约 0.67。", facts)                       # 合理的四舍五入
    assert safe_narrative("AUC 为 0.7123。", facts) is None             # 编造的数字
    assert safe_narrative("提升了 15%。", facts) is None
    assert safe_narrative("实验 t1_e03 表现最好。", facts)                # 标识符里的数字不算


def test_model_card_content(tmp_path):
    cfg = make_toy(tmp_path)
    o = Orchestrator(toy_spec(cfg), cfg, PolicyMockLLM(), FinalGate("toy", cfg, "t1"), "t1")
    o.run()
    card = open(f"{cfg['paths']['reports_root']}/model_card_t1.md").read()
    for sec in ("## 执行摘要", "## 1. 任务规格", "## 2. 数据切分", "## 3. 最终模型", "## 4. 各段指标", "## 5. 分月 AUC", "## 6. 特征清单", "## 7. 实验树", "## 8. 风险提示"):
        assert sec in card
    assert "```mermaid" in card and "holdout" in card and o.p["best_exp_id"] in card
    assert "未生成摘要" not in card                                      # mock 摘要通过了数字校验
    assert "正类比例" in card and "坏样本" not in card                     # 不只聚焦风控：用通用说法
    best = o._result(o.p["best_exp_id"])
    assert f"{best['metrics']['oot_dev_auc']:.4f}" in card              # 指标来自代码，不是 LLM
    assert (tmp_path / "rep" / "experiment_tree_t1.mmd").exists()


def test_bad_narrative_is_dropped(tmp_path):
    cfg = make_toy(tmp_path)
    o = Orchestrator(toy_spec(cfg), cfg, PolicyMockLLM(), FinalGate("toy", cfg, "t1"), "t1")
    o.run()
    liar = ScriptedLLM(["最终 holdout AUC 高达 0.9999，非常出色。"])
    rep = build_report(cfg, o.spec, o.store, o.p, o.p["final"], 3, liar)
    assert rep["narrative"] is None
    assert "0.9999" not in open(rep["model_card"]).read()


def test_model_card_survives_explain_failure(tmp_path, monkeypatch):
    """最终检验已出结果后，特征贡献算不出来只影响模型卡那一节，不能让整次运行失败、丢掉成绩。"""
    import report.model_card as mc

    def boom(*a, **k):
        raise AttributeError("'XSpec' object has no attribute '_transform'")

    cfg = make_toy(tmp_path)
    monkeypatch.setattr(mc, "explain", boom)
    o = Orchestrator(toy_spec(cfg), cfg, PolicyMockLLM(), FinalGate("toy", cfg, "t1"), "t1")
    assert o.run().value == "DONE"
    card = open(f"{cfg['paths']['reports_root']}/model_card_t1.md").read()
    assert "## 6. 特征清单" in card and "特征贡献计算失败" in card and "## 8. 风险提示" in card


def test_overfit_to_oot_dev_is_relative_to_reference_drift(tmp_path):
    """固定的 0.02 在 hotel 上每次都报（默认参照模型自己 OOT-dev → holdout 就掉 0.043）。最终检验时同时量出参照模型的下滑，
    多掉超过 δ 才算疑似过拟合；agent 运行中仍看不到 holdout。"""
    cfg = make_toy(tmp_path)
    cfg["evaluator"]["overfit_gap"] = {"mode": "relative", "tolerance": 0.05}
    o = Orchestrator(toy_spec(cfg), cfg, PolicyMockLLM(), FinalGate("toy", cfg, "t1"), "t1")
    o.run()
    f = o.p["final"]
    assert f["reference_drop"] is not None
    assert f["overfit_to_oot_dev"] == (f["drop"] - f["reference_drop"] > cfg["final_gate"]["delta"])
    rep = build_report(cfg, o.spec, o.store, o.p, {**f, "overfit_to_oot_dev": True, "drop": f["reference_drop"] + 0.05}, 3, PolicyMockLLM())
    assert "比默认参照模型多掉 0.0500" in open(rep["model_card"]).read()


def test_model_card_lists_tied_race_models(tmp_path):
    cfg = make_toy(tmp_path)
    o = Orchestrator(toy_spec(cfg), cfg, PolicyMockLLM(), FinalGate("toy", cfg, "t1"), "t1")
    o.run()
    p = {**o.p, "race_tie": {"recommended": "lgbm", "tied": ["lgbm", "catboost"], "best": "catboost", "mde": 0.0099}}
    card = open(build_report(cfg, o.spec, o.store, p, o.p["final"], 3, PolicyMockLLM())["model_card"]).read()
    assert "赛跑打平" in card and "lgbm、catboost" in card and "0.0099" in card


def test_final_gate_and_model_card_have_report_metrics(tmp_path):
    """补充参考指标在 holdout 上算一次，写进模型卡；不参与选模型。"""
    cfg = make_toy(tmp_path)
    o = Orchestrator(toy_spec(cfg), cfg, PolicyMockLLM(), FinalGate("toy", cfg, "t1"), "t1")
    o.run()
    f = o.p["final"]
    assert {"holdout_brier", "holdout_lift_top10", "holdout_recall_top10"} <= set(f) and f["holdout_lift_top10"] > 1
    card = open(f"{cfg['paths']['reports_root']}/model_card_t1.md").read()
    assert "头部 10%" in card and "Brier" in card


def test_setup_section_explains_params_for_reproduction():
    from report.model_card import setup_section
    from types import SimpleNamespace
    cfg = {"fidelity": {"full": {"early_stopping_rounds": 50}}, "inner_loop": {"n_rounds_max": 500}, "split": {"valid_frac": 0.2, "seed": 42}}
    spec = SimpleNamespace(oot_windows={"oot_dev": ["2020-01", "2020-03"], "holdout": ["2020-04", "2020-06"]})
    best = {"model": "lgbm", "n_trials": 20, "n_rows": 1000, "best_iter": 87, "n_features": 2,
            "best_params": {"learning_rate": 0.0312345, "num_leaves": 31},
            "config": {"model": "lgbm", "features": ["a", "b_lag1"], "seed": 3}}
    s = "\n".join(setup_section(cfg, spec, best, "auc"))
    assert s.startswith("## 9. 模型设定（复现用）")
    for t in ["随机种子 3", "20 组参数", "valid 段 AUC 最高", "train 段 1,000 行", "随机抽出的 20%（种子 42）", "时间外验证 2020-01 ~ 2020-03", "连续 50 轮", "实际用了 87 棵树",
              "| learning_rate | 0.0312345 | 学习率", "| num_leaves | 31 | ", "`a`、`b_lag1`"]:
        assert t in s, t
