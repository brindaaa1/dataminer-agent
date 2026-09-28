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
    t = mermaid_tree([rec("a-1", None, "ACCEPT", 0.70), rec("b.2", "a-1", "REJECT", 0.69, codes=["PLATEAU"]), rec("c3", "a-1", "ACCEPT", 0.71, inv=True)])
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
