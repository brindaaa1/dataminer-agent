"""第 11 项测试：约束向量（先验、别名、数据一致性守卫）、lgbm 确实单调、动作校验、evaluator 的非劣效规则。"""
import numpy as np
import pandas as pd
import pytest

from agent.actions import ActionError, apply_action
from agent.schemas import Budget, TaskSpec
from data.knowhow import load_config
from evaluation.evaluator import Evaluator, Verdict
from modeling.monotone import constraints_for, priors_for
from modeling.zoo import ZOO

CFG = load_config()


def test_priors_alias_and_constraints_vector():
    cfg = {**CFG, "task_specs": {**CFG["task_specs"], "lending_club": {**CFG["task_specs"]["lending_club"]}}}
    pri = priors_for("lending_club", cfg, {})
    assert pri["dti"] == 1 and pri["fico_range_low"] == -1
    assert pri["earliest_cr_line"] == -1                                # 别名：预处理后即"信用历史月数"
    assert pri["mths_since_last_delinq"] == 0                            # 缺失有含义 → 不约束


def test_constraints_skip_no_prior_zero_prior_categorical_and_data_disagreement():
    r = np.random.default_rng(0)
    n = 5000
    y = pd.Series((r.random(n) < 0.3).astype(int))
    X = pd.DataFrame({
        "dti": y * 1.0 + r.normal(size=n),                               # 先验 +1，数据也是 +：约束
        "pub_rec": -y * 1.0 + r.normal(size=n),                          # 先验 +1，数据是 −：与先验相反 → 跳过
        "mths_since_last_delinq": r.normal(size=n),                      # 先验 0
        "unknown_feat": r.normal(size=n),                                # 没有先验
        "purpose": pd.Categorical(r.choice(["a", "b"], n))})            # 类别
    vec, rep = constraints_for("lending_club", CFG, {}, X, y)
    assert dict(zip(X.columns, vec)) == {"dti": 1, "pub_rec": 0, "mths_since_last_delinq": 0, "unknown_feat": 0, "purpose": 0}
    assert rep["constrained"] == ["dti"] and "pub_rec" in rep["disagree"] and rep["zero_prior"] == ["mths_since_last_delinq"]
    assert rep["no_prior"] == ["unknown_feat"] and rep["categorical"] == ["purpose"]


def test_lgbm_respects_constraint_on_nonmonotone_truth():
    """真实关系是 U 形：不加约束会学出非单调曲线；加约束后预测对该特征严格单调。"""
    r = np.random.default_rng(0)
    n = 6000
    x = r.uniform(-3, 3, n)
    y = pd.Series((r.random(n) < 1 / (1 + np.exp(-(x ** 2 - 2)))).astype(int))
    X = pd.DataFrame({"x": x})
    cfg = {**CFG, "inner_loop": {**CFG["inner_loop"], "n_threads": 2, "n_rounds_max": 60}}
    grid = pd.DataFrame({"x": np.linspace(-3, 3, 61)})
    params = {"learning_rate": 0.1, "num_leaves": 16, "min_child_samples": 20, "feature_fraction": 1.0, "bagging_fraction": 1.0, "lambda_l2": 1.0}
    free = ZOO["lgbm"](params, cfg, 0, None).fit(X, y, X, y, 10)
    mono = ZOO["lgbm"](params, cfg, 0, [1]).fit(X, y, X, y, 10)
    d_free, d_mono = np.diff(free.predict(grid)), np.diff(mono.predict(grid))
    assert (d_free < -1e-6).any()                                        # 不约束：出现下降段（U 形）
    assert (d_mono >= -1e-9).all()                                       # 约束：严格不降


def _spec():
    return TaskSpec(dataset="lending_club", budget=Budget(tokens=1, cpu_minutes=1, wall_minutes=1))


class R:  # 最小的 ExperimentRecord 替身
    exp_id, fidelity, invalidated = "e1", "full", False


def test_tune_monotone_action_and_validation():
    node = {"exp_id": "e1", "config": {"model": "lgbm", "features": ["a"], "space": None, "fidelity": "full", "n_trials": None}}
    kind, cfg, parent = apply_action("TUNE", {"monotone": True}, node, {}, _spec())
    assert kind == "train" and cfg["monotone"] is True and parent == "e1"
    kind, cfg2, _ = apply_action("TUNE", {"monotone": False}, {**node, "config": cfg}, {}, _spec())
    assert "monotone" not in cfg2
    with pytest.raises(ActionError, match="不支持单调约束"):
        apply_action("TUNE", {"monotone": True}, {**node, "config": {**node["config"], "model": "lr_scorecard"}}, {}, _spec())
    with pytest.raises(ActionError):
        apply_action("TUNE", {}, node, {}, _spec())                     # 既没有 space 也没有 monotone


@pytest.fixture
def ev(tmp_path):
    from data.access import DataAccess  # noqa
    r = np.random.default_rng(0)
    n = 6000
    y = r.integers(0, 2, n)
    d = tmp_path / "art" / "splits" / "toy"
    d.mkdir(parents=True)
    pd.DataFrame({"_label": y}).to_parquet(d / "oot_dev.parquet")
    base = y * 1.0 + r.normal(size=n)
    preds = {"base": base, "mono_same": base + r.normal(scale=0.02, size=n), "mono_worse": y * 0.6 + r.normal(size=n)}
    for k, p in preds.items():
        (tmp_path / "art" / "experiments" / k).mkdir(parents=True)
        np.save(tmp_path / "art" / "experiments" / k / "oot_dev_pred.npy", p)
    cfg = {**CFG, "paths": {**CFG["paths"], "artifacts_root": str(tmp_path / "art")}, "evaluator": {**CFG["evaluator"], "bootstrap_n": 200}}
    M = {"train_auc": .8, "valid_auc": .7, "oot_dev_auc": .69, "gap": .01}
    mk = lambda k, psi, mono=False: {"exp_id": k, "model": "lgbm", "fidelity": "full", "n_features": 5, "metrics": {**M, "score_psi": psi},
                                     "config": {"model": "lgbm", "features": ["a"], **({"monotone": True} if mono else {})}}
    return Evaluator(cfg, "toy", "t"), mk


def test_monotone_kept_only_if_not_worse_and_psi_better(ev):
    e, mk = ev
    base = mk("base", 0.08)
    o = e.evaluate(mk("mono_same", 0.03, True), base)                    # AUC 无显著差异 + PSI 更优 → 保留
    assert o["verdict"] == Verdict.ACCEPT and "monotone_noninferior" in o
    o = e.evaluate(mk("mono_same", 0.12, True), base)                    # AUC 无显著差异但 PSI 更差 → 不保留
    assert o["verdict"] == Verdict.INCONCLUSIVE
    o = e.evaluate(mk("mono_worse", 0.01, True), base)                   # AUC 显著变差 → 即使 PSI 更优也拒绝
    assert o["verdict"] == Verdict.REJECT
    o = e.evaluate(mk("mono_same", 0.03, False), base)                   # 不是约束实验：不适用非劣效规则
    assert o["verdict"] != Verdict.ACCEPT


def test_psi_improvement_must_exceed_margin(ev):
    """回归：A6 首次运行中 PSI 0.0024 vs 0.0024（差在第 5 位小数）被当成『更优』而 ACCEPT。"""
    e, mk = ev
    base = mk("base", 0.0024)
    assert e.evaluate(mk("mono_same", 0.00235, True), base)["verdict"] == Verdict.INCONCLUSIVE   # 噪声级改善：不保留
    assert e.evaluate(mk("mono_same", 0.0005, True), base)["verdict"] == Verdict.INCONCLUSIVE    # 改善 0.0019 < margin 0.002
    assert e.evaluate(mk("mono_same", 0.0001, True), base)["verdict"] == Verdict.ACCEPT          # 改善 0.0023 ≥ margin：保留
