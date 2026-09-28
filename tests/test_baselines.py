"""第 8 项测试：B0/B1 使用与 agent 相同的数据与协议；holdout 只经 FinalGate 评一次；不绕过隔离。"""
import json
from pathlib import Path

import numpy as np
import pytest

from baselines import b0_default_lgbm, b1_automl
from evaluation.final_gate import FinalGate, FinalGateAlreadyRun


def test_b0_and_b1_run_and_use_holdout_once(toy_cfg):
    r0 = b0_default_lgbm.run("toy", toy_cfg)
    assert r0["oot_dev_auc"] > 0.6 and r0["holdout_auc"] > 0.6 and r0["n_features"] == 4
    r1 = b1_automl.run("toy", toy_cfg, time_budget=5)
    assert r1["oot_dev_auc"] > 0.6 and r1["holdout_auc"] > 0.6 and r1["best_estimator"]
    for n in ("b0_toy", "b1_toy"):                                        # 预测已保存，可与 agent 做配对 bootstrap
        assert np.load(Path(toy_cfg["paths"]["artifacts_root"]) / "experiments" / n / "oot_dev_pred.npy").shape == (1500,)


def test_baseline_holdout_evaluated_only_once(toy_cfg):
    b0_default_lgbm.run("toy", toy_cfg)
    fg = FinalGate("toy", toy_cfg, "b0_toy")
    first = fg.run_predictor("b0_toy", lambda h: h["x1"].values, 0.7)     # 幂等：同名实验返回缓存，不重新读
    assert first["exp_id"] == "b0_toy"
    with pytest.raises(FinalGateAlreadyRun):
        fg.run_predictor("another_model", lambda h: h["x1"].values, 0.7)  # 换一个模型再评一次 → 拒绝


def test_baselines_never_touch_holdout_files_directly():
    for f in (Path(__file__).resolve().parents[1] / "baselines").glob("*.py"):
        src = f.read_text()
        assert "holdout.parquet" not in src and "artifacts/holdout" not in src, f
        assert "issue_holdout_token" not in src, f


def test_baselines_use_same_wide_table_as_agent(toy_cfg):
    from baselines.common import load_wide
    from data.access import DataAccess
    from modeling.prep import feature_cols
    Xtr, ytr, Xva, yva, Xoo, yoo, _ = load_wide("toy", toy_cfg)
    assert list(Xtr.columns) == feature_cols(DataAccess("toy", toy_cfg["paths"]["artifacts_root"]).load("train"))
    assert len(Xoo) == 1500 and len(Xva) == 1500
