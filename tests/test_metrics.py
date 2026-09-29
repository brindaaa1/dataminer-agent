import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from agent.schemas import TaskSpec
from data.knowhow import load_config
from evaluation.metrics import METRICS, all_metrics


def test_registry_values():
    r = np.random.default_rng(0)
    y = r.integers(0, 2, 2000)
    p = y + r.normal(size=2000)
    m = all_metrics(y, p)
    assert set(m) == {"auc", "pr_auc", "ks"}
    assert abs(m["auc"] - roc_auc_score(y, p)) < 1e-9
    assert abs(m["pr_auc"] - average_precision_score(y, p)) < 1e-9
    assert 0 < m["ks"] < 1
    assert METRICS["auc"].gap_max == 0.03


def test_taskspec_reads_metric_from_config():
    cfg = load_config()
    assert TaskSpec.from_knowhow("hotel_bookings", cfg).metric.guards == {"pr_auc": 0.02}


def test_bootstrap_with_ks_matches_direct():
    from evaluation.compare import paired_bootstrap
    r = np.random.default_rng(1)
    y = r.integers(0, 2, 1500)
    a, b = y + r.normal(size=1500), r.normal(size=1500)
    out = paired_bootstrap(y, a, b, 0.05, 100, 0, metric_fn=METRICS["ks"].fn)
    assert abs(out["delta"] - (METRICS["ks"].fn(y, a) - METRICS["ks"].fn(y, b))) < 1e-9


def test_overfit_gap_uses_primary_metric():
    from evaluation.guardrails import diagnose
    cfg = load_config()
    cfg["metric"] = {"primary": "ks", "guards": {}}
    m = {"valid_auc": .80, "oot_dev_auc": .79, "valid_ks": .50, "oot_dev_ks": .40}
    assert "OVERFIT_GAP" in diagnose(m, cfg, n_features=5)["codes"]          # ks 差 0.10 > 0.05
    cfg["metric"]["primary"] = "auc"
    assert "OVERFIT_GAP" not in diagnose(m, cfg, n_features=5)["codes"]      # auc 差 0.01


def test_ks_fast_on_mid_size_samples():
    import time
    r = np.random.default_rng(2)
    y = r.integers(0, 2, 8000)
    p = y + r.normal(size=8000)
    t0 = time.time()
    for _ in range(50):
        METRICS["ks"].fn(y, p)
    assert time.time() - t0 < 0.5                     # 精确 p 值路径约 40ms/次，只要统计量应在毫秒级


def test_lesson_and_topk_use_primary_metric(tmp_path):
    from memory.schema import Cost, ExperimentRecord
    from memory.store import Store
    s = Store(str(tmp_path / "s.db"))
    mk = lambda e, auc, ks: ExperimentRecord(exp_id=e, task_id="t", parent_exp_id=None, action_type="TUNE", hypothesis="",
                                             diff={}, config={}, config_hash=e, fidelity="full", metrics={"oot_dev_auc": auc, "oot_dev_ks": ks},
                                             diagnosis_codes=[], cost=Cost(), verdict="ACCEPT")
    s.add(mk("a", .80, .30)); s.add(mk("b", .70, .40))
    assert [r.exp_id for r in s.top_k("t", 2, "ks")] == ["b", "a"]


def test_native_early_stopping_metric_follows_primary():
    import pandas as pd
    from modeling.zoo import ZOO
    cfg = load_config()
    cfg["inner_loop"]["n_threads"] = 2
    cfg["metric"] = {"primary": "pr_auc", "guards": {}}
    r = np.random.default_rng(3)
    X = pd.DataFrame({"x": r.normal(size=600)})
    y = pd.Series((X.x + r.normal(size=600) > 0).astype(int))
    lg = ZOO["lgbm"]({"learning_rate": .1, "num_leaves": 8, "min_child_samples": 20, "feature_fraction": 1.0,
                      "bagging_fraction": 1.0, "lambda_l2": 1.0}, cfg).fit(X, y, X, y, 5)
    assert lg.model.params["metric"] == "average_precision"
    cb = ZOO["catboost"]({"depth": 3, "learning_rate": .1, "l2_leaf_reg": 3.0, "iterations": 20}, cfg).fit(X, y, X, y, 5)
    assert cb.model.get_params()["eval_metric"] == "PRAUC"
