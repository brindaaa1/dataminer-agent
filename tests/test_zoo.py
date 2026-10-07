import numpy as np
import pandas as pd

from data.knowhow import load_config
from modeling.zoo import ZOO


def _data(n, seed):
    r = np.random.default_rng(seed)
    cat = pd.Categorical(r.choice(["a", "b", "c", None], n), categories=["a", "b", "c"])
    x = r.normal(size=n)
    x[r.random(n) < .1] = np.nan
    y = pd.Series((r.random(n) < 1 / (1 + np.exp(-(np.nan_to_num(x) + (cat == "a"))))).astype(int))
    return pd.DataFrame({"x": x, "cat": cat}), y


def test_new_models_fit_predict_with_missing_and_unseen_categories():
    cfg = load_config()
    cfg["inner_loop"]["n_threads"] = 2
    X, y = _data(800, 0)
    Xv, yv = _data(300, 1)
    Xv["cat"] = pd.Categorical([None] * 300, categories=["a", "b", "c"])     # 全部为未见/缺失类别
    params = {"catboost": {"depth": 4, "learning_rate": .1, "l2_leaf_reg": 3.0, "iterations": 50},
              "random_forest": {"max_depth": 5, "min_samples_leaf": 5, "max_features": .7, "n_estimators": 50}}
    for name, p in params.items():
        pred = ZOO[name](p, cfg).fit(X, y, Xv, yv, 10).predict(Xv)
        assert pred.shape == (300,) and np.isfinite(pred).all()
    assert all(ZOO[m].profile for m in ZOO)


def test_every_model_in_zoo_has_feature_contributions():
    """模型卡要给每个可能成为最终模型的模型算特征贡献（v0 评测：random_forest 被采纳后在生成模型卡时崩溃）。"""
    from modeling.explain import contributions
    cfg = load_config()
    cfg["inner_loop"]["n_threads"] = 2
    X, y = _data(800, 0)
    Xv, yv = _data(300, 1)
    params = {"lgbm": {"learning_rate": .1, "num_leaves": 15, "min_child_samples": 20, "feature_fraction": 1.0, "bagging_fraction": 1.0, "lambda_l2": 0.0},
              "lr_scorecard": {"C": 1.0, "n_bins": 10}, "catboost": {"depth": 4, "learning_rate": .1, "l2_leaf_reg": 3.0, "iterations": 50},
              "random_forest": {"max_depth": 5, "min_samples_leaf": 5, "max_features": .7, "n_estimators": 50}}
    assert set(params) == set(ZOO)                                       # 新加模型时这里要补参数
    for name, p in params.items():
        m = ZOO[name](p, cfg).fit(X, y, Xv, yv, 10)
        sv = contributions(m, name, Xv)
        assert sv.shape == (300, 2) and np.isfinite(sv).all(), name
        assert np.abs(sv[:, 0]).mean() > 0, name                         # x 与标签相关，贡献不为 0
