"""B1：FLAML AutoML + 原始宽表。time_budget 与 agent 的训练耗时对齐（--match-task），用相同线程数。
评估协议与 agent 相同：valid 作为 FLAML 的验证集，OOT-dev/holdout 完全不参与搜索。"""
import time

from flaml import AutoML

from baselines.common import finish, load_wide


def run(dataset: str, cfg: dict, time_budget: float, feature_sets=None, name=None) -> dict:
    Xtr, ytr, Xva, yva, Xoo, yoo, prep = load_wide(dataset, cfg, feature_sets)
    automl = AutoML()
    t0 = time.time()
    automl.fit(Xtr, ytr, X_val=Xva, y_val=yva, task="classification", metric="roc_auc", time_budget=time_budget,
               n_jobs=cfg["inner_loop"]["n_threads"], seed=0, verbose=0)
    wall = time.time() - t0
    return finish(name or f"b1_{dataset}", dataset, cfg, lambda X: automl.predict_proba(X)[:, 1], prep, Xoo, yoo, wall,
                  {"n_features": Xtr.shape[1], "best_estimator": automl.best_estimator, "time_budget": time_budget,
                   "best_config": automl.best_config})
