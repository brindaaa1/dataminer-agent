"""B0：LightGBM 默认参数 + 原始宽表（下限）。仅用 valid 做早停轮数，不调任何参数。"""
import time

import lightgbm as lgb

from baselines.common import finish, load_wide


def run(dataset: str, cfg: dict, feature_sets=None, name=None) -> dict:
    Xtr, ytr, Xva, yva, Xoo, yoo, prep = load_wide(dataset, cfg, feature_sets)
    t0 = time.time()
    m = lgb.train({"objective": "binary", "metric": "auc", "verbosity": -1, "seed": 0,
                   "num_threads": cfg["inner_loop"]["n_threads"]},
                  lgb.Dataset(Xtr, ytr), cfg["inner_loop"]["n_rounds_max"], valid_sets=[lgb.Dataset(Xva, yva)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    wall = time.time() - t0
    return finish(name or f"b0_{dataset}", dataset, cfg, lambda X: m.predict(X, num_iteration=m.best_iteration), prep, Xoo, yoo, wall,
                  {"n_features": Xtr.shape[1], "best_iter": m.best_iteration})
