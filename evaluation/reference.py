"""参照漂移：默认参数 LightGBM（同 B0）在本数据上的 valid − OOT-dev 主指标差，代表这份数据正常的时间漂移。
OVERFIT_GAP 按它做相对判定（v1 评测：hotel 的默认模型自己的 gap 就有 0.07，固定阈值 0.03 会拒掉所有有用的模型）。"""
import lightgbm as lgb

from baselines.common import load_wide
from evaluation.mde import oot_power
from evaluation.metrics import METRICS, primary


def reference_gap(dataset: str, cfg: dict) -> dict:
    Xtr, ytr, Xva, yva, Xoo, yoo, _ = load_wide(dataset, cfg)
    m = lgb.train({"objective": "binary", "metric": "auc", "verbosity": -1, "seed": 0, "num_threads": cfg["inner_loop"]["n_threads"]},
                  lgb.Dataset(Xtr, ytr), cfg["inner_loop"]["n_rounds_max"], valid_sets=[lgb.Dataset(Xva, yva)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    f = METRICS[primary(cfg)].fn
    p_oo = m.predict(Xoo, num_iteration=m.best_iteration)
    v, o = f(yva, m.predict(Xva, num_iteration=m.best_iteration)), f(yoo, p_oo)
    return {"valid": v, "oot_dev": o, "gap": v - o, "oot_power": oot_power(yoo, p_oo, cfg)}
