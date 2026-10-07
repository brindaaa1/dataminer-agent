"""参照漂移：默认参数 LightGBM（同 B0）在本数据上的 valid − OOT-dev 主指标差，代表这份数据正常的时间漂移。
OVERFIT_GAP 按它做相对判定（v1 评测：hotel 的默认模型自己的 gap 就有 0.07，固定阈值 0.03 会拒掉所有有用的模型）。"""
from pathlib import Path

import lightgbm as lgb

from baselines.common import load_wide
from evaluation.mde import oot_power
from evaluation.metrics import METRICS, primary


def _train(Xtr, ytr, Xva, yva, cfg: dict):
    return lgb.train({"objective": "binary", "metric": "auc", "verbosity": -1, "seed": 0, "num_threads": cfg["inner_loop"]["n_threads"]},
                     lgb.Dataset(Xtr, ytr), cfg["inner_loop"]["n_rounds_max"], valid_sets=[lgb.Dataset(Xva, yva)],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def _fit(dataset: str, cfg: dict):
    """开始时训练一次、按最优迭代数存盘；最终检验时直接读（同一任务的产物目录、同一切分）。"""
    Xtr, ytr, Xva, yva, Xoo, yoo, prep = load_wide(dataset, cfg)
    path = Path(cfg["paths"]["artifacts_root"]) / f"reference_{dataset}.txt"
    if not path.exists():
        m = _train(Xtr, ytr, Xva, yva, cfg)
        m.save_model(str(path), num_iteration=m.best_iteration)
    return lgb.Booster(model_file=str(path)), prep, Xva, yva, Xoo, yoo


def reference_gap(dataset: str, cfg: dict) -> dict:
    m, _, Xva, yva, Xoo, yoo = _fit(dataset, cfg)
    f = METRICS[primary(cfg)].fn
    p_oo = m.predict(Xoo)
    v, o = f(yva, m.predict(Xva)), f(yoo, p_oo)
    return {"valid": v, "oot_dev": o, "gap": v - o, "oot_power": oot_power(yoo, p_oo, cfg)}


def reference_drop(dataset: str, cfg: dict, holdout) -> float:
    """参照模型 OOT-dev → holdout 的主指标下滑：这份数据正常的时间漂移。只在 FinalGate 里调用（唯一能读 holdout 的地方）。"""
    m, prep, _, _, Xoo, yoo = _fit(dataset, cfg)
    f = METRICS[primary(cfg)].fn
    return f(yoo, m.predict(Xoo)) - f(holdout["_label"], m.predict(prep(holdout, "holdout")))
