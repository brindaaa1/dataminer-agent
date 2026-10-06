"""run_experiment 工具：先估算成本，超出剩余预算返回 BUDGET_EXCEEDED；否则跑一个 study（§4.6）。"""
import pandas as pd

from modeling.inner_loop import effective_frac, run_study


def estimate_cost_minutes(n_rows, n_features, n_trials, cfg, model: str = "lgbm", obs: dict | None = None) -> float:
    """CPU 分钟。obs：本任务同一模型的一次实测（模型赛跑），按行数、特征数、trial 数的比例外推；
    没有实测时用静态公式：系数 × 模型倍数 × 行数 × 特征数 × 有效树数 × trial 数（config.inner_loop，lending_club/hotel 上校准，
    v6 评测 home_credit 122 个特征时高估 37–44 倍）。"""
    if obs:
        return obs["cpu_minutes"] * n_rows / obs["n_rows"] * n_features / obs["n_features"] * n_trials / obs["n_trials"]
    rounds = cfg["inner_loop"]["n_rounds_max"] / 5     # 早停后的有效树数（粗略）
    factor = cfg["inner_loop"].get("cost_model_factor", {}).get(model, 1.0)
    return cfg["inner_loop"]["cost_coef_sec"] * factor * n_rows * n_features * rounds * n_trials / 60


def run_experiment(exp_id, dataset, model, cfg, remaining_cpu_minutes=None, cost_obs=None, **kw):
    fid = cfg["fidelity"][kw.get("fidelity", "low")]
    n_all = len(pd.read_parquet(f"{cfg['paths']['artifacts_root']}/splits/{dataset}/train.parquet", columns=["_label"]))
    n_train = int(n_all * effective_frac(fid, n_all))
    n_feat = len(kw["features"]) if kw.get("features") else 30
    est = estimate_cost_minutes(n_train, n_feat, kw.get("n_trials") or fid["n_trials"], cfg, model, cost_obs)
    if remaining_cpu_minutes is not None and est > remaining_cpu_minutes:
        return {"status": "BUDGET_EXCEEDED", "est_cpu_minutes": est, "remaining": remaining_cpu_minutes}
    res = run_study(exp_id, dataset, model, cfg, **kw)
    return {"status": "OK", "est_cpu_minutes": est, **res}
