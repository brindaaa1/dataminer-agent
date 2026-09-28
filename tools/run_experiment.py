"""run_experiment 工具：先估算成本，超出剩余预算返回 BUDGET_EXCEEDED；否则跑一个 study（§4.6）。"""
import pandas as pd

from modeling.inner_loop import run_study


def estimate_cost_minutes(n_rows, n_features, n_trials, cfg) -> float:
    rounds = cfg["inner_loop"]["n_rounds_max"] / 5     # 早停后的有效树数（粗略）
    return cfg["inner_loop"]["cost_coef_sec"] * n_rows * n_features * rounds * n_trials / 60


def run_experiment(exp_id, dataset, model, cfg, remaining_cpu_minutes=None, **kw):
    fid = cfg["fidelity"][kw.get("fidelity", "low")]
    n_train = int(len(pd.read_parquet(f"{cfg['paths']['artifacts_root']}/splits/{dataset}/train.parquet",
                                      columns=["_label"])) * fid["sample_frac"])
    n_feat = len(kw["features"]) if kw.get("features") else 30
    est = estimate_cost_minutes(n_train, n_feat, kw.get("n_trials") or fid["n_trials"], cfg)
    if remaining_cpu_minutes is not None and est > remaining_cpu_minutes:
        return {"status": "BUDGET_EXCEEDED", "est_cpu_minutes": est, "remaining": remaining_cpu_minutes}
    res = run_study(exp_id, dataset, model, cfg, **kw)
    return {"status": "OK", "est_cpu_minutes": est, **res}
