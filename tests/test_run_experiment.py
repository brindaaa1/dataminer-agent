"""第 2 项测试：run_experiment（Optuna + lgbm + lr_scorecard）、warm-start ≤10%、预算、study 断点续跑。"""
import numpy as np
import optuna
import pandas as pd
import pytest

from data.knowhow import load_config
from modeling.inner_loop import clip_to_space, run_study
from modeling.zoo import ZOO
from tools.run_experiment import run_experiment


@pytest.fixture
def env(tmp_path):
    rng = np.random.default_rng(0)

    def part(n, start):
        x1, x2 = rng.normal(size=n), rng.normal(size=n)
        cat = rng.choice(["a", "b", "c"], n)
        y = (rng.random(n) < 1 / (1 + np.exp(-(1.2 * x1 - 0.8 * x2 + (cat == "a") - 1.5)))).astype(int)
        t = pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, 90, n), unit="D")
        x2[rng.random(n) < 0.1] = np.nan
        return pd.DataFrame({"_row_id": np.arange(n).astype(str), "_label": y, "_split_time": t, "_obs_time": t,
                             "x1": x1, "x2": x2, "cat": cat,
                             "opened": [(tt - pd.Timedelta(days=int(k))).strftime("%b-%Y") for tt, k in zip(t, rng.integers(100, 3000, n))]})

    art = tmp_path / "art" / "splits" / "toy"
    art.mkdir(parents=True)
    for s, st in (("train", "2012-01-01"), ("valid", "2012-01-01"), ("oot_dev", "2013-01-01")):
        part(3000 if s == "train" else 1500, st).to_parquet(art / f"{s}.parquet")
    kh = tmp_path / "kh" / "toy"
    kh.mkdir(parents=True)
    (kh / "data_dictionary.yaml").write_text("dataset: toy\nfields:\n  opened: {type: date}\n")
    cfg = load_config()
    cfg["paths"] = {"knowhow_root": str(tmp_path / "kh"), "artifacts_root": str(tmp_path / "art"), "reports_root": str(tmp_path / "rep")}
    cfg["fidelity"] = {"low": {"sample_frac": 1.0, "n_trials": 10, "early_stopping_rounds": 5},
                       "full": {"sample_frac": 1.0, "n_trials": 10, "early_stopping_rounds": 5}}
    cfg["inner_loop"].update(n_threads=2, n_rounds_max=60, report_every=10, pruner_startup_trials=2)
    return cfg


@pytest.mark.parametrize("model", ["lgbm", "lr_scorecard"])
def test_models_learn_signal(env, model):
    r = run_experiment(f"t_{model}", "toy", model, env, n_trials=5)
    assert r["status"] == "OK"
    assert r["metrics"]["valid_auc"] > 0.65 and r["metrics"]["oot_dev_auc"] > 0.65
    assert set(r["metrics"]) >= {"train_auc", "valid_auc", "oot_dev_auc", "oot_dev_ks", "gap"}
    assert np.load(f"{env['paths']['artifacts_root']}/experiments/t_{model}/oot_dev_pred.npy").shape == (1500,)


def test_zoo_registry():
    assert {"lgbm", "lr_scorecard"} <= set(ZOO) and "xgb" not in ZOO


def test_warm_start_capped_at_10_percent(env):
    pts = [{"learning_rate": 0.05 + i * 0.001, "num_leaves": 20} for i in range(8)]
    r = run_study("t_warm", "toy", "lgbm", env, n_trials=20, warm_start=pts)
    assert r["n_warm_start"] == 2                                   # 20 × 10%
    study = optuna.load_study(study_name="t_warm", storage=f"sqlite:///{env['paths']['artifacts_root']}/optuna.db")
    assert study.trials[0].params["learning_rate"] == pytest.approx(0.05)   # 初始点确实被注入


def test_warm_start_clipped_to_new_space():
    space = {"learning_rate": {"low": 0.01, "high": 0.1, "log": True}}
    assert clip_to_space({"learning_rate": 0.5, "other": 1}, space) == {"learning_rate": 0.1}


def test_budget_exceeded(env):
    r = run_experiment("t_budget", "toy", "lgbm", env, remaining_cpu_minutes=0.0, n_trials=5)
    assert r["status"] == "BUDGET_EXCEEDED"


def test_study_resume_does_not_redo_trials(env):
    run_study("t_res", "toy", "lgbm", env, n_trials=4)
    r = run_study("t_res", "toy", "lgbm", env, n_trials=6)          # 中断后续跑：只补剩余 2 个
    assert r["n_trials"] == 6


def test_prepare_handles_real_datetime_columns():
    """回归：数据里的日期类型列（如放回的泄漏字段 reservation_status_date）要能被模型消费。"""
    from modeling.prep import prepare
    t = pd.Series(pd.to_datetime(["2017-05-01", "2017-05-10"]))
    df = pd.DataFrame({"_obs_time": t, "d": t + pd.to_timedelta([3, -2], unit="D"), "x": [1.0, 2.0]})
    X, _ = prepare(df, [])
    assert X["d"].tolist() == [3.0, -2.0] and X["x"].tolist() == [1.0, 2.0]
