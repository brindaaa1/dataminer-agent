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


def test_tune_does_not_replay_parent_trials(env):
    """v4 评测 lending_club：TUNE 与父实验同种子、同空间，采样序列逐点重放，调参等于重跑（AUC 一位不差）。
    同一个 seed 下，不同实验的采样点应不同；同一实验重跑仍可复现。"""
    run_study("t_parent", "toy", "lgbm", env, n_trials=4, seed=0)
    run_study("t_child", "toy", "lgbm", env, n_trials=4, seed=0)
    db = f"sqlite:///{env['paths']['artifacts_root']}/optuna.db"
    params = lambda n: [t.params for t in optuna.load_study(study_name=n, storage=db).trials]
    assert params("t_parent") != params("t_child")


def test_prepare_turns_nullable_dtypes_into_float():
    """v5 评测 home_credit：切分里有 pandas 可空类型（Yes/No → boolean、带缺失的整数 → Int64），缺失是 pd.NA，
    CatBoost 转 float 直接报错。prepare 统一转成 float64，缺失为 NaN。"""
    from modeling.prep import prepare
    df = pd.DataFrame({"flag": pd.array([True, None, False], dtype="boolean"), "days": pd.array([-5, None, 3], dtype="Int64")})
    X, _ = prepare(df, [])
    assert all(X[c].dtype == "float64" for c in X) and X["flag"].tolist()[0] == 1.0 and np.isnan(X["days"][1])


def test_low_fidelity_has_a_row_floor():
    """A/A 诊断：hotel 训练集 2.1 万行，30% 只剩 6300 行，同一模型赛跑 OOT-dev 随采样种子差 0.010；全量 10 trial 只差 0.0016。
    低保真抽样比例 = max(sample_frac, min_rows / 训练行数)，最多 1；大数据集仍按比例抽。"""
    from modeling.inner_loop import effective_frac
    fid = {"sample_frac": 0.3, "min_rows": 20_000}
    assert effective_frac(fid, 21_096) == pytest.approx(20_000 / 21_096)      # hotel：约 95%
    assert effective_frac(fid, 8_000) == 1.0
    assert effective_frac(fid, 50_000) == pytest.approx(0.4)
    assert effective_frac(fid, 72_000) == 0.3
    assert effective_frac({"sample_frac": 0.3}, 21_096) == 0.3                 # 没配下限：照旧


def test_run_study_uses_row_floor_and_reports_rows(env):
    env["fidelity"]["low"].update(sample_frac=0.3, min_rows=2000)
    r = run_study("t_floor", "toy", "lgbm", env, fidelity="low", n_trials=2)
    assert r["n_rows"] == 2000                                                 # 3000 行 × max(0.3, 2000/3000)
