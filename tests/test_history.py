"""历史值可用（Elec2 端到端测试暴露）：当前值是泄漏、但预测时点之前的取值已知的列（如过去时段的电价、上一条记录的结果）。
代码按预测时点排序生成滞后值和过去 N 条的均值，原列照样剔除；这些特征可得时间为 history，泄漏扫描不再按单特征 AUC 误杀。"""
import numpy as np
import pandas as pd
import pytest

from data.splits import build_splits

DICT = """
dataset: hs
label: {source_col: dir, positive: ["UP"], negative: ["DOWN"]}
observation_time_col: t
suggested_split: {train_valid: ["2012-01", "2012-12"], oot_dev: ["2013-01", "2013-06"], holdout: ["2013-07", "2013-12"]}
fields:
  hour:  {availability: at_prediction, type: numeric}
  price: {availability: post_outcome, type: numeric, history: {lags: [1, 2], means: [3]}}
  dir:   {availability: post_outcome, type: categorical, history: {lags: [1]}}
"""


@pytest.fixture
def hs(tmp_path):
    rng = np.random.default_rng(0)
    t = pd.date_range("2012-01-01", "2013-12-31", freq="6h")
    price = np.cumsum(rng.normal(size=len(t)))
    d = np.where(price > pd.Series(price).rolling(4, min_periods=1).mean().values, "UP", "DOWN")
    df = pd.DataFrame({"t": t.strftime("%Y-%m-%d %H:%M"), "hour": t.hour, "price": price, "dir": d})
    df.sample(frac=1, random_state=0).to_csv(tmp_path / "hs.csv", index=False)          # 文件里的行序打乱：滞后必须按时间排
    kh = tmp_path / "knowhow" / "hs"
    kh.mkdir(parents=True)
    (kh / "data_dictionary.yaml").write_text(DICT)
    (kh / "blacklist.yaml").write_text("leakage: {fields: [price, dir]}\n")
    cfg = {"paths": {"knowhow_root": str(tmp_path / "knowhow"), "artifacts_root": str(tmp_path / "art"), "reports_root": str(tmp_path / "rep")},
           "datasets": {"hs": {"tables": {"main": str(tmp_path / "hs.csv")}}}, "split": {"valid_frac": 0.2, "seed": 1},
           "thresholds": {"maturity_current_share_max": 0.05, "missing_shift_report": 0.2},
           "task_specs": {"hs": {"row_filter": "true", "split_time_expr": "t",
                                 "observation_time_expr": "t"}}}
    return cfg, df


def test_history_features_are_lagged_in_time_order(hs):
    cfg, df = hs
    build_splits("hs", cfg)
    from data.access import DataAccess
    tr = pd.concat([DataAccess("hs", cfg["paths"]["artifacts_root"]).load(s) for s in ("train", "valid")]).sort_values("_obs_time")
    assert "price" not in tr and "dir" not in tr                                  # 当前值照样剔除
    ref = df.assign(ts=pd.to_datetime(df.t)).set_index("ts")
    ref["lag1"], ref["lag2"] = ref.price.shift(1), ref.price.shift(2)
    ref["mean3"] = ref.price.shift(1).rolling(3, min_periods=1).mean()
    ref["dir1"] = (ref.dir == "UP").astype(float).shift(1)
    got = tr.set_index("_obs_time")
    for col, r in (("price_lag1", "lag1"), ("price_lag2", "lag2"), ("price_mean3", "mean3"), ("dir_lag1", "dir1")):
        assert np.allclose(got[col].values, ref.loc[got.index, r].values, equal_nan=True), col


def test_history_features_are_trusted_by_the_leak_scan(hs):
    """上一条的结果单独就有很高的区分力（Elec2：约 0.85），按"单特征 AUC > 0.75"会被误判泄漏；它们由代码按时间生成，结构上不会泄漏。"""
    cfg, _ = hs
    build_splits("hs", cfg)
    from data.access import DataAccess
    from data.knowhow import load_knowhow
    from evaluation.evaluator import scan_features
    tr = DataAccess("hs", cfg["paths"]["artifacts_root"]).load("train")
    s = scan_features(tr, ["hour", "price_lag1", "dir_lag1"], load_knowhow("hs", cfg), cfg, "hs")
    assert s["unavailable_fields"] == [] and "dir_lag1" not in s["feature_aucs"] and "hour" in s["feature_aucs"]
