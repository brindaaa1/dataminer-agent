"""第 6 项测试：模板实例化的 PIT 正确性、窗口/派生/哨兵值、筛选漏斗各环节、注册表 attach。"""
import numpy as np
import pandas as pd
import pytest
import yaml

from data.access import DataAccess
from data.knowhow import load_config
from features.factory import compute, parse_agg
from features.registry import Registry
from features.screen import iv, screen
from tools.generate_features import generate_features

TEMPLATES = """
templates:
  - id: t_inst
    source: hist
    entity: cid
    time_field: DAYS_X
    time_unit: days
    windows: [30, 90, all]
    aggs: [count, "avg(AMT_PAY < AMT_DUE)", "max(SENT)"]
    derived: ["ratio(30/90)", "trend(30 vs all)", "recency: -max(DAYS_X)", "recency: -max(DAYS_X) where Refused"]
    monotone_prior: +1
    rationale: 测试
  - id: t_req
    source: app
    entity: cid
    windows: [current]
    aggs: ["AMT / INC", "AMT / NULLIF(INC, 0)   # monotone_prior: 0"]
    monotone_prior: +1
"""
DICT = "dataset: toyhc\nentity_key: cid\nlabel: {source_table: app, col: y}\nsentinel_values:\n  - {value: 365243, fields: [SENT], table: hist, meaning: 占位}\n"


@pytest.fixture
def hc(tmp_path):
    hist = pd.DataFrame({
        "cid":   [1, 1, 1, 1, 2, 2, 3],
        "DAYS_X": [-10, -50, -200, 5, -20, -25, 3],           # cid=1 有一条未来记录(+5)；cid=3 只有未来记录
        "AMT_DUE": [100, 100, 100, 100, 100, 100, 100],
        "AMT_PAY": [50, 100, 50, 10, 100, 100, 10],
        "SENT": [1, 2, 365243, 9, 3, 4, 5]})
    app = pd.DataFrame({"cid": [1, 2, 3, 4], "AMT": [10.0, 20, 30, 40], "INC": [5.0, 0, 10, 20], "y": [0, 1, 0, 1]})
    hist.to_csv(tmp_path / "hist.csv", index=False)
    app.to_csv(tmp_path / "app.csv", index=False)
    kh = tmp_path / "kh"
    (kh / "templates").mkdir(parents=True)
    (kh / "templates" / "toyhc.yaml").write_text(TEMPLATES)
    (kh / "toyhc").mkdir()
    (kh / "toyhc" / "data_dictionary.yaml").write_text(DICT)
    cfg = load_config()
    cfg["paths"] = {"knowhow_root": str(kh), "artifacts_root": str(tmp_path / "art"), "reports_root": str(tmp_path / "rep")}
    cfg["datasets"]["toyhc"] = {"tables": {"hist": str(tmp_path / "hist.csv"), "app": str(tmp_path / "app.csv")}}
    cfg["task_specs"]["toyhc"] = {"pit_extra_filters": {}}
    return cfg


def test_parse_agg():
    assert parse_agg("count")["expr"] == "count(*)"
    assert parse_agg("count_distinct(X)")["expr"] == "count(DISTINCT X)"
    assert parse_agg("avg(A < B)")["expr"] == "avg(CAST((A < B) AS DOUBLE))"
    assert parse_agg("AMT / X   # monotone_prior: -1")["prior"] == -1
    assert parse_agg("share of months with STATUS in ('1')") is None      # 散文，不支持


def test_pit_windows_derived_sentinel(hc):
    df, defs = compute("t_inst", "toyhc", hc, pd.Series([1, 2, 3, 9]))
    r = df.set_index("_row_id")
    # cid=1：未来记录 DAYS_X=+5 被 PIT 排除；w30 只有 -10；w90 有 -10,-50；all 有 -10,-50,-200
    assert r.loc[1, "t_inst__a0__w30"] == 1 and r.loc[1, "t_inst__a0__w90"] == 2 and r.loc[1, "t_inst__a0__wall"] == 3
    assert r.loc[1, "t_inst__a1__w90"] == pytest.approx(0.5)                # (50<100, 100<100)→ 1/2
    assert pd.isna(r.loc[1, "t_inst__a2__wall"]) is False and r.loc[1, "t_inst__a2__wall"] == 2   # 哨兵 365243 → NULL，未来记录也不参与
    assert r.loc[1, "t_inst__a0__ratio_30_90"] == pytest.approx(0.5)
    assert r.loc[1, "t_inst__a1__trend_30_all"] == pytest.approx(1.0 - 2 / 3)
    assert r.loc[1, "t_inst__recency2"] == 10                              # -max(DAYS_X)=10
    # cid=3 只有未来记录 → 视同没有历史；cid=9 不存在 → 缺失
    assert pd.isna(r.loc[3, "t_inst__a0__w30"]) and pd.isna(r.loc[9, "t_inst__a0__w30"])
    assert not any("where" in d["expr"] for d in defs)                     # 依赖取值的散文写法被跳过


def test_row_level_template_and_priors(hc):
    df, defs = compute("t_req", "toyhc", hc, pd.Series([1, 2]))
    assert df["t_req__a0"].iloc[0] == 2.0 and not np.isfinite(df["t_req__a0"].iloc[1])   # 20/0：未做保护的表达式原样保留
    assert pd.isna(df["t_req__a1"].iloc[1])                                # NULLIF 防除零
    assert [d["prior"] for d in defs] == [1, 0]                            # agg 上的 monotone_prior 优先于模板


# ---------- 筛选漏斗 ----------
@pytest.fixture
def funnel_data():
    r = np.random.default_rng(0)
    n = 6000
    y = pd.Series(r.integers(0, 2, n))
    good = y * 0.8 + r.normal(size=n)
    X = pd.DataFrame({
        "good": good, "good_dup": good * 2 + r.normal(scale=0.01, size=n),        # 与 good 高度相关
        "noise": r.normal(size=n), "mostly_nan": np.where(r.random(n) < 0.97, np.nan, 1.0),
        "leaky": y * 3 + r.normal(scale=0.1, size=n), "drift": y * 0.8 + r.normal(size=n)})
    Xo = X.copy()
    Xo["drift"] = Xo["drift"] + 3.0                                                # 训练/OOT 分布漂移
    return X, y, Xo


def test_screen_funnel(funnel_data):
    X, y, Xo = funnel_data
    cfg = load_config()
    rep = screen(X, y, Xo, cfg)
    assert rep["dropped"]["mostly_nan"] == "missing"
    assert rep["dropped"]["noise"] == "low_iv"
    assert "leaky" in rep["leak_suspect"] and "leaky" not in rep["kept"]           # 报警，不进入 kept
    assert rep["dropped"]["good_dup"].startswith("corr") or rep["dropped"]["good"].startswith("corr")
    assert sum(f in rep["kept"] for f in ("good", "good_dup")) == 1                # 相关对只留一个
    assert rep["dropped"]["drift"] == "psi"
    assert rep["kept"] == ["good"] or rep["kept"] == ["good_dup"]


def test_iv_signal_vs_noise(funnel_data):
    X, y, _ = funnel_data
    assert iv(X["good"], y) > 0.1 > iv(X["noise"], y)


# ---------- 注册表 + attach ----------
def test_generate_and_attach(hc):
    cfg = hc
    r = np.random.default_rng(0)
    n = 400
    ids = np.arange(1000, 1000 + n)
    hist = pd.DataFrame({"cid": np.repeat(ids, 3), "DAYS_X": r.integers(-300, 5, n * 3), "AMT_DUE": 100.0,
                         "AMT_PAY": r.choice([10.0, 100.0], n * 3), "SENT": 1})
    y = (hist.groupby("cid")["AMT_PAY"].apply(lambda g: (g < 100).mean()).values + r.normal(scale=0.3, size=n) > 0.6).astype(int)
    hist.to_csv(cfg["datasets"]["toyhc"]["tables"]["hist"], index=False)
    for s, sl in (("train", slice(0, 200)), ("valid", slice(200, 300)), ("oot_dev", slice(300, 350)), ("holdout", slice(350, 400))):
        d = ("holdout" if s == "holdout" else "splits")
        p = pd.DataFrame({"_row_id": ids[sl], "_label": y[sl]})
        pth = __import__("pathlib").Path(cfg["paths"]["artifacts_root"]) / d / "toyhc"
        pth.mkdir(parents=True, exist_ok=True)
        p.to_parquet(pth / f"{s}.parquet")
    cfg["screen"]["iv_min"] = 0.0                                                  # 小样本下放宽，本测试只验证流程
    out = generate_features("t_inst", "toyhc", cfg)
    assert out["feature_set_id"].startswith("fs_") and out["kept"]
    reg = Registry(cfg)
    tr = reg.attach(DataAccess("toyhc", cfg["paths"]["artifacts_root"]).load("train"), [out["feature_set_id"]], "train")
    assert set(out["kept"]) <= set(tr.columns) and len(tr) == 200
    # holdout：缓存里没有，attach 现算（由 final_gate 触发）
    ho = pd.DataFrame({"_row_id": ids[350:400], "_label": y[350:400]})
    ho2 = reg.attach(ho, [out["feature_set_id"]], "holdout")
    assert set(out["kept"]) <= set(ho2.columns) and len(ho2) == 50
