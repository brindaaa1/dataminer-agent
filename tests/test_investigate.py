"""INVESTIGATE 的只读查询：只看训练集、只返回聚合值、不限于风控的通用指标。"""
import numpy as np
import pandas as pd
import pytest

from tools.query_data import QUESTIONS, answer


@pytest.fixture
def df():
    """signal：与目标单调相关；bump：非单调（中间取值正类多）；noise：无关；dup：signal 的复制；
    shifted：后半段时间整体偏移；miss：缺失本身有信息；cat：类别特征。"""
    r = np.random.default_rng(0)
    n = 4000
    signal = r.normal(size=n)
    bump = r.normal(size=n)
    y = (signal + 2 * (np.abs(bump) < 0.5) + r.normal(scale=1.0, size=n) > 1.0).astype(int)
    miss = r.normal(size=n)
    miss[(y == 1) & (r.random(n) < 0.6)] = np.nan
    t = pd.date_range("2020-01-01", periods=n, freq="h")
    shifted = r.normal(size=n) + np.where(np.arange(n) > n / 2, 3.0, 0.0)
    cat = np.where(y == 1, r.choice(["a", "b"], n, p=[0.8, 0.2]), r.choice(["a", "b"], n, p=[0.3, 0.7]))
    return pd.DataFrame({"signal": signal, "bump": bump, "noise": r.normal(size=n), "dup": signal * 2 + 0.001,
                         "shifted": shifted, "miss": miss, "cat": cat, "_obs_time": t, "_label": y})


FEATS = ["signal", "bump", "noise", "dup", "shifted", "miss", "cat"]


def test_questions_are_generic():
    assert set(QUESTIONS) >= {"single_feature_auc", "missing_rate", "iv", "mutual_info", "target_rate_by_bin",
                              "stability_psi", "missing_vs_target", "redundant_pairs"}


def test_iv_and_mutual_info_rank_informative_features(df):
    iv = answer(df, "iv", FEATS)
    assert list(iv)[0] in {"signal", "dup", "cat", "miss"} and iv.get("noise", 0) < 0.05
    mi = answer(df, "mutual_info", FEATS)
    assert mi["bump"] > mi["noise"]                      # 非单调关系 AUC 看不出来，互信息看得出


def test_target_rate_by_bin_shows_where_the_signal_is(df):
    r = answer(df, "target_rate_by_bin", ["signal"])
    bins = r["signal"]
    assert len(bins) >= 5 and bins[-1]["target_rate"] > bins[0]["target_rate"] and {"bin", "n", "target_rate", "lift"} <= set(bins[0])


def test_stability_missing_and_redundancy(df):
    psi = answer(df, "stability_psi", FEATS)
    assert list(psi)[0] == "shifted" and psi["shifted"] > 0.25
    mv = answer(df, "missing_vs_target", FEATS)
    assert list(mv) == ["miss"] and mv["miss"]["target_rate_missing"] > mv["miss"]["target_rate_present"]
    pairs = answer(df, "redundant_pairs", FEATS)
    assert pairs[0]["pair"] == ["signal", "dup"] and pairs[0]["corr"] > 0.99


def test_unknown_question_is_refused(df):
    with pytest.raises(ValueError):
        answer(df, "nope", FEATS)
