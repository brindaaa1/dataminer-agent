import numpy as np
import pandas as pd
import pytest

from eval.planted import issues


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 1000
    return pd.DataFrame({"id": np.arange(n), "x": rng.normal(size=n), "cat": rng.choice(["RENT", "OWN"], n),
                         "x_na": np.where(rng.random(n) < 0.2, np.nan, 1.0), "days": -rng.integers(0, 3000, n)})


def test_sentinel_replaces_share_of_rows(df):
    out, a = issues.sentinel(df, "t", "x", -1, 0.05, np.random.default_rng(1))
    assert (out["x"] == -1).sum() == 50 and a.kind == "sentinel" and a.column == "x" and a.detail["value"] == -1
    assert (df["x"] == -1).sum() == 0                    # 原表不变


def test_sentinel_only_missing_fills_missing_rows(df):
    out, a = issues.sentinel(df, "t", "x_na", 999, 0.5, np.random.default_rng(1), only_missing=True)
    filled = out["x_na"] == 999
    assert df.loc[filled, "x_na"].isna().all() and filled.sum() == int(df["x_na"].isna().sum() * 0.5)


def test_sentinel_rejects_value_already_present(df):
    with pytest.raises(AssertionError):
        issues.sentinel(df.assign(x=df["x"].where(df.index != 3, -1)), "t", "x", -1, 0.05, np.random.default_rng(1))


def test_duplicate_rows_adds_exact_copies(df):
    out, a = issues.duplicate_rows(df, "t", 0.02, np.random.default_rng(1))
    assert len(out) == 1020 and out.duplicated().sum() == 20 and a.column is None


def test_duplicate_key_reuses_existing_keys(df):
    out, a = issues.duplicate_key(df, "t", "id", 0.01, np.random.default_rng(1))
    assert out["id"].duplicated().sum() == 10 and a.column == "id"


def test_category_variants_rewrites_value(df):
    out, a = issues.category_variants(df, "t", "cat", "RENT", ["rent", " RENT"], 0.1, np.random.default_rng(1))
    assert set(out["cat"]) == {"RENT", "OWN", "rent", " RENT"} and a.detail["variants"] == ["rent", " RENT"]


def test_category_variants_rejects_variant_already_present(df):
    with pytest.raises(AssertionError):
        issues.category_variants(df, "t", "cat", "RENT", ["OWN"], 0.1, np.random.default_rng(1))


def test_missing_months_drops_interior_months():
    period = pd.Series(["2016-01", "2016-02", "2016-03", "2016-04"] * 5)
    d = pd.DataFrame({"m": period})
    out, a = issues.missing_months(d, "t", "m", period, ["2016-02"])
    assert len(out) == 15 and "2016-02" not in set(out["m"]) and a.detail["n_rows"] == 5


def test_missing_months_rejects_edge_month():
    period = pd.Series(["2016-01", "2016-02", "2016-03"])
    with pytest.raises(AssertionError):
        issues.missing_months(pd.DataFrame({"m": period}), "t", "m", period, ["2016-01"])


def test_post_outcome_tracks_label(df):
    y = (df["x"] > 0).astype(int).to_numpy()
    out, a = issues.post_outcome(df, "t", "leak", y, 0.1, np.random.default_rng(1))
    assert np.corrcoef(out["leak"], y)[0, 1] > 0.7 and a.kind == "post_outcome"


def test_after_cutoff_moves_events_after_application(df):
    out, a = issues.after_cutoff(df, "t", "days", 0.01, np.random.default_rng(1))
    assert (out["days"] > 0).sum() == 10 and a.kind == "after_cutoff"


from pathlib import Path

from eval.planted import build as planted_build


@pytest.fixture
def tiny_hotel(tmp_path, monkeypatch):
    """用仓库自带的酒店小样本代替 data_raw/eval，避免测试依赖本机数据。"""
    from data.sample import SAMPLE_CSV
    src = tmp_path / "eval" / "hotel_bookings"
    src.mkdir(parents=True)
    (src / "hotel_bookings.csv").write_bytes(Path(SAMPLE_CSV).read_bytes())
    monkeypatch.setattr(planted_build, "SOURCE", tmp_path / "eval")
    return tmp_path / "planted"


def test_build_writes_answers_and_values_survive_csv(tiny_hotel):
    answers = planted_build.build("hotel_bookings", tiny_hotel)
    d = pd.read_csv(tiny_hotel / "hotel_bookings" / "hotel_bookings.csv", low_memory=False)   # 工具默认的读法
    kinds = {a["kind"] for a in answers}
    assert {"sentinel", "category_variants", "missing_months", "duplicate_rows", "post_outcome"} <= kinds
    for a in answers:
        if a["kind"] == "sentinel" and not a["native"]:
            assert (d[a["column"]] == a["detail"]["value"]).sum() == a["detail"]["n_rows"]
        if a["kind"] == "category_variants":
            assert set(a["detail"]["variants"]) <= set(d[a["column"]])
    assert (tiny_hotel / "hotel_bookings" / "answers.json").exists()


def test_build_records_native_duplicates_instead_of_planting(tiny_hotel):
    answers = planted_build.build("hotel_bookings", tiny_hotel)
    dup = [a for a in answers if a["kind"] == "duplicate_rows"]
    assert len(dup) == 1 and dup[0]["native"]


from eval.planted.score import score, to_markdown

ANS = [{"kind": "sentinel", "table": "a", "column": "x", "must_find": True, "optional": False},
       {"kind": "duplicate_rows", "table": "a", "column": None, "must_find": False, "optional": False},
       {"kind": "post_outcome", "table": "a", "column": "adr", "must_find": False, "optional": True}]


def test_score_matches_by_table_and_column():
    f = [{"table": "a", "column": "x", "kind": "Outlier", "source": "t"},
         {"table": "a", "column": None, "kind": "Dups", "source": "t"}]
    r = score(ANS, f)
    assert r["n_answers"] == 2 and r["found_rate"] == 1.0 and r["must_find_ok"] and r["not_problems"] == []


def test_score_counts_not_problems_once_per_column_and_ignores_optional():
    f = [{"table": "a", "column": "y", "kind": "Skewed", "source": "t"},
         {"table": "a", "column": "y", "kind": "Zeros", "source": "t"},
         {"table": "a", "column": "adr", "kind": "Skewed", "source": "t"}]
    r = score(ANS, f)
    assert r["not_problems"] == [("a", "y")] and r["found_rate"] == 0 and not r["must_find_ok"]
    assert len(r["missed"]) == 2


def test_to_markdown_mentions_rates():
    md = to_markdown("x", score(ANS, []))
    assert "0/2" in md and "DAYS_EMPLOYED" not in md


def test_duplicate_key_keeps_every_original_row(df):
    """编号重复用追加的方式放：原有的行和编号都不动，明细表的关联不会断。"""
    out, a = issues.duplicate_key(df, "t", "id", 0.01, np.random.default_rng(1))
    assert len(out) == 1010 and set(out["id"]) == set(df["id"]) and out["id"].duplicated().sum() == 10
    assert not out.duplicated().any()                       # 编号相同，但内容不同


def test_recount_duplicate_key_ignores_exact_copies(df):
    rng = np.random.default_rng(1)
    d, a_key = issues.duplicate_key(df, "t", "id", 0.01, rng)
    d, a_rows = issues.duplicate_rows(d, "t", 0.02, rng)
    planted_build._recount({"t": d}, [a_key, a_rows])
    assert a_key.detail["n_rows"] == 10 and a_rows.detail["n_rows"] == 20


def test_score_table_level_answer_needs_a_duplicate_finding():
    f = [{"table": "a", "column": None, "kind": "Conflicting Labels", "source": "t"}]
    r = score(ANS, f)
    assert r["n_found"] == 0 and r["not_problems"] == [("a", None)]
    r = score(ANS, [{"table": "a", "column": None, "kind": "Data Duplicates", "source": "t"}])
    assert r["n_found"] == 1


def test_score_lists_what_matched_each_answer():
    r = score(ANS, [{"table": "a", "column": "x", "kind": "Outlier", "source": "t"}])
    assert r["matched_by"] == {"a.x": ["Outlier"]}


def test_to_markdown_lists_check_per_answer():
    md = to_markdown("x", score(ANS, [{"table": "a", "column": "x", "kind": "Outlier", "source": "t"}]))
    assert "a.x：Outlier" in md
