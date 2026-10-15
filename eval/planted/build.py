"""三份测试数据：在 data_raw/eval/ 的干净子集里人为放进已知问题，连同原生问题一起写成标准答案。

    python -m eval.planted.build                 # 三份都生成
    python -m eval.planted.build home_credit
"""
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.sample import ROOT
from eval.planted import issues
from eval.planted.issues import Answer

SOURCE = ROOT / "data_raw" / "eval"
PLANTED = ROOT / "data_raw" / "eval" / "planted"
KNOWHOW = ROOT / "knowhow"
LABELS = {"lending_club": ("lending_club", "loan_status"), "hotel_bookings": ("hotel_bookings", "is_canceled"),
          "home_credit": ("application_train", "TARGET")}


def _interior_months(period: pd.Series) -> list[str]:
    u = sorted(period.unique())
    return [u[int(len(u) * 0.4)], u[int(len(u) * 0.6)]]


def _native_duplicates(table: str, df: pd.DataFrame) -> list[Answer]:
    n = int(df.duplicated().sum())
    return [Answer("duplicate_rows", table, None, {"n_rows": n}, native=True)] if n else []


def lending_club(src: Path, rng) -> tuple[dict, list[Answer]]:
    df = pd.read_csv(src / "lending_club.csv", low_memory=False)
    t = "lending_club"
    period = pd.to_datetime(df["issue_d"], format="%b-%Y").dt.strftime("%Y-%m")
    df, a = issues.missing_months(df, t, "issue_d", period, _interior_months(period))
    ans = [a]
    for f in (lambda d: issues.sentinel(d, t, "mths_since_last_delinq", 999, 0.5, rng, only_missing=True),
              lambda d: issues.sentinel(d, t, "revol_util", -1, 0.02, rng),
              lambda d: issues.sentinel(d, t, "earliest_cr_line", "Jan-1900", 0.01, rng),
              lambda d: issues.duplicate_key(d, t, "id", 0.005, rng),
              lambda d: issues.category_variants(d, t, "home_ownership", "RENT", ["rent", " RENT", "Rent"], 0.05, rng),
              lambda d: issues.post_outcome(d, t, "last_pymnt_score", (d["loan_status"] == "Charged Off").astype(int).to_numpy(), 0.1, rng),
              lambda d: issues.duplicate_rows(d, t, 0.01, rng)):
        df, a = f(df)
        ans.append(a)
    return {t: df}, ans


def hotel_bookings(src: Path, rng) -> tuple[dict, list[Answer]]:
    df = pd.read_csv(src / "hotel_bookings.csv", low_memory=False)
    t = "hotel_bookings"
    period = pd.to_datetime(df["arrival_date_year"].astype(str) + "-" + df["arrival_date_month"] + "-1",
                            format="%Y-%B-%d").dt.strftime("%Y-%m")
    df, a = issues.missing_months(df, t, "arrival_date_month", period, _interior_months(period))
    ans = [a] + _native_duplicates(t, df)
    bl = yaml.safe_load((KNOWHOW / "hotel_bookings" / "blacklist.yaml").read_text())
    ans += [Answer("post_outcome", t, c, native=True) for c in bl["leakage"]["fields"] if c in df.columns]
    ans += [Answer("post_outcome", t, c, native=True, optional=True) for c in bl["unknown_availability"]["fields"] if c in df.columns]
    for f in (lambda d: issues.sentinel(d, t, "lead_time", 999, 0.01, rng),
              lambda d: issues.sentinel(d, t, "adr", -1, 0.02, rng),
              lambda d: issues.category_variants(d, t, "meal", "BB", ["bb", " BB", "BB "], 0.05, rng)):
        df, a = f(df)
        ans.append(a)
    return {t: df}, ans


def _native_sentinels(tables: dict) -> list[Answer]:
    """数据字典里登记的占位值，只算表里确实出现的。"""
    d = yaml.safe_load((KNOWHOW / "home_credit" / "data_dictionary.yaml").read_text())
    out = []
    for e in d["sentinel_values"]:
        t = e.get("table", "application_train")
        for c in e["fields"]:
            if c in tables[t].columns and tables[t][c].eq(e["value"]).any():
                out.append(Answer("sentinel", t, c, {"value": e["value"]}, must_find=(c == "DAYS_EMPLOYED"), native=True))
    return out


def home_credit(src: Path, rng) -> tuple[dict, list[Answer]]:
    tables = {t: pd.read_csv(src / f"{t}.csv", low_memory=False) for t in ("application_train", "bureau", "previous_application")}
    ans = _native_sentinels(tables) + [a for t, d in tables.items() for a in _native_duplicates(t, d)]
    t, df = "application_train", tables["application_train"]
    for f in (lambda d: issues.sentinel(d, t, "OWN_CAR_AGE", 99, 0.3, rng, only_missing=True),
              lambda d: issues.sentinel(d, t, "AMT_REQ_CREDIT_BUREAU_YEAR", -1, 0.02, rng),
              lambda d: issues.duplicate_key(d, t, "SK_ID_CURR", 0.003, rng),
              lambda d: issues.category_variants(d, t, "NAME_INCOME_TYPE", "Working", ["working", "WORKING "], 0.05, rng),
              lambda d: issues.post_outcome(d, t, "dpd_after_3m", d["TARGET"].to_numpy(), 0.1, rng)):
        df, a = f(df)
        ans.append(a)
    tables[t] = df
    tables["bureau"], a = issues.after_cutoff(tables["bureau"], "bureau", "DAYS_CREDIT", 0.01, rng)
    return tables, ans + [a]


def _recount(tables: dict, ans: list[Answer]) -> None:
    """后放的问题会改动先放的（重复行会把占位值行也复制一份），行数按成品表重新数。"""
    for a in ans:
        df = tables[a.table]
        if a.kind == "sentinel" and not a.native:
            a.detail["n_rows"] = int(df[a.column].eq(a.detail["value"]).sum())
        elif a.kind == "category_variants":
            a.detail["n_rows"] = int(df[a.column].isin(a.detail["variants"]).sum())
        elif a.kind == "duplicate_key":          # 完全相同的重复行也会让编号重复，那部分算在重复行里
            a.detail["n_rows"] = int(df.drop_duplicates()[a.column].duplicated().sum())
        elif a.kind == "duplicate_rows":
            a.detail["n_rows"] = int(df.duplicated().sum())


RECIPES = {"lending_club": lending_club, "hotel_bookings": hotel_bookings, "home_credit": home_credit}


def build(name: str, out_root: Path = PLANTED, seed: int = 0) -> list[dict]:
    src, out = SOURCE / name, Path(out_root) / name
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    tables, ans = RECIPES[name](src, np.random.default_rng(seed))
    _recount(tables, ans)
    for t, df in tables.items():
        df.to_csv(out / f"{t}.csv", index=False)
    for f in src.glob("*.csv"):                       # 没改动的明细表原样复制，保持多表完整
        if not (out / f.name).exists():
            shutil.copy(f, out / f.name)
    answers = [asdict(a) for a in ans]
    (out / "answers.json").write_text(json.dumps(answers, ensure_ascii=False, indent=1, default=str))
    return answers


if __name__ == "__main__":
    for n in sys.argv[1:] or list(RECIPES):
        print(n, len(build(n)), "条答案")
