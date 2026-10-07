"""评测集的数据：从原始数据抽出固定的子集，写到 data_raw/eval/<数据集>/（不提交）。随机种子固定，可重复生成。

    python -m eval.suite.prepare_data            # 三个都生成
    python -m eval.suite.prepare_data hotel_bookings

- hotel_bookings：data_raw/hotel_bookings/hotel_bookings.csv（TidyTuesday 原文件，出处见 data_sample/hotel_bookings/NOTICE.md），
  按 到店年月 × is_canceled 分层抽 4 万行，做法同 data_sample/make_hotel_sample.py。
- lending_club：原始 CSV 已不在本机，从早期运行留下的切分数据还原（data_raw/lending_club/processed/*.parquet）：
  _label → loan_status，_split_time → issue_d（"Dec-2015"），_row_id → id。当时加载就剔掉了泄漏字段，
  所以这份数据只能测建模，测不了"能否发现泄漏"。按 放款月 × label 分层抽 6 万行。
- home_credit：data_raw/home_credit/raw/*.parquet（七张原始表），随机抽 12 万个申请人，子表只留这些人的记录
  （bureau_balance 按抽中的 SK_ID_BUREAU 过滤）。原来抽 3 万：OOT-dev 只有约 240 个坏样本，只能分辨 ≥0.027 的 AUC 提升
  （v5 评测，见 evaluation/mde.py）；12 万约 970 个坏样本，MDE 约 0.0135。
"""
import sys
from pathlib import Path

import duckdb
import pandas as pd

from data.sample import ROOT

RAW, OUT, SEED = ROOT / "data_raw", ROOT / "data_raw" / "eval", 42
N = {"hotel_bookings": 40_000, "lending_club": 60_000, "home_credit": 120_000}
HC_CHILD = {"bureau": "SK_ID_CURR", "previous_application": "SK_ID_CURR", "installments_payments": "SK_ID_CURR",
            "POS_CASH_balance": "SK_ID_CURR", "credit_card_balance": "SK_ID_CURR"}


def _stratified(df: pd.DataFrame, key: pd.Series, n: int) -> pd.DataFrame:
    return df.groupby(key, group_keys=False).sample(frac=n / len(df), random_state=SEED).sort_index()


def hotel_bookings() -> dict:
    df = pd.read_csv(RAW / "hotel_bookings" / "hotel_bookings.csv", dtype=str, keep_default_na=False)   # 全按字符串读，写回保持原值
    s = _stratified(df, df["arrival_date_year"] + "-" + df["arrival_date_month"] + "-" + df["is_canceled"], N["hotel_bookings"])
    out = OUT / "hotel_bookings" / "hotel_bookings.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(out, index=False)
    return {"main": str(out)}


def lending_club() -> dict:
    df = pd.concat([pd.read_parquet(p) for p in sorted((RAW / "lending_club" / "processed").glob("*.parquet"))])
    df = df.drop(columns=[c for c in ("__index_level_0__", "_obs_time") if c in df])
    df.insert(0, "id", df.pop("_row_id"))
    df["loan_status"] = df.pop("_label").map({1: "Charged Off", 0: "Fully Paid"})
    t = df.pop("_split_time")
    df["issue_d"] = t.dt.strftime("%b-%Y")
    s = _stratified(df.reset_index(drop=True), t.dt.strftime("%Y-%m").values + df["loan_status"].values, N["lending_club"])
    out = OUT / "lending_club" / "lending_club.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(out, index=False)
    return {"main": str(out)}


def home_credit() -> dict:
    src, out = RAW / "home_credit" / "raw", OUT / "home_credit"
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"CREATE TABLE ids AS SELECT SK_ID_CURR FROM read_parquet('{src}/application_train.parquet') "
                f"USING SAMPLE {N['home_credit']} ROWS (reservoir, {SEED})")
    q = {"application_train": f"SELECT a.* FROM read_parquet('{src}/application_train.parquet') a JOIN ids USING (SK_ID_CURR)"}
    q |= {t: f"SELECT c.* FROM read_parquet('{src}/{t}.parquet') c JOIN ids USING ({k})" for t, k in HC_CHILD.items()}
    q["bureau_balance"] = (f"SELECT bb.* FROM read_parquet('{src}/bureau_balance.parquet') bb WHERE SK_ID_BUREAU IN "
                           f"(SELECT SK_ID_BUREAU FROM read_parquet('{src}/bureau.parquet') JOIN ids USING (SK_ID_CURR))")
    tables = {}
    for t, sql in q.items():
        con.execute(f"COPY ({sql}) TO '{out / t}.csv' (HEADER)")
        tables[t] = str(out / f"{t}.csv")
    return tables


BUILDERS = {"hotel_bookings": hotel_bookings, "lending_club": lending_club, "home_credit": home_credit}


def tables(dataset: str) -> dict:
    """评测时用的表路径（给 cfg['datasets'][dataset]['tables']）；文件由 prepare 生成。"""
    d = OUT / dataset
    if dataset == "home_credit":
        return {t: str(d / f"{t}.csv") for t in ("application_train", "bureau_balance", *HC_CHILD)}
    return {"main": str(d / f"{dataset}.csv")}


if __name__ == "__main__":
    for name in sys.argv[1:] or BUILDERS:
        t = BUILDERS[name]()
        rows = {k: sum(1 for _ in open(v)) - 1 for k, v in t.items()}
        print(name, rows)
