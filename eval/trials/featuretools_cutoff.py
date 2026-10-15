"""试验：Featuretools 按截止时间汇总 Home Credit 明细表，是否只用了申请之前的记录；顺带记录运行时间。
用 .venvs/ft 运行：.venvs/ft/bin/python eval/trials/featuretools_cutoff.py"""
import time
from pathlib import Path

import featuretools as ft
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REF = pd.Timestamp("2020-01-01")
AGG = ["count", "mean", "max", "min", "sum"]


def entityset(d: Path):
    app = pd.read_csv(d / "application_train.csv", usecols=["SK_ID_CURR"]).drop_duplicates("SK_ID_CURR")
    app["app_time"] = REF
    bur = pd.read_csv(d / "bureau.csv")
    bur["event_time"] = REF + pd.to_timedelta(bur["DAYS_CREDIT"], unit="D")
    prev = pd.read_csv(d / "previous_application.csv")
    prev["event_time"] = REF + pd.to_timedelta(prev["DAYS_DECISION"], unit="D")
    es = ft.EntitySet("home_credit")
    es.add_dataframe(dataframe_name="app", dataframe=app, index="SK_ID_CURR", time_index="app_time")
    es.add_dataframe(dataframe_name="bureau", dataframe=bur, index="SK_ID_BUREAU", time_index="event_time")
    es.add_dataframe(dataframe_name="prev", dataframe=prev, index="SK_ID_PREV", time_index="event_time")
    es.add_relationship("app", "SK_ID_CURR", "bureau", "SK_ID_CURR")
    es.add_relationship("app", "SK_ID_CURR", "prev", "SK_ID_CURR")
    return es, app, bur


def run(d: Path):
    es, app, bur = entityset(d)
    cutoff = pd.DataFrame({"SK_ID_CURR": app["SK_ID_CURR"], "time": REF})
    t0 = time.time()
    fm, _ = ft.dfs(entityset=es, target_dataframe_name="app", agg_primitives=AGG, trans_primitives=[], max_depth=1,
                   cutoff_time=cutoff, cutoff_time_in_index=False)
    return fm, bur, time.time() - t0


def main():
    md = ["# Featuretools 按截止时间汇总明细表是否可靠", "", f"Featuretools {ft.__version__}", ""]
    fm, bur, sec = run(ROOT / "data_raw" / "eval" / "planted" / "home_credit")
    expected = bur[bur["DAYS_CREDIT"] <= 0].groupby("SK_ID_CURR").size().reindex(fm.index, fill_value=0)
    got = fm["COUNT(bureau)"].fillna(0).astype(int)
    md += ["## 有申请之后记录的数据", "",
           f"- 申请人 {len(fm)} 个，生成特征 {fm.shape[1]} 个，用时 {sec:.0f} 秒",
           f"- bureau 里申请之后的记录：{int((bur['DAYS_CREDIT'] > 0).sum())} 条；恰好在申请当天的：{int((bur['DAYS_CREDIT'] == 0).sum())} 条",
           f"- 每人的 bureau 记录数与「只数申请当天及之前」不一致的人数：{int((got != expected).sum())}",
           f"- `MAX(bureau.DAYS_CREDIT)` 大于 0 的人数：{int((fm['MAX(bureau.DAYS_CREDIT)'] > 0).sum())}",
           ""]
    kept = bur[bur["DAYS_CREDIT"] <= 0]
    md += ["## 截止时间管不到的地方", "",
           "Featuretools 只按每条记录的时间（这里是 DAYS_CREDIT）过滤；记录本身在申请前，但其中某些字段记的是申请之后的事：", "",
           f"- 保留下来的记录里 `DAYS_ENDDATE_FACT > 0`（申请之后才结清）：{int((kept['DAYS_ENDDATE_FACT'] > 0).sum())} 条",
           f"- 保留下来的记录里 `DAYS_CREDIT_UPDATE > 0`（申请之后才更新）：{int((kept['DAYS_CREDIT_UPDATE'] > 0).sum())} 条", ""]
    fm_clean, _, sec_clean = run(ROOT / "data_raw" / "eval" / "home_credit")
    fm_clean.to_parquet(ROOT / "data_raw" / "eval" / "planted" / "ft_home_credit_clean.parquet")
    md += ["## 干净数据", "", f"- 生成特征 {fm_clean.shape[1]} 个，用时 {sec_clean:.0f} 秒，已存给提升试验用", ""]
    out = ROOT / "eval" / "trials" / "results" / "featuretools_cutoff.md"
    out.write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
