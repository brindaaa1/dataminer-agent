"""PROFILE 阶段一次性执行：读取原始表 → 打 label → 按 know-how 剔除字段 → 四段切分 → 写 artifacts。"""
import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from data.access import numpy_types
from data.knowhow import column_policy, field_table, label_sql, load_config, load_knowhow

ID, LABEL, OBS, SPLIT_T = "_row_id", "_label", "_obs_time", "_split_time"


def _month_range(pair):
    return pd.Timestamp(pair[0] + "-01"), pd.Timestamp(pair[1] + "-01") + pd.offsets.MonthBegin(1)


def _open(path: str, spec: dict):
    con = duckdb.connect()
    nulls = spec.get("csv_null_values")               # 数据里代表缺失的字符串（如 "NA"），否则整列会被推断成文本
    con.execute(f"CREATE VIEW raw AS SELECT * FROM read_csv('{path}', sample_size=-1, header=true{f', nullstr={nulls}' if nulls else ''})")
    aliases = spec.get("sql_aliases", {})              # 派生表达式里用到的、由多个原始列拼出的量（如到店日期）
    if aliases:
        con.execute("CREATE VIEW raw_a AS SELECT *, " + ", ".join(f"{v} AS {k}" for k, v in aliases.items()) + " FROM raw")
    return con, ("raw_a" if aliases else "raw")


def _features(dataset: str, kh: dict, spec: dict, cols: list[str]):
    """字段策略 + 派生特征 + 哨兵值 → (policy, SELECT 列表, 警告)。"""
    d, main_table = kh["dictionary"], kh["dictionary"].get("label", {}).get("source_table")
    # classic_derived：know-how 里的派生特征。可得时间 = 声明值与输入推断值中较弱者（§4.11）
    known, dexpr, dwarn = field_table(kh, main_table, spec.get("derived_expr_overrides"))
    # 历史值：按预测时点排序（有 history_by 时按它分组）的窗口，只取当前行之前的记录
    by = d.get("history_by")
    win = (f'PARTITION BY "{by}" ' if by else "") + f"ORDER BY {spec.get('observation_time_expr')}"
    dexpr = {k: v.replace("__HIST__", win) for k, v in dexpr.items()}
    pol = column_policy(dataset, cols + list(dexpr), kh, spec.get("use_incumbent", False), main_table,
                        spec.get("unregistered_columns", "unknown"), {n_: known[n_] for n_ in dexpr})

    def inline(e):                                     # 派生特征引用其他派生特征时，内联展开（不依赖 SELECT 内的别名顺序）
        for n_, x_ in dexpr.items():
            e = re.sub(rf"\b{n_}\b", f"({x_})", e)
        return e
    # 哨兵值（如 DAYS_EMPLOYED=365243）→ 缺失 + 标志位；来自 data_dictionary.sentinel_values（只处理主表字段）
    exprs = {}
    for s_ in (x for x in d.get("sentinel_values", []) if x.get("table") in (None, main_table)):
        lit = f"'{s_['value']}'" if isinstance(s_["value"], str) else s_["value"]
        for f in (f for f in s_["fields"] if f in pol["keep"]):
            exprs[f] = (f'CASE WHEN "{f}" = {lit} THEN NULL ELSE "{f}" END', f'("{f}" = {lit})::int AS "{f}__is_sentinel"')
    sel = lambda c: (f'{exprs[c][0]} AS "{c}"' if c in exprs else
                     f'TRY_CAST(({inline(dexpr[c])}) AS DOUBLE) AS "{c}"' if c in dexpr else f'"{c}"')
    select = ", ".join([sel(c) for c in pol["keep"]] + [e[1] for e in exprs.values()])
    pol["keep"] = pol["keep"] + [f"{f}__is_sentinel" for f in exprs]
    return pol, select, list(dwarn)


def _maturity(con, label: dict, spec: dict, sp: dict, cfg: dict, report: dict):
    """成熟度检查（只对 label 由状态字段生成的数据集）：按月统计被排除状态（如 Current）占比。"""
    q = ", ".join("'" + x + "'" for x in label["exclude"])
    m = con.execute(f"SELECT strftime({spec['split_time_expr']}, '%Y-%m') AS m, "
                    f"avg(({label['source_col']} IN ({q}))::int) AS share, count(*) AS n "
                    f"FROM raw WHERE {spec['row_filter']} AND {spec['split_time_expr']} IS NOT NULL GROUP BY 1 ORDER BY 1").df()
    m = m[(m.m >= sp["train_valid"][0]) & (m.m <= sp["holdout"][1])]
    bad = m[m.share > cfg["thresholds"]["maturity_current_share_max"]]
    report["maturity_unfinished_share_by_month"] = dict(zip(m.m, m.share.round(4)))
    if len(bad):
        report["warnings"].append({"code": "DATA_FATAL", "msg": "以下月份未完结状态占比超过阈值，label 成熟度不足",
                                   "months": dict(zip(bad.m, bad.share.round(4)))})


def _write(df: pd.DataFrame, dataset: str, cfg: dict, report: dict):
    art = Path(cfg["paths"]["artifacts_root"])
    (art / "splits" / dataset).mkdir(parents=True, exist_ok=True)
    (art / "holdout" / dataset).mkdir(parents=True, exist_ok=True)
    for s in ("train", "valid", "oot_dev"):
        df[df.split == s].drop(columns="split").to_parquet(art / "splits" / dataset / f"{s}.parquet")
    df[df.split == "holdout"].drop(columns="split").to_parquet(art / "holdout" / dataset / "holdout.parquet")
    Path(cfg["paths"]["reports_root"]).mkdir(exist_ok=True)
    (Path(cfg["paths"]["reports_root"]) / f"profile_{dataset}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))


def build_splits(dataset: str, cfg: dict | None = None, table_path: str | None = None) -> dict:
    cfg = cfg or load_config()
    kh, spec = load_knowhow(dataset, cfg), cfg["task_specs"][dataset]
    d, label, sp = kh["dictionary"], kh["dictionary"]["label"], kh["dictionary"]["suggested_split"]
    path = table_path or (cfg["datasets"][dataset]["tables"].get("main") or cfg["datasets"][dataset]["tables"][label["source_table"]])
    con, src = _open(path, spec)
    cols = [r[0] for r in con.execute("DESCRIBE raw").fetchall()]
    pol, select, warnings = _features(dataset, kh, spec, cols)
    report = {"dataset": dataset, "policy": {k: v for k, v in pol.items() if k != "keep"}, "n_features": len(pol["keep"]), "warnings": warnings}

    time_based = isinstance(sp.get("oot_dev"), list)
    ent = d.get("entity_key")
    tsel = f", {spec['split_time_expr']} AS {SPLIT_T}, {spec['observation_time_expr']} AS {OBS}" if time_based else ""
    base = (f"SELECT {ent if ent in cols else 'row_number() OVER ()'} AS {ID}, {label_sql(label)} AS {LABEL}{tsel}, {select} "
            f"FROM {src} WHERE {spec['row_filter']}")
    if "exclude" in label and time_based:
        _maturity(con, label, spec, sp, cfg, report)
    df = con.execute(f"SELECT * FROM ({base}) WHERE {LABEL} IS NOT NULL").df()
    for c, s in d.get("fields", {}).items():                # label_check_only 字段与 label 的对应关系（如酒店 reservation_status）
        if s.get("role") == "label_check_only":
            ct = con.execute(f"SELECT {c} AS v, {label_sql(label)} AS y, count(*) n FROM raw GROUP BY 1,2 ORDER BY 1,2").df()
            report.setdefault("label_check", {})[c] = ct.astype(str).values.tolist()
    df = _assign_time(df, sp, cfg, report) if time_based else _assign_random(df, sp, cfg)
    report["counts"] = {s: {"n": int((df.split == s).sum()), "bad_rate": round(float(df.loc[df.split == s, LABEL].mean()), 4)}
                        for s in ("train", "valid", "oot_dev", "holdout")}
    tr, oo = df[df.split == "train"], df[df.split == "oot_dev"]
    ms = (tr[pol["keep"]].isna().mean() - oo[pol["keep"]].isna().mean()).abs()
    report["missing_rate_shift"] = ms[ms > cfg["thresholds"]["missing_shift_report"]].round(3).to_dict()
    report["dtype_normalized"] = numpy_types(df)      # 写文件前统一成 numpy 类型：下游只见 float/object/category/datetime
    _write(df, dataset, cfg, report)
    return report


def _assign_time(df, sp, cfg, report):
    t = pd.to_datetime(df[SPLIT_T])
    df["split"] = None
    for name, key in (("train", "train_valid"), ("oot_dev", "oot_dev"), ("holdout", "holdout")):
        lo, hi = _month_range(sp[key])
        df.loc[(t >= lo) & (t < hi), "split"] = name
    df = df[df.split.notna()].copy()
    rng = np.random.default_rng(cfg["split"]["seed"])
    tr = df.index[df.split == "train"]
    df.loc[rng.choice(tr, int(len(tr) * cfg["split"]["valid_frac"]), replace=False), "split"] = "valid"
    return df


def _assign_random(df, sp, cfg):
    rng = np.random.default_rng(cfg["split"]["seed"])
    df["split"] = None
    fr = {k: sp[k] for k in ("train", "valid", "oot_dev", "holdout")}
    for y in (0, 1):  # 分层
        idx = rng.permutation(df.index[df[LABEL] == y])
        cuts = np.cumsum([int(len(idx) * v) for v in fr.values()])
        for name, a, b in zip(fr, np.r_[0, cuts[:-1]], cuts):
            df.loc[idx[a:b], "split"] = name
    return df[df.split.notna()].copy()


if __name__ == "__main__":
    import sys
    for ds in sys.argv[1:] or ["lending_club"]:
        r = build_splits(ds)
        print(json.dumps({k: r[k] for k in ("counts", "warnings", "n_features")}, ensure_ascii=False, indent=1, default=str))
