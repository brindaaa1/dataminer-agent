"""PROFILE 阶段一次性执行：读取原始表 → 打 label → 按 know-how 剔除字段 → 四段切分 → 写 artifacts。"""
import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from data.knowhow import column_policy, field_table, load_config, load_knowhow

ID, LABEL, OBS, SPLIT_T = "_row_id", "_label", "_obs_time", "_split_time"


def _label_sql(label: dict) -> str:
    if "source_col" in label:
        q = lambda xs: ", ".join("'" + x.replace("'", "''") + "'" for x in xs)
        c = f'"{label["source_col"]}"::VARCHAR'          # 列名可能带空格；取值按文本比较（0/1 标签写成 '1'、'0'）
        return (f"CASE WHEN {c} IN ({q(label['positive'])}) THEN 1 "
                f"WHEN {c} IN ({q(label['negative'])}) THEN 0 END")
    return label["col"]


def _month_range(pair):
    return pd.Timestamp(pair[0] + "-01"), pd.Timestamp(pair[1] + "-01") + pd.offsets.MonthBegin(1)


def build_splits(dataset: str, cfg: dict | None = None, table_path: str | None = None, leak_back: bool = False, sample_n: int | None = None,
                 suspects_back: bool = False, split_override: dict | None = None) -> dict:
    cfg = cfg or load_config()
    kh, spec = load_knowhow(dataset, cfg), cfg["task_specs"][dataset]
    d = kh["dictionary"]
    path = table_path or (cfg["datasets"][dataset]["tables"].get("main")
                          or cfg["datasets"][dataset]["tables"][d["label"]["source_table"]])
    con = duckdb.connect()
    nulls = spec.get("csv_null_values")               # 数据里代表缺失的字符串（如 "NA"），否则整列会被推断成文本
    nulls_sql = f", nullstr={nulls}" if nulls else ""
    con.execute(f"CREATE VIEW raw AS SELECT * FROM read_csv('{path}', sample_size=-1, header=true{nulls_sql})")
    cols = [r[0] for r in con.execute("DESCRIBE raw").fetchall()]
    main_table = d.get("label", {}).get("source_table")
    # classic_derived：know-how 里的派生特征。可得时间 = 声明值与输入推断值中较弱者（§4.11）
    known, dexpr, dwarn = field_table(kh, main_table, spec.get("derived_expr_overrides"))
    extra = {n_: known[n_] for n_ in dexpr}

    def inline(e):                                     # 派生特征引用其他派生特征时，内联展开（不依赖 SELECT 内的别名顺序）
        for n_, x_ in dexpr.items():
            e = re.sub(rf"\b{n_}\b", f"({x_})", e)
        return e
    pol = column_policy(dataset, cols + list(dexpr), kh, spec.get("use_incumbent", False), leak_back, main_table,
                        spec.get("unregistered_columns", "unknown"), extra, suspects_back)
    # 哨兵值（如 DAYS_EMPLOYED=365243）→ 缺失 + 标志位；来自 data_dictionary.sentinel_values（只处理主表字段）
    sent = [s_ for s_ in d.get("sentinel_values", []) if s_.get("table") in (None, main_table)]
    exprs = {}
    for s_ in sent:
        for f in s_["fields"]:
            if f in pol["keep"]:
                v = s_["value"]
                lit = f"'{v}'" if isinstance(v, str) else v
                exprs[f] = (f'CASE WHEN "{f}" = {lit} THEN NULL ELSE "{f}" END', f'("{f}" = {lit})::int AS "{f}__is_sentinel"')
    flags = [e[1] for e in exprs.values()]
    label = d["label"]
    aliases = spec.get("sql_aliases", {})              # 派生表达式里用到的、由多个原始列拼出的量（如到店日期）
    if aliases:
        con.execute("CREATE VIEW raw_a AS SELECT *, " + ", ".join(f"{v} AS {k}" for k, v in aliases.items()) + " FROM raw")
    src_view = "raw_a" if aliases else "raw"
    report = {"dataset": dataset, "leak_back": leak_back, "policy": {k: v for k, v in pol.items() if k != "keep"},
              "n_features": len(pol["keep"]), "warnings": list(dwarn)}

    sp = {**d["suggested_split"], **(split_override or {})}     # T4：可用不同年份段构造"历史任务/新任务"
    time_based = "oot_dev" in sp and isinstance(sp["oot_dev"], list)
    ent = d.get("entity_key")
    ent_sql = ent if ent in cols else "row_number() OVER ()"
    sel = lambda c: (f'{exprs[c][0]} AS "{c}"' if c in exprs else
                     f'TRY_CAST(({inline(dexpr[c])}) AS DOUBLE) AS "{c}"' if c in dexpr else f'"{c}"')
    feats = ", ".join([sel(c) for c in pol["keep"]] + flags)
    if exprs:
        pol["keep"] = pol["keep"] + [f"{f}__is_sentinel" for f in exprs]
    tsel = f", {spec['split_time_expr']} AS {SPLIT_T}, {spec['observation_time_expr']} AS {OBS}" if time_based else ""
    base = (f"SELECT {ent_sql} AS {ID}, {_label_sql(label)} AS {LABEL}{tsel}, {feats} FROM {src_view} "
            f"WHERE {spec['row_filter']}")

    # 成熟度检查（只对 label 由状态字段生成的数据集）：按月统计被排除状态（如 Current）占比
    if "exclude" in label and time_based:
        q = ", ".join("'" + x + "'" for x in label["exclude"])
        m = con.execute(f"SELECT strftime({spec['split_time_expr']}, '%Y-%m') AS m, "
                        f"avg(({label['source_col']} IN ({q}))::int) AS share, count(*) AS n "
                        f"FROM raw WHERE {spec['row_filter']} AND {spec['split_time_expr']} IS NOT NULL GROUP BY 1 ORDER BY 1").df()
        lo, hi = sp["train_valid"][0], sp["holdout"][1]
        m = m[(m.m >= lo) & (m.m <= hi)]
        bad = m[m.share > cfg["thresholds"]["maturity_current_share_max"]]
        report["maturity_unfinished_share_by_month"] = dict(zip(m.m, m.share.round(4)))
        if len(bad):
            report["warnings"].append({"code": "DATA_FATAL", "msg": "以下月份未完结状态占比超过阈值，label 成熟度不足",
                                       "months": dict(zip(bad.m, bad.share.round(4)))})

    df = con.execute(f"SELECT * FROM ({base}) WHERE {LABEL} IS NOT NULL").df()
    # label_check_only 字段与 label 的对应关系（如酒店 reservation_status）
    for c, s in d.get("fields", {}).items():
        if s.get("role") == "label_check_only":
            ct = con.execute(f"SELECT {c} AS v, {_label_sql(label)} AS y, count(*) n FROM raw GROUP BY 1,2 ORDER BY 1,2").df()
            report.setdefault("label_check", {})[c] = ct.astype(str).values.tolist()
    if time_based:
        df = _assign_time(df, sp, cfg, report)
    else:
        df = _assign_random(df, sp, cfg)
    if sample_n and len(df) > sample_n:              # 开发/演示期抽样：在切分之后，各段按同一比例缩小（最终结果用全量重跑）
        df = df.groupby("split", group_keys=False).apply(
            lambda g: g.sample(frac=sample_n / len(df), random_state=cfg["split"]["seed"]), include_groups=True).reset_index(drop=True)
    report["counts"] = {s: {"n": int((df.split == s).sum()), "bad_rate": round(float(df.loc[df.split == s, LABEL].mean()), 4)}
                        for s in ("train", "valid", "oot_dev", "holdout")}
    tr, oo = df[df.split == "train"], df[df.split == "oot_dev"]
    ms = (tr[pol["keep"]].isna().mean() - oo[pol["keep"]].isna().mean()).abs()
    report["missing_rate_shift"] = ms[ms > cfg["thresholds"]["missing_shift_report"]].round(3).to_dict()

    art = Path(cfg["paths"]["artifacts_root"])
    (art / "splits" / dataset).mkdir(parents=True, exist_ok=True)
    (art / "holdout" / dataset).mkdir(parents=True, exist_ok=True)
    for s in ("train", "valid", "oot_dev"):
        df[df.split == s].drop(columns="split").to_parquet(art / "splits" / dataset / f"{s}.parquet")
    df[df.split == "holdout"].drop(columns="split").to_parquet(art / "holdout" / dataset / "holdout.parquet")
    Path(cfg["paths"]["reports_root"]).mkdir(exist_ok=True)
    (Path(cfg["paths"]["reports_root"]) / f"profile_{dataset}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))
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
