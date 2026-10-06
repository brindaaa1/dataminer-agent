"""特征工厂：把 knowhow/templates/<dataset>.yaml 中的模板展开成 DuckDB SQL，按 point-in-time 规则聚合（§4.11）。
不含任何数据集特有的字段名：表、字段、窗口、聚合全部来自模板和数据字典。"""
import re
from pathlib import Path

import duckdb
import pandas as pd
import yaml

ID = "_row_id"


def load_templates(dataset: str, cfg: dict) -> dict:
    p = Path(cfg["paths"]["knowhow_root"]) / "templates" / f"{dataset}.yaml"
    return {t["id"]: t for t in yaml.safe_load(open(p))["templates"]}


def supported(t: dict) -> bool:
    """多级聚合 / 跨表关联模板还没实现，不给 LLM 选（v5 评测 home_credit：选了就 NotImplementedError）。"""
    return not (t.get("via") or t.get("join_with"))


def parse_agg(a: str) -> dict | None:
    """把模板里的聚合写法规整成 DuckDB 表达式。无法解析（散文描述）返回 None。"""
    expr, _, comment = a.partition("#")
    expr = expr.strip()
    m = re.search(r"monotone_prior:\s*(-?\d+)", comment)
    prior = int(m.group(1)) if m else None
    if expr == "count":
        expr = "count(*)"
    elif (m2 := re.fullmatch(r"count_distinct\((.+)\)", expr)):
        expr = f"count(DISTINCT {m2.group(1)})"
    elif (m2 := re.fullmatch(r"avg\((.+)\)", expr)):
        expr = f"avg(CAST(({m2.group(1)}) AS DOUBLE))"      # avg(布尔表达式) 在 DuckDB 中需要显式转换
    if re.match(r"^[A-Za-z]+( [A-Za-z]+){2,}", expr):        # 以 ≥3 个英文单词开头 → 散文描述，不是 SQL
        return None
    return {"expr": expr, "prior": prior, "raw": a}


def _sentinel_select(table: str, dictionary: dict, pq: str) -> str:
    """来源表的哨兵值 → NULL（data_dictionary.sentinel_values 中指定了 table 的条目）。"""
    reps = []
    for s in dictionary.get("sentinel_values", []):
        if s.get("table") == table:
            v = f"'{s['value']}'" if isinstance(s["value"], str) else s["value"]
            reps += [f'CASE WHEN "{f}" = {v} THEN NULL ELSE "{f}" END AS "{f}"' for f in s["fields"]]
    return f"SELECT * REPLACE ({', '.join(reps)}) FROM '{pq}'" if reps else f"SELECT * FROM '{pq}'"


def _pq_path(dataset, table, cfg):
    root = Path(cfg["paths"]["artifacts_root"]) / f"{dataset}_parquet"
    root.mkdir(parents=True, exist_ok=True)
    pq = root / f"{table}.parquet"
    if not pq.exists():
        csv = cfg["datasets"][dataset]["tables"][table]
        duckdb.connect().execute(f"COPY (SELECT * FROM read_csv('{csv}', sample_size=-1)) TO '{pq}' (FORMAT parquet)")
    return str(pq)


def compute(template_id: str, dataset: str, cfg: dict, ids: pd.Series, dictionary: dict | None = None):
    """返回 (DataFrame[_row_id + 特征列], 特征定义列表)。ids：要计算的实体 id。"""
    from data.knowhow import load_knowhow
    dictionary = dictionary or load_knowhow(dataset, cfg)["dictionary"]
    t = load_templates(dataset, cfg)[template_id]
    if not supported(t):
        raise NotImplementedError(f"{template_id}: 多级聚合/跨表关联模板尚未支持")
    extra = cfg["task_specs"][dataset].get("pit_extra_filters", {}).get(t["source"])
    ent, unit, tf = t["entity"], t.get("time_unit"), t.get("time_field")
    aggs = [a for a in (parse_agg(x) for x in t["aggs"]) if a]
    windows = t.get("windows", ["current"])
    con = duckdb.connect()
    con.register("ids", pd.DataFrame({ent: ids.values}))
    src = _sentinel_select(t["source"], dictionary, _pq_path(dataset, t["source"], cfg))
    defs, cols = [], []
    if windows == ["current"]:                                   # 行级表达式（如申请表上的比例特征）
        sel = []
        for i, a in enumerate(aggs):
            name = f"{template_id}__a{i}"
            sel.append(f"({a['expr']}) AS {name}")
            defs.append({"name": name, "expr": a["expr"], "window": "current", "prior": a["prior"] if a["prior"] is not None else t.get("monotone_prior")})
        df = con.execute(f"SELECT {ent} AS {ID}, {', '.join(sel)} FROM ({src}) s WHERE {ent} IN (SELECT {ent} FROM ids)").df()
    else:
        pit = {"days": 0, "months": -1}[unit]                    # PIT：事件时间必须在观察点之前
        base = f"{tf} <= {pit}" + (f" AND {extra}" if extra else "") + (f" AND {t['filter']}" if t.get("filter") else "")
        sel = []
        for i, a in enumerate(aggs):
            for w in windows:
                name = f"{template_id}__a{i}__w{w}"
                flt = "" if w == "all" else f" FILTER (WHERE {tf} >= -{w})"
                sel.append(f"{a['expr']}{flt} AS {name}")
                defs.append({"name": name, "expr": a["expr"], "window": w, "prior": a["prior"] if a["prior"] is not None else t.get("monotone_prior")})
        for j, dv in enumerate(t.get("derived", [])):
            if " where " in dv:                                  # 依赖具体取值的散文写法，不支持
                continue
            if (m := re.fullmatch(r"recency:\s*(.+)", dv)):
                name = f"{template_id}__recency{j}"
                sel.append(f"{m.group(1)} AS {name}")
                defs.append({"name": name, "expr": m.group(1), "window": "all", "prior": t.get("monotone_prior")})
        inner = con.execute(
            f"SELECT {ent} AS {ID}, {', '.join(sel)} FROM ({src}) s WHERE {ent} IN (SELECT {ent} FROM ids) AND {base} GROUP BY {ent}").df()
        out = {ID: inner[ID]}
        for d_ in defs:
            out[d_["name"]] = inner[d_["name"]]
        wset = {str(w) for w in windows}
        for dv in t.get("derived", []):
            m = re.fullmatch(r"(ratio|trend)\((\w+)\s*(?:/|vs)\s*(\w+)\)", dv)
            if not m or not {m.group(2), m.group(3)} <= wset:
                continue
            kind, a_, b_ = m.groups()
            for i, a in enumerate(aggs):
                x, y = inner[f"{template_id}__a{i}__w{a_}"], inner[f"{template_id}__a{i}__w{b_}"]
                name = f"{template_id}__a{i}__{kind}_{a_}_{b_}"
                out[name] = x / y.where(y != 0) if kind == "ratio" else x - y
                defs.append({"name": name, "expr": f"{kind}({a_},{b_}) of {a['expr']}", "window": f"{a_}/{b_}", "prior": None})
        df = pd.DataFrame(out)
    full = pd.DataFrame({ID: ids.values}).merge(df, on=ID, how="left")     # 没有历史记录的实体 → 缺失
    for d_ in defs:
        d_.update(template=template_id, rationale=t.get("rationale", ""), source=t["source"])
    return full, defs
