"""代码校验草稿：字段存在、时间表达式在 DuckDB 里实跑、日期字段真能解析成日期、切分窗口有序且每段都有正负样本；
必答项必须有用户确认，且写出的值就是确认过的值。"""
import json

import duckdb
from pydantic import BaseModel

from data.knowhow import domains, load_config
from intake.schemas import REQUIRED, IntakeDraft, Question

CONFIRM = {"label": "标签列、正类取值和业务定义是否正确？",
           "observation_time_expr": "预测时点（观察时间）这样计算是否正确？",
           "windows": "训练、OOT-dev、holdout 的时间窗口这样划分是否合适？",
           "metric": "主指标和护栏指标这样定是否合适？"}
MIN_NONNULL = 0.99


def _con(csv_path: str, nulls: list[str]):
    """数据先读进内存表，再关掉外部访问：LLM 写的表达式只能读这张表，读不到本机其他文件。"""
    con = duckdb.connect()
    ns = f", nullstr={nulls}" if nulls else ""
    con.execute(f"CREATE TABLE raw AS SELECT * FROM read_csv('{csv_path}', sample_size=-1, header=true{ns})")
    con.execute("SET enable_external_access = false")
    return con


def _ratio(con, expr: str) -> float:
    n, ok = con.execute(f"SELECT count(*), count({expr}) FROM raw").fetchone()
    return ok / n


def validate(d: IntakeDraft, csv_path: str) -> list[str]:
    """返回错误列表，空表示通过。错误原样回灌给 LLM 重写。"""
    errs = [f"{k} 未给出" for k in (*REQUIRED, "split_time_expr") if getattr(d, k) is None]
    if errs:
        return errs
    names = {x["name"] for x in domains(load_config())}
    if d.domain not in names:
        errs.append(f"domain {d.domain} 不是可选的领域，只能取 {sorted(names)} 之一")
    con = _con(csv_path, d.csv_null_values)
    cols = {r[0] for r in con.execute("DESCRIBE raw").fetchall()}
    errs += [f"字段 {c} 不在表头中" for c in (d.label.col, *d.fields, *d.leakage, *d.quarantine, *([d.history_by] if d.history_by else [])) if c not in cols]
    for k in ("observation_time_expr", "split_time_expr"):
        try:
            r = _ratio(con, getattr(d, k))
            if r < MIN_NONNULL:
                errs.append(f"{k} 只有 {r:.1%} 的行能算出时间，需要 ≥ {MIN_NONNULL:.0%}")
        except duckdb.Error as e:
            errs.append(f"{k} 在 DuckDB 中执行失败：{str(e)[:200]}")
    # 标为 date 的字段会在建模时按日期解析；解析不了的（如整数年份、英文月份名）会悄悄变成错误的特征
    for f, fd in d.fields.items():
        if fd.type == "date" and f in cols:
            n, ok = con.execute(f'SELECT count("{f}"), count(TRY_CAST("{f}"::VARCHAR AS DATE)) FROM raw').fetchone()
            if n and ok / n < MIN_NONNULL:
                errs.append(f"字段 {f} 标为 date，但只有 {ok / n:.1%} 的值能解析成日期；请改成 numeric 或 categorical")
    w = d.windows
    segs = [w.train_valid, w.oot_dev, w.holdout]
    if any(lo > hi for lo, hi in segs) or not (w.train_valid[1] < w.oot_dev[0] and w.oot_dev[1] < w.holdout[0]):
        errs.append("切分窗口需要按 train_valid、oot_dev、holdout 的时间先后排列，且互不重叠")
    if errs:
        return errs
    for seg, (lo, hi) in w.model_dump().items():
        n_pos, n_neg = con.execute(
            f'SELECT count(*) FILTER ("{d.label.col}"::VARCHAR = ?), count(*) FILTER ("{d.label.col}"::VARCHAR = ?) FROM raw '
            f"WHERE strftime({d.split_time_expr}, '%Y-%m') BETWEEN ? AND ?", [str(d.label.positive), str(d.label.negative), lo, hi]).fetchone()
        if not (n_pos and n_neg):
            errs.append(f"{seg} 窗口 {lo}~{hi} 内正样本 {n_pos} 个、负样本 {n_neg} 个，每段都需要两类样本")
    return errs


def value(d: IntakeDraft, key: str):
    """必答项的可比较值。标签的业务定义只是说明文字，措辞变化不要求重新确认。"""
    v = getattr(d, key)
    v = v.model_dump() if isinstance(v, BaseModel) else v
    return {k: str(x) for k, x in v.items() if k != "definition"} if key == "label" and v else v


def missing_confirmations(d: IntakeDraft, confirmed: dict) -> list[Question]:
    """必答项：LLM 给了推荐值也必须经用户确认；草稿里的值和用户确认过的不一致（没确认过、或之后被改了），
    就由代码补成问题，推荐答案取草稿里的当前值。"""
    def shown(k):
        v = getattr(d, k)
        return json.dumps(v.model_dump() if isinstance(v, BaseModel) else v, ensure_ascii=False)
    return [Question(key=k, question=CONFIRM[k], recommended=shown(k)) for k in REQUIRED
            if k not in confirmed or confirmed[k]["value"] != value(d, k)]
