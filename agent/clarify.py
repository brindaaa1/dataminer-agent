"""前置检查缺必答项（label 定义、观察时间、OOT 窗口）时怎么问人：LLM 根据数据字典和表头起草推荐答案，代码校验取值；
人确认才写进 spec，不同意就由 LLM 把人的原话解析成取值，再请人确认一次。约定与接入阶段（intake/session.py）一致。"""
import fnmatch
import json
from pathlib import Path

import pandas as pd

from data.knowhow import header
from intake.session import ACCEPT

SYSTEM = (Path(__file__).parent / "prompts" / "clarify.md").read_text()


def columns(dataset: str, cfg: dict) -> list[str]:
    """原始表的全部列名（多表合并）；没登记原始表时用已有切分的列。"""
    tables = cfg.get("datasets", {}).get(dataset, {}).get("tables") or {}
    cols = [c for p in tables.values() for c in (header(p) if str(p).endswith(".csv") else pd.read_parquet(p).columns)]
    if not cols:
        tr = Path(cfg["paths"]["artifacts_root"]) / "splits" / dataset / "train.parquet"
        cols = list(pd.read_parquet(tr).columns) if tr.exists() else []
    return list(dict.fromkeys(cols))


def context(dataset: str, cfg: dict, keys: list[str], replies: dict | None = None) -> dict:
    root = Path(cfg["paths"]["knowhow_root"]) / dataset / "data_dictionary.yaml"
    return {"mode": "CLARIFY", "missing": keys, "user_replies": replies or {},
            "data_dictionary": root.read_text()[:12000] if root.exists() else "", "columns": columns(dataset, cfg)}


def validate(key: str, value, cols: list[str]) -> list[str]:
    if not isinstance(value, dict):
        return [f"{key}.value 必须是对象"]
    if key == "label_def":
        return [] if str(value.get("definition") or "").strip() else ["label_def.value.definition 不能为空"]
    if key == "oot_windows":
        return [] if value.get("oot_dev") is not None else ["oot_windows.value 至少要有 oot_dev"]
    if key == "observation_time_col":
        if ("col" in value) == ("relative_to" in value):
            return ["observation_time_col.value 只能二选一：{col} 或 {relative_to, offset_cols}"]
        if "col" in value:
            return [] if value["col"] in cols else [f"列 {value['col']} 不存在"]
        pats = value.get("offset_cols") or []
        bad = [p for p in pats if not fnmatch.filter(cols, p)]
        return [f"offset_cols 里 {bad} 匹配不到任何列"] if bad or not pats else []
    return [f"未知的必答项 {key}"]


def apply(spec: dict, key: str, value: dict) -> dict:
    """确认过的取值写进 spec（dict）。"""
    if key == "label_def":
        return {**spec, "label_def": value["definition"]}
    if key == "oot_windows":
        return {**spec, "oot_windows": value}
    if "col" in value:
        return {**spec, "observation_time_col": value["col"], "observation_time_relative": None}
    return {**spec, "observation_time_col": None, "observation_time_relative": value}


def accepted(reply: str, proposal: dict | None) -> bool:
    return bool(proposal and proposal.get("value") is not None
                and (reply.strip() in ACCEPT or reply == proposal.get("recommended")))


def draft(llm_json, dataset: str, cfg: dict, keys: list[str], replies: dict | None = None, max_retries: int = 2) -> dict:
    """返回 {key: {question, recommended, reason, value}}。校验不过的项回灌重写（过了的保留）；仍不过就只问问题、不给推荐。"""
    cols, good, todo, errs = columns(dataset, cfg), {}, list(keys), {}
    for _ in range(max_retries + 1):
        ctx = context(dataset, cfg, todo, {k: v for k, v in (replies or {}).items() if k in todo})
        if errs:
            ctx["feedback"] = [e for es in errs.values() for e in es]
        try:
            out = (llm_json(SYSTEM, "```json\n" + json.dumps(ctx, ensure_ascii=False) + "\n```") or {}).get("proposals") or {}
        except ValueError as e:                 # 格式错误也回灌重写，与接入阶段一致（v5 评测 home_credit s0：多了一个右括号）
            errs = {k: [f"输出不是合法 JSON：{str(e)[:200]}"] for k in todo}
            continue
        errs = {k: validate(k, (out.get(k) or {}).get("value"), cols) if k in out else [f"缺少 {k} 的推荐"] for k in todo}
        good |= {k: {f: out[k].get(f) for f in ("question", "recommended", "reason", "value")} for k in todo if not errs[k]}
        todo = [k for k in todo if errs[k]]
        if not todo:
            return good
    return good | {k: {"question": f"请提供 {k}", "recommended": None, "reason": "推荐答案未通过校验：" + "；".join(errs[k]), "value": None}
                   for k in todo}
