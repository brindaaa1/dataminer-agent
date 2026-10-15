"""试验：只给大模型每列的统计（不给原始行），看能找出多少占位值。对照答案里 kind = sentinel 的条目。
    python eval/trials/llm_placeholders.py deepseek          # 结果文件以 llm_placeholders_deepseek 开头
    python eval/trials/llm_placeholders.py deepseek run3     # 重跑时加个标记，不覆盖之前的结果"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.llm import extract_json, make_llm  # noqa: E402
from data.knowhow import load_config  # noqa: E402
from eval.planted.build import PLANTED  # noqa: E402
from eval.planted.score import score, to_markdown  # noqa: E402

SYSTEM = """你是做建模的数据分析师。下面是几张表每一列的统计，没有原始数据行。
找出看起来是"用一个特殊值代替缺失或未知"的情况：比如一个现实中不可能出现的数值，或者一个表示未知的字符串，并且它出现的次数明显偏多。
只输出 JSON：{"placeholders": [{"table": "表名", "column": "列名", "value": "那个值", "reason": "一句话依据"}]}。没有就输出空列表。"""


def column_stats(s: pd.Series) -> dict:
    vc = s.value_counts(dropna=True)
    out = {"n": int(len(s)), "missing_rate": round(float(s.isna().mean()), 4), "n_unique": int(s.nunique()),
           "top_values": [[str(v), int(c)] for v, c in vc.head(10).items()]}
    if pd.api.types.is_numeric_dtype(s) and s.notna().any():
        q = s.quantile([0.01, 0.5, 0.99])
        out.update(min=float(s.min()), max=float(s.max()), p1=float(q[0.01]), p50=float(q[0.5]), p99=float(q[0.99]),
                   n_at_min=int((s == s.min()).sum()), n_at_max=int((s == s.max()).sum()))
    return out


def main(provider: str = "deepseek", tag: str = ""):
    out = ROOT / "eval" / "trials" / "results" / "_".join(["llm_placeholders", provider] + ([tag] if tag else []))
    llm = make_llm(provider, load_config())
    md = [f"# 大模型只看统计能否找出占位值（{provider}）", ""]
    for d in sorted(p for p in PLANTED.iterdir() if p.is_dir()):
        answers = [a for a in json.loads((d / "answers.json").read_text()) if a["kind"] == "sentinel"]
        tables = sorted({a["table"] for a in json.loads((d / "answers.json").read_text())})
        stats = {t: {c: column_stats(s) for c, s in pd.read_csv(d / f"{t}.csv", low_memory=False).items()} for t in tables}
        text, _ = llm.complete(SYSTEM, "```json\n" + json.dumps(stats, ensure_ascii=False) + "\n```", 0.0)
        Path(f"{out}_{d.name}_raw.txt").write_text(text)
        found = [{**p, "kind": "llm", "source": provider} for p in extract_json(text)["placeholders"]]
        Path(f"{out}_{d.name}.json").write_text(
            json.dumps(found, ensure_ascii=False, indent=1))
        md.append(to_markdown(d.name, score(answers, found)))
    Path(f"{out}.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main(*sys.argv[1:])
