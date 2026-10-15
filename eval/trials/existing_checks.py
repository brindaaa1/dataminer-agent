"""试验：Deepchecks 数据完整性检查、ydata-profiling 告警，能直接找出测试数据里多少问题。
用 .venvs/checks 运行：.venvs/checks/bin/python eval/trials/existing_checks.py"""
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eval.planted.build import LABELS, PLANTED  # noqa: E402
from eval.planted.score import score, to_markdown  # noqa: E402

OUT = ROOT / "eval" / "trials" / "results"


def deepchecks_findings(table: str, df: pd.DataFrame, label: str | None) -> list[dict]:
    from deepchecks.tabular import Dataset
    from deepchecks.tabular.suites import data_integrity
    cats = [c for c in df.columns if df[c].dtype == object and c != label]
    res = data_integrity().run(Dataset(df, label=label, cat_features=cats))
    out = []
    for r in res.get_not_passed_checks():
        details = " ".join(c.details for c in r.conditions_results if not c.is_pass())
        cols = [c for c in df.columns if re.search(rf"['\"]{re.escape(c)}['\"]", details)]
        out += [{"table": table, "column": c, "kind": r.get_header(), "source": "deepchecks"} for c in cols] or \
               [{"table": table, "column": None, "kind": r.get_header(), "source": "deepchecks"}]
    return out


def ydata_findings(table: str, df: pd.DataFrame) -> list[dict]:
    from ydata_profiling import ProfileReport
    alerts = ProfileReport(df, minimal=True, progress_bar=False).description_set.alerts
    return [{"table": table, "column": a.column_name or None, "kind": a.alert_type.name, "source": "ydata"} for a in alerts]


def main():
    md = ["# Deepchecks、ydata-profiling 能直接找出多少数据问题", ""]
    for name, (main_table, label) in LABELS.items():
        d = PLANTED / name
        answers = json.loads((d / "answers.json").read_text())
        tables = sorted({a["table"] for a in answers} | {main_table})
        frames = {t: pd.read_csv(d / f"{t}.csv", low_memory=False) for t in tables}
        for tool in ("deepchecks", "ydata"):
            t0, findings = time.time(), []
            for t, df in frames.items():
                findings += deepchecks_findings(t, df, label if t == main_table else None) if tool == "deepchecks" else ydata_findings(t, df)
            (OUT / f"existing_checks_{name}_{tool}.json").write_text(json.dumps(findings, ensure_ascii=False, indent=1))
            md.append(to_markdown(f"{name} · {tool}（{time.time() - t0:.0f} 秒）", score(answers, findings)))
    (OUT / "existing_checks.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main()
