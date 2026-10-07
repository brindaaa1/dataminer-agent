"""数据画像：LLM 只看这份统计，不看原始数据行。"""
import re

import duckdb
import pandas as pd

DATE_NAME = re.compile(r"year|month|day|date|time", re.I)


def _parses_as_date(s: pd.Series) -> bool:
    return s.dtype == object and pd.to_datetime(s.dropna().head(50), errors="coerce", format="mixed").notna().mean() > .9


def profile(csv_path: str) -> dict:
    df = pd.read_csv(csv_path, keep_default_na=False, na_values=["", "NA", "NULL"])
    # 表达式在 DuckDB 里执行，列类型以 DuckDB 读到的为准（如日期字符串会被读成 DATE，不能再 strptime）
    sql = {r[0]: r[1] for r in duckdb.sql(f"DESCRIBE SELECT * FROM read_csv('{csv_path}', sample_size=-1, header=true)").fetchall()}
    cols = {}
    for c in df.columns:
        s = df[c]
        info = {"sql_type": sql[c], "missing_rate": round(float(s.isna().mean()), 4), "n_unique": int(s.nunique()),
                "examples": [str(v) for v in s.dropna().unique()[:5]]}
        if pd.api.types.is_numeric_dtype(s):
            info.update(min=float(s.min()), max=float(s.max()))
        cols[c] = info
    return {"n_rows": len(df), "columns": cols,
            "date_candidates": [c for c in df.columns if DATE_NAME.search(c) or _parses_as_date(df[c])],
            "label_candidates": [c for c in df.columns if df[c].nunique() == 2 and df[c].dropna().isin([0, 1]).all()]}
