"""情景用例的公共部分（spec 2026-10-05 §6）：读底座主表、抽样、复制并修改数据字典、写变体文件。只改副本，不动原件。
多表数据（home_credit）不抽样：子表要跟着主表一起过滤，情景只改 label 所在的主表。"""
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml


@dataclass
class Variant:
    tables: dict                       # 数据集登记的表路径（主表换成变体文件）
    knowhow_root: str                  # 修改过的 know-how 副本根目录
    answers: dict = field(default_factory=dict)          # 停下问人时的模拟回答
    prepare: Callable[[dict], None] | None = None        # 运行前对产物目录做的准备（如预先切分再缩小 OOT）


def dictionary(base: str, cfg: dict) -> dict:
    return yaml.safe_load((Path(cfg["paths"]["knowhow_root"]) / base / "data_dictionary.yaml").read_text())


def main_table(base: str, cfg: dict) -> str:
    t = cfg["datasets"][base]["tables"]
    return "main" if "main" in t else dictionary(base, cfg)["label"]["source_table"]


def load_main(base: str, cfg: dict, n_rows: int | None = None, seed: int = 0) -> pd.DataFrame:
    tables = cfg["datasets"][base]["tables"]
    df = pd.read_csv(tables[main_table(base, cfg)], low_memory=False)
    if n_rows and len(tables) == 1 and len(df) > n_rows:
        df = df.sample(n=n_rows, random_state=seed)
    return df


def copy_knowhow(base: str, cfg: dict, out: Path) -> Path:
    src, dst = Path(cfg["paths"]["knowhow_root"]), Path(out) / "knowhow"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src / base, dst / base)
    if (src / "domain_priors.yaml").exists():          # 单调约束方向、模型卡先验都读它
        shutil.copy(src / "domain_priors.yaml", dst / "domain_priors.yaml")
    t = src / "templates" / f"{base}.yaml"
    if t.exists():
        (dst / "templates").mkdir(parents=True, exist_ok=True)
        shutil.copy(t, dst / "templates" / t.name)
    return dst


def edit_dictionary(kh_root: Path, base: str, fn) -> None:
    p = Path(kh_root) / base / "data_dictionary.yaml"
    d = yaml.safe_load(p.read_text())
    fn(d)
    p.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False))


def label_col(label: dict) -> str:
    return label.get("source_col") or label["col"]


def _vals(v) -> list:
    return list(v) if isinstance(v, (list, tuple)) else [v]       # hotel 写 positive: 1，lending_club 写取值列表


def label_mask(df: pd.DataFrame, label: dict) -> pd.Series:
    """有明确正负取值的行（排除 Current 之类）；没写 negative 时全部非空行都算。"""
    if "negative" in label:
        return df[label_col(label)].isin(_vals(label["positive"]) + _vals(label["negative"]))
    return df[label_col(label)].notna()


def label_binary(df: pd.DataFrame, label: dict) -> np.ndarray:
    """正类 = 1。home_credit 的 TARGET 没写 positive，默认取值 1 为正类。"""
    return df[label_col(label)].isin(_vals(label.get("positive", 1))).to_numpy().astype(int)


def write_variant(base: str, cfg: dict, df: pd.DataFrame, out: Path) -> dict:
    tables, m = dict(cfg["datasets"][base]["tables"]), main_table(base, cfg)
    Path(out).mkdir(parents=True, exist_ok=True)
    p = Path(out) / f"{m}.csv"
    df.to_csv(p, index=False)
    tables[m] = str(p)
    return tables


def final_gate(events: list[dict]) -> dict | None:
    return next((e["payload"] for e in events if e["type"] == "final_gate"), None)
