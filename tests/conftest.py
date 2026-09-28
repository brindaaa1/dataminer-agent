import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from agent.schemas import TaskSpec
from data.knowhow import load_config


def make_toy(root: Path, leak: bool = False, seed: int = 0):
    """合成数据集 + know-how + 缩小的 config。返回 cfg。"""
    rng = np.random.default_rng(seed)

    def part(n, start, ho=False):
        x1, x2 = rng.normal(size=n), rng.normal(size=n)
        cat = rng.choice(["a", "b", "c"], n)
        y = (rng.random(n) < 1 / (1 + np.exp(-(0.8 * x1 - 0.5 * x2 + 0.5 * (cat == "a") - 1.5)))).astype(int)
        t = pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, 90, n), unit="D")
        x2[rng.random(n) < 0.1] = np.nan
        df = pd.DataFrame({"_row_id": np.arange(n).astype(str), "_label": y, "_split_time": t, "_obs_time": t,
                           "x1": x1, "x2": x2, "cat": cat,
                           "opened": [(tt - pd.Timedelta(days=int(k))).strftime("%b-%Y") for tt, k in zip(t, rng.integers(100, 3000, n))]})
        if leak:
            df["leaky"] = y + rng.normal(scale=0.3, size=n)
        return df

    for sub, s, n, st in (("splits", "train", 3000, "2012-01-01"), ("splits", "valid", 1500, "2012-01-01"),
                          ("splits", "oot_dev", 1500, "2013-01-01"), ("holdout", "holdout", 1500, "2013-07-01")):
        d = root / "art" / sub / "toy"
        d.mkdir(parents=True, exist_ok=True)
        part(n, st).to_parquet(d / f"{s}.parquet")
    kh = root / "kh" / "toy"
    kh.mkdir(parents=True, exist_ok=True)
    fields = "  x1: {availability: application}\n  x2: {availability: application}\n  cat: {availability: application}\n  opened: {availability: application, type: date}\n"
    if leak:
        fields += "  leaky: {availability: application}\n"
    (kh / "data_dictionary.yaml").write_text("dataset: toy\nfields:\n" + fields)
    (kh / "blacklist.yaml").write_text("leakage_patterns: ['pymnt']\n")
    cfg = load_config()
    cfg["paths"] = {"knowhow_root": str(root / "kh"), "artifacts_root": str(root / "art"), "reports_root": str(root / "rep")}
    cfg["memory"]["db"] = str(root / "memory.db")          # 跨任务记忆库也放临时目录，否则测试会往真实的 artifacts/memory.db 写玩具任务
    cfg["fidelity"] = {"low": {"sample_frac": 1.0, "n_trials": 3, "early_stopping_rounds": 5},
                       "full": {"sample_frac": 1.0, "n_trials": 4, "early_stopping_rounds": 5}}
    cfg["inner_loop"].update(n_threads=2, n_rounds_max=40, report_every=10, pruner_startup_trials=2)
    cfg["evaluator"]["bootstrap_n"] = 100
    cfg["run"]["max_rounds"] = 6
    return cfg


def toy_spec(cfg, **kw):
    return TaskSpec(dataset="toy", label_def="toy label", label_col="y", observation_time_col="t",
                    oot_windows={"oot_dev": ["2013-01", "2013-06"], "holdout": ["2013-07", "2013-12"]},
                    budget=cfg["run"]["budget"], **kw)


@pytest.fixture
def toy_cfg(tmp_path):
    return make_toy(tmp_path)
