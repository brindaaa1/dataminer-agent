"""run_code：LLM 编写的 Python 代码，作为特征工厂之外的逃生通道（§4.6、README §4.0-4）。

设计取舍（刻意从简，见 docs/DESIGN_NOTES.md）：
- 隔离手段是「子进程 + 超时」，只挂载单个切分的 parquet 文件，不注入 holdout 路径、不给 final_gate 的 token。
  这能防止误用（拿不到 holdout 的引用，也没有 DataAccess 能读到它）；
  但不是操作系统级沙箱（同一用户的子进程理论上仍能用绝对路径读磁盘上的任意文件），
  真正的强隔离需要容器/chroot，超出本项目范围，此处如实说明，不假装做到了。
- 产出**强制**登记（register_feature）并进入筛选漏斗（§4.0-4），且**不管数据集的字段策略如何，
  一律视为可得时间不可信（LEAK_SUSPECT）**：因为它绕过了 know-how 的可得时间标注，
  这是"generate 端放开、verify 端收紧"原则在这里最重的一处落点。只有人工批准（handled_leaks）才能真正被使用。
"""
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from data.access import DataAccess
from features.registry import Registry
from features.screen import screen

ID = "_row_id"
WORKER = Path(__file__).parent / "_run_code_worker.py"


class RunCodeError(RuntimeError):
    pass


def apply_code_to_df(code: str, df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """在子进程中对单个切分执行 compute(df)。返回 [_row_id + 产出列]。"""
    timeout = cfg["run_code"]["timeout_sec"]
    with tempfile.TemporaryDirectory() as td:
        inp, outp, cp = Path(td) / "in.parquet", Path(td) / "out.parquet", Path(td) / "code.py"
        df.to_parquet(inp)
        cp.write_text(code)
        try:
            r = subprocess.run([sys.executable, str(WORKER), str(inp), str(outp), str(cp), ID],
                               capture_output=True, text=True, timeout=timeout, cwd=td)
        except subprocess.TimeoutExpired:
            raise RunCodeError(f"超时（>{timeout}s），已终止")
        if r.returncode != 0:
            raise RunCodeError(f"执行失败：{r.stderr[-1500:]}")
        return pd.read_parquet(outp)


def run_code(code: str, dataset: str, cfg: dict) -> dict:
    """在 train/valid/oot_dev 上执行代码（从不触碰 holdout），登记为特征集并过筛选漏斗。"""
    c = cfg["run_code"]
    da, reg = DataAccess(dataset, cfg["paths"]["artifacts_root"]), Registry(cfg)
    fs_id = "fs_code_" + hashlib.sha1(code.encode()).hexdigest()[:10]
    frames = {s: da.load(s) for s in ("train", "valid", "oot_dev")}
    feats = {s: apply_code_to_df(code, df, cfg) for s, df in frames.items()}
    cols = [x for x in feats["train"].columns if x != ID]
    if not cols:
        raise RunCodeError("compute(df) 没有产出任何列")
    if len(cols) > c["max_features"]:
        raise RunCodeError(f"一次最多产出 {c['max_features']} 个特征，实际 {len(cols)} 个")
    if any(set(feats[s].columns) != set(feats["train"].columns) for s in feats):
        raise RunCodeError("各切分产出的列不一致")
    rep = screen(feats["train"][cols], frames["train"]["_label"].reset_index(drop=True), feats["oot_dev"][cols], cfg)
    for s, f in feats.items():
        p = reg.cache_path(fs_id, s)
        p.parent.mkdir(parents=True, exist_ok=True)
        f.to_parquet(p)
    defs = [{"name": f, "expr": "run_code", "window": "n/a", "prior": None, "template": "run_code",
            "rationale": "由 run_code 生成，可得时间未知，默认按 LEAK_SUSPECT 处理，需人工批准后才可使用",
            "source": "run_code"} for f in cols]
    reg.register(fs_id, dataset, "run_code", "run_code", rep["kept"], defs, {k: rep[k] for k in ("dropped", "leak_suspect", "funnel")}, code=code)
    return {"feature_set_id": fs_id, "kept": rep["kept"], "leak_suspect": rep["leak_suspect"], "funnel": rep["funnel"]}
