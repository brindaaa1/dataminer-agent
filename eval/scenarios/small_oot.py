"""情景 c：OOT 太小。预先切分，再把 OOT-dev 抽到指定的坏样本数（默认 45 个，坏样本率保持不变）。
通过条件：PROFILE 发出 MDE 警告（underpowered）；警告进入决策上下文；没有非首个实验在不显著的情况下被采纳。"""
from pathlib import Path

import pandas as pd

from eval.scenarios.common import Variant, copy_knowhow, load_main, write_variant


def build(base, cfg, out, params):
    kh = copy_knowhow(base, cfg, out)
    df = load_main(base, cfg, params.get("n_rows"), params.get("seed", 0))
    n_pos, seed = params.get("n_pos", 45), params.get("seed", 0)

    def prepare(run_cfg):
        from data.splits import build_splits
        build_splits(base, run_cfg)
        p = Path(run_cfg["paths"]["artifacts_root"]) / "splits" / base / "oot_dev.parquet"
        oo = pd.read_parquet(p)
        pos, neg = oo[oo._label == 1], oo[oo._label == 0]
        k = min(n_pos, len(pos))
        m = min(len(neg), round(k * len(neg) / max(len(pos), 1)))
        pd.concat([pos.sample(k, random_state=seed), neg.sample(m, random_state=seed)]).sort_index().to_parquet(p)
    return Variant(write_variant(base, cfg, df, out), str(kh), prepare=prepare)


def check(events, records, params):
    pw = next((e for e in events if e["type"] == "oot_power"), None)
    if not pw or pw["payload"].get("underpowered") is not True:
        return {"passed": False, "reason": "OOT-dev 很小却没有发出 MDE 警告", "evidence": [pw["id"]] if pw else []}
    if not any(e["type"] == "llm_call" and e["payload"].get("stage") == "DECIDE" and '"underpowered": true' in (e["payload"].get("prompt") or "")
               for e in events):
        return {"passed": False, "reason": "MDE 警告没有进入决策上下文", "evidence": [pw["id"]]}
    bad = [e["exp_id"] for e in events if e["type"] == "evaluated" and e["payload"].get("verdict") == "ACCEPT"
           and e["payload"].get("compare") and not e["payload"]["compare"].get("significant")]
    if bad:
        return {"passed": False, "reason": "有实验在不显著的情况下被采纳", "evidence": bad}
    return {"passed": True, "reason": f"MDE {pw['payload'].get('mde')} 超过目标，已警告并进入决策上下文；没有不显著的采纳",
            "evidence": [pw["id"]]}
