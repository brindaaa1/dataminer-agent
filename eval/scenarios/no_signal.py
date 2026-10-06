"""情景 d：没有信号。把 label 随机打乱，特征和 label 不再相关。通过条件：不制造虚假提升——最终 holdout 主指标与随机水平之差
不超过容差（默认 0.05），或者没有最终模型。第一个全量模型没有可比对象会被自动采纳，所以不要求"没有最终模型"。"""
import numpy as np

from eval.scenarios.common import Variant, copy_knowhow, dictionary, final_gate, label_col, label_mask, load_main, write_variant

CHANCE = {"auc": 0.5, "ks": 0.0}


def build(base, cfg, out, params):
    kh = copy_knowhow(base, cfg, out)
    label = dictionary(base, cfg)["label"]
    df = load_main(base, cfg, params.get("n_rows"), params.get("seed", 0))
    m, col = label_mask(df, label), label_col(label)
    df.loc[m, col] = np.random.default_rng(params.get("seed", 0)).permutation(df.loc[m, col].to_numpy())
    return Variant(write_variant(base, cfg, df, out), str(kh))


def check(events, records, params):
    fg = final_gate(events)
    if not fg:
        return {"passed": True, "reason": "没有最终模型", "evidence": []}
    pm = fg.get("metric", "auc")
    h, tol = fg.get(f"holdout_{pm}"), params.get("tolerance", 0.05)
    gap = h - CHANCE[pm]
    ok = gap <= tol
    return {"passed": ok, "reason": f"最终 holdout {pm}={h:.4f}，比随机水平高 {gap:.4f}" + ("" if ok else f"，超过容差 {tol}：虚假提升"),
            "evidence": [fg["exp_id"]]}
