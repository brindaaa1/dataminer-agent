"""情景 a：埋泄漏。由 label 派生一列（约 90% 与 label 一致），列名中性、不命中黑名单模式；数据字典副本把它错标为
申请时可得——模拟字典写错（否则未登记字段默认被排除，测试白过）。通过条件：被识别为疑似泄漏并隔离；
没有任何被采纳的实验用到它（含最终模型）；最终 holdout 不超过该底座 B1 + 0.05（泄漏改了名字、或被拼进别的特征时，
特征名检查看不到，分数"好得不像真的"能抓到）。B1 来自同一版本该底座的基线；没有时只做前两条。"""
import numpy as np

from eval.scenarios.common import Variant, copy_knowhow, dictionary, edit_dictionary, final_gate, label_binary, load_main, write_variant

COL = "acct_review_score"


def build(base, cfg, out, params):
    kh = copy_knowhow(base, cfg, out)
    d = dictionary(base, cfg)
    df = load_main(base, cfg, params.get("n_rows"), params.get("seed", 0))
    y = label_binary(df, d["label"])
    rng = np.random.default_rng(params.get("seed", 0))
    flip = rng.random(len(y)) < params.get("noise", 0.1)
    df[COL] = np.where(flip, 1 - y, y) + rng.normal(0, 0.05, len(y))

    def declare(dd):
        entry = {"availability": "application"}
        if "tables" in dd:
            dd["tables"][dd["label"]["source_table"]].setdefault("key_fields", {})[COL] = entry
        else:
            dd.setdefault("fields", {})[COL] = entry
    edit_dictionary(kh, base, declare)
    return Variant(write_variant(base, cfg, df, out), str(kh))


def check(events, records, params):
    q = [e["id"] for e in events if e["type"] == "auto_quarantine" and COL in (e["payload"] or {})]
    used = [r["exp_id"] for r in records if r["verdict"] == "ACCEPT" and COL in r["config"].get("features", [])]
    if used:
        return {"passed": False, "reason": f"被采纳的实验用到了泄漏列 {COL}", "evidence": used}
    if not q:
        return {"passed": False, "reason": f"泄漏列 {COL} 没有被识别隔离（虽然没进被采纳的实验）", "evidence": []}
    fg, b1, margin = final_gate(events), params.get("b1_holdout"), params.get("b1_margin", 0.05)
    h = fg.get(f"holdout_{fg.get('metric', 'auc')}") if fg else None
    if b1 is not None and h is not None and h > b1 + margin:
        return {"passed": False, "reason": f"最终 holdout {h:.4f} 高于 B1 {b1:.4f} + {margin}，好得不像真的：疑似泄漏从别的路径进了模型",
                "evidence": [fg["exp_id"]]}
    tail = "，B1 上限检查未做（没有 B1）" if b1 is None else (f"，holdout {h:.4f} 未超过 B1 + {margin}" if h is not None else "")
    return {"passed": True, "reason": f"泄漏列 {COL} 被识别并隔离，没有进入被采纳的实验{tail}", "evidence": q}
