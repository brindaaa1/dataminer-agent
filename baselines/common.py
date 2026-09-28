"""B0/B1 共用：与 agent 完全相同的数据、相同的预处理（原始宽表），只是不做特征挖掘/搜索。"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from data.access import DataAccess
from evaluation.final_gate import FinalGate
from features.registry import Registry
from modeling.inner_loop import date_spec, ks
from modeling.prep import prepare


def load_wide(dataset: str, cfg: dict, feature_sets: list[str] | None = None):
    """返回 (Xtr, ytr, Xva, yva, Xoo, yoo, prep)。原始宽表 = 数据层切分出的特征（已按 know-how 剔除泄漏/合规字段）。
    feature_sets 非空时附加特征工厂产出的特征（用于『FLAML + 工厂特征』的公平对照）。"""
    da = DataAccess(dataset, cfg["paths"]["artifacts_root"])
    date_fields, dfmt = date_spec(dataset, cfg)
    reg = Registry(cfg) if feature_sets else None
    fr = {s: da.load(s) for s in ("train", "valid", "oot_dev")}
    if reg:
        fr = {s: reg.attach(f, feature_sets, s) for s, f in fr.items()}
    Xtr, cats = prepare(fr["train"], date_fields, None, dfmt)
    cols = list(Xtr.columns)

    def prep(df, split):
        if reg and split == "holdout":
            df = reg.attach(df, feature_sets, "holdout")
        return prepare(df, date_fields, cats, dfmt)[0][cols]
    return (Xtr, fr["train"]["_label"], prep(fr["valid"], "valid"), fr["valid"]["_label"],
            prep(fr["oot_dev"], "oot_dev"), fr["oot_dev"]["_label"], prep)


def finish(name: str, dataset: str, cfg: dict, predict, prep, Xoo, yoo, wall_sec: float, extra: dict) -> dict:
    """OOT-dev 指标 + 保存预测（供与 agent 做配对 bootstrap）+ holdout 只评一次。"""
    p_oo = predict(Xoo)
    art = Path(cfg["paths"]["artifacts_root"]) / "experiments" / name
    art.mkdir(parents=True, exist_ok=True)
    np.save(art / "oot_dev_pred.npy", p_oo)
    auc = float(roc_auc_score(yoo, p_oo))
    fg = FinalGate(dataset, cfg, name).run_predictor(name, lambda h: predict(prep(h, "holdout")), auc)
    res = {"name": name, "oot_dev_auc": auc, "oot_dev_ks": ks(yoo.values, p_oo), "holdout_auc": fg["holdout_auc"],
           "wall_sec": wall_sec, **extra}
    (art / "result.json").write_text(json.dumps(res, indent=1, default=float))
    return res
