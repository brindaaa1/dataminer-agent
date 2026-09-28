"""generate_features 工具：模板 ID → 实例化（train/valid/oot_dev）→ 筛选漏斗 → 注册，返回 feature_set_id（§4.6）。
只使用 train/valid/oot_dev 的实体 id；holdout 的特征由 final_gate 在验收时通过 registry.attach 现算。"""
import pandas as pd

from data.access import DataAccess
from features import factory
from features.registry import Registry
from features.screen import screen

ID = "_row_id"


def generate_features(template_id: str, dataset: str, cfg: dict) -> dict:
    da, reg = DataAccess(dataset, cfg["paths"]["artifacts_root"]), Registry(cfg)
    fs_id = reg.fs_id(dataset, template_id, cfg["paths"]["artifacts_root"])
    frames = {s: da.load(s)[[ID, "_label"]] for s in ("train", "valid", "oot_dev")}
    feats, defs = {}, None
    for s, f in frames.items():
        feats[s], defs = factory.compute(template_id, dataset, cfg, f[ID])
        p = reg.cache_path(fs_id, s)
        p.parent.mkdir(parents=True, exist_ok=True)
        feats[s].to_parquet(p)
    cols = [c for c in feats["train"].columns if c != ID]
    rep = screen(feats["train"][cols], frames["train"]["_label"].reset_index(drop=True), feats["oot_dev"][cols], cfg)
    reg.register(fs_id, dataset, template_id, "template", rep["kept"], defs, {k: rep[k] for k in ("dropped", "leak_suspect", "funnel")})
    return {"feature_set_id": fs_id, "template_id": template_id, "kept": rep["kept"], "leak_suspect": rep["leak_suspect"],
            "funnel": rep["funnel"], "dropped": rep["dropped"], "metrics": rep["metrics"]}
