"""草稿 → 与手写格式相同的 know-how YAML；注册到 cfg 后，后续流程与手写的数据集完全一致。"""
from pathlib import Path

import yaml

from data.knowhow import load_knowhow
from intake.schemas import IntakeDraft


def write(d: IntakeDraft, dataset_id: str, csv_path: str, out_dir: str) -> dict:
    root = Path(out_dir) / "knowhow"
    kd = root / dataset_id
    kd.mkdir(parents=True, exist_ok=True)
    dic = {"dataset": dataset_id,
           # 按取值打标签（与 Lending Club 的写法相同）：正类取值由用户确认，切分时真正用到
           "label": {"source_col": d.label.col, "definition": d.label.definition,
                     "positive": [str(d.label.positive)], "negative": [str(d.label.negative)]},
           "observation_time_col": d.observation_time_expr,
           "suggested_split": d.windows.model_dump(),
           "fields": {k: {"availability": f.availability, "type": f.type, **({"note": f.reason} if f.reason else {})}
                      for k, f in d.fields.items()}}
    bl = {"leakage": {"action": "drop_at_load", "fields": d.leakage},
          "unknown_availability": {"policy": "quarantine", "fields": d.quarantine},
          "meta": {"action": "drop_at_load", "fields": [k for k, f in d.fields.items() if f.availability == "meta"]}}   # 不是特征（如只用于切分的年份）
    texts = {"data_dictionary.yaml": yaml.safe_dump(dic, allow_unicode=True, sort_keys=False),
             "blacklist.yaml": yaml.safe_dump(bl, allow_unicode=True, sort_keys=False)}
    for n, t in texts.items():
        (kd / n).write_text(t)
    ts = {"row_filter": "true", "split_time_expr": d.split_time_expr, "observation_time_expr": d.observation_time_expr,
          "use_incumbent": False, "metric": d.metric.model_dump(),
          **({"csv_null_values": d.csv_null_values} if d.csv_null_values else {})}
    return {"dataset_id": dataset_id, "knowhow_root": str(root), "csv": csv_path, "task_spec": ts, "yaml": texts}


def register(cfg: dict, w: dict) -> dict:
    """把生成的数据集登记进 cfg：know-how 目录、数据文件、task_specs。之后 TaskSpec.from_knowhow 照常使用。"""
    cfg["paths"]["knowhow_root"] = w["knowhow_root"]
    cfg["datasets"][w["dataset_id"]] = {"tables": {"main": w["csv"]}}
    cfg["task_specs"][w["dataset_id"]] = w["task_spec"]
    return cfg


def from_knowhow_dir(dataset: str, knowhow_root: str, task_spec: dict) -> IntakeDraft:
    """把手写的 know-how 读成一份"完美草稿"：用于端到端测试，也说明草稿与手写配置表达的是同一套信息。
    手写配置里 blacklist.meta 的字段（仅用于切分，如 arrival_date_year）在草稿里标为 availability: meta。"""
    kh = load_knowhow(dataset, {"paths": {"knowhow_root": knowhow_root}})
    d, b = kh["dictionary"], kh["blacklist"]
    types = ("numeric", "categorical", "binary", "date")
    meta = set(b.get("meta", {}).get("fields", []))
    return IntakeDraft(label={k: d["label"][k] for k in ("col", "definition", "positive", "negative")},
                       observation_time_expr=task_spec["observation_time_expr"], split_time_expr=task_spec["split_time_expr"],
                       windows={k: d["suggested_split"][k] for k in ("train_valid", "oot_dev", "holdout")},
                       fields={k: {"type": v["type"] if v.get("type") in types else "text",
                                   "availability": "meta" if k in meta else v["availability"]}
                               for k, v in d["fields"].items() if v.get("availability")},
                       leakage=b["leakage"]["fields"],
                       quarantine=b["unknown_availability"]["fields"],
                       metric=task_spec["metric"], csv_null_values=task_spec.get("csv_null_values", []))
