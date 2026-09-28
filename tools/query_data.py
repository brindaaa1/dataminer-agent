"""INVESTIGATE 用的只读查询：只访问 train 切分，只返回聚合值，有行数上限（§4.6）。"""
from data.access import DataAccess
from evaluation.guardrails import single_feature_auc

QUESTIONS = ("single_feature_auc", "missing_rate")
MAX_ROWS, TOP = 100_000, 10


def investigate(dataset: str, cfg: dict, question: str, features: list[str]) -> dict:
    df = DataAccess(dataset, cfg["paths"]["artifacts_root"]).load("train")
    df = df.sample(min(len(df), MAX_ROWS), random_state=0)
    if question == "single_feature_auc":
        v = {f: round(single_feature_auc(df[f], df["_label"]), 4) for f in features if f in df}
    elif question == "missing_rate":
        v = {f: round(float(df[f].isna().mean()), 4) for f in features if f in df}
    else:
        raise ValueError(question)
    return dict(sorted(v.items(), key=lambda x: -x[1])[:TOP])
