"""指标注册表：主指标和护栏指标都从这里取。全部越大越好。"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.metrics import average_precision_score

from evaluation.compare import _auc


def ks(y, p):
    from scipy.stats import ks_2samp
    y, p = np.asarray(y), np.asarray(p)
    return float(ks_2samp(p[y == 1], p[y == 0], method="asymp").statistic)     # 只用统计量；asymp 跳过精确 p 值，bootstrap 里快两个数量级


@dataclass(frozen=True)
class Metric:
    fn: Callable
    gap_max: float          # OVERFIT_GAP：valid − OOT-dev 超过即拒
    doc: str


METRICS = {
    "auc": Metric(lambda y, p: float(_auc(np.asarray(y), np.asarray(p))), 0.03, "排序能力，通用"),
    "pr_auc": Metric(lambda y, p: float(average_precision_score(y, p)), 0.05, "正样本稀少时更敏感"),
    "ks": Metric(ks, 0.05, "正负样本分数分布的最大差距"),
}


def primary(cfg) -> str:
    return cfg["metric"]["primary"]


def all_metrics(y, p) -> dict:
    return {k: m.fn(y, p) for k, m in METRICS.items()}


def report_metrics(y, p) -> dict:
    """补充参考指标：只写进报告，不参与选模型、护栏和过拟合判定。
    brier 越小越好（预测概率与实际是否一致）；lift_top10 / recall_top10：分数最高的 10% 里正类浓度是整体的几倍、覆盖了多少正类。"""
    y, p = np.asarray(y, float), np.asarray(p, float)
    k = max(1, int(round(len(y) * 0.1)))
    top = np.argsort(-p, kind="stable")[:k]
    base = y.mean()
    return {"brier": float(np.mean((p - y) ** 2)), "lift_top10": float(y[top].mean() / base) if base else float("nan"),
            "recall_top10": float(y[top].sum() / y.sum()) if y.sum() else float("nan")}
