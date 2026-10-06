"""OOT-dev 能可靠分辨的最小主指标提升（MDE）。v5 评测 home_credit：OOT-dev 只有 242 个坏样本，加特征 +0.0107 的 AUC 提升
置信区间是 [−0.007, 0.029]，怎么调规则都过不了；瓶颈是数据量。PROFILE 时算出来，超过目标值就警告。

AUC 的标准误用 DeLong et al. (1988) 的 placement values；两个候选在同一 OOT 上比较，差的标准误 = SE × √(2(1−r))，
r 是两个 AUC 估计的相关系数（config.evaluator.mde.auc_corr）。MDE = (z(1−α) + z(power)) × SE_Δ，SE 随坏样本数按 1/√n 缩小。"""
import numpy as np
import pandas as pd
from scipy.stats import norm


def auc_se(y, p) -> float:
    y, p = np.asarray(y).astype(bool), np.asarray(p, dtype=float)
    pos, neg = p[y], p[~y]
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    v10 = (r[:len(pos)] - pd.Series(pos).rank().to_numpy()) / len(neg)            # 每个正样本排在多少负样本之上
    v01 = 1 - (r[len(pos):] - pd.Series(neg).rank().to_numpy()) / len(pos)
    return float(np.sqrt(v10.var(ddof=1) / len(pos) + v01.var(ddof=1) / len(neg)))


def mde(se: float, *, alpha: float, power: float, auc_corr: float) -> float:
    return float((norm.ppf(1 - alpha) + norm.ppf(power)) * se * np.sqrt(2 * (1 - auc_corr)))


def oot_power(y, p, cfg: dict) -> dict:
    """用参照模型在 OOT-dev 上的预测估算 MDE（α 取第一次比较的 alpha_0；之后每多比一次 α 收紧，MDE 只会更大）。"""
    c = cfg["evaluator"]["mde"]
    se = auc_se(y, p)
    m = mde(se, alpha=cfg["oot_budget"]["alpha_0"], power=c["power"], auc_corr=c["auc_corr"])
    n_pos = int(np.asarray(y).sum())
    out = {"n_oot": int(len(y)), "n_pos": n_pos, "auc_se": round(se, 5), "mde": round(m, 4), "target": c["target"],
           "underpowered": m > c["target"]}
    if out["underpowered"]:
        need = int(np.ceil(n_pos * (m / c["target"]) ** 2))
        out["warning"] = (f"OOT-dev 只有 {n_pos} 个正样本，只能可靠分辨 ≥{m:.4f} 的 AUC 提升（目标 {c['target']}）；"
                          f"小于这个量级的改进大多判不出显著。要达到目标约需 {need} 个正样本：加数据或调大 OOT-dev 比例")
    return out
