"""OOT-dev 能分辨多小的提升（MDE）：DeLong 方差 + 样本量公式。v5 评测 home_credit：OOT-dev 只有 242 个坏样本，
实测只能可靠分辨 ≥0.027 的 AUC 提升，加特征的 +0.0107 不可能显著。"""
import numpy as np
import pytest

from evaluation.mde import auc_se, mde


def scores(n, bad, seed=0):
    r = np.random.default_rng(seed)
    y = (r.random(n) < bad).astype(int)
    return y, y * 0.8 + r.normal(size=n)


def test_auc_se_matches_bootstrap_and_shrinks_with_sample_size():
    from sklearn.metrics import roc_auc_score
    y, p = scores(3000, 0.08)
    r = np.random.default_rng(1)
    boot = [roc_auc_score(y[i], p[i]) for i in (r.integers(0, len(y), len(y)) for _ in range(400))]
    assert auc_se(y, p) == pytest.approx(np.std(boot), rel=0.15)
    y4, p4 = scores(12000, 0.08)
    assert auc_se(y4, p4) == pytest.approx(auc_se(y, p) / 2, rel=0.15)        # 样本量 ×4 → SE 减半


def test_mde_formula():
    """MDE = (z(1−α) + z(power)) × SE × √(2(1−r))：α=0.05 单侧、power 0.8、r=0.82 时系数约 2.49 × 0.6。"""
    out = mde(0.016, alpha=0.05, power=0.8, auc_corr=0.82)
    assert out == pytest.approx((1.6449 + 0.8416) * 0.016 * np.sqrt(2 * 0.18), rel=1e-3)
    assert mde(0.008, alpha=0.05, power=0.8, auc_corr=0.82) == pytest.approx(out / 2)
