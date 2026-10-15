"""试验：Home Credit 只用主表 vs 主表 + Featuretools 汇总的明细表特征，同一切分、同一个默认 LightGBM，比检验集 AUC。
    python eval/trials/agg_gain.py"""
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.knowhow import load_config  # noqa: E402
from evaluation.compare import paired_bootstrap  # noqa: E402
from evaluation.mde import auc_se, mde  # noqa: E402


def fit_predict(Xtr, ytr, Xva, yva, Xte):
    m = lgb.train({"objective": "binary", "metric": "auc", "verbosity": -1, "seed": 0}, lgb.Dataset(Xtr, ytr), 2000,
                  valid_sets=[lgb.Dataset(Xva, yva)], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m.predict(Xte, num_iteration=m.best_iteration)


def main():
    cfg = load_config()
    app = pd.read_csv(ROOT / "data_raw" / "eval" / "home_credit" / "application_train.csv", low_memory=False).set_index("SK_ID_CURR")
    y = app.pop("TARGET")
    app = app.astype({c: "category" for c in app.columns if app[c].dtype == object})
    agg = pd.read_parquet(ROOT / "data_raw" / "eval" / "planted" / "ft_home_credit_clean.parquet").reindex(app.index)
    idx_tr, idx_te = train_test_split(app.index, test_size=0.2, stratify=y, random_state=42)
    idx_tr, idx_va = train_test_split(idx_tr, test_size=0.25, stratify=y[idx_tr], random_state=42)
    preds = {}
    for name, X in (("只用主表", app), ("主表 + 汇总特征", app.join(agg))):
        preds[name] = fit_predict(X.loc[idx_tr], y[idx_tr], X.loc[idx_va], y[idx_va], X.loc[idx_te])
    yte = y[idx_te].to_numpy()
    base, new = preds["只用主表"], preds["主表 + 汇总特征"]
    c = cfg["evaluator"]["mde"]
    m = mde(auc_se(yte, base), alpha=0.05, power=c["power"], auc_corr=c["auc_corr"])
    cmp = paired_bootstrap(yte, new, base, alpha=0.05)
    md = ["# 明细表汇总特征能带来多少提升（Home Credit）", "",
          f"检验集 {len(yte)} 人，其中坏客户 {int(yte.sum())} 人；同一切分、默认 LightGBM。", "",
          "| 特征 | 检验集 AUC |", "|---|---|"] + [f"| {k} | {roc_auc_score(yte, p):.4f} |" for k, p in preds.items()] + [
          "", f"- 差值 {cmp['delta']:+.4f}，90% 区间（下界即单侧 95% 下界）[{cmp['ci_low']:+.4f}, {cmp['ci_high']:+.4f}]",
          f"- 这份检验集能分辨的最小提升约 {m:.4f}",
          "- 参照：现有 agent 在 Home Credit 上用特征模板得到的最终检验 AUC 是 0.769（examples/baselines_home_credit.json，切分不同，只作量级参考）", ""]
    out = ROOT / "eval" / "trials" / "results" / "agg_gain.md"
    out.write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
