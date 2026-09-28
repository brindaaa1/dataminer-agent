"""Model zoo：注册表 + 统一接口。每个模型：默认搜索空间、fit(trial_params)、predict_proba。"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ZOO: dict = {}


def register_model(name):
    def deco(cls):
        cls.name = name
        ZOO[name] = cls
        return cls
    return deco


def sample_params(trial, space: dict) -> dict:
    """space: {param: {low, high, log?, type?}} 或 {param: {choices: [...]}}"""
    out = {}
    for k, s in space.items():
        if "choices" in s:
            out[k] = trial.suggest_categorical(k, s["choices"])
        elif s.get("type") == "int":
            out[k] = trial.suggest_int(k, s["low"], s["high"], log=s.get("log", False))
        else:
            out[k] = trial.suggest_float(k, s["low"], s["high"], log=s.get("log", False))
    return out


@register_model("lgbm")
class LGBMSpec:
    preprocessing = ["native_missing", "categorical_native"]
    supports = {"monotone_constraints": True, "shap": True}
    default_space = {
        "learning_rate": {"low": 0.01, "high": 0.2, "log": True},
        "num_leaves": {"low": 8, "high": 128, "type": "int", "log": True},
        "min_child_samples": {"low": 20, "high": 500, "type": "int", "log": True},
        "feature_fraction": {"low": 0.5, "high": 1.0},
        "bagging_fraction": {"low": 0.5, "high": 1.0},
        "lambda_l2": {"low": 1e-3, "high": 100.0, "log": True},
    }

    def __init__(self, params, cfg, seed=0, monotone=None):
        self.params, self.cfg, self.seed, self.monotone = params, cfg, seed, monotone
        self.model = None

    def fit(self, X, y, Xv, yv, early_stopping_rounds, callbacks=()):
        import lightgbm as lgb
        p = dict(objective="binary", metric="auc", verbosity=-1, seed=self.seed, bagging_freq=1,
                 num_threads=self.cfg["inner_loop"]["n_threads"], **self.params)
        if self.monotone is not None and any(self.monotone):
            p["monotone_constraints"] = list(self.monotone)
        cb = [lgb.early_stopping(early_stopping_rounds, verbose=False), *callbacks]
        self.model = lgb.train(p, lgb.Dataset(X, y), self.cfg["inner_loop"]["n_rounds_max"],
                               valid_sets=[lgb.Dataset(Xv, yv)], callbacks=cb)
        self.best_iter = self.model.best_iteration or self.model.current_iteration()
        return self

    def predict(self, X):
        return self.model.predict(X, num_iteration=self.best_iter)


@register_model("lr_scorecard")
class LRScorecardSpec:
    """分位数分箱 + WOE + LR。缺失值单独一箱；类别变量每个取值一箱（稀有类合并到 __rare__）。"""
    preprocessing = ["woe_binning"]
    supports = {"monotone_constraints": False, "shap": False}
    default_space = {"C": {"low": 1e-3, "high": 10.0, "log": True},
                     "n_bins": {"low": 5, "high": 20, "type": "int"}}

    def __init__(self, params, cfg, seed=0, monotone=None):      # 不支持单调约束：忽略（动作校验已拦截）
        self.params, self.cfg, self.seed = params, cfg, seed

    def _bin(self, s: pd.Series, edges):
        if edges is None:
            return s.astype(object).where(s.notna(), "__na__")
        return pd.cut(s, edges, labels=False, include_lowest=True).astype("float").fillna(-1)

    def _fit_bins(self, X, y):
        self.edges, self.woe = {}, {}
        nb, prior = int(self.params["n_bins"]), y.mean()
        for c in X.columns:
            s = X[c]
            if str(s.dtype) == "category":
                s = s.astype(object)
                vc = s.value_counts()
                s = s.where(s.isin(vc[vc >= 50].index), "__rare__")
                self.edges[c] = None
            else:
                e = np.unique(np.nanquantile(s, np.linspace(0, 1, nb + 1))) if s.notna().any() else np.array([0.0])
                self.edges[c] = e if len(e) > 2 else np.array([-np.inf, np.inf])
            b = self._bin(s, self.edges[c])
            g = pd.DataFrame({"b": b, "y": y.values}).groupby("b", observed=True)["y"].agg(["sum", "count"])
            bad, good = g["sum"] + prior * 2, g["count"] - g["sum"] + (1 - prior) * 2   # 平滑
            self.woe[c] = np.log((bad / bad.sum()) / (good / good.sum())).to_dict()

    def _transform(self, X):
        out = {}
        for c in X.columns:
            s = X[c]
            if self.edges[c] is None:
                s = s.astype(object)
                s = s.where(s.isin(self.woe[c].keys()), "__rare__")
            b = self._bin(s, self.edges[c])
            out[c] = b.map(self.woe[c]).fillna(0.0).values
        return pd.DataFrame(out, index=X.index)

    def fit(self, X, y, Xv, yv, early_stopping_rounds=None, callbacks=()):
        self._fit_bins(X, y)
        self.lr = LogisticRegression(C=self.params["C"], max_iter=500).fit(self._transform(X), y)
        return self

    def predict(self, X):
        return self.lr.predict_proba(self._transform(X))[:, 1]
