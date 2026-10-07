"""Model zoo：注册表 + 统一接口。每个模型：默认搜索空间、fit(trial_params)、predict_proba；
profile 是给 LLM 看的模型档案（适合什么数据、主要风险），圈候选和换模型时要引用。"""
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
    profile = "通用强基线，数值与低基数类别都能处理；叶子数大时容易在同期验证集上过拟合。"
    native_metric = {"auc": "auc", "pr_auc": "average_precision", "ks": "auc"}   # 早停与剪枝用的内置指标；KS 无内置实现，用 AUC 近似
    preprocessing = ["native_missing", "categorical_native"]
    supports = {"monotone_constraints": True, "shap": True}
    train_doc = "目标函数 binary，缺失值和类别列交给 LightGBM 原生处理；在 train 段训练，valid 段早停（连续 {es} 轮没有提升就停，最多 {max_trees} 棵树），实际用了 {best_iter} 棵树"
    param_doc = {"learning_rate": "学习率：每棵树修正结果的幅度，越小越稳，但需要更多树",
                 "num_leaves": "每棵树的叶子数：越大越能拟合复杂关系，也越容易过拟合",
                 "min_child_samples": "每个叶子至少包含的样本数：越大越保守",
                 "feature_fraction": "每棵树随机使用的特征比例",
                 "bagging_fraction": "每棵树随机使用的样本比例",
                 "lambda_l2": "L2 正则：越大，叶子上的分数越向 0 收缩"}
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
        p = dict(objective="binary", metric=self.native_metric[self.cfg["metric"]["primary"]], verbosity=-1, seed=self.seed, bagging_freq=1,
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
    profile = "分箱 + WOE + 逻辑回归，可解释、跨时间最稳；表达不了非线性交互，上限低。"
    preprocessing = ["woe_binning"]
    supports = {"monotone_constraints": False, "shap": False}
    train_doc = "数值特征按分位数分箱、类别特征每个取值一箱（出现少于 50 次的合并），缺失单独一箱，各箱转成 WOE 后在 train 段训练逻辑回归"
    param_doc = {"C": "正则强度的倒数：越小，系数越向 0 收缩",
                 "n_bins": "数值特征分成的箱数"}
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


def _cats(X):
    return [c for c in X.columns if str(X[c].dtype) == "category"]


@register_model("catboost")
class CatBoostSpec:
    profile = "高基数类别变量多时占优（原生处理类别，不需要编码）；有序提升对过拟合更克制。训练最慢。"
    native_metric = {"auc": "AUC", "pr_auc": "PRAUC", "ks": "AUC"}
    preprocessing = ["categorical_native"]
    supports = {"monotone_constraints": False, "shap": True}
    train_doc = "类别列转成字符串原样输入（缺失单独一类）；在 train 段训练，valid 段早停（连续 {es} 轮没有提升就停），实际用了 {best_iter} 棵树"
    param_doc = {"depth": "每棵树的深度：越深越能拟合复杂关系，也越容易过拟合",
                 "learning_rate": "学习率：每棵树修正结果的幅度，越小越稳，但需要更多树",
                 "l2_leaf_reg": "L2 正则：越大，叶子上的分数越向 0 收缩",
                 "iterations": "最多训练的树数（早停可能提前结束）"}
    default_space = {
        "depth": {"low": 3, "high": 8, "type": "int"},
        "learning_rate": {"low": 0.01, "high": 0.2, "log": True},
        "l2_leaf_reg": {"low": 1.0, "high": 30.0, "log": True},
        "iterations": {"low": 100, "high": 800, "type": "int", "log": True},
    }

    def __init__(self, params, cfg, seed=0, monotone=None):
        self.params, self.cfg, self.seed = params, cfg, seed

    def _x(self, X):
        X = X.copy()
        for c in self.cats:                                   # catboost 的类别列要求字符串，缺失单独成一类
            X[c] = X[c].astype(str).where(X[c].notna(), "__na__")
        return X

    def fit(self, X, y, Xv, yv, early_stopping_rounds=None, callbacks=()):
        from catboost import CatBoostClassifier
        self.cats = _cats(X)
        self.model = CatBoostClassifier(**self.params, random_seed=self.seed, verbose=False, allow_writing_files=False,
                                        eval_metric=self.native_metric[self.cfg["metric"]["primary"]],
                                        thread_count=self.cfg["inner_loop"]["n_threads"], cat_features=self.cats)
        self.model.fit(self._x(X), y, eval_set=(self._x(Xv), yv), early_stopping_rounds=early_stopping_rounds)
        self.best_iter = self.model.get_best_iteration()
        return self

    def predict(self, X):
        return self.model.predict_proba(self._x(X))[:, 1]


@register_model("random_forest")
class RandomForestSpec:
    profile = "bagging，方差低，valid 与 OOT-dev 的差距通常更小；OVERFIT_GAP 频繁时是稳健的替代。排序能力上限一般低于提升树。"
    preprocessing = ["ordinal_codes"]
    supports = {"monotone_constraints": False, "shap": False}
    train_doc = "类别列转成整数编码，在 train 段训练，不用早停"
    param_doc = {"max_depth": "每棵树的最大深度",
                 "min_samples_leaf": "每个叶子至少包含的样本数：越大越保守",
                 "max_features": "每次分裂随机考虑的特征比例",
                 "n_estimators": "树的数量"}
    default_space = {
        "max_depth": {"low": 4, "high": 16, "type": "int"},
        "min_samples_leaf": {"low": 5, "high": 200, "type": "int", "log": True},
        "max_features": {"low": 0.2, "high": 1.0},
        "n_estimators": {"low": 100, "high": 500, "type": "int"},
    }

    def __init__(self, params, cfg, seed=0, monotone=None):
        self.params, self.cfg, self.seed = params, cfg, seed

    @staticmethod
    def _x(X):
        X = X.copy()
        for c in _cats(X):                                    # 整数编码，缺失交给 sklearn 原生处理
            X[c] = X[c].cat.codes.astype(float).where(X[c].notna())
        return X

    def fit(self, X, y, Xv, yv, early_stopping_rounds=None, callbacks=()):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(**self.params, random_state=self.seed, n_jobs=self.cfg["inner_loop"]["n_threads"])
        self.model.fit(self._x(X), y)
        return self

    def predict(self, X):
        return self.model.predict_proba(self._x(X))[:, 1]
