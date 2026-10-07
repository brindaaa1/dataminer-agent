"""唯一可读 holdout 的模块（§4.0-1）。holdout 只跑一次；不允许在此之后回头重调（§4.9）。"""
import json
import sqlite3
from pathlib import Path

from sklearn.metrics import roc_auc_score

from data.access import DataAccess, issue_holdout_token
from evaluation.compare import monthly_auc
from evaluation.metrics import all_metrics, primary
from modeling.inner_loop import fit_and_predict


class FinalGateAlreadyRun(RuntimeError):
    pass


class FinalGate:
    def __init__(self, dataset: str, cfg: dict, task_id: str):
        self._token = issue_holdout_token()
        self.dataset, self.cfg, self.task_id = dataset, cfg, task_id
        art = Path(cfg["paths"]["artifacts_root"])
        self._data = DataAccess(dataset, str(art))
        self.db = sqlite3.connect(art / "state.db")
        self.db.execute("CREATE TABLE IF NOT EXISTS final_gate (task_id TEXT PRIMARY KEY, result TEXT)")

    def load_holdout(self):
        return self._data.load("holdout", token=self._token)

    def _finish(self, exp_id, h, p, oot, extra=None, ref_drop=None) -> dict:
        """oot：该实验 OOT-dev 上的主指标。drop 和『疑似对 OOT-dev 过拟合』都按主指标算，其余指标并列记录。
        ref_drop：默认参照模型自己的 OOT-dev → holdout 下滑；有它时多掉超过 δ 才算疑似过拟合（hotel 参照自己就掉 0.043）。"""
        pm = primary(self.cfg)
        hm = all_metrics(h["_label"].values, p)
        drop = float(oot - hm[pm])
        res = {"exp_id": exp_id, "metric": pm, **{f"holdout_{k}": v for k, v in hm.items()},
               f"oot_dev_{pm}": float(oot), "drop": drop, "reference_drop": ref_drop,
               "overfit_to_oot_dev": bool(drop - (ref_drop or 0.0) > self.cfg["final_gate"]["delta"]),
               "n_holdout": int(len(h)), "holdout_bad_rate": float(h["_label"].mean()), **(extra or {})}
        tcol = "_split_time" if "_split_time" in h else "_obs_time"      # 分月按『切分所用的时间』（如到店日），而不是观察时间
        if tcol in h:
            res["holdout_monthly_auc"] = monthly_auc(h[tcol], h["_label"], p)
        self.db.execute("INSERT INTO final_gate VALUES (?, ?)", (self.task_id, json.dumps(res)))
        self.db.commit()
        return res

    def _cached(self, exp_id):
        row = self.db.execute("SELECT result FROM final_gate WHERE task_id=?", (self.task_id,)).fetchone()
        if row:
            if json.loads(row[0]).get("exp_id") != exp_id:
                raise FinalGateAlreadyRun("holdout 已用于另一个实验，不允许重复验收")
            return json.loads(row[0])

    def run_once(self, best: dict, mask_fields: list[str] | None = None) -> dict:
        """best: 当前最优实验的 result dict（含 config、best_params、metrics）。第二次调用直接抛异常；
        崩溃后重放时若已有结果则原样返回（幂等），不重新读 holdout。"""
        if (c := self._cached(best["exp_id"])):
            return c
        h = self.load_holdout()
        hm = h.copy()
        for c_ in mask_fields or []:
            if c_ in hm:
                hm[c_] = None if hm[c_].dtype == object else float("nan")   # 保持原 dtype，数值列置 NaN
        preds = fit_and_predict(self.dataset, self.cfg, best["config"], best["best_params"],
                                {"holdout": h, **({"masked": hm} if mask_fields else {})})
        extra = {"holdout_auc_masked": float(roc_auc_score(h["_label"], preds["masked"]))} if mask_fields else None
        ref = None
        if self.cfg["evaluator"].get("overfit_gap", {}).get("mode") == "relative":
            from evaluation.reference import reference_drop
            ref = float(reference_drop(self.dataset, self.cfg, h))
        return self._finish(best["exp_id"], h, preds["holdout"], best["metrics"][f"oot_dev_{primary(self.cfg)}"], extra, ref)

    def run_predictor(self, exp_id: str, predict_fn, oot_dev_score: float) -> dict:
        """基线（B0/B1）用：predict_fn(holdout_df) → 分数。与 run_once 相同的『只评一次』约束。"""
        if (c := self._cached(exp_id)):
            return c
        h = self.load_holdout()
        return self._finish(exp_id, h, predict_fn(h), oot_dev_score)
