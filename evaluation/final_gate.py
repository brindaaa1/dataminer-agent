"""唯一可读 holdout 的模块（§4.0-1）。holdout 只跑一次；不允许在此之后回头重调（§4.9）。"""
import json
import sqlite3
from pathlib import Path

from sklearn.metrics import roc_auc_score

from data.access import DataAccess, issue_holdout_token
from evaluation.compare import monthly_auc
from modeling.inner_loop import fit_and_predict, ks


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

    def _finish(self, exp_id, h, p, oot, extra=None) -> dict:
        auc = float(roc_auc_score(h["_label"], p))
        res = {"exp_id": exp_id, "holdout_auc": auc, "holdout_ks": ks(h["_label"].values, p),
               "oot_dev_auc": float(oot), "drop": float(oot - auc),
               "overfit_to_oot_dev": bool((oot - auc) > self.cfg["final_gate"]["delta"]),
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
        return self._finish(best["exp_id"], h, preds["holdout"], best["metrics"]["oot_dev_auc"], extra)

    def run_predictor(self, exp_id: str, predict_fn, oot_dev_auc: float) -> dict:
        """基线（B0/B1）用：predict_fn(holdout_df) → 分数。与 run_once 相同的『只评一次』约束。"""
        if (c := self._cached(exp_id)):
            return c
        h = self.load_holdout()
        return self._finish(exp_id, h, predict_fn(h), oot_dev_auc)
