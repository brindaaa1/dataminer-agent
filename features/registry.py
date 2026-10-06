"""特征注册表（§4.11）：每个特征带来源（模板 / run_code）、定义、筛选指标；特征集缓存到 parquet。"""
import hashlib
import json
import sqlite3
from pathlib import Path

import pandas as pd

from features import factory

ID = "_row_id"


class Registry:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.art = Path(cfg["paths"]["artifacts_root"])
        self.art.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.art / "state.db")
        self.db.execute("CREATE TABLE IF NOT EXISTS feature_sets (fs_id TEXT PRIMARY KEY, dataset TEXT, template_id TEXT, source TEXT, kept TEXT, defs TEXT, screen TEXT, code TEXT)")
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(feature_sets)")}
        if "code" not in cols:                        # 迁移：早期版本的表没有这一列（run_code 后补）
            self.db.execute("ALTER TABLE feature_sets ADD COLUMN code TEXT")
        self.db.commit()

    @staticmethod
    def fs_id(dataset, template_id, root) -> str:
        return "fs_" + hashlib.sha1(f"{dataset}|{template_id}|{root}".encode()).hexdigest()[:8]

    def register(self, fs_id, dataset, template_id, source, kept, defs, screen, code=None):
        self.db.execute("INSERT OR REPLACE INTO feature_sets VALUES (?,?,?,?,?,?,?,?)",
                        (fs_id, dataset, template_id, source, json.dumps(kept), json.dumps(defs), json.dumps(screen, default=float), code))
        self.db.commit()

    def get(self, fs_id) -> dict:
        r = self.db.execute("SELECT dataset, template_id, source, kept, defs, screen, code FROM feature_sets WHERE fs_id=?", (fs_id,)).fetchone()
        return {"dataset": r[0], "template_id": r[1], "source": r[2], "kept": json.loads(r[3]), "defs": json.loads(r[4]),
                "screen": json.loads(r[5]), "code": r[6]}

    def cache_path(self, fs_id, split) -> Path:
        return self.art / "feature_sets" / fs_id / f"{split}.parquet"

    def attach(self, df: pd.DataFrame, feature_sets: list[str], split: str) -> pd.DataFrame:
        """把特征集里保留的特征按 _row_id 左连接到 df。缓存没有时（如 holdout，由 final_gate 触发）现算。"""
        for fs in feature_sets or []:
            meta = self.get(fs)
            p = self.cache_path(fs, split)
            if p.exists():
                feat = pd.read_parquet(p)
            elif meta["source"] == "run_code":            # holdout：final_gate 拿到 df 后现算（同一份沙箱代码，同一套子进程隔离）
                from tools.run_code import apply_code_to_df
                feat = apply_code_to_df(meta["code"], df, self.cfg, (meta["defs"] or [{}])[0].get("inputs"))   # 与生成时同一份输入白名单
                p.parent.mkdir(parents=True, exist_ok=True)
                feat.to_parquet(p)
            else:
                feat, _ = factory.compute(meta["template_id"], meta["dataset"], self.cfg, df[ID])
                p.parent.mkdir(parents=True, exist_ok=True)
                feat.to_parquet(p)
            df = df.merge(feat[[ID] + meta["kept"]], on=ID, how="left")
        return df
