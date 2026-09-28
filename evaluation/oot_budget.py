"""OOT-dev 使用预算：每次 compare 计数 +1，显著性水平逐步收紧，达到上限 K 触发硬停止（§4.9）。"""
import math
import sqlite3
from pathlib import Path

POLICIES = {"log": lambda a0, k: a0 / (1 + math.log(k)),
            "constant": lambda a0, k: a0}      # 可插拔：后续可换 Thresholdout


class OOTBudget:
    """计数持久化在 SQLite，resume 后不会归零。"""

    def __init__(self, cfg: dict, db_path: str, task_id: str):
        c = cfg["oot_budget"]
        self.alpha_0, self.K, self.policy = c["alpha_0"], c["max_compares"], POLICIES[c["policy"]]
        self.task_id = task_id
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path)
        self.db.execute("CREATE TABLE IF NOT EXISTS oot_usage (task_id TEXT PRIMARY KEY, k INTEGER)")
        self.db.execute("CREATE TABLE IF NOT EXISTS oot_compares (task_id TEXT, key TEXT, k INTEGER, alpha REAL, PRIMARY KEY (task_id, key))")
        self.db.execute("INSERT OR IGNORE INTO oot_usage VALUES (?, 0)", (task_id,))
        self.db.commit()

    @property
    def used(self) -> int:
        return self.db.execute("SELECT k FROM oot_usage WHERE task_id=?", (self.task_id,)).fetchone()[0]

    @property
    def exhausted(self) -> bool:
        return self.used >= self.K

    def current_alpha(self) -> float:
        """下一次 compare 将使用的 alpha。"""
        return self.policy(self.alpha_0, self.used + 1)

    def consume(self, key: str | None = None) -> tuple[int, float]:
        """登记一次 compare，返回 (k, alpha_k)。超过上限抛异常。
        key（如 exp_id）相同的重复登记直接返回原结果：resume 重放 EVALUATE 时不会重复计数。"""
        if key is not None:
            row = self.db.execute("SELECT k, alpha FROM oot_compares WHERE task_id=? AND key=?", (self.task_id, key)).fetchone()
            if row:
                return row
        if self.exhausted:
            raise OOTBudgetExhausted(f"OOT-dev 使用次数已达上限 K={self.K}")
        k = self.used + 1
        alpha = self.policy(self.alpha_0, k)
        self.db.execute("UPDATE oot_usage SET k=? WHERE task_id=?", (k, self.task_id))
        if key is not None:
            self.db.execute("INSERT INTO oot_compares VALUES (?,?,?,?)", (self.task_id, key, k, alpha))
        self.db.commit()
        return k, alpha


class OOTBudgetExhausted(RuntimeError):
    pass
