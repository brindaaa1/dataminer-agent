"""实验记录存储（episodic，只追加）+ 结构化查询（§4.13 上下文组装的数据来源）。"""
import json
import sqlite3

from memory.schema import ExperimentRecord


class Store:
    def __init__(self, db_path: str):
        self.db = sqlite3.connect(db_path)
        self.db.execute("CREATE TABLE IF NOT EXISTS experiments (exp_id TEXT PRIMARY KEY, task_id TEXT, seq INTEGER, record TEXT)")
        self.db.commit()

    def add(self, rec: ExperimentRecord):
        """幂等：重放 RECORD 时同一 exp_id 不会重复写入。"""
        n = self.db.execute("SELECT count(*) FROM experiments WHERE task_id=?", (rec.task_id,)).fetchone()[0]
        self.db.execute("INSERT OR IGNORE INTO experiments VALUES (?,?,?,?)", (rec.exp_id, rec.task_id, n, rec.model_dump_json()))
        self.db.commit()

    def all(self, task_id: str) -> list[ExperimentRecord]:
        rows = self.db.execute("SELECT record FROM experiments WHERE task_id=? ORDER BY seq", (task_id,)).fetchall()
        return [ExperimentRecord.model_validate_json(r[0]) for r in rows]

    def get(self, exp_id: str) -> ExperimentRecord | None:
        r = self.db.execute("SELECT record FROM experiments WHERE exp_id=?", (exp_id,)).fetchone()
        return ExperimentRecord.model_validate_json(r[0]) if r else None

    def taboo_hashes(self, task_id: str) -> set[str]:
        return {r.config_hash for r in self.all(task_id) if not r.invalidated and r.fidelity != "none"}

    def path_to_root(self, exp_id: str | None) -> list[ExperimentRecord]:
        out = []
        while exp_id:
            r = self.get(exp_id)
            if not r:
                break
            out.append(r)
            exp_id = r.parent_exp_id
        return out[::-1]

    def top_k(self, task_id: str, k: int) -> list[ExperimentRecord]:
        rs = [r for r in self.all(task_id) if r.metrics and not r.invalidated and "oot_dev_auc" in r.metrics]
        return sorted(rs, key=lambda r: -r.metrics["oot_dev_auc"])[:k]

    def recent_rejected(self, task_id: str, k: int) -> list[ExperimentRecord]:
        return [r for r in self.all(task_id) if r.verdict == "REJECT"][-k:]

    def invalidate(self, task_id: str, predicate) -> list[str]:
        """约束改变后标记失效（不删除，§4.14）。"""
        ids = [r.exp_id for r in self.all(task_id) if predicate(r)]
        for i in ids:
            r = self.get(i)
            r.invalidated = True
            self.db.execute("UPDATE experiments SET record=? WHERE exp_id=?", (r.model_dump_json(), i))
        self.db.commit()
        return ids


class MemoryStore:
    """跨任务记忆：任务指纹 + Prior + Lesson。独立于单任务的 state.db。"""

    def __init__(self, db_path: str):
        from pathlib import Path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path)
        self.db.execute("CREATE TABLE IF NOT EXISTS tasks (task_id TEXT PRIMARY KEY, fingerprint TEXT, prior TEXT, final TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS lessons (lesson_id TEXT PRIMARY KEY, lesson TEXT)")
        self.db.commit()

    def put_task(self, fp: dict, prior: dict | None, final: dict | None):
        self.db.execute("INSERT OR REPLACE INTO tasks VALUES (?,?,?,?)", (fp["task_id"], json.dumps(fp), json.dumps(prior), json.dumps(final)))
        self.db.commit()

    def tasks(self) -> list[dict]:
        return [{"fingerprint": json.loads(f), "prior": json.loads(p), "final": json.loads(fi)}
                for f, p, fi in self.db.execute("SELECT fingerprint, prior, final FROM tasks")]

    def get_lesson(self, lesson_id):
        r = self.db.execute("SELECT lesson FROM lessons WHERE lesson_id=?", (lesson_id,)).fetchone()
        return json.loads(r[0]) if r else None

    def put_lesson(self, lesson: dict):
        self.db.execute("INSERT OR REPLACE INTO lessons VALUES (?,?)", (lesson["lesson_id"], json.dumps(lesson, ensure_ascii=False)))
        self.db.commit()

    def lessons(self) -> list[dict]:
        return [json.loads(r[0]) for r in self.db.execute("SELECT lesson FROM lessons")]
