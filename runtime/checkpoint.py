"""检查点：每个状态结束时写 (task_id, state, state_payload, last_event_id)；resume = 读最后一条。"""
import json
import sqlite3


class Checkpoint:
    def __init__(self, db_path: str, task_id: str):
        self.db, self.task_id = sqlite3.connect(db_path), task_id
        self.db.execute("CREATE TABLE IF NOT EXISTS checkpoints (task_id TEXT PRIMARY KEY, state TEXT, payload TEXT, last_event_id INTEGER)")
        self.db.commit()

    def save(self, state: str, payload: dict, last_event_id: int):
        self.db.execute("INSERT OR REPLACE INTO checkpoints VALUES (?,?,?,?)",
                        (self.task_id, state, json.dumps(payload, default=str), last_event_id))
        self.db.commit()

    def load(self):
        r = self.db.execute("SELECT state, payload, last_event_id FROM checkpoints WHERE task_id=?", (self.task_id,)).fetchone()
        return (r[0], json.loads(r[1]), r[2]) if r else None
