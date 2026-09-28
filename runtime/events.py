"""事件日志：append-only（§4.0-5, §4.14）。所有状态转换、工具调用、LLM 决策、人工指令都写在这里，挂在 exp_id 下。"""
import json
import sqlite3
import time


class EventLog:
    def __init__(self, db_path: str, task_id: str):
        self.db, self.task_id = sqlite3.connect(db_path), task_id
        self.db.execute("CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, ts REAL,"
                        " type TEXT, exp_id TEXT, payload TEXT)")
        self.db.commit()

    def append(self, type_: str, payload: dict | None = None, exp_id: str | None = None) -> int:
        cur = self.db.execute("INSERT INTO events (task_id, ts, type, exp_id, payload) VALUES (?,?,?,?,?)",
                              (self.task_id, time.time(), type_, exp_id, json.dumps(payload or {}, default=str, ensure_ascii=False)))
        self.db.commit()
        return cur.lastrowid

    def query(self, type_: str | None = None, exp_id: str | None = None) -> list[dict]:
        q, a = "SELECT id, ts, type, exp_id, payload FROM events WHERE task_id=?", [self.task_id]
        if type_:
            q += " AND type=?"; a.append(type_)
        if exp_id:
            q += " AND exp_id=?"; a.append(exp_id)
        return [{"id": r[0], "ts": r[1], "type": r[2], "exp_id": r[3], "payload": json.loads(r[4])}
                for r in self.db.execute(q + " ORDER BY id", a)]
