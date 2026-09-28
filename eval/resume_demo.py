"""Resume 演示：运行到某个实验的训练中途，kill -9 杀掉进程，再用同一条命令恢复，展示
  ① 断点在哪、Optuna study 已完成多少 trial；② 恢复后从断点继续，已完成的实验不重算；③ 最终结果与不中断的运行对比。
python -m eval.resume_demo [--compare-task hotel_mock2]"""
import argparse
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--task-id", default="resume_demo")
ap.add_argument("--compare-task", default="hotel_mock2", help="同配置、不中断的参照运行")
a = ap.parse_args()
CMD = [sys.executable, "-u", "-m", "eval.run_task", "--dataset", "hotel_bookings", "--task-id", a.task_id, "--llm", "mock",
       "--autonomy", "L0", "--full-trials", "15", "--max-rounds", "5"]
db = sqlite3.connect("artifacts/state.db")
q = lambda sql, *p: db.execute(sql, p).fetchall()
if q("SELECT 1 FROM checkpoints WHERE task_id=?", a.task_id):
    sys.exit(f"{a.task_id} 已存在，请换一个 --task-id")

p = subprocess.Popen(CMD, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
while p.poll() is None:                                        # 等到第 2 个实验（EXECUTE）开始训练，再多等几秒让 study 跑到一半
    ck = q("SELECT state FROM checkpoints WHERE task_id=?", a.task_id)
    if ck and ck[0][0] == "EXECUTE" and q("SELECT count(*) FROM events WHERE task_id=? AND type='recorded'", a.task_id)[0][0] >= 1:
        time.sleep(4)
        break
    time.sleep(0.3)
os.kill(p.pid, signal.SIGKILL)
p.wait()
state = q("SELECT state FROM checkpoints WHERE task_id=?", a.task_id)[0][0]
done_exps = q("SELECT exp_id FROM experiments WHERE task_id=?", a.task_id)
odb = sqlite3.connect("artifacts/optuna.db")
half = odb.execute("SELECT s.study_name, sum(t.state='COMPLETE'), sum(t.state='RUNNING'), count(*) FROM studies s JOIN trials t USING(study_id) "
                   "WHERE s.study_name LIKE ? GROUP BY s.study_name ORDER BY max(t.trial_id) DESC LIMIT 1", (f"{a.task_id}_%",)).fetchone()
print(f"[kill -9] 进程被杀；检查点状态={state}；已记录实验={len(done_exps)}；被打断的 study={half[0]}（已完成 {half[1]} 个 trial，{half[2]} 个在跑）")

t0 = time.time()
r = subprocess.run(CMD, capture_output=True, text=True)
print(f"[恢复] 同一条命令重跑，耗时 {time.time() - t0:.0f}s，退出码 {r.returncode}")

exps = q("SELECT exp_id, record FROM experiments WHERE task_id=? ORDER BY seq", a.task_id)
rows, dup = [], []
for eid, rec in exps:
    d = json.loads(rec)
    n = odb.execute("SELECT count(*) FROM trials t JOIN studies s USING(study_id) WHERE s.study_name=? AND t.state='COMPLETE'", (eid,)).fetchone()[0]
    rows.append((eid, d["action_type"], d["verdict"], d["metrics"].get("oot_dev_auc"), n))
ref = {}
if q("SELECT 1 FROM checkpoints WHERE task_id=?", a.compare_task):
    ref = [(json.loads(rec)["metrics"].get("oot_dev_auc"), json.loads(rec)["verdict"]) for _, rec in q("SELECT exp_id, record FROM experiments WHERE task_id=? ORDER BY seq", a.compare_task)]
out = ["| 实验 | 动作 | 判定 | OOT-dev AUC | 完成的 trial 数（不含剪枝）| 不中断参照运行的 AUC / 判定 |", "|---|---|---|---|---|---|"]
for i, (eid, act, v, auc, n) in enumerate(rows):
    rf = f"{ref[i][0]:.4f} / {ref[i][1]}" if ref and i < len(ref) and ref[i][0] else "-"
    out.append(f"| {eid} | {act} | {v} | {auc:.4f} | {n} | {rf} |")
fin = json.loads(q("SELECT result FROM final_gate WHERE task_id=?", a.task_id)[0][0])
print("\n".join(out))
print(f"\n最终 holdout AUC={fin['holdout_auc']:.4f}（final_gate 只评一次）；events 中 state_transition 共 {q('SELECT count(*) FROM events WHERE task_id=? AND type=?', a.task_id, 'state_transition')[0][0]} 条")
open("reports/resume_demo.md", "w").write(
    f"# Resume 演示\n\n- 在 `{state}` 状态、study `{half[0]}` 跑到 {half[1]} 个 trial 时 `kill -9`\n- 用同一条命令恢复：{CMD[3:]}\n\n" + "\n".join(out) + "\n")
