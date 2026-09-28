# Resume 演示

- 在 `EXECUTE` 状态、study `resume_demo_e02` 跑到 11 个 trial 时 `kill -9`
- 用同一条命令恢复：['eval.run_task', '--dataset', 'hotel_bookings', '--task-id', 'resume_demo', '--llm', 'mock', '--autonomy', 'L0', '--full-trials', '15', '--max-rounds', '5']

| 实验 | 动作 | 判定 | OOT-dev AUC | 完成的 trial 数 | 不中断参照运行的 AUC / 判定 |
|---|---|---|---|---|---|
| resume_demo_race_lgbm | RACE | REJECT | 0.8284 | 12 | 0.8284 / REJECT |
| resume_demo_race_lr_scorecard | RACE | REJECT | 0.8173 | 20 | 0.8173 / REJECT |
| resume_demo_e01 | PROMOTE_FIDELITY | REJECT | 0.8234 | 8 | 0.8234 / REJECT |
| resume_demo_e02 | PROMOTE_FIDELITY | ACCEPT | 0.8279 | 15 | 0.8292 / ACCEPT |
| resume_demo_e03 | TUNE | INCONCLUSIVE | 0.8283 | 8 | 0.8283 / INCONCLUSIVE |
