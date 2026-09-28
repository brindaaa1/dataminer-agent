# Memory 轻量演示

小样本：每个任务约 60000 行；mock LLM（策略固定）；full 保真度 15 个 trial。

## 1. 指纹与检索

| 任务 | 数据 | n_samples | n_features | bad_rate | 检索结果（距离；阈值 2.0） |
|---|---|---|---|---|---|
| H | lending_club | 28219 | 35 | 0.132 | 冷启动（无历史任务） |
| N_on | lending_club | 31461 | 35 | 0.125 | 最近=H；距离 0.31；复用 prior；ACTIVE 经验 0 条 |
| T | hotel_bookings | 31642 | 27 | 0.364 | 最近=无；距离 4.98；冷启动；ACTIVE 经验 0 条 |

## 2. Lesson 生命周期

| key | 极性 | 状态 | 支撑任务 | 置信度 |
|---|---|---|---|---|
| promote:lgbm | POSITIVE | **ACTIVE** | H,N_on | 0.9 |
| race:lgbm | POSITIVE | **ACTIVE** | H,N_on | 0.9 |
| race:lr_scorecard | NEGATIVE | **ACTIVE** | H,N_on | 0.9 |
| switch_to:lr_scorecard | NEGATIVE | **ACTIVE** | H,N_on | 0.9 |
| tune:lgbm:num_leaves | NEGATIVE | **ACTIVE** | H,N_on | 0.9 |
| promote:lgbm | NEGATIVE | **CANDIDATE** | T | 0.3 |
| promote:lr_scorecard | POSITIVE | **CANDIDATE** | T | 0.3 |
| race:lgbm | NEGATIVE | **CANDIDATE** | T | 0.3 |
| race:lr_scorecard | NEGATIVE | **CANDIDATE** | T | 0.3 |
| tune:lr_scorecard:n_bins | NEGATIVE | **CANDIDATE** | T | 0.3 |

N_on 的 CONSOLIDATE：LLM 提出 5 条，规则准入 5 条，丢弃 0 条；本次状态变化：{'e81ea5be0d': ['race:lgbm', 'VALIDATED', 'ACTIVE'], 'e7969212c5': ['race:lr_scorecard', 'VALIDATED', 'ACTIVE'], '88c45d2de4': ['promote:lgbm', 'VALIDATED', 'ACTIVE'], 'fce9e2a399': ['switch_to:lr_scorecard', 'VALIDATED', 'ACTIVE'], '503a562e58': ['tune:lgbm:num_leaves', 'VALIDATED', 'ACTIVE']}

## 3. 收敛对比（N 的数据，lgbm，20 个 trial，5 个种子，best-so-far valid AUC 的均值 ± 标准差）

warm-start 使用 H 的 prior：从历史任务取 3 组参数，上限为 trial 总数的 10%（= 2 个初始点）。

| 已跑 trial 数 | 1 | 2 | 3 | 5 | 10 | 15 |
|---|---|---|---|---|---|---|
| 冷启动 | 0.6555±0.0046 | 0.6584±0.0026 | 0.6594±0.0016 | 0.6603±0.0013 | 0.6603±0.0013 | 0.6610±0.0009 |
| warm-start（用 H 的 prior） | 0.6561±0.0026 | 0.6561±0.0026 | 0.6571±0.0030 | 0.6594±0.0016 | 0.6598±0.0019 | 0.6608±0.0019 |

达到目标（冷启动最终均值 − 0.002 = 0.6590）所需 trial 数：冷启动 [11, 2, 1, 1, 5]，warm-start [11, 4, 3, 1, None]（None = 20 个 trial 内没达到）。
