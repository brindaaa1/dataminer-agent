# 模型卡：a3_hotel_bookings_guardrail_on

> 数据集 `hotel_bookings`；本文档数据全部由代码生成。

## 执行摘要

本次共运行 5 个实验，其中 1 个被接受。最终模型为 lr_scorecard，OOT-dev AUC 为 0.8292。

## 1. 任务规格

| 项 | 值 |
|---|---|
| label 定义 | 预订最终被取消（通常包含 No-Show），加载时用 reservation_status 校验该对应关系，校验后 reservation_status 立即剔除 |
| 观察时间列 | arrival_date - lead_time |
| OOT 窗口 | {"oot_dev": ["2017-01", "2017-04"], "holdout": ["2017-05", "2017-08"]} |
| 目标指标 | auc（目标值 None） |
| 约束 | {'interpretability': 'none', 'max_features': None, 'banned_models': []} |
| 预算 | {'tokens': 200000.0, 'cpu_minutes': 240.0, 'wall_minutes': 120.0} |
| 自治级别 | L0 |

## 2. 数据切分

| 切分 | 样本数 | 坏样本率 |
|---|---|---|
| train | 62,963 | 0.3612 |
| valid | 15,740 | 0.3644 |
| oot_dev | 18,489 | 0.3648 |
| holdout（仅 final_gate） | 22,198 | 0.4055 |

## 3. 最终模型

- 实验：`a3_hotel_bookings_guardrail_on_e02`；模型：`lr_scorecard`；保真度：full；trial 数：15（剪枝 0）
- 特征数：27；最优参数：`{"C": 1.9728399969456505, "n_bins": 17}`

## 4. 各段指标

| 段 | AUC | KS |
|---|---|---|
| train | 0.8393 | |
| valid | 0.8360 | |
| OOT-dev | 0.8292 | 0.5102 |
| holdout | 0.7752 | 0.4104 |

- valid − OOT-dev gap：0.0068；分数 PSI（valid→OOT-dev）：0.1206；OOT-dev 使用次数：2

## 5. 分月 AUC、坏样本率与分数 PSI

（月份按 `_split_time` 划分）

| 月份 | 段 | n | AUC | 坏样本率 | 去年同期坏样本率 | 分数 PSI（相对 valid） |
|---|---|---|---|---|---|---|
| 2017-01 | OOT-dev | 3681 | 0.8601 | 0.340 | 0.248 | 0.2570 |
| 2017-02 | OOT-dev | 4177 | 0.8021 | 0.325 | 0.344 | 0.1396 |
| 2017-03 | OOT-dev | 4970 | 0.8184 | 0.336 | 0.306 | 0.1608 |
| 2017-04 | OOT-dev | 5661 | 0.8283 | 0.435 | 0.380 | 0.1500 |
| 2017-05 | holdout | 6313 | 0.8194 | 0.438 | 0.350 | |
| 2017-06 | holdout | 5647 | 0.7986 | 0.432 | 0.396 | |
| 2017-07 | holdout | 5313 | 0.7386 | 0.373 | 0.328 | |
| 2017-08 | holdout | 4925 | 0.7141 | 0.369 | 0.360 | |

> 酒店业务季节性很强：oot_dev 覆盖冬春，holdout 覆盖夏季。OOT 上的漂移有一部分来自季节而非模型退化， 报告中需要按月份对比同期（2016 年同月）的坏样本率，区分季节效应与真实漂移。

## 6. 特征清单（按平均 |SHAP| 排序）

| 特征 | 来源 | 定义/说明 | IV | 贡献占比 | 备注 |
|---|---|---|---|---|---|
| deposit_type | 原始字段 | 数据中 Non Refund 的取消率反直觉地高（可能与团体/渠道订单有关）。critic 标注但不自动剔除 | 2.083 | 28.5% |  |
| agent | 原始字段 | ID，高基数，含 NULL 字符串 | 0.910 | 15.2% |  |
| customer_type | 原始字段 | categorical | 0.049 | 13.9% |  |
| lead_time | 原始字段 | numeric | 0.663 | 12.9% |  |
| cancel_history_ratio | 原始字段 |  | 0.312 | 11.0% |  |
| hotel | 原始字段 | categorical | 0.100 | 2.2% |  |
| weekend_share | 原始字段 |  | 0.047 | 1.6% |  |
| company | 原始字段 | ID，高基数，大量 NULL | 0.106 | 1.5% |  |
| stays_in_weekend_nights | 原始字段 | 对已入住的订单，可能反映实际入住晚数而非预订晚数；列入审查 | 0.001 | 1.5% |  |
| stays_in_week_nights | 原始字段 | 同上 | 0.070 | 1.5% |  |
| arrival_date_week_number | 原始字段 | numeric | 0.021 | 1.3% |  |
| market_segment | 原始字段 | categorical | 0.384 | 1.2% |  |
| total_nights | 原始字段 |  | 0.108 | 1.1% |  |
| total_guests | 原始字段 |  | 0.042 | 1.1% |  |
| reserved_room_type | 原始字段 | categorical | 0.040 | 1.0% |  |
| arrival_date_day_of_month | 原始字段 | numeric | 0.003 | 0.8% |  |
| children | 原始字段 | 有少量缺失 | 0.000 | 0.6% |  |
| distribution_channel | 原始字段 | categorical | 0.157 | 0.6% |  |
| booking_month | 原始字段 |  | 0.036 | 0.6% |  |
| meal | 原始字段 | Undefined 与 SC 含义相同，合并 | 0.014 | 0.5% |  |
| previous_cancellations | 原始字段 | numeric | 0.000 | 0.5% |  |
| adults | 原始字段 | numeric | 0.031 | 0.4% |  |
| arrival_date_month | 原始字段 | categorical | 0.022 | 0.2% |  |
| babies | 原始字段 | numeric | 0.000 | 0.0% |  |
| is_repeated_guest | 原始字段 | binary | 0.000 | 0.0% |  |
| previous_bookings_not_canceled | 原始字段 | numeric | 0.000 | 0.0% |  |
| has_kids | 原始字段 |  | 0.000 | 0.0% |  |

## 7. 实验树

```mermaid
flowchart TD
  ROOT((start))
  n_a3_hotel_bookings_guardrail_on_race_lgbm["a3_hotel_bookings_guardrail_on_race_lgbm<br/>RACE [low]<br/>AUC 1.0000<br/>REJECT<br/>LEAK_SUSPECT,NO_CONVERGE"]:::reject
  n_a3_hotel_bookings_guardrail_on_race_lr_scorecard["a3_hotel_bookings_guardrail_on_race_lr_scorecard<br/>RACE [low]<br/>AUC 1.0000<br/>REJECT<br/>LEAK_SUSPECT,PSI_DRIFT,PLATEAU"]:::reject
  n_a3_hotel_bookings_guardrail_on_e01["a3_hotel_bookings_guardrail_on_e01<br/>PROMOTE_FIDELITY [full]<br/>AUC 0.8234 (-0.1766)<br/>REJECT<br/>OVERFIT_GAP,PSI_DRIFT"]:::reject
  n_a3_hotel_bookings_guardrail_on_e02["a3_hotel_bookings_guardrail_on_e02<br/>PROMOTE_FIDELITY [full]<br/>AUC 0.8292 (-0.1708)<br/>ACCEPT<br/>PSI_DRIFT"]:::accept
  n_a3_hotel_bookings_guardrail_on_e03["a3_hotel_bookings_guardrail_on_e03<br/>TUNE [full]<br/>AUC 0.8283 (-0.0009)<br/>INCONCLUSIVE<br/>PSI_DRIFT,PLATEAU"]:::inconclusive
  ROOT --> n_a3_hotel_bookings_guardrail_on_race_lgbm
  ROOT --> n_a3_hotel_bookings_guardrail_on_race_lr_scorecard
  n_a3_hotel_bookings_guardrail_on_race_lgbm --> n_a3_hotel_bookings_guardrail_on_e01
  n_a3_hotel_bookings_guardrail_on_race_lr_scorecard --> n_a3_hotel_bookings_guardrail_on_e02
  n_a3_hotel_bookings_guardrail_on_e02 --> n_a3_hotel_bookings_guardrail_on_e03
  classDef accept fill:#c8e6c9,stroke:#2e7d32
  classDef reject fill:#ffcdd2,stroke:#c62828
  classDef inconclusive fill:#fff9c4,stroke:#f9a825
  classDef promising fill:#bbdefb,stroke:#1565c0
  classDef none fill:#eeeeee,stroke:#757575
  classDef accept_inv,reject_inv,inconclusive_inv,promising_inv,none_inv fill:#f5f5f5,stroke:#9e9e9e,stroke-dasharray:4 3
```

| 实验 | 父节点 | 动作 | 保真度 | OOT-dev AUC | 判定 | 诊断码 |
|---|---|---|---|---|---|---|
| a3_hotel_bookings_guardrail_on_race_lgbm | - | RACE | low | 1.0000 | REJECT | LEAK_SUSPECT,NO_CONVERGE |
| a3_hotel_bookings_guardrail_on_race_lr_scorecard | - | RACE | low | 1.0000 | REJECT | LEAK_SUSPECT,PSI_DRIFT,PLATEAU |
| a3_hotel_bookings_guardrail_on_e01 | a3_hotel_bookings_guardrail_on_race_lgbm | PROMOTE_FIDELITY | full | 0.8234 | REJECT | OVERFIT_GAP,PSI_DRIFT |
| a3_hotel_bookings_guardrail_on_e02 | a3_hotel_bookings_guardrail_on_race_lr_scorecard | PROMOTE_FIDELITY | full | 0.8292 | ACCEPT | PSI_DRIFT |
| a3_hotel_bookings_guardrail_on_e03 | a3_hotel_bookings_guardrail_on_e02 | TUNE | full | 0.8283 | INCONCLUSIVE | PSI_DRIFT,PLATEAU |

## 8. 风险提示

- holdout AUC 比 OOT-dev 低 0.0541，超过 δ=0.02：**疑似对 OOT-dev 过拟合**。
- 分月 AUC 极差 0.146 > 0.05（sanity_rules）。注意月度样本量差异较大。
- 被隔离的疑似泄漏特征（11 个）：`reservation_status`、`country`、`assigned_room_type`、`booking_changes`、`days_in_waiting_list`、`adr`、`required_car_parking_spaces`、`total_of_special_requests`、`reservation_status_date`、`price_per_guest`、`room_mismatch`
- 停止原因：LLM_STOP: mock: 剩余方向期望价值低于成本。
