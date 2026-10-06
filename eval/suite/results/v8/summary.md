# 评测 v8

设置：max_rounds=8，low_sample_frac=0.3，low_trials=10，full_trials=20，split_seed=42，autonomy=L0，llm=deepseek，scenario={'max_rounds': 5, 'n_rows': 20000, 'seed': 0}，tier=fast，seeds=[1]

## hotel_bookings

| 种子 | 状态 | 最终模型 | holdout | Δ vs B0 | OOT-dev | 过拟合 OOT-dev | 轮数 | 被拒决策 | LLM 失败 | 停止原因 | 输入 token（命中/未命中） | 输出（推理） | 美元 | 耗时(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | ok | catboost | 0.7882 | +0.0121 | 0.8246 | 是 | 8 | 0 | 0 | MAX_ROUNDS | 9728/26343 | 32658（26003） | 0.0129 | 838.2 |
| 1 | ok | catboost | 0.7939 | +0.0179 | 0.8281 | 是 | 4 | 0 | 1 | NO_PROGRESS | 11520/8076 | 13176（8671） | 0.0049 | 2057.9 |

- 赢 B0 的种子：2/2；本数据集花费合计 0.0177 美元
- B0：holdout 0.7761，OOT-dev 0.8195，耗时 1.1s
- B1：holdout 0.7839，OOT-dev 0.8253，耗时 838.0s

## lending_club

| 种子 | 状态 | 最终模型 | holdout | Δ vs B0 | OOT-dev | 过拟合 OOT-dev | 轮数 | 被拒决策 | LLM 失败 | 停止原因 | 输入 token（命中/未命中） | 输出（推理） | 美元 | 耗时(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | ok | lgbm | 0.6585 | +0.0111 | 0.6561 |  | 8 | 0 | 0 | MAX_ROUNDS | 9596/28045 | 31022（24566） | 0.0126 | 337.2 |
| 1 | ok | catboost | 0.6623 | +0.0149 | 0.6560 |  | 8 | 0 | 0 | MAX_ROUNDS | 9724/26496 | 25721（18656） | 0.0109 | 416.9 |

- 赢 B0 的种子：2/2；本数据集花费合计 0.0236 美元
- B0：holdout 0.6474，OOT-dev 0.6410，耗时 0.2s
- B1：holdout 0.6607，OOT-dev 0.6564，耗时 337.1s

## 回归对比：v8 vs v7b（fast 档）

**结论：有退步需看（2 项）**

| 数据集 | 指标 | 新 | 旧 | 差 | 配对数 | 判定 | 变化最大的运行 |
|---|---|---|---|---|---|---|---|
| hotel_bookings | holdout | 0.7911 | 0.7935 | -0.0024 | 2 | 无显著变化 | v8_hotel_bookings_s0 |
| hotel_bookings | delta_b1 | 0.0072 | 0.0105 | -0.0033 | 2 | 无显著变化 | v8_hotel_bookings_s0 |
| hotel_bookings | wasted_rate | 0.0625 | 0.0 | +0.0625 | 2 | 无显著变化 | v8_hotel_bookings_s0 |
| hotel_bookings | rejected_decisions | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v8_hotel_bookings_s0 |
| hotel_bookings | upgrade_hit_rate | 1.0 | 1.0 | +0.0 | 1 | 无显著变化 | v8_hotel_bookings_s0 |
| hotel_bookings | fe_in_final | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v8_hotel_bookings_s0 |
| lending_club | holdout | 0.6604 | 0.6621 | -0.0017 | 2 | 无显著变化 | v8_lending_club_s0 |
| lending_club | delta_b1 | -0.0004 | 0.0035 | -0.0038 | 2 | 无显著变化 | v8_lending_club_s0 |
| lending_club | wasted_rate | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v8_lending_club_s0 |
| lending_club | rejected_decisions | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v8_lending_club_s0 |
| lending_club | fe_in_final | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v8_lending_club_s0 |

**硬性标记**

- cost_up：wall_sec 308.2 → 754.1（阈值 ×1.33）
- cost_up：cost_usd 0.01136 → 0.02358（阈值 ×1.34）

## 过程指标

### hotel_bookings

| 种子 | 无效轮次 | 被拒决策 | 决策失败收尾 | 升全量命中 | 特征工程进最终模型 | 最终模型来源 |
|---|---|---|---|---|---|---|
| 0 | 12% | 0 |  | 100% |  | 赛跑 catboost → 升全量 |
| 1 | 0% | 0 |  | — |  | 赛跑 catboost → 升全量 |

### lending_club

| 种子 | 无效轮次 | 被拒决策 | 决策失败收尾 | 升全量命中 | 特征工程进最终模型 | 最终模型来源 |
|---|---|---|---|---|---|---|
| 0 | 0% | 0 |  | 100% |  | 赛跑 lgbm → 升全量 |
| 1 | 0% | 0 |  | 100% |  | 赛跑 catboost → 升全量 |

## 情景用例

| 情景 | 底座 | 结果 | 原因 | 任务 |
|---|---|---|---|---|
| leak | lending_club | 通过 | 泄漏列 acct_review_score 被识别并隔离，没有进入被采纳的实验，holdout 0.6270 未超过 B1 + 0.05 | v8_scn_leak_lending_club |
| missing_spec | lending_club | 通过 | 实验前就停下问 label_def，推荐「二分类违约标签：loan_status 属于 Charged Off、Defau」，确认后跑完 | v8_scn_missing_spec_lending_club |
| small_oot | lending_club | 通过 | MDE 0.0688 超过目标，已警告并进入决策上下文；没有不显著的采纳 | v8_scn_small_oot_lending_club |
| no_signal | lending_club | 通过 | 最终 holdout auc=0.4938，比随机水平高 -0.0062 | v8_scn_no_signal_lending_club |
