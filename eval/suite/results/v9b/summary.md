# 评测 v9b

设置：max_rounds=8，low_sample_frac=0.3，low_trials=10，full_trials=20，split_seed=42，autonomy=L0，llm=deepseek，scenario={'max_rounds': 5, 'n_rows': 20000, 'seed': 0}，tier=fast，seeds=[0, 1]

## hotel_bookings

| 种子 | 状态 | 最终模型 | holdout | Δ vs B0 | OOT-dev | 过拟合 OOT-dev | 轮数 | 被拒决策 | LLM 失败 | 停止原因 | 输入 token（命中/未命中） | 输出（推理） | 美元 | 耗时(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | ok | catboost | 0.7916 | +0.0155 | 0.8281 | 是 | 8 | 0 | 0 | MAX_ROUNDS | 12544/22882 | 32527（25209） | 0.0123 | 949.9 |
| 1 | ok | random_forest | 0.7803 | +0.0043 | 0.8224 | 是 | 4 | 0 | 0 | NO_PROGRESS | 5120/13400 | 14794（10807） | 0.0060 | 202.0 |

- 赢 B0 的种子：2/2；本数据集花费合计 0.0184 美元
- B0：holdout 0.7761，OOT-dev 0.8195，耗时 0.3s
- B1：holdout 0.7766，OOT-dev 0.8230，耗时 202.0s

## lending_club

| 种子 | 状态 | 最终模型 | holdout | Δ vs B0 | OOT-dev | 过拟合 OOT-dev | 轮数 | 被拒决策 | LLM 失败 | 停止原因 | 输入 token（命中/未命中） | 输出（推理） | 美元 | 耗时(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | ok | catboost | 0.6641 | +0.0167 | 0.6581 |  | 4 | 0 | 0 | NO_PROGRESS | 5116/13173 | 15173（10757） | 0.0061 | 652.9 |
| 1 | ok | lgbm | 0.6594 | +0.0120 | 0.6567 |  | 4 | 1 | 0 | NO_PROGRESS | 9340/13820 | 15914（11025） | 0.0064 | 115.9 |

- 赢 B0 的种子：2/2；本数据集花费合计 0.0125 美元
- B0：holdout 0.6474，OOT-dev 0.6410，耗时 0.1s
- B1：holdout 0.6627，OOT-dev 0.6581，耗时 115.9s

## 回归对比：v9b vs v9a（fast 档）

**结论：有退步需看（3 项）**

| 数据集 | 指标 | 新 | 旧 | 差 | 配对数 | 判定 | 变化最大的运行 |
|---|---|---|---|---|---|---|---|
| hotel_bookings | holdout | 0.786 | 0.7928 | -0.0069 | 2 | 无显著变化 | v9b_hotel_bookings_s1 |
| hotel_bookings | delta_b1 | 0.0093 | 0.0096 | -0.0003 | 2 | 无显著变化 | v9b_hotel_bookings_s1 |
| hotel_bookings | wasted_rate | 0.0625 | 0.0 | +0.0625 | 2 | 无显著变化 | v9b_hotel_bookings_s0 |
| hotel_bookings | rejected_decisions | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v9b_hotel_bookings_s0 |
| hotel_bookings | fe_in_final | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v9b_hotel_bookings_s0 |
| lending_club | holdout | 0.6617 | 0.66 | +0.0017 | 2 | 无显著变化 | v9b_lending_club_s1 |
| lending_club | delta_b1 | -0.001 | -0.0027 | +0.0017 | 2 | 无显著变化 | v9b_lending_club_s1 |
| lending_club | wasted_rate | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v9b_lending_club_s0 |
| lending_club | rejected_decisions | 0.5 | 0.0 | +0.5 | 2 | 无显著变化 | v9b_lending_club_s1 |
| lending_club | fe_in_final | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v9b_lending_club_s0 |

**用时与花费**（配对种子合计）

| 数据集 | 项 | 新 | 旧 | 倍数 | 标记阈值 |
|---|---|---|---|---|---|
| hotel_bookings | 用时（秒） | 1152 | 1084 | ×1.06 | ×3.31 |
| hotel_bookings | 花费（美元） | 0.0184 | 0.011 | ×1.67 | ×1.33 |
| lending_club | 用时（秒） | 768.8 | 291.3 | ×2.64 | ×1.33 |
| lending_club | 花费（美元） | 0.0125 | 0.011 | ×1.14 | ×1.34 |

**硬性标记**

- cost_up：cost_usd 0.011 → 0.01838（阈值 ×1.33）
- cost_up：wall_sec 291.3 → 768.8（阈值 ×1.33）
- new_reject_kind：新出现的被拒原因：duplicate（v9b_lending_club_s1）

## 过程指标

### hotel_bookings

| 种子 | 无效轮次 | 被拒决策 | 决策失败收尾 | 升全量命中 | 特征工程进最终模型 | 最终模型来源 |
|---|---|---|---|---|---|---|
| 0 | 12% | 0 |  | 100% |  | 赛跑 catboost → 升全量 → 删特征 |
| 1 | 0% | 0 |  | — |  | 赛跑 random_forest → 升全量 |

### lending_club

| 种子 | 无效轮次 | 被拒决策 | 决策失败收尾 | 升全量命中 | 特征工程进最终模型 | 最终模型来源 |
|---|---|---|---|---|---|---|
| 0 | 0% | 0 |  | — |  | 赛跑 catboost → 升全量 |
| 1 | 0% | 1 |  | — |  | 赛跑 lgbm → 升全量 |

## 情景用例

| 情景 | 底座 | 结果 | 原因 | 任务 |
|---|---|---|---|---|
| leak | lending_club | 通过 | 泄漏列 acct_review_score 被识别并隔离，没有进入被采纳的实验，holdout 0.6369 未超过 B1 + 0.05 | v9b_scn_leak_lending_club |
| missing_spec | lending_club | 通过 | 实验前就停下问 label_def，推荐「二分类违约 label：loan_status 为 Charged Off、De」，确认后跑完 | v9b_scn_missing_spec_lending_club |
| small_oot | lending_club | 通过 | MDE 0.0688 超过目标，已警告并进入决策上下文；没有不显著的采纳 | v9b_scn_small_oot_lending_club |
| no_signal | lending_club | 通过 | 最终 holdout auc=0.4989，比随机水平高 -0.0011 | v9b_scn_no_signal_lending_club |
