# 评测 v7b

设置：max_rounds=8，low_sample_frac=0.3，low_trials=10，full_trials=20，split_seed=42，autonomy=L0，llm=deepseek，scenario={'max_rounds': 5, 'n_rows': 20000, 'seed': 0}，tier=fast，seeds=[0, 1]

## hotel_bookings

| 种子 | 状态 | 最终模型 | holdout | Δ vs B0 | OOT-dev | 过拟合 OOT-dev | 轮数 | 被拒决策 | LLM 失败 | 停止原因 | 输入 token（命中/未命中） | 输出（推理） | 美元 | 耗时(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | ok | catboost | 0.7947 | +0.0187 | 0.8386 | 是 | 8 | 0 | 0 | MAX_ROUNDS | 9728/28546 | 37964（30355） | 0.0147 | 875.2 |
| 1 | ok | catboost | 0.7922 | +0.0161 | 0.8325 | 是 | 7 | 0 | 0 | NO_PROGRESS | 8576/23502 | 28044（21901） | 0.0112 | 640.1 |

- 赢 B0 的种子：2/2；本数据集花费合计 0.0258 美元
- B0：holdout 0.7761，OOT-dev 0.8195，耗时 0.8s
- B1：holdout 0.7830，OOT-dev 0.8262，耗时 639.9s

## lending_club

| 种子 | 状态 | 最终模型 | holdout | Δ vs B0 | OOT-dev | 过拟合 OOT-dev | 轮数 | 被拒决策 | LLM 失败 | 停止原因 | 输入 token（命中/未命中） | 输出（推理） | 美元 | 耗时(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | ok | catboost | 0.6650 | +0.0176 | 0.6533 |  | 4 | 0 | 0 | NO_PROGRESS | 5116/13337 | 10337（6510） | 0.0048 | 135.2 |
| 1 | ok | catboost | 0.6592 | +0.0118 | 0.6571 |  | 4 | 0 | 0 | NO_PROGRESS | 5116/13310 | 16801（12458） | 0.0066 | 173.0 |

- 赢 B0 的种子：2/2；本数据集花费合计 0.0114 美元
- B0：holdout 0.6474，OOT-dev 0.6410，耗时 0.4s
- B1：holdout 0.6586，OOT-dev 0.6522，耗时 135.2s

## 回归对比：v7b vs v7a（fast 档）

**结论：有退步需看（4 项）**（噪声带未校准：用默认值，先跑一次 A/A）

| 数据集 | 指标 | 新 | 旧 | 差 | 配对数 | 判定 | 变化最大的运行 |
|---|---|---|---|---|---|---|---|
| hotel_bookings | holdout | 0.7935 | 0.7774 | +0.016 | 2 | ↑ 变好 | v7b_hotel_bookings_s0 |
| hotel_bookings | delta_b1 | 0.0105 | 0.0013 | +0.0092 | 2 | ↑ 变好 | v7b_hotel_bookings_s0 |
| hotel_bookings | wasted_rate | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v7b_hotel_bookings_s0 |
| hotel_bookings | rejected_decisions | 0.0 | 0.5 | -0.5 | 2 | 无显著变化 | v7b_hotel_bookings_s0 |
| hotel_bookings | upgrade_hit_rate | 1.0 | 1.0 | +0.0 | 1 | 无显著变化 | v7b_hotel_bookings_s0 |
| hotel_bookings | fe_in_final | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v7b_hotel_bookings_s0 |
| lending_club | holdout | 0.6621 | 0.6605 | +0.0016 | 2 | 无显著变化 | v7b_lending_club_s1 |
| lending_club | delta_b1 | 0.0035 | -0.0022 | +0.0057 | 2 | ↑ 变好 | v7b_lending_club_s1 |
| lending_club | wasted_rate | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v7b_lending_club_s0 |
| lending_club | rejected_decisions | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v7b_lending_club_s0 |
| lending_club | fe_in_final | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v7b_lending_club_s0 |

**硬性标记**

- cost_up：wall_sec 459.1 → 1515
- cost_up：cost_usd 0.01945 → 0.02582
- overfit：疑似对 OOT-dev 过拟合（v7b_hotel_bookings_s0）
- overfit：疑似对 OOT-dev 过拟合（v7b_hotel_bookings_s1）

## 过程指标

### hotel_bookings

| 种子 | 无效轮次 | 被拒决策 | 决策失败收尾 | 升全量命中 | 特征工程进最终模型 | 最终模型来源 |
|---|---|---|---|---|---|---|
| 0 | 0% | 0 |  | 100% |  | 赛跑 catboost → 升全量 → 删特征 |
| 1 | 0% | 0 |  | 100% |  | 赛跑 catboost → 升全量 → 删特征 |

### lending_club

| 种子 | 无效轮次 | 被拒决策 | 决策失败收尾 | 升全量命中 | 特征工程进最终模型 | 最终模型来源 |
|---|---|---|---|---|---|---|
| 0 | 0% | 0 |  | — |  | 赛跑 catboost → 升全量 |
| 1 | 0% | 0 |  | — |  | 赛跑 catboost → 升全量 |

## 情景用例

| 情景 | 底座 | 结果 | 原因 | 任务 |
|---|---|---|---|---|
| leak | lending_club | 通过 | 泄漏列 acct_review_score 被识别并隔离，没有进入被采纳的实验，holdout 0.6321 未超过 B1 + 0.05 | v7b_scn_leak_lending_club |
| missing_spec | lending_club | 通过 | 实验前就停下问 label_def，推荐「目标变量定义：贷款在观察期内是否最终坏账。正样本 = loan_status ∈」，确认后跑完 | v7b_scn_missing_spec_lending_club |
| small_oot | lending_club | 通过 | MDE 0.0688 超过目标，已警告并进入决策上下文；没有不显著的采纳 | v7b_scn_small_oot_lending_club |
| no_signal | lending_club | 通过 | 最终 holdout auc=0.5066，比随机水平高 0.0066 | v7b_scn_no_signal_lending_club |
