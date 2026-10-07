# 评测 v11

设置：max_rounds=6，low_sample_frac=0.3，low_trials=10，full_trials=20，split_seed=42，autonomy=L0，llm=deepseek，scenario={'max_rounds': 5, 'n_rows': 20000, 'seed': 0}，tier=fast，seeds=[0, 1]

## lending_club

| 种子 | 状态 | 最终模型 | holdout | Δ vs B0 | OOT-dev | 过拟合 OOT-dev | 轮数 | 被拒决策 | LLM 失败 | 停止原因 | 输入 token（命中/未命中） | 输出（推理） | 美元 | 耗时(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | ok | catboost | 0.6628 | +0.0154 | 0.6589 |  | 4 | 0 | 0 | NO_PROGRESS | 4220/8853 | 11593（8235） | 0.0045 | 339.2 |
| 1 | ok | lgbm | 0.6616 | +0.0142 | 0.6570 |  | 4 | 1 | 0 | NO_PROGRESS | 7804/9629 | 16280（12082） | 0.0059 | 261.4 |

- 赢 B0 的种子：2/2；本数据集花费合计 0.0104 美元
- B0：holdout 0.6474，OOT-dev 0.6410，耗时 0.1s
- B1：holdout 0.6607，OOT-dev 0.6564，耗时 300.0s

## hotel_bookings

| 种子 | 状态 | 最终模型 | holdout | Δ vs B0 | OOT-dev | 过拟合 OOT-dev | 轮数 | 被拒决策 | LLM 失败 | 停止原因 | 输入 token（命中/未命中） | 输出（推理） | 美元 | 耗时(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | ok | lgbm | 0.7758 | -0.0002 | 0.8211 |  | 4 | 0 | 0 | NO_PROGRESS | 3968/9575 | 12828（9046） | 0.0049 | 407.8 |
| 1 | ok | catboost | 0.7918 | +0.0158 | 0.8307 |  | 4 | 0 | 0 | NO_PROGRESS | 3968/9471 | 15891（12625） | 0.0058 | 824.6 |

- 赢 B0 的种子：1/2；本数据集花费合计 0.0107 美元
- B0：holdout 0.7761，OOT-dev 0.8195，耗时 0.3s
- B1：holdout 0.7766，OOT-dev 0.8230，耗时 300.5s

## 回归对比：v11 vs v10c（fast 档）

**结论：有退步需看（1 项）**

| 数据集 | 指标 | 新 | 旧 | 差 | 配对数 | 判定 | 变化最大的运行 |
|---|---|---|---|---|---|---|---|
| lending_club | holdout | 0.6622 | 0.6575 | +0.0047 | 2 | 无显著变化 | v11_lending_club_s1 |
| lending_club | delta_b1 | 0.0014 | 0.0037 | -0.0023 | 2 | 无显著变化 | v11_lending_club_s1 |
| lending_club | wasted_rate | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v11_lending_club_s1 |
| lending_club | rejected_decisions | 0.5 | 0.0 | +0.5 | 2 | 无显著变化 | v11_lending_club_s1 |
| lending_club | fe_in_final | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v11_lending_club_s1 |
| hotel_bookings | holdout | 0.7838 | 0.7939 | -0.01 | 2 | 无显著变化 | v11_hotel_bookings_s0 |
| hotel_bookings | delta_b1 | 0.0072 | 0.0106 | -0.0034 | 2 | 无显著变化 | v11_hotel_bookings_s0 |
| hotel_bookings | wasted_rate | 0.0 | 0.167 | -0.167 | 2 | ↑ 变好 | v11_hotel_bookings_s0 |
| hotel_bookings | rejected_decisions | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v11_hotel_bookings_s0 |
| hotel_bookings | fe_in_final | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v11_hotel_bookings_s0 |

**用时与花费**（配对种子合计）

| 数据集 | 项 | 新 | 旧 | 倍数 | 标记阈值 |
|---|---|---|---|---|---|
| lending_club | 花费（美元） | 0.0104 | 0.0088 | ×1.19 | ×1.3 |
| hotel_bookings | 花费（美元） | 0.0107 | 0.0147 | ×0.73 | ×1.68 |

**硬性标记**

- new_reject_kind：新出现的被拒原因：format（v11_lending_club_s1）

## 过程指标

### lending_club

| 种子 | 无效轮次 | 被拒决策 | 决策失败收尾 | 升全量命中 | 特征工程进最终模型 | 最终模型来源 |
|---|---|---|---|---|---|---|
| 0 | 0% | 0 |  | — |  | 赛跑 catboost → 升全量 |
| 1 | 0% | 1 |  | — |  | 赛跑 lgbm → 升全量 |

### hotel_bookings

| 种子 | 无效轮次 | 被拒决策 | 决策失败收尾 | 升全量命中 | 特征工程进最终模型 | 最终模型来源 |
|---|---|---|---|---|---|---|
| 0 | 0% | 0 |  | — |  | 赛跑 lgbm → 升全量 |
| 1 | 0% | 0 |  | — |  | 赛跑 catboost → 升全量 |

## 情景用例

| 情景 | 底座 | 结果 | 原因 | 任务 |
|---|---|---|---|---|
| missing_spec | lending_club | 通过 | 实验前就停下问 label_def，推荐「二分类违约标签：正样本 = loan_status ∈ {Charged Off」，确认后跑完 | v11_scn_missing_spec_lending_club |
| no_signal | lending_club | 通过 | 最终 holdout auc=0.5185，比随机水平高 0.0185 | v11_scn_no_signal_lending_club |
| small_oot | lending_club | 通过 | MDE 0.0688 超过目标，已警告并进入决策上下文；没有不显著的采纳 | v11_scn_small_oot_lending_club |
| leak | lending_club | 通过 | 泄漏列 acct_review_score 被识别并隔离，没有进入被采纳的实验，holdout 0.6167 未超过 B1 + 0.05 | v11_scn_leak_lending_club |
