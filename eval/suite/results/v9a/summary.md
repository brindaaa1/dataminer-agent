# 评测 v9a

设置：max_rounds=8，low_sample_frac=0.3，low_trials=10，full_trials=20，split_seed=42，autonomy=L0，llm=deepseek，scenario={'max_rounds': 5, 'n_rows': 20000, 'seed': 0}，tier=fast，seeds=[0, 1]

## hotel_bookings

| 种子 | 状态 | 最终模型 | holdout | Δ vs B0 | OOT-dev | 过拟合 OOT-dev | 轮数 | 被拒决策 | LLM 失败 | 停止原因 | 输入 token（命中/未命中） | 输出（推理） | 美元 | 耗时(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | ok | catboost | 0.7913 | +0.0152 | 0.8303 | 是 | 4 | 0 | 0 | NO_PROGRESS | 5120/13593 | 14258（10505） | 0.0059 | 432.3 |
| 1 | ok | catboost | 0.7944 | +0.0183 | 0.8283 | 是 | 4 | 0 | 0 | NO_PROGRESS | 7808/10410 | 12904（8463） | 0.0051 | 651.3 |

- 赢 B0 的种子：2/2；本数据集花费合计 0.0110 美元
- B0：holdout 0.7761，OOT-dev 0.8195，耗时 0.4s
- B1：holdout 0.7833，OOT-dev 0.8251，耗时 432.7s

## lending_club

| 种子 | 状态 | 最终模型 | holdout | Δ vs B0 | OOT-dev | 过拟合 OOT-dev | 轮数 | 被拒决策 | LLM 失败 | 停止原因 | 输入 token（命中/未命中） | 输出（推理） | 美元 | 耗时(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | ok | lgbm | 0.6589 | +0.0115 | 0.6588 |  | 4 | 0 | 0 | NO_PROGRESS | 5116/13158 | 14039（9688） | 0.0058 | 156.9 |
| 1 | ok | lgbm | 0.6612 | +0.0138 | 0.6555 |  | 4 | 0 | 0 | NO_PROGRESS | 5116/13257 | 11831（7696） | 0.0052 | 134.4 |

- 赢 B0 的种子：2/2；本数据集花费合计 0.0110 美元
- B0：holdout 0.6474，OOT-dev 0.6410，耗时 0.2s
- B1：holdout 0.6627，OOT-dev 0.6581，耗时 134.4s

## 回归对比：v9a vs v8（fast 档）

**结论：无退步**

| 数据集 | 指标 | 新 | 旧 | 差 | 配对数 | 判定 | 变化最大的运行 |
|---|---|---|---|---|---|---|---|
| hotel_bookings | holdout | 0.7928 | 0.7911 | +0.0018 | 2 | 无显著变化 | v9a_hotel_bookings_s1 |
| hotel_bookings | delta_b1 | 0.0096 | 0.0072 | +0.0024 | 2 | 无显著变化 | v9a_hotel_bookings_s1 |
| hotel_bookings | wasted_rate | 0.0 | 0.0625 | -0.0625 | 2 | 无显著变化 | v9a_hotel_bookings_s1 |
| hotel_bookings | rejected_decisions | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v9a_hotel_bookings_s0 |
| hotel_bookings | fe_in_final | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v9a_hotel_bookings_s0 |
| lending_club | holdout | 0.66 | 0.6604 | -0.0004 | 2 | 无显著变化 | v9a_lending_club_s1 |
| lending_club | delta_b1 | -0.0027 | -0.0004 | -0.0023 | 2 | 无显著变化 | v9a_lending_club_s1 |
| lending_club | wasted_rate | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v9a_lending_club_s0 |
| lending_club | rejected_decisions | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v9a_lending_club_s0 |
| lending_club | fe_in_final | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v9a_lending_club_s0 |

**用时与花费**（配对种子合计）

| 数据集 | 项 | 新 | 旧 | 倍数 | 标记阈值 |
|---|---|---|---|---|---|
| hotel_bookings | 用时（秒） | 1084 | 2896 | ×0.37 | ×3.31 |
| hotel_bookings | 花费（美元） | 0.011 | 0.0177 | ×0.62 | ×1.33 |
| lending_club | 用时（秒） | 291.3 | 754.1 | ×0.39 | ×1.33 |
| lending_club | 花费（美元） | 0.011 | 0.0236 | ×0.47 | ×1.34 |

## 过程指标

### hotel_bookings

| 种子 | 无效轮次 | 被拒决策 | 决策失败收尾 | 升全量命中 | 特征工程进最终模型 | 最终模型来源 |
|---|---|---|---|---|---|---|
| 0 | 0% | 0 |  | — |  | 赛跑 catboost → 升全量 |
| 1 | 0% | 0 |  | — |  | 赛跑 catboost → 升全量 |

### lending_club

| 种子 | 无效轮次 | 被拒决策 | 决策失败收尾 | 升全量命中 | 特征工程进最终模型 | 最终模型来源 |
|---|---|---|---|---|---|---|
| 0 | 0% | 0 |  | — |  | 赛跑 lgbm → 升全量 |
| 1 | 0% | 0 |  | — |  | 赛跑 lgbm → 升全量 |

## 情景用例

| 情景 | 底座 | 结果 | 原因 | 任务 |
|---|---|---|---|---|
| leak | lending_club | 通过 | 泄漏列 acct_review_score 被识别并隔离，没有进入被采纳的实验，holdout 0.6350 未超过 B1 + 0.05 | v9a_scn_leak_lending_club |
| missing_spec | lending_club | 通过 | 实验前就停下问 label_def，推荐「二分类违约标签：loan_status ∈ {Charged Off, Defa」，确认后跑完 | v9a_scn_missing_spec_lending_club |
| small_oot | lending_club | 通过 | MDE 0.0688 超过目标，已警告并进入决策上下文；没有不显著的采纳 | v9a_scn_small_oot_lending_club |
| no_signal | lending_club | 通过 | 最终 holdout auc=0.4981，比随机水平高 -0.0019 | v9a_scn_no_signal_lending_club |
