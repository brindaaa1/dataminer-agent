# 评测 v7a

设置：max_rounds=8，low_sample_frac=0.3，low_trials=10，full_trials=20，split_seed=42，autonomy=L0，llm=deepseek，scenario={'max_rounds': 5, 'n_rows': 20000, 'seed': 0}，tier=fast，seeds=[0, 1]

## hotel_bookings

| 种子 | 状态 | 最终模型 | holdout | Δ vs B0 | OOT-dev | 过拟合 OOT-dev | 轮数 | 被拒决策 | LLM 失败 | 停止原因 | 输入 token（命中/未命中） | 输出（推理） | 美元 | 耗时(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | ok | lgbm | 0.7801 | +0.0041 | 0.8264 | 是 | 8 | 0 | 0 | MAX_ROUNDS | 8320/27321 | 32002（25049） | 0.0128 | 335.3 |
| 1 | ok | lgbm | 0.7748 | -0.0013 | 0.8243 | 是 | 4 | 1 | 0 | NO_PROGRESS | 9344/13351 | 16945（12116） | 0.0066 | 123.8 |

- 赢 B0 的种子：1/2；本数据集花费合计 0.0194 美元
- B0：holdout 0.7761，OOT-dev 0.8195，耗时 0.4s
- B1：holdout 0.7761，OOT-dev 0.8218，耗时 123.9s

## lending_club

| 种子 | 状态 | 最终模型 | holdout | Δ vs B0 | OOT-dev | 过拟合 OOT-dev | 轮数 | 被拒决策 | LLM 失败 | 停止原因 | 输入 token（命中/未命中） | 输出（推理） | 美元 | 耗时(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | ok | random_forest | 0.6582 | +0.0108 | 0.6506 |  | 4 | 0 | 0 | NO_PROGRESS | 4224/13795 | 14541（10155） | 0.0060 | 172.3 |
| 1 | ok | catboost | 0.6628 | +0.0154 | 0.6565 |  | 7 | 0 | 0 | NO_PROGRESS | 8188/23602 | 20922（13902） | 0.0092 | 237.5 |

- 赢 B0 的种子：2/2；本数据集花费合计 0.0152 美元
- B0：holdout 0.6474，OOT-dev 0.6410，耗时 0.1s
- B1：holdout 0.6627，OOT-dev 0.6581，耗时 172.3s

## 回归对比

没有可比的同档上一版本（旧版本没有 version.json）。

## 过程指标

### hotel_bookings

| 种子 | 无效轮次 | 被拒决策 | 决策失败收尾 | 升全量命中 | 特征工程进最终模型 | 最终模型来源 |
|---|---|---|---|---|---|---|
| 0 | 0% | 0 |  | 100% |  | 赛跑 lgbm → 升全量 |
| 1 | 0% | 1 |  | — |  | 赛跑 lgbm → 升全量 |

### lending_club

| 种子 | 无效轮次 | 被拒决策 | 决策失败收尾 | 升全量命中 | 特征工程进最终模型 | 最终模型来源 |
|---|---|---|---|---|---|---|
| 0 | 0% | 0 |  | — |  | 赛跑 random_forest → 升全量 |
| 1 | 0% | 0 |  | 100% |  | 赛跑 catboost → 升全量 |

## 情景用例

| 情景 | 底座 | 结果 | 原因 | 任务 |
|---|---|---|---|---|
| leak | lending_club | 通过 | 泄漏列 acct_review_score 被识别并隔离，没有进入被采纳的实验，holdout 0.6434 未超过 B1 + 0.05 | v7a_scn_leak_lending_club |
| missing_spec | lending_club | 通过 | 实验前就停下问 label_def，推荐「二分类违约标签（1=坏，0=好）：坏样本为 loan_status ∈ {Cha」，确认后跑完 | v7a_scn_missing_spec_lending_club |
| small_oot | lending_club | 通过 | MDE 0.0688 超过目标，已警告并进入决策上下文；没有不显著的采纳 | v7a_scn_small_oot_lending_club |
| no_signal | lending_club | 通过 | 最终 holdout auc=0.5073，比随机水平高 0.0073 | v7a_scn_no_signal_lending_club |
