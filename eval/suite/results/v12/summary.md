# 评测 v12

设置：max_rounds=6，low_sample_frac=0.3，low_trials=10，full_trials=20，split_seed=42，autonomy=L0，llm=deepseek，scenario={'max_rounds': 5, 'n_rows': 20000, 'seed': 0}，tier=fast，seeds=[0, 1]

## lending_club

| 种子 | 状态 | 最终模型 | holdout | Δ vs B0 | OOT-dev | 过拟合 OOT-dev | 轮数 | 被拒决策 | LLM 失败 | 停止原因 | 输入 token（命中/未命中） | 输出（推理） | 美元 | 耗时(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | ok | catboost | 0.6607 | +0.0133 | 0.6539 |  | 4 | 0 | 0 | NO_PROGRESS | 3456/11210 | 14442（9896） | 0.0056 | 397.1 |
| 1 | ok | catboost | 0.6625 | +0.0151 | 0.6594 |  | 4 | 0 | 0 | NO_PROGRESS | 3456/11179 | 11391（7279） | 0.0048 | 494.8 |

- 赢 B0 的种子：2/2；本数据集花费合计 0.0104 美元
- B0：holdout 0.6474，OOT-dev 0.6410，耗时 0.1s
- B1：holdout 0.6607，OOT-dev 0.6564，耗时 300.0s

## hotel_bookings

| 种子 | 状态 | 最终模型 | holdout | Δ vs B0 | OOT-dev | 过拟合 OOT-dev | 轮数 | 被拒决策 | LLM 失败 | 停止原因 | 输入 token（命中/未命中） | 输出（推理） | 美元 | 耗时(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | ok | catboost | 0.7856 | +0.0095 | 0.8241 |  | 4 | 0 | 0 | NO_PROGRESS | 3456/10266 | 15481（11935） | 0.0058 | 806.8 |
| 1 | ok | catboost | 0.7941 | +0.0181 | 0.8331 |  | 6 | 1 | 0 | MAX_ROUNDS | 9600/16804 | 23032（18057） | 0.0088 | 1375.1 |

- 赢 B0 的种子：2/2；本数据集花费合计 0.0146 美元
- B0：holdout 0.7761，OOT-dev 0.8195，耗时 0.3s
- B1：holdout 0.7766，OOT-dev 0.8230，耗时 300.5s

## 回归对比：v12 vs v11（fast 档）

**结论：有退步需看（1 项）**

| 数据集 | 指标 | 新 | 旧 | 差 | 配对数 | 判定 | 变化最大的运行 |
|---|---|---|---|---|---|---|---|
| lending_club | holdout | 0.6616 | 0.6622 | -0.0006 | 2 | 无显著变化 | v12_lending_club_s0 |
| lending_club | delta_b1 | 0.0009 | 0.0014 | -0.0006 | 2 | 无显著变化 | v12_lending_club_s0 |
| lending_club | wasted_rate | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v12_lending_club_s0 |
| lending_club | rejected_decisions | 0.0 | 0.5 | -0.5 | 2 | 无显著变化 | v12_lending_club_s0 |
| lending_club | fe_in_final | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v12_lending_club_s0 |
| hotel_bookings | holdout | 0.7899 | 0.7838 | +0.006 | 2 | 无显著变化 | v12_hotel_bookings_s1 |
| hotel_bookings | delta_b1 | 0.0132 | 0.0072 | +0.006 | 2 | 无显著变化 | v12_hotel_bookings_s1 |
| hotel_bookings | wasted_rate | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v12_hotel_bookings_s0 |
| hotel_bookings | rejected_decisions | 0.5 | 0.0 | +0.5 | 2 | 无显著变化 | v12_hotel_bookings_s1 |
| hotel_bookings | fe_in_final | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v12_hotel_bookings_s0 |

**用时与花费**（配对种子合计）

| 数据集 | 项 | 新 | 旧 | 倍数 | 标记阈值 |
|---|---|---|---|---|---|
| lending_club | 花费（美元） | 0.0104 | 0.0104 | ×1.0 | ×1.3 |
| hotel_bookings | 花费（美元） | 0.0146 | 0.0107 | ×1.36 | ×1.68 |

**硬性标记**

- new_reject_kind：新出现的被拒原因：other（v12_hotel_bookings_s1）

## 过程指标

### lending_club

| 种子 | 无效轮次 | 被拒决策 | 决策失败收尾 | 升全量命中 | 特征工程进最终模型 | 最终模型来源 |
|---|---|---|---|---|---|---|
| 0 | 0% | 0 |  | — |  | 赛跑 catboost → 升全量 |
| 1 | 0% | 0 |  | — |  | 赛跑 catboost → 升全量 |

### hotel_bookings

| 种子 | 无效轮次 | 被拒决策 | 决策失败收尾 | 升全量命中 | 特征工程进最终模型 | 最终模型来源 |
|---|---|---|---|---|---|---|
| 0 | 0% | 0 |  | — |  | 赛跑 catboost → 升全量 |
| 1 | 0% | 1 |  | 100% |  | 赛跑 catboost → 升全量 → 删特征 |

## 情景用例

| 情景 | 底座 | 结果 | 原因 | 任务 |
|---|---|---|---|---|
| no_signal | lending_club | 通过 | 最终 holdout auc=0.4909，比随机水平高 -0.0091 | v12_scn_no_signal_lending_club |
| missing_spec | lending_club | 通过 | 实验前就停下问 label_def，推荐「label = 1（坏）当 loan_status ∈ {Charged Off」，确认后跑完 | v12_scn_missing_spec_lending_club |
| small_oot | lending_club | 通过 | MDE 0.0688 超过目标，已警告并进入决策上下文；没有不显著的采纳 | v12_scn_small_oot_lending_club |
| leak | lending_club | 通过 | 泄漏列 acct_review_score 被识别并隔离，没有进入被采纳的实验，holdout 0.6428 未超过 B1 + 0.05 | v12_scn_leak_lending_club |
