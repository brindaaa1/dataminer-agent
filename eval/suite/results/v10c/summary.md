# 评测 v10c

设置：max_rounds=6，low_sample_frac=0.3，low_trials=10，full_trials=20，split_seed=42，autonomy=L0，llm=deepseek，scenario={'max_rounds': 5, 'n_rows': 20000, 'seed': 0}，tier=fast，seeds=[0, 1]

## hotel_bookings

| 种子 | 状态 | 最终模型 | holdout | Δ vs B0 | OOT-dev | 过拟合 OOT-dev | 轮数 | 被拒决策 | LLM 失败 | 停止原因 | 输入 token（命中/未命中） | 输出（推理） | 美元 | 耗时(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | ok | catboost | 0.7922 | +0.0161 | 0.8300 |  | 6 | 0 | 0 | MAX_ROUNDS | 6272/15401 | 17854（13655） | 0.0072 | 469.7 |
| 1 | ok | catboost | 0.7956 | +0.0195 | 0.8302 |  | 6 | 0 | 0 | MAX_ROUNDS | 6272/16963 | 18279（13137） | 0.0075 | 638.5 |

- 赢 B0 的种子：2/2；本数据集花费合计 0.0147 美元
- B0：holdout 0.7761，OOT-dev 0.8195，耗时 0.4s
- B1：holdout 0.7833，OOT-dev 0.8251，耗时 469.9s

## lending_club

| 种子 | 状态 | 最终模型 | holdout | Δ vs B0 | OOT-dev | 过拟合 OOT-dev | 轮数 | 被拒决策 | LLM 失败 | 停止原因 | 输入 token（命中/未命中） | 输出（推理） | 美元 | 耗时(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | ok | lgbm | 0.6554 | +0.0080 | 0.6549 |  | 4 | 0 | 0 | NO_PROGRESS | 3964/10393 | 11951（8326） | 0.0048 | 133.4 |
| 1 | ok | catboost | 0.6596 | +0.0122 | 0.6580 |  | 4 | 0 | 0 | NO_PROGRESS | 3964/10228 | 8980（5383） | 0.0040 | 2374.5 |

- 赢 B0 的种子：2/2；本数据集花费合计 0.0088 美元
- B0：holdout 0.6474，OOT-dev 0.6410，耗时 0.5s
- B1：holdout 0.6537，OOT-dev 0.6433，耗时 133.4s

## 回归对比：v10c vs v10b（fast 档）

**结论：有退步需看（1 项）**

| 数据集 | 指标 | 新 | 旧 | 差 | 配对数 | 判定 | 变化最大的运行 |
|---|---|---|---|---|---|---|---|
| lending_club | holdout | 0.6575 | 0.6555 | +0.002 | 2 | 无显著变化 | v10c_lending_club_s0 |
| lending_club | delta_b1 | 0.0037 | -0.0031 | +0.0069 | 2 | ↑ 变好 | v10c_lending_club_s0 |
| lending_club | wasted_rate | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v10c_lending_club_s0 |
| lending_club | rejected_decisions | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v10c_lending_club_s0 |
| lending_club | fe_in_final | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v10c_lending_club_s0 |

**用时与花费**（配对种子合计）

| 数据集 | 项 | 新 | 旧 | 倍数 | 标记阈值 |
|---|---|---|---|---|---|
| hotel_bookings | 用时（秒） | 0 | 0 | — | ×1.3 |
| hotel_bookings | 花费（美元） | 0 | 0 | — | ×1.68 |
| lending_club | 用时（秒） | 2508 | 235.8 | ×10.64 | ×1.3 |
| lending_club | 花费（美元） | 0.0088 | 0.0088 | ×0.99 | ×1.3 |

**硬性标记**

- cost_up：wall_sec 235.8 → 2508（阈值 ×1.3）

## 过程指标

### hotel_bookings

| 种子 | 无效轮次 | 被拒决策 | 决策失败收尾 | 升全量命中 | 特征工程进最终模型 | 最终模型来源 |
|---|---|---|---|---|---|---|
| 0 | 17% | 0 |  | 100% |  | 赛跑 catboost → 升全量 |
| 1 | 17% | 0 |  | 100% |  | 赛跑 catboost → 升全量 |

### lending_club

| 种子 | 无效轮次 | 被拒决策 | 决策失败收尾 | 升全量命中 | 特征工程进最终模型 | 最终模型来源 |
|---|---|---|---|---|---|---|
| 0 | 0% | 0 |  | — |  | 赛跑 lgbm → 升全量 |
| 1 | 0% | 0 |  | — |  | 赛跑 catboost → 升全量 |

## 情景用例

| 情景 | 底座 | 结果 | 原因 | 任务 |
|---|---|---|---|---|
| leak | lending_club | 通过 | 泄漏列 acct_review_score 被识别并隔离，没有进入被采纳的实验，holdout 0.6168 未超过 B1 + 0.05 | v10c_scn_leak_lending_club |
| missing_spec | lending_club | 通过 | 实验前就停下问 label_def，推荐「二分类违约标签：loan_status 属于 Charged Off、Defau」，确认后跑完 | v10c_scn_missing_spec_lending_club |
| small_oot | lending_club | 通过 | MDE 0.0688 超过目标，已警告并进入决策上下文；没有不显著的采纳 | v10c_scn_small_oot_lending_club |
| no_signal | lending_club | 通过 | 最终 holdout auc=0.5090，比随机水平高 0.0090 | v10c_scn_no_signal_lending_club |
