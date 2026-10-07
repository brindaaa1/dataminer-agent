# 评测 v13

设置：max_rounds=6，low_sample_frac=0.3，low_trials=10，full_trials=20，split_seed=42，autonomy=L0，llm=deepseek，scenario={'max_rounds': 5, 'n_rows': 20000, 'seed': 0}，tier=fast，seeds=[0, 1]

## lending_club

| 种子 | 状态 | 最终模型 | holdout | Δ vs B0 | OOT-dev | 过拟合 OOT-dev | 轮数 | 被拒决策 | LLM 失败 | 停止原因 | 输入 token（命中/未命中） | 输出（推理） | 美元 | 耗时(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | ok | catboost | 0.6652 | +0.0178 | 0.6565 |  | 4 | 0 | 0 | NO_PROGRESS | 4480/10662 | 13902（10165） | 0.0054 | 423.1 |
| 1 | ok | catboost | 0.6644 | +0.0170 | 0.6565 |  | 4 | 0 | 0 | NO_PROGRESS | 4480/10586 | 12752（8067） | 0.0051 | 420.0 |

- 赢 B0 的种子：2/2；本数据集花费合计 0.0105 美元
- B0：holdout 0.6474，OOT-dev 0.6410，耗时 0.1s
- B1：holdout 0.6607，OOT-dev 0.6564，耗时 300.0s

## hotel_bookings

| 种子 | 状态 | 最终模型 | holdout | Δ vs B0 | OOT-dev | 过拟合 OOT-dev | 轮数 | 被拒决策 | LLM 失败 | 停止原因 | 输入 token（命中/未命中） | 输出（推理） | 美元 | 耗时(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | ok | catboost | 0.7948 | +0.0187 | 0.8297 |  | 5 | 0 | 0 | NO_PROGRESS | 5632/13413 | 26479（21893） | 0.0093 | 951.8 |
| 1 | ok | catboost | 0.7950 | +0.0189 | 0.8322 |  | 6 | 0 | 0 | MAX_ROUNDS | 7040/17545 | 22110（17074） | 0.0087 | 1059.5 |

- 赢 B0 的种子：2/2；本数据集花费合计 0.0180 美元
- B0：holdout 0.7761，OOT-dev 0.8195，耗时 0.3s
- B1：holdout 0.7766，OOT-dev 0.8230，耗时 300.5s

## 回归对比：v13 vs v12（fast 档）

**结论：无退步**

| 数据集 | 指标 | 新 | 旧 | 差 | 配对数 | 判定 | 变化最大的运行 |
|---|---|---|---|---|---|---|---|
| lending_club | holdout | 0.6648 | 0.6616 | +0.0032 | 2 | 无显著变化 | v13_lending_club_s1 |
| lending_club | delta_b1 | 0.0041 | 0.0009 | +0.0032 | 2 | 无显著变化 | v13_lending_club_s1 |
| lending_club | wasted_rate | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v13_lending_club_s1 |
| lending_club | rejected_decisions | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v13_lending_club_s1 |
| lending_club | fe_in_final | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v13_lending_club_s1 |
| hotel_bookings | holdout | 0.7949 | 0.7899 | +0.005 | 2 | 无显著变化 | v13_hotel_bookings_s1 |
| hotel_bookings | delta_b1 | 0.0182 | 0.0132 | +0.005 | 2 | 无显著变化 | v13_hotel_bookings_s1 |
| hotel_bookings | wasted_rate | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v13_hotel_bookings_s0 |
| hotel_bookings | rejected_decisions | 0.0 | 0.5 | -0.5 | 2 | 无显著变化 | v13_hotel_bookings_s0 |
| hotel_bookings | fe_in_final | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v13_hotel_bookings_s0 |

**用时与花费**（配对种子合计）

| 数据集 | 项 | 新 | 旧 | 倍数 | 标记阈值 |
|---|---|---|---|---|---|
| lending_club | 花费（美元） | 0.0105 | 0.0104 | ×1.01 | ×1.3 |
| hotel_bookings | 花费（美元） | 0.018 | 0.0146 | ×1.23 | ×1.68 |

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
| 1 | 0% | 0 |  | — |  | 赛跑 catboost → 升全量 |

