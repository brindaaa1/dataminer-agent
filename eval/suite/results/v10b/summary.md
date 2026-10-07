# 评测 v10b

设置：max_rounds=6，low_sample_frac=0.3，low_trials=10，full_trials=20，split_seed=42，autonomy=L0，llm=deepseek，scenario={'max_rounds': 5, 'n_rows': 20000, 'seed': 0}，tier=fast，seeds=[0, 1]

## lending_club

| 种子 | 状态 | 最终模型 | holdout | Δ vs B0 | OOT-dev | 过拟合 OOT-dev | 轮数 | 被拒决策 | LLM 失败 | 停止原因 | 输入 token（命中/未命中） | 输出（推理） | 美元 | 耗时(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | ok | lgbm | 0.6545 | +0.0072 | 0.6513 |  | 4 | 0 | 0 | NO_PROGRESS | 3964/9249 | 11579（8045） | 0.0045 | 141.3 |
| 1 | ok | lgbm | 0.6564 | +0.0090 | 0.6539 |  | 4 | 0 | 0 | NO_PROGRESS | 3964/10288 | 10130（6846） | 0.0043 | 94.5 |

- 赢 B0 的种子：2/2；本数据集花费合计 0.0088 美元
- B0：holdout 0.6474，OOT-dev 0.6410，耗时 0.2s
- B1：holdout 0.6586，OOT-dev 0.6522，耗时 94.5s

## 回归对比：v10b vs v10a（fast 档）

**结论：无退步**

| 数据集 | 指标 | 新 | 旧 | 差 | 配对数 | 判定 | 变化最大的运行 |
|---|---|---|---|---|---|---|---|
| lending_club | holdout | 0.6555 | 0.6606 | -0.0052 | 2 | 无显著变化 | v10b_lending_club_s1 |
| lending_club | delta_b1 | -0.0031 | 0.002 | -0.0052 | 2 | 无显著变化 | v10b_lending_club_s1 |
| lending_club | wasted_rate | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v10b_lending_club_s0 |
| lending_club | rejected_decisions | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v10b_lending_club_s0 |
| lending_club | fe_in_final | 0.0 | 0.0 | +0.0 | 2 | 无显著变化 | v10b_lending_club_s0 |

**用时与花费**（配对种子合计）

| 数据集 | 项 | 新 | 旧 | 倍数 | 标记阈值 |
|---|---|---|---|---|---|
| lending_club | 用时（秒） | 235.8 | 281.1 | ×0.84 | ×1.33 |
| lending_club | 花费（美元） | 0.0088 | 0.0086 | ×1.03 | ×1.3 |

## 过程指标

### lending_club

| 种子 | 无效轮次 | 被拒决策 | 决策失败收尾 | 升全量命中 | 特征工程进最终模型 | 最终模型来源 |
|---|---|---|---|---|---|---|
| 0 | 0% | 0 |  | — |  | 赛跑 lgbm → 升全量 |
| 1 | 0% | 0 |  | — |  | 赛跑 lgbm → 升全量 |

