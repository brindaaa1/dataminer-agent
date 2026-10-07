# 评测 v3

设置：max_rounds=5，low_sample_frac=0.3，low_trials=10，full_trials=20，split_seed=42，autonomy=L0，llm=deepseek，seeds=[0, 1, 2]

## lending_club

| 种子 | 状态 | 最终模型 | holdout | Δ vs B0 | OOT-dev | 过拟合 OOT-dev | 轮数 | 被拒决策 | LLM 失败 | 停止原因 | 输入 token（命中/未命中） | 输出（推理） | 美元 | 耗时(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | ok | catboost | 0.6620 | +0.0147 | 0.6574 |  | 4 | 0 | 0 | NO_PROGRESS | 4224/14034 | 16551（12092） | 0.0066 | 242.8 |
| 1 | ok | catboost | 0.6624 | +0.0150 | 0.6586 |  | 4 | 1 | 0 | NO_PROGRESS | 8832/13400 | 20297（15109） | 0.0076 | 162.1 |
| 2 | ok | random_forest | 0.6587 | +0.0113 | 0.6513 |  | 4 | 0 | 0 | NO_PROGRESS | 4992/13081 | 13879（9562） | 0.0057 | 164.4 |

- 赢 B0 的种子：3/3；本数据集花费合计 0.0199 美元
- B0：holdout 0.6474，OOT-dev 0.6410，耗时 0.1s
- B1：holdout 0.6586，OOT-dev 0.6522，耗时 164.3s
