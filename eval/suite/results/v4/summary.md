# 评测 v4

设置：max_rounds=5，low_sample_frac=0.3，low_trials=10，full_trials=20，split_seed=42，autonomy=L0，llm=deepseek，seeds=[0, 1, 2]

## lending_club

| 种子 | 状态 | 最终模型 | holdout | Δ vs B0 | OOT-dev | 过拟合 OOT-dev | 轮数 | 被拒决策 | LLM 失败 | 停止原因 | 输入 token（命中/未命中） | 输出（推理） | 美元 | 耗时(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | ok | catboost | 0.6620 | +0.0147 | 0.6574 |  | 4 | 0 | 0 | NO_PROGRESS | 4992/13071 | 13547（9478） | 0.0056 | 161.2 |
| 1 | ok | catboost | 0.6624 | +0.0150 | 0.6586 |  | 4 | 0 | 1 | NO_PROGRESS | 7680/9968 | 13963（10299） | 0.0053 | 1933.5 |
| 2 | ok | random_forest | 0.6587 | +0.0113 | 0.6513 |  | 4 | 0 | 0 | NO_PROGRESS | 4992/12738 | 11044（7012） | 0.0049 | 150.9 |

- 赢 B0 的种子：3/3；本数据集花费合计 0.0159 美元
- B0：holdout 0.6474，OOT-dev 0.6410，耗时 0.1s
- B1：holdout 0.6607，OOT-dev 0.6564，耗时 161.2s
