# 评测 v5

设置：max_rounds=5，low_sample_frac=0.3，low_trials=10，full_trials=20，split_seed=42，autonomy=L0，llm=deepseek，seeds=[0, 1, 2]

## home_credit

| 种子 | 状态 | 最终模型 | holdout | Δ vs B0 | OOT-dev | 过拟合 OOT-dev | 轮数 | 被拒决策 | LLM 失败 | 停止原因 | 输入 token（命中/未命中） | 输出（推理） | 美元 | 耗时(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | ok | lgbm | 0.7513 | +0.0122 | 0.7144 |  | 5 | 0 | 0 | MAX_ROUNDS | 14976/23449 | 17194（11952） | 0.0081 | 8.3 |
| 1 | ok | lgbm | 0.7532 | +0.0142 | 0.7233 |  | 5 | 0 | 0 | MAX_ROUNDS | 14848/22757 | 15047（10510） | 0.0074 | 6.5 |
| 2 | ok | random_forest | 0.7346 | -0.0044 | 0.7155 |  | 5 | 4 | 0 | DECIDE_FAILED | 30078/22556 | 32205（25691） | 0.0123 | 13.8 |

- 赢 B0 的种子：2/3；本数据集花费合计 0.0278 美元
- B0：holdout 0.7390，OOT-dev 0.7074，耗时 0.2s
- B1：holdout 0.7473，OOT-dev 0.7201，耗时 30.1s
