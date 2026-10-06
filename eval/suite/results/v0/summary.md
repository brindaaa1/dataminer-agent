# 评测 v0

设置：max_rounds=5，low_sample_frac=0.3，low_trials=10，full_trials=20，split_seed=42，autonomy=L0，llm=deepseek，seeds=[0, 1, 2]

## hotel_sample

| 种子 | 状态 | 最终模型 | holdout | Δ vs B0 | OOT-dev | 过拟合 OOT-dev | 轮数 | 被拒决策 | LLM 失败 | 停止原因 | 输入 token（命中/未命中） | 输出（推理） | 美元 | 耗时(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | ok | 无最终模型 | — | — | — |  | 3 | 0 | 0 | NO_PROGRESS | 1792/10884 | 10832（7735） | 0.0046 | 170.1 |
| 1 | 失败：AttributeError: 'RandomForestSpec' object has no attribute ' | | | | | | | | | | 11264/11379 | 25991（21976） | 0.0089 | |
| 2 | ok | lr_scorecard | 0.7412 | -0.0207 | 0.8165 | 是 | 5 | 0 | 0 | LLM_STOP: final_model 已 ACCEPT | 4992/15658 | 13455（8666） | 0.0060 | 96.8 |

- 赢 B0 的种子：0/3；本数据集花费合计 0.0194 美元
- B0：holdout 0.7619，OOT-dev 0.8257，耗时 0.6s
- B1：holdout 0.7772，OOT-dev 0.8372，耗时 170.1s
