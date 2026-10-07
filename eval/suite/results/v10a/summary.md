# 评测 v10a

设置：max_rounds=6，low_sample_frac=0.3，low_trials=10，full_trials=20，split_seed=42，autonomy=L0，llm=deepseek，scenario={'max_rounds': 5, 'n_rows': 20000, 'seed': 0}，tier=fast，seeds=[0, 1]

## lending_club

| 种子 | 状态 | 最终模型 | holdout | Δ vs B0 | OOT-dev | 过拟合 OOT-dev | 轮数 | 被拒决策 | LLM 失败 | 停止原因 | 输入 token（命中/未命中） | 输出（推理） | 美元 | 耗时(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | ok | lgbm | 0.6597 | +0.0123 | 0.6524 |  | 4 | 0 | 0 | NO_PROGRESS | 3580/10820 | 10269（6553） | 0.0044 | 172.4 |
| 1 | ok | lgbm | 0.6616 | +0.0142 | 0.6545 |  | 4 | 0 | 0 | NO_PROGRESS | 3964/10665 | 9522（5850） | 0.0042 | 108.7 |

- 赢 B0 的种子：2/2；本数据集花费合计 0.0086 美元
- B0：holdout 0.6474，OOT-dev 0.6410，耗时 0.1s
- B1：holdout 0.6586，OOT-dev 0.6522，耗时 108.7s

## 回归对比

不能和 v9b 对比：{'max_rounds': (6, 8)}

## 过程指标

### lending_club

| 种子 | 无效轮次 | 被拒决策 | 决策失败收尾 | 升全量命中 | 特征工程进最终模型 | 最终模型来源 |
|---|---|---|---|---|---|---|
| 0 | 0% | 0 |  | — |  | 赛跑 lgbm → 升全量 |
| 1 | 0% | 0 |  | — |  | 赛跑 lgbm → 升全量 |

