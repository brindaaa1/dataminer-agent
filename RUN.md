# 运行说明

## 1. 环境

```bash
pip install -r requirements.txt
cp .env.example .env      # 填入要用的 key；.env 不会被提交
```

## 2. 数据

数据集本体不随仓库分发（单个数据集就有几百 MB 到几 GB），`knowhow/` 里的 YAML 是唯一需要提交的部分。去 Kaggle 下载后按下面的结构放好，
路径在 `config.yaml` 的 `datasets` 段配置，可以自行调整：

```
data_raw/
├── lending_club/accepted_2007_to_2018Q4.csv          # Kaggle: Lending Club Loan Data
├── hotel_bookings/hotel_bookings.csv                 # Kaggle: Hotel Booking Demand
└── home_credit/
    ├── application_train.csv                         # Kaggle: Home Credit Default Risk
    ├── bureau.csv
    ├── bureau_balance.csv
    ├── previous_application.csv
    ├── installments_payments.csv
    ├── POS_CASH_balance.csv
    └── credit_card_balance.csv
```

只想看代码怎么运作、不想下载几个 GB 的数据？`tests/` 全部用合成的小数据集（`tests/conftest.py::make_toy`），跑 `pytest` 不需要任何真实数据；
`examples/` 目录也存了几组真实数据集上跑出来的结果，可以直接看。

## 3. 测试

```bash
pytest -q     # 从仓库根目录运行；~100 个测试，几十秒内跑完，不需要数据、不需要网络
```

## 4. 数据切分 + 字段校验（一次性）

```bash
python -m data.splits lending_club        # 也可以是 hotel_bookings / home_credit
```

会打印字段校验报告（know-how 里声明的字段名和真实表头是否一致），并把四段切分写入 `artifacts/`（holdout 单独存放，只有验收模块能读）。

## 5. 跑 agent

```bash
# mock：不联网，几分钟跑完，用来验证状态机本身
python -m eval.run_task --dataset lending_club --task-id demo1 --llm mock --full-trials 15 --max-rounds 5

# 接真实模型：需要 .env 里配好对应的 key
python -m eval.run_task --dataset lending_club --task-id demo2 --llm anthropic --full-trials 15 --max-rounds 5
python -m eval.run_task --dataset lending_club --task-id demo3 --llm kimi      --full-trials 15 --max-rounds 5
```

模型卡（含 Mermaid 实验树）写在 `reports/model_card_<task-id>.md`；对已完成的任务重新生成报告：

```bash
python -m eval.make_report --task-id demo1
```

## 6. 核心实验的复现命令

```bash
# B0 / B1 基线对比
python -m eval.run_baselines --dataset lending_club --agent-exp demo1_e03 --agent-task demo1

# 泄漏注入演示：把泄漏字段放回数据、不告诉 agent，对比 guardrail 开/关
python -m eval.a3_leak_demo --dataset hotel_bookings --clean-task demo1 --include-suspects

# 多表特征工厂 vs 原始宽表
python -m eval.hc_feature_eval --trials 20 --per-template

# 单调约束 开 vs 关
python -m eval.a6_monotone --trials 15

# 跨任务 Memory（小样本，几分钟）
python -m eval.memory_demo

# 训练中途 kill -9，同一条命令恢复
python -m eval.resume_demo
```

这些脚本预先跑好的结果在 `examples/` 里，不需要真的下载数据也能看懂发生了什么。

## 7. 进度可视化（可选）

Optuna study 存在 SQLite 里，直接用官方 dashboard 看每个 trial 的得分和参数重要性：

```bash
optuna-dashboard sqlite:///artifacts/optuna.db
```
