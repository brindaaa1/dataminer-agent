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

模型卡（含 Mermaid 实验树）写在 `reports/model_card_<task-id>.md`。

## 6. 核心实验

泄漏检查消融、多表特征工厂、单调约束、跨任务 Memory、中断恢复等实验的结果保存在 `examples/`；产生它们的一次性脚本已移除，可从 git 历史找回。agent 本身的评测（固定数据集、A/A 噪声带、回归对比、情景用例）见 `eval/suite/` 和 `docs/eval/JOURNAL.md`：

```bash
caffeinate -i python -m eval.suite.run --tier fast --label v14 --jobs 4     # macOS 上用 caffeinate 防止睡眠中断
```

## 7. 进度可视化（可选）

Optuna study 存在 SQLite 里，直接用官方 dashboard 看每个 trial 的得分和参数重要性：

```bash
optuna-dashboard sqlite:///artifacts/optuna.db
```

## 工作台

上传数据和业务说明 → 对话确认配置 → 看建模过程（可中止、续跑）→ 确认报告并打包下载。

    pip install -r requirements.txt
    cd web && npm install && npm run build && cd ..
    uvicorn server.app:app --port 8000          # 打开 http://127.0.0.1:8000

- 不填 key：LLM 选 mock，上传 `data_sample/electricity/` 或 `data_sample/hotel_bookings/` 下的 CSV 和业务说明即可走完整流程（mock 的接入起草直接给出录制好的配置：电价用一次 DeepSeek 确认过的草稿，酒店用手写配置）。
- 用真实模型：在 `.env` 里配好对应 provider 的 key，界面上选 provider。
- 左侧任务栏的"示例"是录制的真实运行，只读，不需要 key。
- 每个任务的数据、配置、事件和产出都在 `workspace/tasks/<任务 id>/`；跨任务经验库在 `workspace/memory.db`。
- 开发前端：另开终端 `cd web && npm run dev`，`/api` 会代理到 8000 端口；后端改代码用 `uvicorn server.app:app --reload`。
- 测试：`python -m pytest -q`（后端）、`cd web && npm test`（前端组件）、`cd web && npm run e2e`（端到端，用 mock 走完整流程）。
