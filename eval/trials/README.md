# 可行性试验

动手做新功能之前，先用小试验回答几个最没把握的问题。结论汇总在 `results/summary.md`，怎么据此安排实现见 `docs/BLUEPRINT.md`。

## 测试数据：人为放进已知问题

在三份干净数据里人为放进已知的数据问题，同时记下标准答案，用来给各种检查打分。代码在 `eval/planted/`：

| 文件 | 做什么 |
|---|---|
| `issues.py` | 每种问题一个函数：占位值、重复行、编号重复、同一类别几种写法、某几个月缺失、结果出来之后才有的字段、明细表用到申请日之后的记录。每个函数返回改过的表和一条答案 |
| `build.py` | 按每份数据的配方放问题；数据原本就有的问题（数据字典里登记的占位值、酒店的事后字段）也写进答案，标为「原生」；最后按成品表重新数一遍每条答案的行数 |
| `score.py` | 把检查找出来的问题和答案按（表，列）对照，算出找出来的比例、必须找到的有没有找到、提出来但其实不是问题的有几个 |

生成：

```bash
python -m eval.suite.prepare_data      # 先抽出干净子集到 data_raw/eval/（只需一次）
python -m eval.planted.build           # 放问题，写到 data_raw/eval/planted/<数据集>/，含 answers.json
```

三份数据的标准答案：酒店 14 条、Lending Club 8 条、Home Credit 14 条。Home Credit 原本就有的 `DAYS_EMPLOYED=365243` 标为每次都必须找到。测试见 `tests/test_planted.py`。

数据本身不提交（`data_raw/` 被忽略）。Lending Club 和 Home Credit 来自 Kaggle，需要自己下载；酒店数据的出处见 `data_sample/hotel_bookings/NOTICE.md`。

## 六项试验

每个用到新库的试验都在 `.venvs/` 下单独装一个环境，不动主环境。原因是这些库普遍跟不上新版依赖（见下表「环境」一列）。

| 试验 | 问题 | 环境 | 运行 | 报告 |
|---|---|---|---|---|
| `existing_checks.py` | Deepchecks、ydata-profiling 能直接找出多少数据问题 | `checks`：Deepchecks 0.19.1、ydata-profiling 4.18.4，需要 numpy<2、scikit-learn<1.6 | `.venvs/checks/bin/python eval/trials/existing_checks.py` | `results/existing_checks.md` |
| `llm_placeholders.py` | 大模型只看每列的统计，能否找出占位值 | 主环境，key 从 `.env` 读 | `python eval/trials/llm_placeholders.py deepseek` | `results/llm_placeholders_deepseek.md`、`_run2.md` |
| `featuretools_cutoff.py` | Featuretools 按截止时间汇总明细表是否可靠 | `ft`：Featuretools 1.31，需要 pandas<3 | `.venvs/ft/bin/python eval/trials/featuretools_cutoff.py` | `results/featuretools_cutoff.md` |
| `agg_gain.py` | 明细表汇总特征能带来多少提升（要先跑上一项） | 主环境 | `python eval/trials/agg_gain.py` | `results/agg_gain.md` |
| `autogluon_vs_race.py` | AutoGluon 是否比现有建模好（约 1 小时） | `ag`：AutoGluon 1.6.3 tabular | `caffeinate -i .venvs/ag/bin/python eval/trials/autogluon_vs_race.py` | `results/autogluon_vs_race.md` |
| `hindsight_memory.py` | Hindsight 记数据经验的效果 | `hs`：hindsight-all，约 1.7 GB；第一次启动要几分钟 | `.venvs/hs/bin/python eval/trials/hindsight_memory.py` | `results/hindsight_memory.md` |

装环境（用 uv）：

```bash
uv venv .venvs/checks --python 3.12
uv pip install --python .venvs/checks/bin/python deepchecks ydata-profiling pandas pyarrow pyyaml \
    "setuptools<81" "numpy<2" "scikit-learn<1.6" "category_encoders<2.7"

uv venv .venvs/ft --python 3.12
uv pip install --python .venvs/ft/bin/python featuretools pyarrow "pandas<3" "setuptools<81"

uv venv .venvs/ag --python 3.12
uv pip install --python .venvs/ag/bin/python -r requirements.txt "autogluon.tabular[lightgbm,catboost,xgboost]"

uv venv .venvs/hs --python 3.12
uv pip install --python .venvs/hs/bin/python hindsight-all python-dotenv
```

## 几点说明

- **结果文件**：`results/` 里的 `.md` 是报告，结尾的「结论」是看完结果后手写的；`.json` 是各检查找出来的原始内容，`_raw.txt` 是大模型的原始回复。重跑大模型试验时加一个标记（如 `deepseek run3`），否则会覆盖第一次的报告。
- **大模型只试了 DeepSeek**：`llm_placeholders.py` 和 `hindsight_memory.py` 都用 `.env` 里的 DeepSeek key。`llm_placeholders.py` 换模型只需换第一个参数（`config.yaml` 里 `llm.providers` 登记过的名字）。
- **「其实不是问题的」不全是误报**：大模型多报的里面，有 10 处是数据字典漏登记的真实占位值。标准答案故意不改，这些单独记在 `results/llm_placeholders_deepseek.md` 里。
- **AutoGluon 试验的参照**：现有 agent 的数字取自 `eval/suite/results/` 里旧版本的评测记录（酒店、Lending Club 用 v13，Home Credit 用 v6），切分相同，但不是同一次运行。
