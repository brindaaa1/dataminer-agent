# Know-how 文件格式

这个目录是"换数据集只换配置、不改代码"的落点：数据集特有的字段语义、可得时间、黑名单、特征模板全部用下面的 YAML 格式描述；`data/`、`features/`、`evaluation/` 下的代码只消费这些文件，不硬编码任何字段名或业务规则。新增一个数据集，只需要照着这份格式写 YAML。

```
knowhow/
├── domains/                     领域手册：按业务领域组织，接入时按业务说明推荐、用户确认
│   ├── _default/playbook.yaml   通用：任何二分类都成立的规则（指标建议、泄漏线索、合理性规则）
│   ├── credit_risk/playbook.yaml 信贷风控（第一份填好的示例）：推荐指标、泄漏线索、特征概念
│   └── _template/playbook.yaml  空白骨架，按注释填写新领域（如用户流失、营销响应）
├── templates/
│   └── <dataset>.yaml           在具体表/字段上实例化的模板
└── <dataset>/
    ├── data_dictionary.yaml     字段语义、可得时间、切分建议、（多表）表结构；domain 声明所属领域
    ├── priors.yaml              目标方向先验（单调约束、模型卡用）
    └── blacklist.yaml           字段使用策略：剔除 / 隔离 / 合规 / 命名模式
```

## `<dataset>/data_dictionary.yaml`

### 单表数据集（`lending_club`、`hotel_bookings`）

```yaml
dataset: <名字>
entity_key: <主键列名>
observation_time_col: <观察时间列>       # 或用 observation_time: {definition, ...} 描述如何从多列拼出观察时间

label:
  source_col: <原始状态列>              # 或直接给 col: <已经是 0/1 的列>
  definition: "人话描述"
  positive: [...]                       # source_col 取这些值 → 1
  negative: [...]                       # source_col 取这些值 → 0
  exclude: [...]                        # 取这些值的整行剔除（如仍在还款中，成熟度不足）
  maturity_rule: "为什么这样限定时间窗口"

suggested_split:
  train_valid: ["起始月", "结束月"]      # valid 从中随机切出
  oot_dev: ["起始月", "结束月"]
  holdout: ["起始月", "结束月"]
  caveat: "如果 OOT 的漂移里混了季节效应之类，在这里写清楚，会被写进模型卡"

fields:
  <字段名>:
    availability: <见下表>
    type: numeric | categorical | date | text | binary | ordinal
    compliance: exclude | review          # 可选
    structural_missing_before: "年月"     # 可选：该字段在此之前整体缺失，跨期会漂移
    note: "人话说明"                      # 可选，会原样出现在模型卡的特征清单里

classic_derived:                          # 可选：广为使用的派生特征，代码会展开并做「可得时间继承」
  - {name: ..., expr: "SQL 表达式（DuckDB 方言）", availability: <声明值，可选，缺省则完全由输入推断>}
```

`availability` 的取值因数据集而异（信贷用"贷前/贷后"，酒店用"预订时/入住后"），但都遵循同一套强弱语义，`evaluation/evaluator.py::scan_features` 按这套语义判断是否标记 `LEAK_SUSPECT`：

| 强弱 | 典型取值 | 含义 |
|---|---|---|
| 安全 | `at_prediction` / `history` / `application` / `loan_terms` / `bureau_at_orig` | 预测时点已知 |
| 策略问题 | `incumbent_model` | 在位模型的输出，合法但要不要用是业务决定 |
| 存疑 → 默认隔离 | `unknown` / `updated_until_event` | 可得时间不确定 |
| 泄漏 → 强制剔除 | `before_outcome` / `post_outcome` / `post_origination` | 预测时点之后才确定 |
| 不参与 | `meta` | 标识符/元数据 |

`classic_derived` 的可得时间继承规则：**取输入字段中最弱的一个**；如果显式声明的 `availability` 比从输入推断出的更强，会被拉回到更弱的那个并在 PROFILE 报告里告警（对应 `data/knowhow.py::derive_availability`）。

### 多表数据集（`home_credit`）

```yaml
dataset: home_credit
label: {source_table: <主表名>, col: <0/1 列>}
suggested_split: {method: stratified_random, train: 0.6, valid: 0.15, oot_dev: 0.1, holdout: 0.15,
                  caveat: "没有绝对时间，OOT 是随机切分近似，模型卡里会注明"}
pit_rules: ["用人话写的过滤规则，供 features/factory.py 的开发者读；结构化的时间过滤见每张表的 time_field"]
sentinel_values:
  - {value: 365243, fields: [...], table: <可选，缺省=主表>, meaning: "占位值的实际含义，会被转成缺失"}

tables:
  <表名>:
    grain: "一行代表什么"
    key: <该表主键>
    parent: {table: <父表>, on: <关联键>}     # 明细表才有；用于多级聚合
    time_field: <事件时间字段>                # 用于 point-in-time 过滤：只保留早于观察时点的记录
    key_fields:
      <字段名>: {type: ..., availability: ..., note: ...}
    classic_derived: ["字符串形式，仅供人读，不参与自动展开"]

blacklist:                                    # 多表数据集把 blacklist 内嵌在 data_dictionary 里，不单独成文件
  compliance_exclude: [...]
  incumbent_model: [...]
  incumbent_policy: include_by_default | exclude_by_default
  incumbent_rationale: "为什么这个数据集的默认策略和别的不一样"
```

## `<dataset>/blacklist.yaml`（单表数据集独立成文件）

```yaml
leakage:
  action: drop_at_load
  fields: [...]                  # 精确字段名，加载时直接剔除
  field_prefixes: [...]          # 前缀匹配（如同一次事件产生的一组字段：hardship_*）
  derived_fields: [...]          # classic_derived 里因继承规则被判定为泄漏的派生特征
  eval_ground_truth: true        # 供泄漏注入实验（放回这些字段、不告诉 agent）用作标准答案

incumbent_model:
  action: exclude_by_default | include_by_default
  fields: [...]
  rationale: "为什么默认这样处理"

compliance:
  exclude: [...]      # 直接剔除（如可能成为受保护特征代理变量的字段）
  review: [...]       # 保留但在模型卡的风险提示里标注

unknown_availability:
  fields: [...]                    # 简单形式；或用下面的完整形式：
  # policy: quarantine
  # derived_fields: [...]
  # eval_ground_truth: suspected   # 泄漏注入实验里单独统计"是否被识别为可疑"（不算作硬性泄漏）

meta:
  action: drop_at_load
  fields: [...]

leakage_patterns: ["正则表达式，不区分大小写；命中新字段名或 run_code 产出即标记 LEAK_SUSPECT"]
```

## `templates/<dataset>.yaml` —— 特征模板

```yaml
templates:
  - id: <模板 ID，会成为生成特征的名字前缀>
    family: <对应领域手册 feature_concepts 里的概念 id，如 multi_lending>
    source: <明细表名>
    entity: <聚合到哪个实体的主键>
    time_field: <用于 point-in-time 过滤的时间列>
    time_unit: days | months
    windows: [7, 30, 90, ...]              # 或 [current]：不做窗口聚合，直接算行级比例特征
    aggs: ["count", "avg(布尔或数值表达式)", "count_distinct(...)", "-min(...)  # 取负变成正向"]
    derived: ["ratio(w1/w2)", "trend(w1 vs w2)", "recency: -max(...)"]
    monotone_prior: +1 | -1 | 0             # 目标方向（+1 = 值越大越可能是正类）；单个 agg 后面可以用注释 `# monotone_prior: -1` 覆盖模板默认值
    rationale: "业务解释，会原样写进模型卡的特征清单"
```

`features/factory.py` 把这份配置展开成 DuckDB SQL：`time_unit: days` 时按 `time_field >= -window` 过滤，`months` 时按 `MONTHS_BALANCE >= -window` 过滤；派生的 `ratio`/`trend` 由同一批聚合结果两两运算得到，`recency` 直接翻译成 SQL 表达式。写成散文（不含运算符的自然语言）的聚合定义会被自动跳过，不会被当成 SQL 执行。

## `domains/<领域>/playbook.yaml` —— 领域手册

字段说明见 `domains/_template/playbook.yaml`。`_default` 打底，具体领域覆盖标量和字典（如 `metric`）、追加列表（`leakage_name_patterns`、`leakage_desc_keywords`、`sanity_rules`）。谁在用：

- 接入新数据：LLM 按业务说明和 `match_keywords` 推荐一个领域，用户在确认环节确认或修改；推荐的主指标和护栏来自手册的 `metric`。
- 规划（PLAN）：手册的标题、正类含义、`feature_concepts`、泄漏线索进入上下文，作为提出特征假设时的概念词表。
- 数据集在 `data_dictionary.yaml` 里用 `domain: <领域名>` 声明；不声明用 `_default`。

## `<dataset>/priors.yaml` —— 目标方向先验

```yaml
target_direction:
  <字段名或特征名>: +1 | -1 | 0     # +1 = 值越大越可能是正类；用于 monotone_constraints；模板产出的特征优先用模板里的 monotone_prior
```

没有先验的特征不加单调约束；LLM 可以提议方向，但要写业务理由并过 compare 才能保留。泄漏的实际判定靠 `blacklist.yaml` 的 `leakage_patterns`（正则）和 `data_dictionary.yaml` 的 `availability` 标注；领域手册的泄漏线索用于接入起草和规划时提醒。

## 新增一个数据集需要做什么

1. 在 `knowhow/<new_dataset>/data_dictionary.yaml` 里给每个字段标 `availability`，写清 `label` 和 `suggested_split`。
2. 在 `blacklist.yaml` 里补 `leakage_patterns`（如果字段名不含"贷后/事后"这类明显词根，就必须靠 `availability` 兜底，见 `docs/DESIGN_NOTES.md` 里关于检出召回率的实测）。
3. 在 `config.yaml` 的 `datasets.<name>` 和 `task_specs.<name>` 里补上文件路径，以及"知识库里写不清楚的 SQL 细节"——时间格式、行过滤条件、DuckDB 方言的表达式覆盖。这些是任务规格，不是数据集的业务逻辑，所以放在 config 而不是 knowhow。
4. **不需要改任何 `.py` 文件**。`hotel_bookings` 就是这样接入的：`data/splits.py`、`features/factory.py`、`evaluation/evaluator.py` 全程只读这份 YAML；接入过程中发现并修掉的几处通用性缺陷（写死的日期格式、误判合法派生特征为泄漏等）记在 `docs/DESIGN_NOTES.md`。
