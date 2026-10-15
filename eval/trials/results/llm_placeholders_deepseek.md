# 大模型只看统计能否找出占位值（deepseek）

### home_credit

- 找出来：8/10（80%）
- 每次都必须找到的：找到了
- 其实不是问题的：10 个

没找到的：

- application_train.CODE_GENDER（sentinel，原生）
- application_train.AMT_REQ_CREDIT_BUREAU_YEAR（sentinel）

哪些检查对上了答案：

- llm：8

### hotel_bookings

- 找出来：2/2（100%）
- 每次都必须找到的：找到了
- 其实不是问题的：1 个

哪些检查对上了答案：

- llm：2

### lending_club

- 找出来：3/3（100%）
- 每次都必须找到的：找到了
- 其实不是问题的：0 个

哪些检查对上了答案：

- llm：3

## 结论

两次都用 DeepSeek（`.env` 里没有 Anthropic 的 key，Kimi 账户余额不足）。第二次的结果见 `llm_placeholders_deepseek_run2.md`，和第一次基本一致：Home Credit 第一次 8/10，第二次 9/10，另外两份都是全部找到。有一次请求返回了空内容，重跑就好了。

**只看统计就能找出来的**：人为放进去的 -1、999、99、Jan-1900 全部找到；Home Credit 原生的 365243（主表的 `DAYS_EMPLOYED`，以及历史申请表的 5 个日期字段）两次都找到；酒店的 `meal = Undefined` 也找出来了，这是数据原本就有的「未知餐型」。

**没找到的**：`CODE_GENDER = XNA`，在 12 万行里只有 1 行，不在常见值的前 10 个里，也算不上「出现得多」。这种值找不到问题不大，单独一行对模型影响很小。

**「其实不是问题的」大多是标准答案漏了**：Home Credit 报出的 9 到 10 个，几乎都是真的占位值。历史申请表里 9 个字段的 `XNA`（没有数据）和 `XAP`（不适用），以及 `SELLERPLACE_AREA = -1`，数据字典都没有登记。另外，`AMT_REQ_CREDIT_BUREAU_YEAR` 其实找到了，只是表名写成了 bureau。所以真实的效果比打分还要好。这些不补进标准答案，作为用大模型带来的额外收获单独记下（之后写 README 用）：

| 表 | 字段 | 占位值 | 出现次数 | 数据字典里有没有登记 |
|---|---|---|---|---|
| previous_application | SELLERPLACE_AREA | -1 | 248,317 | 没有 |
| previous_application | NAME_CASH_LOAN_PURPOSE | XAP / XNA | 306,169 / 221,299 | 没有 |
| previous_application | CODE_REJECT_REASON | XAP | 446,267（约 81%） | 没有 |
| previous_application | NAME_GOODS_CATEGORY | XNA | 310,392 | 没有 |
| previous_application | NAME_PORTFOLIO | XNA | 119,959 | 没有 |
| previous_application | NAME_PRODUCT_TYPE | XNA | 350,795 | 没有 |
| previous_application | NAME_SELLER_INDUSTRY | XNA | 279,068 | 没有 |
| previous_application | NAME_YIELD_GROUP | XNA | 167,238 | 没有 |
| previous_application | NAME_PAYMENT_TYPE | XNA | 203,343 | 没有 |
| hotel_bookings | meal | Undefined | 409 | 没有（标准答案里只记了人为放的写法变体） |

人工整理的数据字典只登记了 3 个字段的 XNA 和 6 个字段的 365243；大模型只看统计，又多找出了 10 处。

**和现成工具对比**：Deepchecks 和 ydata-profiling 一个占位值都没真正找出来；大模型只看每列的统计，就能找出几乎全部。这一类检查应该由助手自己做。

**统计要补什么**：`column_stats` 现在给的是每列前 10 个常见值、最小值和最大值、1%/50%/99% 分位数，以及最小值和最大值各出现多少次。这些已经够用，可以直接补进 `intake/profile.py`，替换现在只给 5 个样例值的做法。还要补两点：一是明确要求写对表名（这次出过一次错）；二是对出现次数很少的怪值另外列一份清单，这类值不应该靠大模型来发现。

## 按修正后的打分重新计算（第一次，2026-10-15）

### home_credit

- 找出来：8/10（80%）
- 每次都必须找到的：找到了
- 其实不是问题的：10 个

没找到的：

- application_train.CODE_GENDER（sentinel，原生）
- application_train.AMT_REQ_CREDIT_BUREAU_YEAR（sentinel）

每条找到的答案是哪个检查对上的：

- application_train.DAYS_EMPLOYED：llm
- previous_application.DAYS_FIRST_DRAWING：llm
- previous_application.DAYS_FIRST_DUE：llm
- previous_application.DAYS_LAST_DUE_1ST_VERSION：llm
- previous_application.DAYS_LAST_DUE：llm
- previous_application.DAYS_TERMINATION：llm
- application_train.ORGANIZATION_TYPE：llm
- application_train.OWN_CAR_AGE：llm

### hotel_bookings

- 找出来：2/2（100%）
- 每次都必须找到的：找到了
- 其实不是问题的：1 个

每条找到的答案是哪个检查对上的：

- hotel_bookings.lead_time：llm
- hotel_bookings.adr：llm

### lending_club

- 找出来：3/3（100%）
- 每次都必须找到的：找到了
- 其实不是问题的：0 个

每条找到的答案是哪个检查对上的：

- lending_club.mths_since_last_delinq：llm
- lending_club.revol_util：llm
- lending_club.earliest_cr_line：llm
