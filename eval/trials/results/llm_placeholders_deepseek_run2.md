# 大模型只看统计能否找出占位值（deepseek）

### home_credit

- 找出来：9/10（90%）
- 每次都必须找到的：找到了
- 其实不是问题的：9 个

没找到的：

- application_train.CODE_GENDER（sentinel，原生）

哪些检查对上了答案：

- llm：9

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

## 按修正后的打分重新计算（第二次，2026-10-15）

### home_credit

- 找出来：9/10（90%）
- 每次都必须找到的：找到了
- 其实不是问题的：9 个

没找到的：

- application_train.CODE_GENDER（sentinel，原生）

每条找到的答案是哪个检查对上的：

- application_train.DAYS_EMPLOYED：llm
- previous_application.DAYS_FIRST_DRAWING：llm
- previous_application.DAYS_FIRST_DUE：llm
- previous_application.DAYS_LAST_DUE_1ST_VERSION：llm
- previous_application.DAYS_LAST_DUE：llm
- previous_application.DAYS_TERMINATION：llm
- application_train.ORGANIZATION_TYPE：llm
- application_train.OWN_CAR_AGE：llm
- application_train.AMT_REQ_CREDIT_BUREAU_YEAR：llm

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
