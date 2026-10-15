# Deepchecks、ydata-profiling 能直接找出多少数据问题

### lending_club · deepchecks（10 秒）

- 找出来：3/8（38%）
- 每次都必须找到的：找到了
- 其实不是问题的：6 个

没找到的：

- lending_club.mths_since_last_delinq（sentinel）
- lending_club.revol_util（sentinel）
- lending_club.earliest_cr_line（sentinel）
- lending_club.last_pymnt_score（post_outcome）
- lending_club.None（duplicate_rows）

哪些检查对上了答案：

- Feature-Feature Correlation：2
- String Mismatch：1

### lending_club · ydata（3 秒）

- 找出来：1/8（12%）
- 每次都必须找到的：找到了
- 其实不是问题的：18 个

没找到的：

- lending_club.issue_d（missing_months）
- lending_club.revol_util（sentinel）
- lending_club.earliest_cr_line（sentinel）
- lending_club.id（duplicate_key）
- lending_club.home_ownership（category_variants）
- lending_club.last_pymnt_score（post_outcome）
- lending_club.None（duplicate_rows）

哪些检查对上了答案：

- MISSING：1

### hotel_bookings · deepchecks（12 秒）

- 找出来：5/9（56%）
- 每次都必须找到的：找到了
- 其实不是问题的：2 个

没找到的：

- hotel_bookings.assigned_room_type（post_outcome，原生）
- hotel_bookings.required_car_parking_spaces（post_outcome，原生）
- hotel_bookings.lead_time（sentinel）
- hotel_bookings.adr（sentinel）

哪些检查对上了答案：

- Feature-Feature Correlation：2
- Feature Label Correlation：1
- String Mismatch：1
- Data Duplicates：1

### hotel_bookings · ydata（1 秒）

- 找出来：3/9（33%）
- 每次都必须找到的：找到了
- 其实不是问题的：11 个

没找到的：

- hotel_bookings.arrival_date_month（missing_months）
- hotel_bookings.None（duplicate_rows，原生）
- hotel_bookings.reservation_status（post_outcome，原生）
- hotel_bookings.reservation_status_date（post_outcome，原生）
- hotel_bookings.assigned_room_type（post_outcome，原生）
- hotel_bookings.meal（category_variants）

哪些检查对上了答案：

- ZEROS：6

### home_credit · deepchecks（55 秒）

- 找出来：8/14（57%）
- 每次都必须找到的：找到了
- 其实不是问题的：62 个

没找到的：

- previous_application.DAYS_FIRST_DRAWING（sentinel，原生）
- application_train.CODE_GENDER（sentinel，原生）
- application_train.AMT_REQ_CREDIT_BUREAU_YEAR（sentinel）
- application_train.SK_ID_CURR（duplicate_key）
- application_train.dpd_after_3m（post_outcome）
- bureau.DAYS_CREDIT（after_cutoff）

哪些检查对上了答案：

- Feature-Feature Correlation：8
- String Mismatch：1

### home_credit · ydata（8 秒）

- 找出来：8/14（57%）
- 每次都必须找到的：没找到
- 其实不是问题的：119 个

没找到的：

- application_train.DAYS_EMPLOYED（sentinel，原生）
- application_train.CODE_GENDER（sentinel，原生）
- application_train.ORGANIZATION_TYPE（sentinel，原生）
- application_train.SK_ID_CURR（duplicate_key）
- application_train.NAME_INCOME_TYPE（category_variants）
- bureau.DAYS_CREDIT（after_cutoff）

哪些检查对上了答案：

- MISSING：7
- UNIQUE：1
- ZEROS：1

## 结论

用的版本：Deepchecks 0.19.1、ydata-profiling 4.18.4。Deepchecks 不支持 numpy 2 和新版 scikit-learn，只能在 numpy 1.26、scikit-learn 1.5 下运行，维护已经跟不上。

**上面「找出来」的比例有水分。** 逐条看是哪个检查命中的：Deepchecks 的命中大多来自「特征之间相关」，ydata 的全部来自「有缺失」「有很多 0」「值都不同」，只是碰巧落在答案所在的列上，并没有指出问题本身。只算真正指出问题的：

| 问题 | Deepchecks | ydata-profiling |
|---|---|---|
| 同一类别几种写法 | 3 份数据都找到（String Mismatch） | 没找到 |
| 重复行 | 酒店数据原生的大量重复找到了；Lending Club 里 1% 的重复没报（默认门槛是 5%） | 没找到（minimal 模式不算重复） |
| 结果出来之后才有的字段 | 只找到酒店的 reservation_status；人为放的两列（与结果约九成一致）都没报 | 没找到 |
| 占位值（-1、999、365243、Jan-1900、XNA） | 一个都没真正找出 | 一个都没真正找出 |
| 编号重复 | 没找到 | 没找到 |
| 某几个月数据缺失 | 没找到 | 没找到 |
| 明细表用到申请日之后的记录 | 没找到 | 没找到 |

**直接用现成的**：Deepchecks 的「同一类别几种写法」「重复行」两项可以直接用，重复行的门槛要改成 0。「字段和结果关系强得可疑」只能抓到特别明显的，不能代替现有的单特征 AUC 检查。

**要自己写的**：占位值、编号重复、时间上缺几个月、明细表时间穿越。这几类恰好是业务上最常见、最费精力的。

**其实不是问题的主要来源**：Deepchecks 的「特征之间相关」、ydata 的「有缺失 / 有很多 0 / 分布偏」。这些对找数据问题没用，以后不开。

**ydata-profiling 不用**：它的统计（每个值出现多少次、分布）能用 pandas 几行算出来，下一项试验里的统计就是这么算的。

## 按修正后的打分重新计算（2026-10-15）

检查后改了两处：没有指出列的发现只有说的是重复才算数；每条答案列出是哪个检查对上的。工具结果沿用上面保存的原始输出，没有重跑。

### lending_club · deepchecks

- 找出来：3/8（38%）
- 每次都必须找到的：找到了
- 其实不是问题的：6 个

没找到的：

- lending_club.mths_since_last_delinq（sentinel）
- lending_club.revol_util（sentinel）
- lending_club.earliest_cr_line（sentinel）
- lending_club.last_pymnt_score（post_outcome）
- lending_club.None（duplicate_rows）

每条找到的答案是哪个检查对上的：

- lending_club.issue_d：Feature-Feature Correlation
- lending_club.id：Feature-Feature Correlation
- lending_club.home_ownership：String Mismatch

### lending_club · ydata

- 找出来：1/8（12%）
- 每次都必须找到的：找到了
- 其实不是问题的：18 个

没找到的：

- lending_club.issue_d（missing_months）
- lending_club.revol_util（sentinel）
- lending_club.earliest_cr_line（sentinel）
- lending_club.id（duplicate_key）
- lending_club.home_ownership（category_variants）
- lending_club.last_pymnt_score（post_outcome）
- lending_club.None（duplicate_rows）

每条找到的答案是哪个检查对上的：

- lending_club.mths_since_last_delinq：MISSING

### hotel_bookings · deepchecks

- 找出来：5/9（56%）
- 每次都必须找到的：找到了
- 其实不是问题的：2 个

没找到的：

- hotel_bookings.assigned_room_type（post_outcome，原生）
- hotel_bookings.required_car_parking_spaces（post_outcome，原生）
- hotel_bookings.lead_time（sentinel）
- hotel_bookings.adr（sentinel）

每条找到的答案是哪个检查对上的：

- hotel_bookings.arrival_date_month：Feature-Feature Correlation
- hotel_bookings.None：Data Duplicates
- hotel_bookings.reservation_status：Feature Label Correlation
- hotel_bookings.reservation_status_date：Feature-Feature Correlation
- hotel_bookings.meal：String Mismatch

### hotel_bookings · ydata

- 找出来：3/9（33%）
- 每次都必须找到的：找到了
- 其实不是问题的：11 个

没找到的：

- hotel_bookings.arrival_date_month（missing_months）
- hotel_bookings.None（duplicate_rows，原生）
- hotel_bookings.reservation_status（post_outcome，原生）
- hotel_bookings.reservation_status_date（post_outcome，原生）
- hotel_bookings.assigned_room_type（post_outcome，原生）
- hotel_bookings.meal（category_variants）

每条找到的答案是哪个检查对上的：

- hotel_bookings.required_car_parking_spaces：ZEROS
- hotel_bookings.lead_time：ZEROS
- hotel_bookings.adr：ZEROS

### home_credit · deepchecks

- 找出来：8/14（57%）
- 每次都必须找到的：找到了
- 其实不是问题的：62 个

没找到的：

- previous_application.DAYS_FIRST_DRAWING（sentinel，原生）
- application_train.CODE_GENDER（sentinel，原生）
- application_train.AMT_REQ_CREDIT_BUREAU_YEAR（sentinel）
- application_train.SK_ID_CURR（duplicate_key）
- application_train.dpd_after_3m（post_outcome）
- bureau.DAYS_CREDIT（after_cutoff）

每条找到的答案是哪个检查对上的：

- application_train.DAYS_EMPLOYED：Feature-Feature Correlation
- previous_application.DAYS_FIRST_DUE：Feature-Feature Correlation
- previous_application.DAYS_LAST_DUE_1ST_VERSION：Feature-Feature Correlation
- previous_application.DAYS_LAST_DUE：Feature-Feature Correlation
- previous_application.DAYS_TERMINATION：Feature-Feature Correlation
- application_train.ORGANIZATION_TYPE：Feature-Feature Correlation
- application_train.OWN_CAR_AGE：Feature-Feature Correlation
- application_train.NAME_INCOME_TYPE：Feature-Feature Correlation、String Mismatch

### home_credit · ydata

- 找出来：8/14（57%）
- 每次都必须找到的：没找到
- 其实不是问题的：119 个

没找到的：

- application_train.DAYS_EMPLOYED（sentinel，原生）
- application_train.CODE_GENDER（sentinel，原生）
- application_train.ORGANIZATION_TYPE（sentinel，原生）
- application_train.SK_ID_CURR（duplicate_key）
- application_train.NAME_INCOME_TYPE（category_variants）
- bureau.DAYS_CREDIT（after_cutoff）

每条找到的答案是哪个检查对上的：

- previous_application.DAYS_FIRST_DRAWING：MISSING
- previous_application.DAYS_FIRST_DUE：MISSING
- previous_application.DAYS_LAST_DUE_1ST_VERSION：MISSING
- previous_application.DAYS_LAST_DUE：MISSING
- previous_application.DAYS_TERMINATION：MISSING
- application_train.OWN_CAR_AGE：MISSING
- application_train.AMT_REQ_CREDIT_BUREAU_YEAR：MISSING、ZEROS
- application_train.dpd_after_3m：UNIQUE
