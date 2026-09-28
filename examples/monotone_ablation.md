# A6：单调约束 开 vs 关（lending_club，lgbm，15 trial）

| 配置 | OOT-dev AUC | holdout AUC | 分数 PSI | 分月 AUC 标准差 | 与先验方向相反的特征数 |
|---|---|---|---|---|---|
| 不加约束 | 0.6718 | 0.6772 | 0.0024 | 0.0028 | 1 |
| 加约束 | 0.6720 | 0.6781 | 0.0024 | 0.0029 | 0 |

- 加约束 − 不加约束（OOT-dev，配对 bootstrap）：Δ=+0.0002，95% CI [-0.0009, +0.0014]
- evaluator 判定：**INCONCLUSIVE**
- 与先验相反的特征（不加约束）：['pub_rec_bankruptcies']；（加约束）：无

## 约束的实际覆盖

- 已约束（10）：['annual_inc', 'dti', 'delinq_2yrs', 'earliest_cr_line', 'fico_range_low', 'fico_range_high', 'inq_last_6mths', 'pub_rec', 'revol_util', 'pub_rec_bankruptcies']
- 先验为 0，不约束（2）：['mths_since_last_delinq', 'total_acc']
- 先验与数据相反，跳过（1）：{'loan_amnt': -0.0223}  ← 数据一致性守卫
- 没有先验，不约束：14 个；类别变量：8 个
