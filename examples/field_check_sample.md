# 字段校验报告：lending_club  (OK)

## 表 main（151 列）
- know-how 声明但表头缺失（data_dictionary）: 无
- know-how 声明但表头缺失（blacklist）: 无
- 前缀规则未命中任何列: 无
- 数据里有、know-how 未登记（按 unknown 处理）: 83 列
  - 前缀 `hardship_` 命中 12 列
  - 前缀 `settlement_` 命中 5 列
  - 前缀 `orig_projected_additional_accrued_interest` 命中 1 列
  - 前缀 `deferral_term` 命中 1 列
  - 前缀 `payment_plan_start_date` 命中 1 列
  - 未登记列: acc_open_past_24mths, annual_inc_joint, avg_cur_bal, bc_open_to_buy, bc_util, chargeoff_within_12_mths, debt_settlement_flag_date, deferral_term, delinq_amnt, disbursement_method, dti_joint, hardship_amount, hardship_dpd, hardship_end_date, hardship_last_payment_amount, hardship_length, hardship_loan_status, hardship_payoff_balance_amount, hardship_reason, hardship_start_date, hardship_status, hardship_type, inq_fi, max_bal_bc, mo_sin_old_il_acct, mo_sin_old_rev_tl_op, mo_sin_rcnt_rev_tl_op, mo_sin_rcnt_tl, mort_acc, mths_since_last_major_derog, mths_since_rcnt_il, mths_since_recent_bc, mths_since_recent_bc_dlq, mths_since_recent_inq, mths_since_recent_revol_delinq, num_accts_ever_120_pd, num_actv_bc_tl, num_actv_rev_tl, num_bc_sats, num_bc_tl, num_il_tl, num_op_rev_tl, num_rev_accts, num_rev_tl_bal_gt_0, num_sats, num_tl_120dpd_2m, num_tl_30dpd, num_tl_90g_dpd_24m, num_tl_op_past_12m, open_act_il, open_il_24m, open_rv_12m, open_rv_24m, orig_projected_additional_accrued_interest, payment_plan_start_date, pct_tl_nvr_dlq, percent_bc_gt_75, revol_bal_joint, sec_app_chargeoff_within_12_mths, sec_app_collections_12_mths_ex_med, sec_app_earliest_cr_line, sec_app_fico_range_high, sec_app_fico_range_low, sec_app_inq_last_6mths, sec_app_mort_acc, sec_app_mths_since_last_major_derog, sec_app_num_rev_accts, sec_app_open_acc, sec_app_open_act_il, sec_app_revol_util, settlement_amount, settlement_date, settlement_percentage, settlement_status, settlement_term, tax_liens, tot_hi_cred_lim, total_bal_ex_mort, total_bal_il, total_bc_limit, total_cu_tl, total_il_high_credit_limit, verification_status_joint