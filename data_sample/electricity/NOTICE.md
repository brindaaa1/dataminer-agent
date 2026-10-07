# electricity_nsw.csv：出处与许可

本文件是 Electricity（Elec2）数据集的整理版，用于在工作台里完整体验一次接入和建模流程。

## 出处

M. Harries. *SPLICE-2 Comparative Evaluation: Electricity Pricing*. Technical Report, University of New South Wales, 1999.
数据由 J. Gama 整理发布，OpenML 编号 151（https://www.openml.org/d/151）。

## 许可

OpenML 上该数据集的许可标注为 Public。本整理版沿用同一许可，并注明出处与改动。

## 改动

- 获取来源：OpenML 151 的 CSV（`electricity-normalized`，45,312 行，9 列）。
- 原始的 `date` 列已被归一化到 0~1，不能直接按时间切分。数据是从 1996-05-07 起每半小时一条、每天 48 条的连续记录，因此按行号还原出 `timestamp`（`period` 每 48 行循环一次，还原出的星期与原 `day` 列逐行一致），并删除归一化后的 `date` 列。
- 列名改为更易读的写法：`day` → `day_of_week`，`nswprice` → `nsw_price`，`nswdemand` → `nsw_demand`，`vicprice` → `vic_price`，`vicdemand` → `vic_demand`，`class` → `price_direction`；`period` 由 0~1 还原为 0~47 的时段序号。
- 其余取值（已归一化的价格、需求、输送量）未改动。

## 预测任务

`price_direction` 表示这一时段 NSW 电价相对过去 24 小时平均是涨（UP）还是跌（DOWN）。当前时段的电价就是标签的计算来源，属于泄漏；过去各时段的电价、需求和涨跌在预测时已知，可以作为历史特征。业务说明见 `业务说明.md`。
