# hotel_bookings_sample.csv：出处与许可

本文件是 Hotel Booking Demand 数据集的抽样子集，只用于快速体验和演示，不用于得出任何建模结论。

## 出处

Nuno Antonio, Ana de Almeida, Luis Nunes. *Hotel booking demand datasets*. Data in Brief, 22 (2019) 41–49.
https://doi.org/10.1016/j.dib.2018.11.126

原文说明数据随论文一同发布（"Data is supplied with the paper"），并已删除所有可识别酒店和客户的字段。

## 许可

原论文及随文数据以 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 发布，允许再分发和改编，条件是署名并注明改动。本子集沿用同一许可。

## 改动

- 获取来源：TidyTuesday 2020-02-11 整理版 `hotels.csv`（两家酒店合并为一张表，119,390 行，文件 SHA-1 `72916f571ced08096945c2c632535d198dc09ab9`）。该文件的列名和行数与 Kaggle "Hotel booking demand" 中的 `hotel_bookings.csv` 一致。
- 按 到店年月 × `is_canceled` 分层抽取约 6.7%（7,997 行，随机种子 42），保持原行序。
- 除删除未抽中的行外，列、取值和缺失值写法均未改动。

复现：`python data_sample/make_hotel_sample.py <完整的 hotel_bookings.csv 路径>`

原作者不为本项目或本子集背书。
