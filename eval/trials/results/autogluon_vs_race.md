# AutoGluon 是否比现有建模好

每次训练限时 300 秒（与 B1 相同）；切分种子 42。

| 数据 | 方式 | 时间外验证 AUC | 最终检验 AUC | 比 agent 高（最终检验） | 能分辨的最小提升 | agent 来源 |
|---|---|---|---|---|---|---|
| hotel_bookings | agent 最终模型 | 0.8309 | 0.7949 | — | — | v13 |
| hotel_bookings | B1 FLAML | 0.8230 | 0.7766 | -0.0182 | — | v13 |
| hotel_bookings | AutoGluon 单模型（LightGBM） | 0.8219 | 0.7832 | -0.0117 | 0.0083 | v13 |
| hotel_bookings | AutoGluon 融合（WeightedEnsemble_L3） | 0.8322 | 0.7901 | -0.0048 | 0.0080 | v13 |
| lending_club | agent 最终模型 | 0.6565 | 0.6648 | — | — | v13 |
| lending_club | B1 FLAML | 0.6564 | 0.6607 | -0.0041 | — | v13 |
| lending_club | AutoGluon 单模型（CatBoost） | 0.6609 | 0.6649 | +0.0001 | 0.0097 | v13 |
| lending_club | AutoGluon 融合（WeightedEnsemble_L3） | 0.6590 | 0.6644 | -0.0004 | 0.0097 | v13 |
| home_credit | agent 最终模型 | 0.7486 | 0.7567 | — | — | v6 |
| home_credit | B1 FLAML | 0.7470 | 0.7537 | -0.0030 | — | v6 |
| home_credit | AutoGluon 单模型（CatBoost） | 0.7504 | 0.7555 | -0.0012 | 0.0122 | v6 |
| home_credit | AutoGluon 融合（WeightedEnsemble_L3） | 0.7485 | 0.7561 | -0.0006 | 0.0122 | v6 |

## 结论

用的版本：AutoGluon 1.6.3（tabular）。没装 PyTorch，神经网络类模型被跳过，融合模型只用了树模型。

**没有一处高出能分辨的最小提升**：AutoGluon 的单模型和融合模型在三份数据上，最终检验 AUC 和现有 agent 的差距都在 -0.012 到 +0.0001 之间，没有一次高出那份数据能分辨的最小提升（0.008 到 0.012）。酒店数据上反而低了 0.005 到 0.012。

**比较不完全公平，但不影响结论**：AutoGluon 用的是原始宽表，agent 的最终模型可能还用了它加的特征；agent 的数字来自旧版本的运行（v13、v6），切分相同，但不是同一次运行。即便 AutoGluon 在特征上吃亏，它在 Lending Club 和 Home Credit 上也只是打平，没有体现出「融合一定更好」。

**单模型选中的是 LightGBM 或 CatBoost**：和现有赛跑的胜者是同一类模型。融合模型（WeightedEnsemble_L3）要叠两层，解释起来困难得多，换来的效果却没有提高。

**决定**：不换。建模保留现有的赛跑核心，放到统一的建模入口后面。AutoGluon 以后只在需要快速搭一个参照模型时考虑。
