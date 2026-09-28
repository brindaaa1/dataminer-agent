"""从完整的 hotel_bookings.csv 抽出随仓库分发的小样本（许可与出处见 data_sample/hotel_bookings/NOTICE.md）。

按 到店年月 × is_canceled 分层等比例抽样，保证四段时间切分（train/valid、oot_dev、holdout）都有样本、取消率与全量一致。
除删掉未抽中的行外不做任何改动（列、取值、NULL/NA 写法都保持原样）。

    python data_sample/make_hotel_sample.py data_raw/hotel_bookings/hotel_bookings.csv
"""
import sys
from pathlib import Path

import pandas as pd

N, SEED = 8000, 42
OUT = Path(__file__).parent / "hotel_bookings" / "hotel_bookings_sample.csv"


def sample(src: str, n: int = N, seed: int = SEED) -> pd.DataFrame:
    df = pd.read_csv(src, dtype=str, keep_default_na=False)          # 全按字符串读，写回时保持原值
    key = df["arrival_date_year"] + "-" + df["arrival_date_month"] + "-" + df["is_canceled"]
    out = df.groupby(key, group_keys=False).sample(frac=n / len(df), random_state=seed)
    return out.sort_index()                                           # 保持原文件中的行序


if __name__ == "__main__":
    s = sample(sys.argv[1] if len(sys.argv) > 1 else "data_raw/hotel_bookings/hotel_bookings.csv")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT, index=False)
    print(f"{len(s)} 行 → {OUT}")
