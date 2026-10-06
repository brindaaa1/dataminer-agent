"""评测数据集登记表：评测名 → know-how 里的数据集名 + 表路径。抽样文件由 eval/suite/prepare_data.py 生成。"""
from data.sample import SAMPLE_CSV
from eval.suite.prepare_data import tables

DATASETS = {
    "hotel_sample": {"dataset": "hotel_bookings", "tables": {"main": str(SAMPLE_CSV)}},     # 只做冒烟，不进汇总结论
    "hotel_bookings": {"dataset": "hotel_bookings", "tables": tables("hotel_bookings")},
    "lending_club": {"dataset": "lending_club", "tables": tables("lending_club")},
    "home_credit": {"dataset": "home_credit", "tables": tables("home_credit")},
}
