"""mock provider 的接入起草：直接返回仓库手写的 hotel 配置（knowhow/hotel_bookings），不看上传的数据。
只对 hotel 样本有意义：用来在没有 key 时走通整条流程；上传别的数据会因字段对不上而校验失败。
必答项的确认问题由 IntakeSession 的代码补出，这里不用出问题。"""
from data.knowhow import load_config
from data.sample import ROOT
from intake.writer import from_knowhow_dir


class HotelIntakeMock:
    def complete(self, system, user, temperature=0.0):
        cfg = load_config(str(ROOT / "config.yaml"))
        d = from_knowhow_dir("hotel_bookings", str(ROOT / "knowhow"), cfg["task_specs"]["hotel_bookings"])
        return d.model_dump_json(), 0
