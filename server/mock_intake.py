"""mock provider 的接入起草：不调 LLM，按上传数据的列名认出是仓库自带的哪份样本，返回它的配置。
- 酒店样本：仓库手写的配置（knowhow/hotel_bookings）。
- 电价样本（Elec2）：录制的一次真实接入里用户确认过的配置（data_sample/electricity/mock_draft.json，含历史值）。
用来在没有 key 时走通整条流程；上传别的数据会因字段对不上而校验失败。必答项的确认问题由 IntakeSession 的代码补出。"""
import json

from agent.llm import last_json_block
from data.knowhow import load_config
from data.sample import ROOT
from intake.writer import from_knowhow_dir

ELEC = ROOT / "data_sample" / "electricity" / "mock_draft.json"


class MockIntake:
    def complete(self, system, user, temperature=0.0):
        cols = set(last_json_block(user)["profile"]["columns"])
        if cols == set(json.loads(ELEC.read_text())["fields"]):
            return ELEC.read_text(), 0
        cfg = load_config(str(ROOT / "config.yaml"))
        return from_knowhow_dir("hotel_bookings", str(ROOT / "knowhow"), cfg["task_specs"]["hotel_bookings"]).model_dump_json(), 0
