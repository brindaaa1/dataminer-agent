"""情景 b：缺规格。从数据字典副本删掉一个必答项（默认 label 定义；可配观察时间）。OOT 窗口来自 suggested_split，
删了切分就做不出来，不在可选项里。通过条件：在任何实验之前停下问人且带推荐答案；确认前 spec 不变；
模拟回答"确认"后跑到结束。"""
from eval.scenarios.common import Variant, copy_knowhow, edit_dictionary, load_main, write_variant

REMOVERS = {"label_def": lambda d: d["label"].pop("definition", None),
            "observation_time_col": lambda d: (d.pop("observation_time_col", None), d.pop("observation_time", None))}


def build(base, cfg, out, params):
    key = params.get("key", "label_def")
    kh = copy_knowhow(base, cfg, out)
    edit_dictionary(kh, base, REMOVERS[key])
    df = load_main(base, cfg, params.get("n_rows"), params.get("seed", 0))
    return Variant(write_variant(base, cfg, df, out), str(kh), answers={key: "确认"})


def check(events, records, params):
    key = params.get("key", "label_def")
    wait = next((e for e in events if e["type"] == "await_human"), None)
    if not wait:
        return {"passed": False, "reason": f"缺 {key} 却没有停下问人", "evidence": []}
    first_work = min((e["id"] for e in events if e["type"] in ("profile", "evaluated")), default=None)
    if first_work is not None and first_work < wait["id"]:
        return {"passed": False, "reason": "先开始了数据画像 / 实验才问人", "evidence": [wait["id"]]}
    q = next((x for x in wait["payload"].get("questions", []) if isinstance(x, dict) and x.get("key") == key), None)
    if not q or not q.get("recommended"):
        return {"passed": False, "reason": f"问 {key} 时没有给推荐答案", "evidence": [wait["id"]]}
    ui = next((e["id"] for e in events if e["type"] == "USER_INSTRUCTION"), None)
    conf = next((e["id"] for e in events if e["type"] == "spec_confirmed" and e["payload"].get("key") == key), None)
    if conf is None or ui is None or conf < ui:
        return {"passed": False, "reason": f"{key} 没有在人确认之后才写进 spec", "evidence": [wait["id"]]}
    if not any(e["type"] == "state_transition" and e["payload"].get("to") == "DONE" for e in events):
        return {"passed": False, "reason": "确认之后没有跑到结束", "evidence": [conf]}
    return {"passed": True, "reason": f"实验前就停下问 {key}，推荐「{q['recommended'][:40]}」，确认后跑完", "evidence": [wait["id"], conf]}
