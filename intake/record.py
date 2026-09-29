"""录制一次真实的接入会话，供演示页回放。分步执行，便于在两次调用之间写回答：

    python -m intake.record start --llm deepseek        # 起草，打印问题；状态存到 --state
    python -m intake.record answer answers.json         # 读回答，继续起草；没有问题时写出 YAML 并与手写配置对比
    python -m intake.record rewrite                     # 不调用 LLM：用当前代码从最后一版草稿重新写出 YAML、重新对比
    python -m intake.record export                      # 导出 examples/traces/intake/intake_hotel_deepseek.json
    python -m intake.record upload                      # 把导出的会话补报到 Langfuse（不调用 LLM），打印 trace 链接
"""
import argparse
import json
from pathlib import Path

from agent.llm import make_llm
from data.knowhow import load_config
from data.sample import SAMPLE_CSV
from intake.compare import compare
from intake.session import IntakeSession
from runtime.env import ROOT, load_env

DESC = SAMPLE_CSV.parent / "业务说明.md"
DATASET_ID = "hotel_intake"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["start", "answer", "rewrite", "export", "upload"])
    ap.add_argument("answers", nargs="?")
    ap.add_argument("--llm", default="deepseek")
    ap.add_argument("--state", default="artifacts/intake/record_state.json")
    a = ap.parse_args()
    load_env()
    st_path = Path(a.state)
    if a.cmd == "upload":
        from intake.tracing import report_intake
        from runtime.tracing import langfuse_client
        client = langfuse_client()
        tid = report_intake(json.loads((ROOT / "examples" / "traces" / "intake" / "intake_hotel_deepseek.json").read_text()), client)
        print(client.get_trace_url(trace_id=tid) if tid else "没有配置 LANGFUSE_*，未上报")
    elif a.cmd == "rewrite":
        from intake.schemas import IntakeDraft
        from intake.writer import write
        state = json.loads(st_path.read_text())
        last = [e for e in state["events"] if e["type"] == "draft"][-1]["payload"]
        d = IntakeDraft.model_validate({**last, **{k: v["raw"] for k, v in state["confirmed"].items()}})   # 与写出时相同：必答项用确认过的值
        w = write(d, state["dataset_id"], state["csv"], state["out_dir"])
        for e in state["events"]:
            if e["type"] == "written":
                e["payload"] = {k: v for k, v in w.items() if k != "csv"}
            if e["type"] == "compare":
                e["payload"] = compare(DATASET_ID, w["knowhow_root"], "knowhow", str(SAMPLE_CSV))
        state["written"] = w
        st_path.write_text(json.dumps(state, ensure_ascii=False, indent=1))
        print("已重新写出并对比")
    elif a.cmd == "export":
        state = json.loads(st_path.read_text())
        out = {"dataset_id": state["dataset_id"], "llm": state["llm_name"],
               "model": load_config()["llm"]["providers"][state["llm_name"]]["model"], "description": state["description"],
               "csv": str(SAMPLE_CSV.relative_to(ROOT)), "events": state["events"], "written": state["written"],
               "compare": next(e["payload"] for e in state["events"] if e["type"] == "compare")}
        p = ROOT / "examples" / "traces" / "intake" / "intake_hotel_deepseek.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1).replace(str(ROOT) + "/", ""))
        print(p)
    else:
        if a.cmd == "start":
            name = a.llm
            s = IntakeSession(make_llm(name, load_config()), str(SAMPLE_CSV), DESC.read_text(), DATASET_ID, "artifacts/intake")
        else:
            state = json.loads(st_path.read_text())
            name = state["llm_name"]
            s = IntakeSession.from_json(state, make_llm(name, load_config()))
            s.answer(json.loads(Path(a.answers).read_text()))
        qs = s.step()
        if qs:
            print(json.dumps([q.model_dump() for q in qs], ensure_ascii=False, indent=1))
        else:
            s._log("compare", compare(DATASET_ID, s.written["knowhow_root"], "knowhow", str(SAMPLE_CSV)))
            print("已写出 YAML：", s.written["knowhow_root"])
        st_path.parent.mkdir(parents=True, exist_ok=True)
        st_path.write_text(json.dumps({**s.to_json(), "llm_name": name}, ensure_ascii=False, indent=1))
