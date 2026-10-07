"""把一次接入会话上报到 Langfuse，命名与建模运行一致：trace 名 dataminer · intake · <数据集 id>，session intake_<数据集 id>。

  intake <id>（agent）
  ├─ 数据画像（event）
  ├─ 第 n 轮起草（span）
  │   ├─ 起草 · LLM（generation：prompt、回复）
  │   └─ 校验失败 / 草稿 / 追问 / 用户回答（event）
  ├─ 写出配置（event：生成的 YAML）
  └─ 与手写配置对比（event）

会话按事件顺序上报：现场接入写出配置后调用一次；已录制的会话用 python -m intake.record upload 补报，不重新调用 LLM。
"""
import contextlib

from runtime.tracing import langfuse_client

NAMES = {"validation_failed": "校验失败", "draft": "草稿", "questions": "追问"}


def report_intake(rec: dict, client=None) -> str | None:
    """rec：录制文件的内容（dataset_id、llm、model、description、events、written、compare）。
    返回 trace id；未启用 Langfuse 时返回 None。链接用 client.get_trace_url(trace_id=...) 另查（要联网）。"""
    client = client or langfuse_client()
    if client is None:
        return None
    try:
        from langfuse import propagate_attributes
    except ImportError:
        propagate_attributes = lambda **kw: contextlib.nullcontext()
    did, ev = rec["dataset_id"], rec["events"]
    attrs = lambda: propagate_attributes(session_id=f"intake_{did}", trace_name=f"dataminer · intake · {did}",
                                         tags=["intake", rec.get("llm") or ""], metadata={"dataset": did, "model": rec.get("model") or ""})
    with attrs():
        root = client.start_observation(name=f"intake {did}", as_type="agent", input={"description": rec["description"]})
        root.create_event(name="数据画像", output=next(e["payload"] for e in ev if e["type"] == "profile"))
        for r in range(1, max(e["round"] for e in ev) + 1):
            span = root.start_observation(name=f"第 {r} 轮起草", as_type="span")
            for e in ev:
                if e["round"] != r or e["type"] in ("written", "compare", "answers"):
                    continue
                if e["type"] == "llm_call":
                    g = span.start_observation(name="起草 · LLM", as_type="generation", model=rec.get("model"), input=e["payload"]["prompt"])
                    g.update(output=e["payload"]["response"])
                    g.end()
                else:
                    span.create_event(name=NAMES[e["type"]], output=e["payload"],
                                      level="WARNING" if e["type"] == "validation_failed" else "DEFAULT")
            for e in ev:
                if e["type"] == "answers" and e["round"] == r:
                    span.create_event(name="用户回答", output=e["payload"])
            span.end()
        root.create_event(name="写出配置", output=rec["written"]["yaml"])
        if rec.get("compare"):                                    # 只有录制的酒店会话有手写配置可比
            root.create_event(name="与手写配置对比", output=rec["compare"])
        root.update(output=rec["written"]["task_spec"])
        root.end()
    client.flush()
    return root.trace_id
