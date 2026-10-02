"""接入阶段：驱动 IntakeSession，每一步结束存 intake.json。
事件同步写进任务的 EventLog，类型加 intake. 前缀：建模阶段也有 profile、llm_call，避免两段混在一起；
接入和建模因此共用一条事件流（SSE 只读这一张表）。"""
import json

from agent.llm import make_llm, redact
from data.knowhow import load_config
from data.sample import ROOT
from intake.session import IntakeFailed, IntakeSession
from runtime.events import EventLog
from server import tasks
from server.mock_intake import HotelIntakeMock

_live: set[str] = set()         # 本进程里正在起草的任务：busy 标记还在但不在这里 → 起草被中断（web 服务重启）


def schedule(tid: str):
    """API 在放进后台任务之前调用：先登记再置 busy，后台任务开始前的空档里 refresh 不会误判。"""
    _live.add(tid)
    tasks.update(tid, busy=True)


def refresh(meta: dict) -> dict:
    if meta.get("busy") and meta["id"] not in _live:
        return tasks.update(meta["id"], status="intake_failed", busy=False, error="接入被中断（web 服务重启或出错），请重试")
    return meta


def _guarded(fn):
    """入口统一兜底：step() 之外出错（读 intake.json、answer 本身）也落到"接入失败"，不会停在"起草中"。"""
    def run(tid: str, *a):
        _live.add(tid)
        try:
            fn(tid, *a)
        except Exception as e:
            _fail(tid, e)
        finally:
            _live.discard(tid)
    return run


def _llm(provider: str):
    return HotelIntakeMock() if provider == "mock" else make_llm(provider, load_config(str(ROOT / "config.yaml")))


def _save(tid: str, s: IntakeSession, n0: int):
    log = EventLog(str(tasks.events_db(tid)), tid)
    for e in s.events[n0:]:
        log.append("intake." + e["type"], e["payload"])
    (tasks.task_dir(tid) / "intake.json").write_text(json.dumps(s.to_json(), ensure_ascii=False, default=str))


def _fail(tid: str, e: Exception):
    msg = "；".join(e.args[0]) if isinstance(e, IntakeFailed) else redact(f"{type(e).__name__}: {str(e)[:300]}")
    tasks.update(tid, status="intake_failed", busy=False, error=msg)


def _step(tid: str, s: IntakeSession, n0: int):
    try:
        s.step()
    except Exception as e:          # IntakeFailed：多次起草都没过校验；其他：LLM 调用出错（断网、key 不对）
        _save(tid, s, n0)
        _fail(tid, e)
        return
    _save(tid, s, n0)
    if s.written:
        _report(tid, s)
    tasks.update(tid, status="config_ready" if s.written else "intake", busy=False, error=None)


def _report(tid: str, s: IntakeSession):
    """写出配置后上报一次 Langfuse（配了 LANGFUSE_* 才会上报），与 app.py 的做法一致。"""
    from intake.tracing import report_intake
    from runtime.tracing import langfuse_client
    lf = langfuse_client()
    if lf is None:
        return
    p = tasks.load(tid)["provider"]
    model = "mock" if p == "mock" else load_config(str(ROOT / "config.yaml"))["llm"]["providers"][p]["model"]
    t = report_intake({"dataset_id": s.dataset_id, "llm": p, "model": model, "description": s.description,
                       "events": s.events, "written": s.written}, lf)
    tasks.update(tid, intake_langfuse_url=lf.get_trace_url(trace_id=t) if t else None)


@_guarded
def start(tid: str):
    meta, d = tasks.load(tid), tasks.task_dir(tid)
    try:
        s = IntakeSession(_llm(meta["provider"]), str(d / "input" / "data.csv"), (d / "input" / "description.md").read_text(),
                          tasks.dataset_id(meta), str(d / "config"))
    except Exception as e:          # 数据读不了（不是 CSV 等）
        _fail(tid, e)
        return
    _step(tid, s, 0)


def _load(tid: str) -> IntakeSession:
    state = json.loads((tasks.task_dir(tid) / "intake.json").read_text())
    return IntakeSession.from_json(state, _llm(tasks.load(tid)["provider"]))


@_guarded
def answer(tid: str, replies: dict):
    s = _load(tid)
    n0 = len(s.events)
    s.answer(replies)
    _step(tid, s, n0)


@_guarded
def retry(tid: str):
    if not (tasks.task_dir(tid) / "intake.json").exists():       # 第一轮之前就失败了：从头开始
        start(tid)
        return
    s = _load(tid)
    _step(tid, s, len(s.events))
