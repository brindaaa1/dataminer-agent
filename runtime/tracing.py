"""可选的 Langfuse 上报：运行时把状态机和 LLM 调用实时发到 Langfuse，便于事后细看每一轮。

只有同时满足两点才启用：装了 langfuse（>=4），并且环境变量 / .env 里有 LANGFUSE_PUBLIC_KEY 和 LANGFUSE_SECRET_KEY
（LANGFUSE_HOST 或 LANGFUSE_BASE_URL 可选，默认 Langfuse Cloud）。否则用 NoopTracer，什么也不做。上报出错只记一条警告，不影响运行。

层级和 runtime/trace.py 的分段规则一致：
  task（agent）
  ├─ 准备 / 第 n 轮 / 收尾（span）       ← 每次转入 DECIDE 开新的一轮，转入 FINAL_GATE 进入收尾
  │   └─ 状态（span，如 DECIDE、EVALUATE）
  │       ├─ LLM 调用（generation：prompt、回复、token、耗时）
  │       └─ 事件（decision / evaluated / recorded …）
"""
import contextlib
import logging
import os

from runtime.env import load_env

log = logging.getLogger(__name__)
SKIP_EVENTS = {"llm_call", "state_transition"}          # 前者由 generation 覆盖，后者用来切段
WARN_EVENTS = {"decision_rejected", "plan_invalid", "consolidate_failed", "auto_quarantine", "await_human"}
ERROR_EVENTS = {"decide_failed", "data_fatal"}


class NoopTracer:
    enabled = False

    def url(self):
        return None

    def begin_state(self, state):
        pass

    def on_event(self, type_, payload, exp_id=None):
        pass

    @contextlib.contextmanager
    def generation(self, stage, model, prompt, temperature):
        yield None

    def end_generation(self, g, text=None, tokens=0, error=None):
        pass

    def finish(self, state, output=None, error=None):
        pass


class LangfuseTracer(NoopTracer):
    enabled = True

    def __init__(self, client, task_id: str, meta: dict | None = None):
        try:
            from langfuse import propagate_attributes
        except ImportError:                                 # 测试注入假 client 时可以不装 SDK
            propagate_attributes = lambda **kw: contextlib.nullcontext()
        self.client, self.task_id, self.meta = client, task_id, meta or {}
        self._attrs = lambda: propagate_attributes(session_id=task_id, trace_name=f"dataminer · {task_id}",
                                                   tags=[str(v) for v in (self.meta.get("dataset"), self.meta.get("llm")) if v],
                                                   metadata={k: str(v) for k, v in self.meta.items()})
        self.round_no, self.round, self.state, self.done = 0, None, None, False
        with self._attrs():
            self.root = client.start_observation(name=f"task {task_id}", as_type="agent", input=self.meta)
            self.round = self.root.start_observation(name="准备", as_type="span")

    def url(self):
        """trace 在 Langfuse 上的链接（要联网查项目，按需调用）。"""
        return self._safe(lambda: self.client.get_trace_url(trace_id=self.root.trace_id))

    def _safe(self, fn):
        try:
            return fn()
        except Exception as e:                              # 上报失败不能拖垮建模流程
            log.warning("Langfuse 上报失败：%s", e)
            return None

    def _child(self, parent, **kw):
        with self._attrs():
            return parent.start_observation(**kw)

    def begin_state(self, state):
        self._safe(lambda: setattr(self, "state", self._child(self.round, name=state, as_type="span")))

    def on_event(self, type_, payload, exp_id=None):
        if self.done:
            return
        if type_ == "state_transition":
            self._safe(lambda: self._transition(payload))
            return
        if type_ in SKIP_EVENTS:
            return
        level = "ERROR" if type_ in ERROR_EVENTS else "WARNING" if type_ in WARN_EVENTS else "DEFAULT"
        parent = self.state or self.round
        def _ev():
            with self._attrs():
                parent.create_event(name=type_, output=payload, level=level, metadata={"exp_id": exp_id} if exp_id else None)
        self._safe(_ev)

    def _transition(self, p):
        if self.state:
            self.state.update(output={"next": p.get("to"), "sec": p.get("sec")})
            self.state.end()
            self.state = None
        to = p.get("to")
        if to == "DECIDE":
            self.round.end()
            self.round_no += 1
            self.round = self._child(self.root, name=f"第 {self.round_no} 轮", as_type="span")
        elif to == "FINAL_GATE":
            self.round.end()
            self.round = self._child(self.root, name="收尾", as_type="span")

    @contextlib.contextmanager
    def generation(self, stage, model, prompt, temperature):
        g = self._safe(lambda: self._child(self.state or self.round, name=f"{stage} · LLM", as_type="generation",
                                           model=model, input=prompt, model_parameters={"temperature": temperature}))
        yield g

    def end_generation(self, g, text=None, tokens=0, error=None):
        if g is None:
            return

        def _end():
            if error:
                g.update(level="ERROR", status_message=error)
            else:
                g.update(output=text, usage_details={"total": int(tokens or 0)})
            g.end()
        self._safe(_end)

    def finish(self, state, output=None, error=None):
        if self.done:
            return
        self.done = True

        def _fin():
            for o in (self.state, self.round):
                if o:
                    o.end()
            self.root.update(output={"final_state": state, **(output or {})},
                             **({"level": "ERROR", "status_message": error} if error else {}))
            self.root.end()
            self.client.flush()
        self._safe(_fin)


def langfuse_client():
    """有 key、装了 SDK → Langfuse 客户端；否则 None。建模运行和接入会话共用。"""
    load_env()
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return None
    try:
        from langfuse import Langfuse
    except ImportError:
        log.warning("配置了 LANGFUSE_* 但没有安装 langfuse，跳过上报（pip install 'langfuse>=4'）")
        return None
    return Langfuse(public_key=os.environ["LANGFUSE_PUBLIC_KEY"], secret_key=os.environ["LANGFUSE_SECRET_KEY"],
                    host=os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST") or None)


def make_tracer(task_id: str, meta: dict | None = None, client=None):
    """有 key、装了 SDK → LangfuseTracer；否则 NoopTracer。client 仅供测试注入。"""
    client = client or langfuse_client()
    if client is None:
        return NoopTracer()
    try:
        return LangfuseTracer(client, task_id, meta)
    except Exception as e:
        log.warning("Langfuse 初始化失败，跳过上报：%s", e)
        return NoopTracer()
