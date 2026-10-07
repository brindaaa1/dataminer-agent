"""LLM 接口。complete() 返回 (文本, token 数)。真实模型从环境变量 ANTHROPIC_API_KEY 读取 key。"""
import contextlib
import json
import os
import re
import threading
import time

from runtime.env import load_env


def extract_json(text: str) -> dict:
    m = re.search(r"```json\s*(.*?)```", text, re.S)
    s = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    return json.loads(s)


def last_json_block(text: str) -> dict:
    blocks = re.findall(r"```json\s*(.*?)```", text, re.S)
    return json.loads(blocks[-1])


class AnthropicLLM:
    def __init__(self, model: str, max_tokens: int = 2000, api_key: str | None = None):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self.model, self.max_tokens = model, max_tokens

    def complete(self, system: str, user: str, temperature: float = 0.0):
        r = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, temperature=temperature,
                                        system=system, messages=[{"role": "user", "content": user}])
        return r.content[0].text, r.usage.input_tokens + r.usage.output_tokens


def _usage(u) -> dict | None:
    """分项用量：缓存命中/未命中的输入、输出、其中的推理部分。没有缓存字段的服务商，输入全部按未命中计。"""
    if u is None:
        return None
    hit = getattr(u, "prompt_cache_hit_tokens", None) or 0
    miss = getattr(u, "prompt_cache_miss_tokens", None)
    det = getattr(u, "completion_tokens_details", None)
    prompt, out = getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0
    return {"input_cache_hit": hit, "input_cache_miss": prompt - hit if miss is None else miss,
            "output": out, "reasoning": (getattr(det, "reasoning_tokens", None) or 0) if det else 0}


class OpenAICompatLLM:
    """OpenAI 兼容接口（Kimi/Moonshot、DeepSeek 等）。key 从环境变量 key_env 读取。"""

    def __init__(self, model: str, base_url: str, key_env: str, max_tokens: int = 2000, client=None,
                 fixed_temperature: float | None = None, api_key: str | None = None, thinking: bool | None = None,
                 max_retries: int = 2, timeout: float | None = None, deadline: float | None = None, deadline_retries: int = 0):
        if client is None:
            import openai
            key = api_key or os.environ.get(key_env)
            if not key:
                raise RuntimeError(f"环境变量 {key_env} 未设置：请在启动前 export {key_env}=...")
            client = openai.OpenAI(api_key=key, base_url=base_url, max_retries=max_retries, timeout=timeout)
        self.client, self.model, self.max_tokens, self.fixed_temperature = client, model, max_tokens, fixed_temperature
        self.thinking, self.last_usage = thinking, None
        self.deadline, self.deadline_retries = deadline, deadline_retries

    def _create(self, **kw):
        """总时长上限：SDK 的 timeout 只管两次收到数据的间隔，服务端一直发保活数据时永远不超时（v10c 一次 PLAN 等了 2132 秒）。
        超时就放弃这次（守护线程，不拖住进程退出）、重新发起。"""
        if not self.deadline:
            return self.client.chat.completions.create(**kw)
        for _ in range(self.deadline_retries + 1):
            box = {}

            def run():
                try:
                    box["r"] = self.client.chat.completions.create(**kw)
                except Exception as e:                 # 交回调用方，和不设上限时一样抛出
                    box["e"] = e
            t = threading.Thread(target=run, daemon=True)
            t.start()
            t.join(self.deadline)
            if "e" in box:
                raise box["e"]
            if "r" in box:
                return box["r"]
        raise TimeoutError(f"LLM 调用超过 {self.deadline} 秒，已重试 {self.deadline_retries} 次")

    def complete(self, system: str, user: str, temperature: float = 0.0):
        extra = {"extra_body": {"thinking": {"type": "disabled"}}} if self.thinking is False else {}   # DeepSeek flash 默认先推理
        r = self._create(model=self.model, max_tokens=self.max_tokens,           # 部分模型（如 kimi-k3）只允许固定温度
                         temperature=self.fixed_temperature if self.fixed_temperature is not None else temperature,
                         messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **extra)
        self.last_usage = _usage(r.usage)
        return r.choices[0].message.content or "", (r.usage.total_tokens if r.usage else 0)


def redact(msg: str) -> str:
    """服务商报错里可能带 key 或 access key 片段（如 sk-…、ak-…），写日志、上页面之前抹掉。"""
    return re.sub(r"\b((?:sk|ak)-)[A-Za-z0-9*_\-]+", r"\1***", msg)


class TracedLLM:
    """包一层：每次 complete() 写一条 llm_call 事件（阶段、耗时、token、prompt/回复原文）。失败也记录，再原样抛出。
    stage() 返回调用时所处的状态机状态，用来区分 PLAN / DECIDE / REPORT / CONSOLIDATE。"""

    def __init__(self, inner, emit, stage=lambda: None, tracer=None):
        self.inner, self.emit, self.stage, self.tracer = inner, emit, stage, tracer

    def complete(self, system, user, temperature=0.0):
        t0 = time.time()
        rec = {"stage": self.stage(), "model": getattr(self.inner, "model", type(self.inner).__name__),
               "temperature": temperature, "system_chars": len(system), "prompt": user}
        g = self.tracer.generation(rec["stage"], rec["model"], user, temperature) if self.tracer else contextlib.nullcontext()
        with g as gen:
            try:
                text, toks = self.inner.complete(system, user, temperature)
            except Exception as e:
                err = redact(f"{type(e).__name__}: {str(e)[:300]}")
                self.emit("llm_call", {**rec, "latency_s": round(time.time() - t0, 3), "tokens": 0, "error": err})
                if self.tracer:
                    self.tracer.end_generation(gen, error=err)
                raise
            usage = getattr(self.inner, "last_usage", None)
            self.emit("llm_call", {**rec, "latency_s": round(time.time() - t0, 3), "tokens": toks, "usage": usage, "response": text})
            if self.tracer:
                self.tracer.end_generation(gen, text, toks, usage=usage)
        return text, toks

    def __getattr__(self, name):                 # 其余属性（如 ScriptedLLM.calls）透传给被包装的对象
        return getattr(self.inner, name)


def make_llm(provider: str, cfg: dict, api_key: str | None = None):
    """provider: mock | anthropic | kimi | deepseek …（config.yaml 的 llm.providers 里登记）。
    api_key 不给时从环境变量读取；给了就只交给客户端，不写进环境变量（app.py 里用户临时填的 key 走这条路）。"""
    load_env()                                   # 项目根目录的 .env
    if provider == "mock":
        return PolicyMockLLM()
    p = cfg["llm"]["providers"][provider]
    if p["type"] == "anthropic":
        return AnthropicLLM(p["model"], cfg["llm"]["max_tokens"], api_key=api_key)
    return OpenAICompatLLM(p["model"], p["base_url"], p["key_env"], p.get("max_tokens", cfg["llm"]["max_tokens"]),
                           fixed_temperature=p.get("fixed_temperature"), api_key=api_key, thinking=p.get("thinking"),
                           max_retries=cfg["llm"].get("max_retries", 2), timeout=cfg["llm"]["timeout_sec"],
                           deadline=cfg["llm"]["deadline_sec"], deadline_retries=cfg["llm"]["deadline_retries"])


class ScriptedLLM:
    """测试用：按顺序返回预设文本（字符串或 dict）；也可传 callable(user)->str。"""

    def __init__(self, script):
        self.script, self.calls = (script if callable(script) else list(script)), []

    def complete(self, system, user, temperature=0.0):
        self.calls.append(user)
        s = self.script.pop(0) if not callable(self.script) else self.script(user)
        return (s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)), 0


class PolicyMockLLM:
    """确定性 mock：读取 prompt 里的结构化上下文，按策略先验选动作。用于离线跑通状态机和 resume 测试。"""

    def complete(self, system, user, temperature=0.0):
        ctx = last_json_block(user)
        if ctx.get("mode") == "PLAN":
            return json.dumps({"directions": [{"direction": "先比较模型，再调参", "rationale": "mock"}],
                               "candidate_models": [{"model": "lgbm", "reason": "mock：通用强基线"},
                                                    {"model": "lr_scorecard", "reason": "mock：跨时间最稳"}]}, ensure_ascii=False), 0
        if ctx.get("mode") == "CONSOLIDATE":
            v = ctx["verdicts"]
            out = []
            for k, ev in ctx["allowed_keys"].items():
                pos = any(v[e] in ("ACCEPT", "PROMISING") for e in ev)
                out.append({"key": k, "statement": f"{k} " + ("有效" if pos else "无效"), "evidence": ev if not pos else [e for e in ev if v[e] in ("ACCEPT", "PROMISING")],
                            "polarity": "POSITIVE" if pos else "NEGATIVE"})
            return json.dumps({"lessons": out}, ensure_ascii=False), 0
        if ctx.get("mode") == "REPORT":
            f = ctx["facts"]
            return f"本次共运行 {f['n_experiments']} 个实验，其中 {f['n_accept']} 个被接受。" + (
                f"最终模型为 {f['model']}，OOT-dev {f['metric'].upper()} 为 {f['oot_dev_' + f['metric']]}。" if f.get("model") else "没有产生最终模型。"), 0
        d = self._decide(ctx)
        return json.dumps(d, ensure_ascii=False), 0

    def _decide(self, c):
        base = dict(hypothesis="mock", expected_gain="small", est_cost={"tokens": 0, "cpu_minutes": 0.1},
                    alternatives_considered=["STOP"], rationale="mock policy: follow policy prior")
        if c["unhandled_leaks"]:
            return {**base, "action": "ESCALATE_HUMAN", "params": {"reason": "leak suspect", "options": ["approve", "quarantine"]}}
        promo = [e for e in c["race"] if e["exp_id"] not in c["promoted"]]
        if c["best"] is None and promo:
            return {**base, "action": "PROMOTE_FIDELITY", "params": {"exp_id": promo[0]["exp_id"]}}
        r = c["round_no"]
        cur_model = c["current_config"]["model"]
        script = [("TUNE", {"space": {"learning_rate": {"low": 0.02, "high": 0.1, "log": True}} if cur_model == "lgbm" else {"C": {"low": 0.01, "high": 1.0, "log": True}}, "n_trials": 4 + r}),
                  ("SWITCH_MODEL", {"model": "lr_scorecard" if cur_model == "lgbm" else "lgbm", "fidelity": "full"}),
                  ("TUNE", {"space": {"num_leaves": {"low": 8, "high": 32, "type": "int"}} if cur_model == "lgbm" else {"n_bins": {"low": 5, "high": 10, "type": "int"}}, "n_trials": 5 + r}),
                  ("STOP", {"reason": "mock: 剩余方向期望价值低于成本"})]
        a, p = script[min(r - 1 if c["best"] else r, len(script) - 1) % len(script)]
        return {**base, "action": a, "params": p}
