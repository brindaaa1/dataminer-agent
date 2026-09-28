"""LLM 接口。complete() 返回 (文本, token 数)。真实模型从环境变量 ANTHROPIC_API_KEY 读取 key。"""
import json
import os
import re

from runtime.env import load_env


def extract_json(text: str) -> dict:
    m = re.search(r"```json\s*(.*?)```", text, re.S)
    s = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    return json.loads(s)


def last_json_block(text: str) -> dict:
    blocks = re.findall(r"```json\s*(.*?)```", text, re.S)
    return json.loads(blocks[-1])


class AnthropicLLM:
    def __init__(self, model: str, max_tokens: int = 2000):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model, self.max_tokens = model, max_tokens

    def complete(self, system: str, user: str, temperature: float = 0.0):
        r = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, temperature=temperature,
                                        system=system, messages=[{"role": "user", "content": user}])
        return r.content[0].text, r.usage.input_tokens + r.usage.output_tokens


class OpenAICompatLLM:
    """OpenAI 兼容接口（Kimi/Moonshot、DeepSeek 等）。key 从环境变量 key_env 读取。"""

    def __init__(self, model: str, base_url: str, key_env: str, max_tokens: int = 2000, client=None,
                 fixed_temperature: float | None = None):
        if client is None:
            import openai
            key = os.environ.get(key_env)
            if not key:
                raise RuntimeError(f"环境变量 {key_env} 未设置：请在启动前 export {key_env}=...")
            client = openai.OpenAI(api_key=key, base_url=base_url)
        self.client, self.model, self.max_tokens, self.fixed_temperature = client, model, max_tokens, fixed_temperature

    def complete(self, system: str, user: str, temperature: float = 0.0):
        r = self.client.chat.completions.create(          # 部分模型（如 kimi-k3）只允许固定温度
            model=self.model, max_tokens=self.max_tokens,
            temperature=self.fixed_temperature if self.fixed_temperature is not None else temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return r.choices[0].message.content or "", (r.usage.total_tokens if r.usage else 0)


def make_llm(provider: str, cfg: dict):
    """provider: mock | anthropic | kimi | deepseek …（config.yaml 的 llm.providers 里登记）。"""
    load_env()                                   # 项目根目录的 .env
    if provider == "mock":
        return PolicyMockLLM()
    p = cfg["llm"]["providers"][provider]
    if p["type"] == "anthropic":
        return AnthropicLLM(p["model"], cfg["llm"]["max_tokens"])
    return OpenAICompatLLM(p["model"], p["base_url"], p["key_env"], p.get("max_tokens", cfg["llm"]["max_tokens"]),
                           fixed_temperature=p.get("fixed_temperature"))


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
            return json.dumps({"directions": [{"direction": "先比较模型，再调参", "rationale": "mock"}]}, ensure_ascii=False), 0
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
                f"最终模型为 {f['model']}，OOT-dev AUC 为 {f['oot_dev_auc']}。" if f.get("model") else "没有产生最终模型。"), 0
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
