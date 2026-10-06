"""LLM 适配层：OpenAI 兼容接口（Kimi 等）的调用、token 统计、缺 key 时的报错；不联网。"""
import json
from types import SimpleNamespace

import pytest

from agent.llm import OpenAICompatLLM, extract_json, make_llm
from data.knowhow import load_config


class FakeClient:
    def __init__(self, text):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.text = text

    def _create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.text))], usage=SimpleNamespace(total_tokens=42))


def test_openai_compat_returns_text_and_tokens():
    fc = FakeClient('思考…\n```json\n{"action": "STOP"}\n```')
    llm = OpenAICompatLLM("kimi-x", "http://x", "K", 100, client=fc)
    text, toks = llm.complete("sys", "usr", 0.3)
    assert toks == 42 and extract_json(text) == {"action": "STOP"}
    c = fc.calls[0]
    assert c["model"] == "kimi-x" and c["temperature"] == 0.3 and c["messages"][0] == {"role": "system", "content": "sys"}


def test_missing_key_gives_clear_error(monkeypatch, tmp_path):
    import runtime.env as env
    monkeypatch.setattr(env, "ROOT", tmp_path)                     # 不读真实的 .env（里面可能已有 key）
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MOONSHOT_API_KEY"):
        make_llm("kimi", load_config())


def test_make_llm_mock_and_unknown():
    assert make_llm("mock", load_config()).__class__.__name__ == "PolicyMockLLM"
    with pytest.raises(KeyError):
        make_llm("nope", load_config())


def test_load_env_reads_file_without_overriding(tmp_path, monkeypatch):
    from runtime.env import load_env
    f = tmp_path / ".env"
    f.write_text('# 注释\nA_KEY=abc\nexport B_KEY="quoted"\nC_KEY=\nEXISTING=fromfile\nbadline\n')
    monkeypatch.setenv("EXISTING", "fromshell")
    for k in ("A_KEY", "B_KEY", "C_KEY"):
        monkeypatch.delenv(k, raising=False)
    import os
    assert sorted(load_env(f)) == ["A_KEY", "B_KEY"]                  # 空值不设置；只返回变量名
    assert os.environ["A_KEY"] == "abc" and os.environ["B_KEY"] == "quoted"
    assert os.environ["EXISTING"] == "fromshell"                        # 不覆盖已有环境变量
    assert "C_KEY" not in os.environ
    monkeypatch.delenv("A_KEY"); monkeypatch.delenv("B_KEY")


def test_make_llm_reads_dotenv(tmp_path, monkeypatch):
    import runtime.env as env
    (tmp_path / ".env").write_text("MOONSHOT_API_KEY=dummy-key\n")
    monkeypatch.setattr(env, "ROOT", tmp_path)
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    llm = make_llm("kimi", load_config())                              # 不联网：只是构造客户端
    assert llm.model.startswith("kimi")
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)


class UsageClient(FakeClient):
    def __init__(self, text, usage):
        super().__init__(text)
        self.usage = usage

    def _create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.text))], usage=self.usage)


def test_usage_breakdown_from_deepseek_fields():
    u = SimpleNamespace(total_tokens=150, prompt_tokens=100, completion_tokens=50, prompt_cache_hit_tokens=80,
                        prompt_cache_miss_tokens=20, completion_tokens_details=SimpleNamespace(reasoning_tokens=30))
    llm = OpenAICompatLLM("deepseek-flash", "http://x", "K", 100, client=UsageClient("ok", u))
    assert llm.complete("s", "u")[1] == 150
    assert llm.last_usage == {"input_cache_hit": 80, "input_cache_miss": 20, "output": 50, "reasoning": 30}


def test_usage_without_cache_fields():
    """Kimi 等不返回缓存字段：输入全部按未命中计，推理记 0。"""
    u = SimpleNamespace(total_tokens=42, prompt_tokens=30, completion_tokens=12, completion_tokens_details=None)
    llm = OpenAICompatLLM("kimi-x", "http://x", "K", 100, client=UsageClient("ok", u))
    llm.complete("s", "u")
    assert llm.last_usage == {"input_cache_hit": 0, "input_cache_miss": 30, "output": 12, "reasoning": 0}


def test_thinking_off_is_sent_and_on_is_default():
    off = UsageClient("ok", SimpleNamespace(total_tokens=1, prompt_tokens=1, completion_tokens=0, completion_tokens_details=None))
    OpenAICompatLLM("deepseek-flash", "http://x", "K", 100, client=off, thinking=False).complete("s", "u")
    assert off.calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    on = UsageClient("ok", off.usage)
    OpenAICompatLLM("deepseek-flash", "http://x", "K", 100, client=on).complete("s", "u")
    assert "extra_body" not in on.calls[0]


def test_traced_llm_writes_usage_into_event():
    from agent.llm import TracedLLM
    u = SimpleNamespace(total_tokens=150, prompt_tokens=100, completion_tokens=50, prompt_cache_hit_tokens=80,
                        prompt_cache_miss_tokens=20, completion_tokens_details=SimpleNamespace(reasoning_tokens=30))
    events = []
    t = TracedLLM(OpenAICompatLLM("deepseek-flash", "http://x", "K", 100, client=UsageClient("ok", u)),
                  lambda type_, p: events.append(p), lambda: "DECIDE")
    t.complete("s", "u")
    assert events[0]["usage"] == {"input_cache_hit": 80, "input_cache_miss": 20, "output": 50, "reasoning": 30}


def test_deepseek_provider_config():
    cfg = load_config()
    p = cfg["llm"]["providers"]["deepseek"]
    assert p["model"] == "deepseek-flash" and p["max_tokens"] == 16000 and p["thinking"] is True
    assert cfg["llm"]["prices"]["deepseek-flash"] == {"input_cache_miss": 0.14, "input_cache_hit": 0.0028, "output": 0.28}


def test_openai_client_retries_more_on_flaky_network(monkeypatch):
    """v2 评测 hotel_bookings s2：第一次调用就断网，SDK 默认重试 2 次后整次运行作废。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    cfg = load_config()
    llm = make_llm("deepseek", cfg)
    assert llm.client.max_retries == cfg["llm"]["max_retries"] >= 5


def test_deepseek_max_tokens_leaves_room_for_thinking():
    """v2 评测 lending_club s2：推理用满 8000 个 token，回复为空。"""
    assert load_config()["llm"]["providers"]["deepseek"]["max_tokens"] >= 16000
