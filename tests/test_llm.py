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
