"""调用 LLM 起草接入配置；输出先过 pydantic。"""
import json
from pathlib import Path

from agent.llm import extract_json
from data.knowhow import domains, load_config
from intake.schemas import IntakeDraft

SYSTEM = (Path(__file__).parent / "prompts" / "draft.md").read_text()


def draft(llm, profile: dict, description: str, answers: dict, feedback: list[str] | None = None, prefs: dict | None = None):
    """返回 (回复原文, prompt 的 user 部分)。feedback 是上一次草稿的错误（格式或代码校验），回灌给 LLM 修正。"""
    ctx = {"profile": profile, "description": description, "answers": answers,
           "domains": domains(load_config()),                 # 领域手册：推荐一个领域，主指标和护栏按它的 metric
           **({"preferences": prefs} if prefs else {})}
    user = "```json\n" + json.dumps(ctx, ensure_ascii=False, indent=1) + "\n```\n"
    if feedback:
        user += "\n上一次草稿未通过代码校验，请修正：\n" + "\n".join(f"- {e}" for e in feedback) + "\n"
    text, _ = llm.complete(SYSTEM, user, 0.0)
    return text, user


def parse(text: str) -> IntakeDraft:
    return IntakeDraft.model_validate(extract_json(text))
