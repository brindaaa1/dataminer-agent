"""接入草稿：LLM 的输出先经过这里校验。"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agent.schemas import MetricSpec

Availability = Literal["at_booking", "history", "updated_until_event", "at_checkin", "post_outcome", "meta", "unknown"]
REQUIRED = ("label", "observation_time_expr", "windows", "metric")      # 猜错代价高：必须经用户确认，代码强制


class FieldDraft(BaseModel):
    type: Literal["numeric", "categorical", "binary", "date", "text"]
    availability: Availability
    reason: str = ""


class Question(BaseModel):
    key: str                 # 必答项名，或字段名
    question: str
    recommended: str


class LabelDraft(BaseModel):
    col: str
    positive: int | str = 1
    negative: int | str = 0
    definition: str


class Windows(BaseModel):
    train_valid: list[str]   # ["YYYY-MM", "YYYY-MM"]
    oot_dev: list[str]
    holdout: list[str]


class IntakeDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")
    label: LabelDraft | None = None
    observation_time_expr: str | None = None      # DuckDB 表达式：预测时点
    split_time_expr: str | None = None            # DuckDB 表达式：用于切分的时间
    windows: Windows | None = None
    fields: dict[str, FieldDraft] = Field(default_factory=dict)
    leakage: list[str] = Field(default_factory=list)
    quarantine: list[str] = Field(default_factory=list)
    metric: MetricSpec | None = None
    csv_null_values: list[str] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)
