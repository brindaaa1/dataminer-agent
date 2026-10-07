"""接入草稿：LLM 的输出先经过这里校验。"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agent.schemas import MetricSpec

Availability = Literal["at_prediction", "history", "updated_until_event", "before_outcome", "post_outcome", "meta", "unknown"]
REQUIRED = ("label", "observation_time_expr", "windows", "metric")      # 猜错代价高：必须经用户确认，代码强制


class History(BaseModel):
    lags: list[int] = Field(default_factory=list)     # 前第 k 条的取值
    means: list[int] = Field(default_factory=list)    # 过去 n 条的均值（不含当前行）


class FieldDraft(BaseModel):
    type: Literal["numeric", "categorical", "binary", "date", "text"]
    availability: Availability
    reason: str = ""
    history: History | None = None                    # 当前值不能用、但预测时点之前的取值已知（如过去时段的价格、上一条的结果）


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
    domain: str = "_default"                      # 领域手册名（knowhow/domains/）；没有匹配的用 _default
    data_summary: str = ""
    history_by: str | None = None                 # 有历史值字段时按哪一列分组取前几条（每个实体一条序列）；整表是一条序列时为空                        # LLM 写的几句数据描述，过程页顶部当作数据预览
    csv_null_values: list[str] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)
