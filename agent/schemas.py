"""pydantic schema：LLM 的所有输出都先经过这里校验（§4.0-3）。"""
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ActionType(str, Enum):
    EXPAND_FEATURES = "EXPAND_FEATURES"
    PRUNE_FEATURES = "PRUNE_FEATURES"
    TUNE = "TUNE"
    SWITCH_MODEL = "SWITCH_MODEL"
    PROMOTE_FIDELITY = "PROMOTE_FIDELITY"
    BACKTRACK = "BACKTRACK"
    INVESTIGATE = "INVESTIGATE"
    ESCALATE_HUMAN = "ESCALATE_HUMAN"
    STOP = "STOP"


class EstCost(BaseModel):
    tokens: float = 0
    cpu_minutes: float = 0


class Decision(BaseModel):
    model_config = ConfigDict(extra="ignore")     # LLM 夹带的 verdict 等字段一律丢弃
    action: ActionType
    params: dict = Field(default_factory=dict)
    hypothesis: str
    expected_gain: str
    est_cost: EstCost
    alternatives_considered: list[str] = Field(default_factory=list)
    rationale: str


class Direction(BaseModel):
    direction: str
    rationale: str
    expected_gain: str = ""


class Plan(BaseModel):
    model_config = ConfigDict(extra="ignore")
    directions: list[Direction]


class Constraints(BaseModel):
    interpretability: Literal["none", "high"] = "none"    # high → 只能用 lr_scorecard
    max_features: int | None = None
    banned_models: list[str] = Field(default_factory=list)


class Budget(BaseModel):
    tokens: float
    cpu_minutes: float
    wall_minutes: float


class TaskSpec(BaseModel):
    dataset: str
    entity_key: str | None = None
    label_def: str | None = None
    label_col: str | None = None
    observation_time_col: str | None = None
    oot_windows: dict | None = None
    target_metric: Literal["auc", "ks"] = "auc"
    target_value: float | None = None
    constraints: Constraints = Field(default_factory=Constraints)
    budget: Budget
    autonomy: Literal["L0", "L1", "L2"] = "L1"

    @classmethod
    def from_knowhow(cls, dataset: str, cfg: dict, **kw) -> "TaskSpec":
        from data.knowhow import load_knowhow
        d = load_knowhow(dataset, cfg)["dictionary"]
        lab = d["label"]
        sp = d["suggested_split"]
        return cls(dataset=dataset, entity_key=d.get("entity_key"),
                   label_def=lab.get("definition"), label_col=lab.get("source_col") or lab.get("col"),
                   observation_time_col=d.get("observation_time_col") or (d.get("observation_time") or {}).get("booking_date"),
                   oot_windows={"oot_dev": sp.get("oot_dev"), "holdout": sp.get("holdout")},
                   budget=Budget(**cfg["run"]["budget"]), **kw)

    def missing_required(self) -> list[str]:
        """§4.4：这三个字段推断不出来且猜错代价高，禁止 LLM 假设，缺了就必须问人。"""
        return [f for f in ("label_def", "observation_time_col", "oot_windows") if not getattr(self, f)]
