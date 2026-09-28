"""Memory schema（§4.13）。第 4 项只用到 ExperimentRecord；Lesson/Prior 在第 12 项补充。"""
from typing import Literal

from pydantic import BaseModel

Verdict = Literal["ACCEPT", "REJECT", "INCONCLUSIVE", "PROMISING", "NONE"]   # NONE：不训练的动作（BACKTRACK 等）


class Cost(BaseModel):
    tokens: float = 0
    cpu_minutes: float = 0
    wall_minutes: float = 0


class ExperimentRecord(BaseModel):
    exp_id: str
    task_id: str
    parent_exp_id: str | None
    hypothesis: str
    action_type: str
    diff: dict
    config: dict                       # 本实验的完整配置（model/features/space/fidelity/n_trials）
    config_hash: str
    fidelity: Literal["low", "full", "none"]
    metrics: dict
    diagnosis_codes: list[str]
    cost: Cost
    verdict: Verdict
    invalidated: bool = False
    artifact_refs: dict = {}


class DatasetFingerprint(BaseModel):
    """数据集指纹（§4.13）：用 meta-features 检索相似历史任务。"""
    task_id: str
    n_samples: int
    n_features: int
    bad_rate: float
    missing_rate: float
    time_span_months: int
    label_def: str
    feature_domains: list[str]


class Lesson(BaseModel):
    """跨任务经验（semantic）。LLM 只能提出，状态流转由 consolidate.py 的规则决定。"""
    lesson_id: str
    key: str                              # 由代码从实验树推导的枚举键，如 tune:lgbm:learning_rate
    statement: str
    scope: dict                           # 提出该经验时的数据集指纹（用于检索时判断是否适用）
    evidence: list[str]                   # exp_id 列表
    tasks: list[str]                      # 支撑该经验的任务（>=2 才能 ACTIVE）
    effect_size: float
    confidence: float
    status: Literal["CANDIDATE", "VALIDATED", "ACTIVE", "DEPRECATED"]
    polarity: Literal["POSITIVE", "NEGATIVE"]


class Prior(BaseModel):
    """从历史任务最优实验提取的起点（procedural）：只是 warm-start 的初始点，不是约束。"""
    from_task: str
    distance: float
    model: str
    params: list[dict]                    # 最优实验 study 中排名靠前的 trial 参数
