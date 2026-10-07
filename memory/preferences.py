"""用户偏好（工作台本地单用户）：同一结构的数据确认过的口径（按表头指纹），每个领域上次用的指标。
接入起草时作为推荐答案放进上下文，仍由用户确认，不会自动跳过。"""
import hashlib
import json
from pathlib import Path

KEYS = ("label", "observation_time_expr", "split_time_expr", "windows")


def _fp(columns) -> str:
    return hashlib.sha1("\n".join(sorted(columns)).encode()).hexdigest()[:12]


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text()) if Path(path).exists() else {"data": {}, "metric": {}}


def recall(path: Path, columns) -> dict:
    p = _load(path)
    return {"same_data": p["data"].get(_fp(columns)), "metric_by_domain": p["metric"]}


def remember(path: Path, columns, draft: dict):
    """draft：写出配置时的最终草稿（必答项是用户确认过的值）。"""
    p = _load(path)
    p["data"][_fp(columns)] = {k: draft[k] for k in KEYS}
    p["metric"][draft["domain"]] = draft["metric"]
    Path(path).write_text(json.dumps(p, ensure_ascii=False, indent=1))
