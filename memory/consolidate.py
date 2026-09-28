"""Lesson 的提出、准入与生命周期（§4.13）。
LLM 只能**提出**候选（key 必须是代码从本任务实验树推导出的枚举键；必须引用真实 exp_id；极性必须与证据判定一致）；
状态流转（CANDIDATE → VALIDATED → ACTIVE / DEPRECATED）全部由本文件的规则决定，LLM 无法晋升。"""
import hashlib

from pydantic import BaseModel, ConfigDict, ValidationError

from memory.retrieval import fingerprint_distance
from memory.schema import Lesson

GOOD, BAD = ("ACCEPT", "PROMISING"), ("REJECT", "INCONCLUSIVE")


class Proposal(BaseModel):
    model_config = ConfigDict(extra="ignore")
    key: str
    statement: str
    evidence: list[str]
    polarity: str


class Proposals(BaseModel):
    model_config = ConfigDict(extra="ignore")
    lessons: list[Proposal]


def allowed_keys(records) -> dict[str, list[str]]:
    """代码从实验树推导可总结的经验键 → 对应的证据实验 id。LLM 只能在这些键里选。"""
    keys: dict[str, list[str]] = {}
    for r in records:
        if r.invalidated or not r.metrics:
            continue
        m = r.config.get("model")
        if r.action_type == "RACE":
            k = f"race:{m}"
        elif r.action_type == "TUNE":
            ks = sorted((r.diff.get("space") or {}).keys())
            k = "tune:%s:%s@%s" % (m, ",".join(ks), r.fidelity)      # 保真度决定判定的含义（低保真最多 PROMISING），必须区分
        elif r.action_type == "SWITCH_MODEL":
            k = f"switch_to:{m}@{r.fidelity}"
        elif r.action_type == "PROMOTE_FIDELITY":
            k = f"promote:{m}"
        elif r.action_type == "PRUNE_FEATURES":
            k = f"prune:{m}"
        else:
            continue
        keys.setdefault(k, []).append(r.exp_id)
    return keys


def validate(props: Proposals, records, allowed) -> tuple[list[Proposal], list[dict]]:
    """规则准入：返回 (通过的, 被丢弃的及原因)。"""
    by = {r.exp_id: r for r in records}
    ok, dropped = [], []
    for p in props.lessons:
        why = None
        if p.key not in allowed:
            why = "key 不在代码推导的枚举里"
        elif not p.evidence or any(e not in by or e not in allowed[p.key] for e in p.evidence):
            why = "证据必须引用该 key 对应的真实 exp_id"
        elif p.polarity not in ("POSITIVE", "NEGATIVE"):
            why = "polarity 非法"
        else:
            vs = [by[e].verdict for e in p.evidence]
            if p.polarity == "POSITIVE" and not any(v in GOOD for v in vs):
                why = "POSITIVE 但证据没有 ACCEPT/PROMISING"
            if p.polarity == "NEGATIVE" and any(v in GOOD for v in vs):
                why = "NEGATIVE 但证据里有 ACCEPT/PROMISING"
        (dropped.append({"key": p.key, "why": why}) if why else ok.append(p))
    return ok, dropped


def _id(key, polarity):
    return hashlib.sha1(f"{key}|{polarity}".encode()).hexdigest()[:10]


def _find(mem, key, polarity, fp, cfg):
    """同 key、同极性、scope 与当前指纹匹配的已有 Lesson。scope 不匹配（如换了完全不同的数据集）不能合并证据。"""
    for l in mem.lessons():
        if l["key"] == key and l["polarity"] == polarity and fingerprint_distance(l["scope"], fp, cfg) <= cfg["memory"]["max_distance"]:
            return l

def admit(mem, ok: list[Proposal], records, task_id: str, fp: dict, cfg: dict) -> list[str]:
    """准入为 CANDIDATE；同 key 同极性已存在则合并证据/任务（幂等，resume 重放不会重复计数）。"""
    by = {r.exp_id: r for r in records}
    conf = cfg["memory"]["confidence"]
    ids = []
    for p in ok:
        cur = _find(mem, p.key, p.polarity, fp, cfg)
        lid = cur["lesson_id"] if cur else _id(f"{p.key}@{task_id}", p.polarity)      # 新谱系以首个提出它的任务命名
        deltas = []
        for e in p.evidence:
            par = by.get(by[e].parent_exp_id)
            if par and par.metrics.get("oot_dev_auc") is not None and by[e].metrics.get("oot_dev_auc") is not None:
                deltas.append(by[e].metrics["oot_dev_auc"] - par.metrics["oot_dev_auc"])
        if cur:
            cur["evidence"] = sorted(set(cur["evidence"]) | set(p.evidence))
            cur["tasks"] = sorted(set(cur["tasks"]) | {task_id})
            L = cur
        else:
            L = Lesson(lesson_id=lid, key=p.key, statement=p.statement, scope=fp, evidence=sorted(set(p.evidence)), tasks=[task_id],
                       effect_size=sum(deltas) / len(deltas) if deltas else 0.0, confidence=conf["CANDIDATE"],
                       status="CANDIDATE", polarity=p.polarity).model_dump()
        mem.put_lesson(L)
        ids.append(lid)
    return ids


def update_lifecycle(mem, task_id: str, records, best_exp_id: str | None, final: dict | None, cfg: dict) -> dict:
    """FINAL_GATE 之后按规则推进本任务涉及的 Lesson：
    VALIDATED：证据实验通过 final_gate（holdout 落差 <= δ）；POSITIVE 还要求证据在最优实验的祖先路径上。
    ACTIVE：同 key 同极性在 >=2 个不同任务上都被支持。
    DEPRECATED：出现同 key 相反极性且已 VALIDATED 的新证据。"""
    conf, delta = cfg["memory"]["confidence"], cfg["final_gate"]["delta"]
    by = {r.exp_id: r for r in records}
    path = set()
    e = best_exp_id
    while e and e in by:
        path.add(e)
        e = by[e].parent_exp_id
    passed = final is not None and final["drop"] <= delta
    changes = {}
    lessons = {l["lesson_id"]: l for l in mem.lessons()}
    for l in lessons.values():
        if task_id not in l["tasks"] or l["status"] == "DEPRECATED":
            continue
        old = l["status"]
        if l["status"] == "CANDIDATE" and passed and (l["polarity"] == "NEGATIVE" or set(l["evidence"]) & path):
            l["status"] = "VALIDATED"
        if l["status"] == "VALIDATED" and len(set(l["tasks"])) >= 2:
            l["status"] = "ACTIVE"
        if l["status"] != old:
            l["confidence"] = conf[l["status"]]
            mem.put_lesson(l)
            changes[l["lesson_id"]] = (l["key"], old, l["status"])
    for l in list(lessons.values()):                               # 反例：同 key 相反极性且已 VALIDATED/ACTIVE
        opp = next((o for o in lessons.values() if o["key"] == l["key"] and o["polarity"] != l["polarity"]
                    and fingerprint_distance(o["scope"], l["scope"], cfg) <= cfg["memory"]["max_distance"]), None)
        if opp and opp["status"] in ("VALIDATED", "ACTIVE") and l["status"] in ("VALIDATED", "ACTIVE") \
                and task_id in opp["tasks"] and task_id not in l["tasks"] \
                and fingerprint_distance(l["scope"], opp["scope"], cfg) <= cfg["memory"]["max_distance"]:   # 只在 scope 相同的范围内推翻
            l["status"], l["confidence"] = "DEPRECATED", conf["DEPRECATED"]
            mem.put_lesson(l)
            changes[l["lesson_id"]] = (l["key"], "…", "DEPRECATED")
    return changes
