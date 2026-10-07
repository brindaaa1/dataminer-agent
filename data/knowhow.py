"""加载 knowhow/<dataset>/ 并校验其声明的字段与实际表头是否一致（§4.11 实现要求）。"""
import csv
import re
from pathlib import Path

import yaml


def load_config(path="config.yaml"):
    return yaml.safe_load(open(path))


def _yaml(p: Path):
    return yaml.safe_load(open(p)) if p.exists() else None


def load_knowhow(dataset: str, cfg: dict) -> dict:
    root = Path(cfg["paths"]["knowhow_root"]) / dataset
    return {"dictionary": _yaml(root / "data_dictionary.yaml"),
            "blacklist": _yaml(root / "blacklist.yaml")}


# ---------- 领域手册（knowhow/domains/）：_default 打底，具体领域覆盖标量与字典、追加列表 ----------
APPEND = ("leakage_name_patterns", "leakage_desc_keywords", "sanity_rules")


def _domains_root(cfg: dict) -> Path:
    """领域手册不跟数据集的 know-how 目录走：接入新数据时 knowhow_root 会指向这份数据单独生成的目录。"""
    return Path(cfg["paths"].get("domains_root") or Path(cfg["paths"]["knowhow_root"]) / "domains")


def domains(cfg: dict) -> list[dict]:
    """可选的领域（不含 _template 骨架）：给接入起草时推荐用。"""
    out = []
    for d in sorted(_domains_root(cfg).iterdir()):
        pb = _yaml(d / "playbook.yaml")
        if d.name != "_template" and pb:
            out.append({k: pb.get(k) for k in ("name", "title", "description", "match_keywords", "metric")})
    return out


def load_domain(name: str | None, cfg: dict) -> dict:
    base = _yaml(_domains_root(cfg) / "_default" / "playbook.yaml") or {}
    if not name or name == "_default":
        return base
    dom = _yaml(_domains_root(cfg) / name / "playbook.yaml") or {}
    out = {**base, **{k: v for k, v in dom.items() if k not in APPEND}}
    for k in APPEND:
        out[k] = list(dict.fromkeys((base.get(k) or []) + (dom.get(k) or [])))
    return out


def dataset_domain(dataset: str, cfg: dict) -> str:
    """数据集在 data_dictionary 里声明的领域；没声明用通用手册。"""
    return (load_knowhow(dataset, cfg)["dictionary"] or {}).get("domain") or "_default"


def domain_context(dataset: str, cfg: dict) -> dict:
    """规划（PLAN）时给 LLM 的领域信息：提出特征假设时的概念词表、正类含义、泄漏线索。"""
    d = load_domain(dataset_domain(dataset, cfg), cfg)
    return {"title": d.get("title"), "positive_meaning": (d.get("target") or {}).get("positive_meaning"),
            "feature_concepts": [{k: f.get(k) for k in ("id", "concept", "direction", "rationale")} for f in d.get("feature_concepts") or []],
            "leakage_name_patterns": d.get("leakage_name_patterns") or []}


def target_direction(dataset: str, cfg: dict) -> dict:
    """knowhow/<数据集>/priors.yaml 的目标方向：+1 = 值越大越可能是正类。新数据集可以没有。"""
    return (_yaml(Path(cfg["paths"]["knowhow_root"]) / dataset / "priors.yaml") or {}).get("target_direction") or {}


def header(path: str) -> list[str]:
    with open(path, newline="") as f:
        return next(csv.reader(f))


def declared_fields(dataset: str, kh: dict) -> dict[str, dict[str, set]]:
    """返回 {table: {"dictionary": {...}, "blacklist": {...}}}，全是 know-how 声明过的字段名。"""
    d, b = kh["dictionary"], kh["blacklist"] or {}
    out = {}
    if "tables" in d:  # 多表（home_credit）
        for t, spec in d["tables"].items():
            out[t] = {"dictionary": set(spec.get("key_fields", {}))}
        bl = d.get("blacklist", {})
        out.setdefault("application_train", {"dictionary": set()})["blacklist"] = {
            f for k in ("compliance_exclude", "compliance_review", "incumbent_model") for f in bl.get(k, [])}
    else:
        dic = set(d.get("fields", {}))
        lab = d.get("label", {})
        for k in ("source_col", "col"):
            if lab.get(k):
                dic.add(lab[k])
        if d.get("observation_time_col"):
            dic.add(d["observation_time_col"])
        out["main"] = {"dictionary": dic}
        bl = set()
        for sec in b.values():
            if isinstance(sec, dict):
                bl |= set(sec.get("fields", []))
                bl |= set(sec.get("review", []) or []) | set(sec.get("exclude", []) or [])
        out["main"]["blacklist"] = bl
    return out


def check_fields(dataset: str, cfg: dict) -> dict:
    """比对声明字段与真实表头。返回结构化报告（不静默忽略）。"""
    kh = load_knowhow(dataset, cfg)
    decl = declared_fields(dataset, kh)
    report = {"dataset": dataset, "tables": {}, "ok": True}
    for t, path in cfg["datasets"][dataset]["tables"].items():
        hdr = header(path)
        hs = set(hdr)
        d = decl.get(t, {"dictionary": set(), "blacklist": set()})
        missing_dict = sorted(d["dictionary"] - hs)
        missing_bl = sorted(d.get("blacklist", set()) - hs)
        # 前缀类黑名单：至少要命中一个真实字段才算有效
        undeclared = sorted(hs - d["dictionary"])
        rep = {"n_columns": len(hdr), "missing_in_data_dictionary": missing_dict,
               "missing_in_blacklist": missing_bl, "undeclared_in_data": undeclared}
        prefixes = (kh["blacklist"] or {}).get("leakage", {}).get("field_prefixes", []) if t == "main" else []
        rep["prefix_hits"] = {p: sorted(c for c in hdr if c.startswith(p)) for p in prefixes}
        rep["prefix_no_hit"] = [p for p, v in rep["prefix_hits"].items() if not v]
        if missing_dict or missing_bl:
            report["ok"] = False
        report["tables"][t] = rep
    return report


def format_report(rep: dict) -> str:
    lines = [f"# 字段校验报告：{rep['dataset']}  ({'OK' if rep['ok'] else '有不一致'})"]
    for t, r in rep["tables"].items():
        lines += [f"\n## 表 {t}（{r['n_columns']} 列）",
                  f"- know-how 声明但表头缺失（data_dictionary）: {r['missing_in_data_dictionary'] or '无'}",
                  f"- know-how 声明但表头缺失（blacklist）: {r['missing_in_blacklist'] or '无'}",
                  f"- 前缀规则未命中任何列: {r['prefix_no_hit'] or '无'}",
                  f"- 数据里有、know-how 未登记（按 unknown 处理）: {len(r['undeclared_in_data'])} 列"]
        if r["prefix_hits"]:
            for p, v in r["prefix_hits"].items():
                lines.append(f"  - 前缀 `{p}` 命中 {len(v)} 列")
        lines.append("  - 未登记列: " + ", ".join(r["undeclared_in_data"]))
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    cfg = load_config()
    outdir = Path(cfg["paths"]["reports_root"]); outdir.mkdir(exist_ok=True)
    for ds in (sys.argv[1:] or ["lending_club"]):
        rep = check_fields(ds, cfg)
        txt = format_report(rep)
        (outdir / f"field_check_{ds}.md").write_text(txt)
        print(txt, "\n")


def normalize(kh: dict, main_table: str | None = None) -> tuple[dict, dict]:
    """把不同 know-how 写法统一成 (fields, blacklist)：
    单表（LC/酒店）直接用 fields + blacklist.yaml；多表（Home Credit）取主表 key_fields + 字典内嵌的 blacklist 段。"""
    d, b = kh["dictionary"], kh["blacklist"]
    if "fields" in d:
        return d["fields"], b or {}
    fields = dict(d["tables"][main_table]["key_fields"])
    eb = d.get("blacklist", {})
    b = {"compliance": {"exclude": eb.get("compliance_exclude", []), "review": eb.get("compliance_review", [])},
         "incumbent_model": {"fields": eb.get("incumbent_model", []),
                             "action": "exclude_by_default" if eb.get("incumbent_policy") == "exclude_by_default" else "include_by_default"}}
    return fields, b


# 可得时间的"强弱"：数值越大越弱（越可能是事后信息）。派生特征的可得时间 = 其输入中最弱的一个（§4.11）。
# 通用取值：at_prediction（预测时点已知）、before_outcome（预测时点之后、结果之前才确定）；application 等是信贷数据集更细的说法。
AVAIL_RANK = {"at_prediction": 0, "history": 0, "application": 0, "bureau_at_orig": 0, "loan_terms": 0, "incumbent_model": 0,
              "meta": 0, "updated_until_event": 1, "unknown": 1, "before_outcome": 2, "post_origination": 3, "post_outcome": 3}


def derive_availability(expr: str, known: dict, declared: str | None = None) -> tuple[str, str | None]:
    """按输入字段推断派生特征的可得时间，与声明值取较弱者。返回 (有效可得时间, 警告)。
    known: 已知字段/已派生字段 → {"availability": ...}；表达式里不在 known 中的标识符（函数名、别名）忽略。"""
    ins = [t for t in re.findall(r"[A-Za-z_]\w*", expr) if t in known]
    inferred = max((known[t].get("availability", "unknown") for t in ins), key=lambda a: AVAIL_RANK.get(a, 1), default="application")
    if declared is None:
        return inferred, None
    eff = max((declared, inferred), key=lambda a: AVAIL_RANK.get(a, 1))
    warn = None
    if AVAIL_RANK.get(declared, 1) < AVAIL_RANK.get(inferred, 1):
        warn = f"声明的可得时间 {declared} 比其输入推断出的 {inferred} 更强，已按 {eff} 处理"
    return eff, warn


def label_sql(label: dict) -> str:
    """标签的 SQL：由状态字段按取值打 0/1，或直接用已是 0/1 的列。"""
    if "source_col" in label:
        q = lambda xs: ", ".join("'" + x.replace("'", "''") + "'" for x in xs)
        c = f'"{label["source_col"]}"::VARCHAR'          # 列名可能带空格；取值按文本比较（0/1 标签写成 '1'、'0'）
        return (f"CASE WHEN {c} IN ({q(label['positive'])}) THEN 1 "
                f"WHEN {c} IN ({q(label['negative'])}) THEN 0 END")
    return label["col"]


def _history(d: dict) -> tuple[dict, dict]:
    """历史值可用的字段（data_dictionary 里 history: {lags, means}）：当前值是泄漏，但预测时点之前的取值已知。
    生成 <列>_lag<k>（前第 k 条）和 <列>_mean<n>（过去 n 条的均值，不含当前行）；标签列按 0/1。
    窗口写成占位符 __HIST__，切分时换成按预测时点排序（及分组）的窗口。"""
    known, dexpr = {}, {}
    label = d.get("label", {})
    for c, spec in (d.get("fields") or {}).items():
        h = spec.get("history") or {}
        src = label_sql(label) if c == label.get("source_col") else f'"{c}"'
        for k in h.get("lags", []):
            dexpr[f"{c}_lag{k}"] = f"LAG({src}, {k}) OVER (__HIST__)"
        for n in h.get("means", []):
            dexpr[f"{c}_mean{n}"] = f"AVG({src}) OVER (__HIST__ ROWS BETWEEN {n} PRECEDING AND 1 PRECEDING)"
    return {k: {"availability": "history", "type": "derived"} for k in dexpr}, dexpr


def field_table(kh: dict, main_table: str | None = None, expr_overrides: dict | None = None):
    """返回 (known, dexpr, warnings)：known = 登记字段 + 派生特征（可得时间已按继承规则算好）；
    dexpr = 派生特征名 → 可执行表达式（应用方言覆盖后）。切分（data/splits）与泄漏扫描（evaluator）共用这一份口径。"""
    d = kh["dictionary"]
    known, dexpr, warns = dict(normalize(kh, main_table)[0]), {}, []
    for c in [x for x in d.get("classic_derived", []) if isinstance(x, dict)]:
        eff, w = derive_availability(c["expr"], known, c.get("availability"))
        known[c["name"]] = {"availability": eff, "type": "derived"}
        dexpr[c["name"]] = (expr_overrides or {}).get(c["name"], c["expr"])
        if w:
            warns.append({"code": "DERIVED_AVAILABILITY", "name": c["name"], "msg": w})
    hk, hx = _history(d)
    return {**known, **hk}, {**dexpr, **hx}, warns


def column_policy(dataset: str, columns: list[str], kh: dict, use_incumbent: bool = False,
                  main_table: str | None = None, unregistered: str = "unknown", extra_fields: dict | None = None) -> dict:
    """按 data_dictionary + blacklist 给每一列定性。返回各类别的列名列表。
    unregistered: 主表里未登记字段的处理。'unknown'（默认，LC/酒店）→ 隔离；'application'（Home Credit：
    主表一行=一次当前申请，全部是申请时字段）→ 保留。由 config.task_specs 指定。"""
    d = kh["dictionary"]
    fields, b = normalize(kh, main_table)
    fields = {**fields, **(extra_fields or {})}          # 派生特征：可得时间已按继承规则算好
    leak = b.get("leakage", {})
    drop = {"leakage": set(leak.get("fields", [])) | set(leak.get("derived_fields", [])), "meta": set(b.get("meta", {}).get("fields", [])),
            "compliance_exclude": set(b.get("compliance", {}).get("exclude", []) or [])}
    prefixes = tuple(leak.get("field_prefixes", []))
    drop["leakage"] |= {c for c in columns if prefixes and c.startswith(prefixes)}
    inc = b.get("incumbent_model", {})
    drop["incumbent"] = set(inc.get("fields", [])) if inc.get("action") == "exclude_by_default" and not use_incumbent else set()
    ua = b.get("unknown_availability", {})
    quarantine = set(ua.get("fields", []) + ua.get("derived_fields", []) if isinstance(ua, dict) else ua)
    label = d.get("label", {})
    special = {label.get("source_col"), label.get("col"), d.get("observation_time_col"), d.get("entity_key")}
    if unregistered == "unknown":
        quarantine |= {c for c in columns if c not in fields and c not in special and c not in set().union(*drop.values())}
    for c, spec in fields.items():
        av = spec.get("availability")
        if av in ("post_origination", "post_outcome", "before_outcome"):
            drop["leakage"].add(c)
        elif av in ("unknown", "updated_until_event") and c not in special:      # 字典声明的可得时间存疑 → 隔离
            quarantine.add(c)
    pats = [re.compile(p, re.I) for p in b.get("leakage_patterns", [])]
    dropped = set().union(*drop.values())
    keep = [c for c in columns if c not in dropped and c not in quarantine]
    keep = [c for c in keep if c not in special]
    return {"drop": {k: sorted(v & set(columns)) for k, v in drop.items()},
            "quarantine": sorted(quarantine & set(columns)),
            "pattern_hits_on_kept": sorted(c for c in keep if any(p.search(c) for p in pats)),
            "pattern_hits_on_quarantined": sorted(c for c in quarantine if any(p.search(c) for p in pats)),
            "review_flags": sorted((set(b.get("compliance", {}).get("review", []) or [])) & set(keep)),
            "keep": keep}
