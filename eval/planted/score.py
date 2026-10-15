"""把工具找出来的问题和标准答案对照。按（表，列）对上即算找到；同一列被几个检查重复报出只算一次。"""
from collections import Counter


def _key(x: dict) -> tuple:
    return (x["table"], x.get("column"))


def _usable(f: dict) -> bool:
    """整张表层面的发现（没有列名）只有说的是重复才算数，别的检查没指出列时不能碰巧对上重复行。"""
    return f.get("column") is not None or any(w in f["kind"].lower() for w in ("dup", "重复"))


def score(answers: list[dict], findings: list[dict]) -> dict:
    answer_keys = {_key(a) for a in answers}
    hits = [f for f in findings if _usable(f) and _key(f) in answer_keys]
    keys = {_key(f) for f in hits}
    required = [a for a in answers if not a.get("optional")]
    found = [a for a in required if _key(a) in keys]
    return {"n_answers": len(required), "n_found": len(found),
            "found_rate": len(found) / len(required) if required else 1.0,
            "must_find_ok": all(_key(a) in keys for a in answers if a.get("must_find")),
            "missed": [a for a in required if _key(a) not in keys],
            "not_problems": sorted({_key(f) for f in findings if f not in hits}, key=str),
            "matched_kinds": dict(Counter(f["kind"] for f in hits)),
            "matched_by": {f"{a['table']}.{a['column']}": sorted({f["kind"] for f in hits if _key(f) == _key(a)}) for a in found}}


def to_markdown(title: str, r: dict) -> str:
    lines = [f"### {title}", "",
             f"- 找出来：{r['n_found']}/{r['n_answers']}（{r['found_rate']:.0%}）",
             f"- 每次都必须找到的：{'找到了' if r['must_find_ok'] else '没找到'}",
             f"- 其实不是问题的：{len(r['not_problems'])} 个"]
    if r["missed"]:
        lines += ["", "没找到的：", ""] + [f"- {a['table']}.{a['column']}（{a['kind']}{'，原生' if a.get('native') else ''}）" for a in r["missed"]]
    if r["matched_by"]:
        lines += ["", "每条找到的答案是哪个检查对上的：", ""] + [f"- {k}：{'、'.join(v)}" for k, v in r["matched_by"].items()]
    return "\n".join(lines) + "\n"
