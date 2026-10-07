"""赛跑打平（v7–v9 评测：20 次运行 18 次，赛跑里至少两个模型与最优差距小于 MDE）。
打平就是数据分不出高低：不让 LLM 在噪声里挑，按固定偏好顺序推荐，同样的数据每次推荐同一个。"""


def race_tie(scores: dict[str, float], mde: float, order: list[str]) -> dict:
    """scores：模型 → 赛跑的 OOT-dev 主指标。与最优差距 < mde 的都算效果相当；按 order 推荐，不在 order 里的排后面、按分数。"""
    best = max(scores, key=scores.get)
    rank = lambda m: (order.index(m) if m in order else len(order), -scores[m])
    tied = sorted((m for m, s in scores.items() if scores[best] - s < mde or m == best), key=rank)
    return {"recommended": tied[0], "tied": tied, "best": best, "mde": mde}
