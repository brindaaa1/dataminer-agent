"""试验：Hindsight 能否记住数据经验、整理出对的结论、按标记筛选、删掉错的。
用 .venvs/hs 运行：.venvs/hs/bin/python eval/trials/hindsight_memory.py"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import time

from hindsight import HindsightClient, HindsightServer
from hindsight_client.hindsight_client import _run_async

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
BANK = "trial-dataminer"

# (内容, 数据集, 是否确认过)
EXPERIENCES = [
    ("home_credit 的 application_train 表里，DAYS_EMPLOYED = 365243 表示没有工作或已退休，约占 18%，这些人的违约率更低。", "home_credit", True),
    ("home_credit 的 previous_application 表里，DAYS_FIRST_DRAWING 等 5 个日期字段用 365243 表示缺失。", "home_credit", False),
    ("lending_club 做违约模型时，只保留已结清（Fully Paid）和已核销（Charged Off）的贷款，还在还款中（Current）的不进样本。", "lending_club", True),
    ("hotel_bookings 的 reservation_status 是结果出来之后才有的字段，必须剔除。", "hotel_bookings", True),
    ("lending_club 上给 dti 加单调约束后，分数稳定性变好，AUC 没有明显下降。", "lending_club", False),
    ("home_credit 只用主表时，加派生特征几乎没有提升。", "home_credit", False),
    ("另一份信贷数据里，-1 表示字段未填写。", "other_credit", True),
    ("用户补充：home_credit 的 DAYS_EMPLOYED = 365243 不只是退休人员，也包括失业人员。", "home_credit", True),
]
QUESTIONS = [  # (问题, 期望答案里出现的词)
    ("home_credit 里 DAYS_EMPLOYED 等于 365243 是什么意思？", ["退休", "失业"]),
    ("接手一份新的信贷数据，有哪些占位值要注意？", ["365243", "-1"]),
    ("lending_club 的 target 怎么定义？", ["已结清", "已核销"]),
]
CONSOLIDATE_WAIT_SEC = 60   # 存完后等后台把零散事实整理成结论（observations）


def delete_document(client, bank_id: str, document_id: str):
    """Hindsight 没有删单条记录的接口，按存入时的 document 删（低层接口是异步的）。"""
    return _run_async(client.documents.delete_document(bank_id=bank_id, document_id=document_id))


def texts(res) -> list[str]:
    return [getattr(r, "text", str(r)) for r in getattr(res, "results", res)]


def main():
    md = ["# Hindsight 记数据经验的效果", ""]
    server = HindsightServer(llm_provider="deepseek", llm_model="deepseek-flash", llm_api_key=os.environ["DEEPSEEK_API_KEY"])
    server.start(timeout=900)       # 第一次启动要下载并加载向量模型，默认 30 秒不够
    try:
        c = HindsightClient(base_url=server.url)
        for i, (content, ds, confirmed) in enumerate(EXPERIENCES):
            c.retain(bank_id=BANK, content=content, context=f"数据集 {ds}；{'用户确认过' if confirmed else '助手推测'}",
                     document_id=f"exp{i}", metadata={"dataset": ds}, tags=[ds, "confirmed" if confirmed else "guess"])
        time.sleep(CONSOLIDATE_WAIT_SEC)
        md += ["## 取出来的对不对", ""]
        for q, words in QUESTIONS:
            got = texts(c.recall(bank_id=BANK, query=q))
            hit = [w for w in words if any(w in t for t in got)]
            md += [f"**{q}**", "", f"- 期望出现：{words}；实际出现：{hit}", ""] + [f"  - {t}" for t in got[:5]] + [""]
        md += ["## 整理出来的结论", ""]
        for q in ("根据经验，接手一份新的信贷数据时应该先检查什么？", "home_credit 的 DAYS_EMPLOYED 应该怎么处理？"):
            md += [f"**{q}**", "", str(getattr(c.reflect(bank_id=BANK, query=q), "text", "")), ""]
        md += ["## 按标记筛选", ""]
        got = texts(c.recall(bank_id=BANK, query="有哪些数据经验？", tags=["confirmed"], tags_match="any_strict"))
        md += [f"- 只要确认过的，返回 {len(got)} 条；其中含「助手推测」内容的：{sum('5 个日期字段' in t or '单调约束' in t or '派生特征' in t for t in got)} 条", ""]
        md += ["## 删掉一条", ""]
        delete_document(c, BANK, "exp2")      # lending_club 的 target 定义
        time.sleep(CONSOLIDATE_WAIT_SEC)
        after = texts(c.recall(bank_id=BANK, query="lending_club 的 target 怎么定义？"))
        refl = str(getattr(c.reflect(bank_id=BANK, query="lending_club 的 target 怎么定义？"), "text", ""))
        md += [f"- 删掉：{EXPERIENCES[2][0]}",
               f"- 删掉后直接取，还提到「Current / 还款中 / 已结清」的条数：{sum(any(w in t for w in ('Current', '还款中', '已结清')) for t in after)}",
               f"- 删掉后整理出来的结论：{refl}", ""] + [f"  - {t}" for t in after[:5]] + [""]
        c.delete_bank(bank_id=BANK)
    finally:
        server.stop()
    out = ROOT / "eval" / "trials" / "results" / "hindsight_memory.md"
    out.write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
