"""评测结果写进 Langfuse：每个评测数据集一个 Langfuse 数据集，每个种子一个条目，每个版本号一次实验（run）。
trace 本身由运行时上报（runtime/tracing.py），这里只把它挂到条目上并写分数。没配 Langfuse 时什么都不做。"""
from runtime.tracing import langfuse_client


def report(label: str, rows: list[dict], bases: dict, client=None):
    lf = client or langfuse_client()
    if lf is None:
        return
    for name in dict.fromkeys(r["dataset"] for r in rows):
        lf.create_dataset(name=f"dataminer-eval/{name}", description="DataMiner agent 结果层评测（每个种子一个条目）")
    for r in rows:
        iid = f"{r['dataset']}-s{r['seed']}"
        lf.create_dataset_item(dataset_name=f"dataminer-eval/{r['dataset']}", id=iid, input={"dataset": r["dataset"], "seed": r["seed"]})
        if not r.get("trace_id"):
            continue
        lf.api.dataset_run_items.create(run_name=label, dataset_item_id=iid, trace_id=r["trace_id"])
        b0 = ((bases.get(r["dataset"]) or {}).get("b0") or {}).get("holdout_auc")
        scores = {"holdout": r.get("holdout"), "delta_vs_b0": None if r.get("holdout") is None or b0 is None else r["holdout"] - b0,
                  "cost_usd": r.get("cost_usd"), "output_tokens": r["usage"]["output"]}
        for k, v in scores.items():
            if v is not None:
                lf.create_score(name=k, value=float(v), trace_id=r["trace_id"])
    lf.flush()
