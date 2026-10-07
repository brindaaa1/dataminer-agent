"""生成的配置 vs 手写配置：两边都过 column_policy（与切分时的口径相同），比较剔除、隔离的结果和逐字段的可得时间。"""
from data.knowhow import column_policy, header, load_knowhow


def compare(dataset_id: str, generated_root: str, reference_root: str, csv_path: str, reference_id: str = "hotel_bookings") -> dict:
    cols = header(csv_path)
    g = load_knowhow(dataset_id, {"paths": {"knowhow_root": generated_root}})
    r = load_knowhow(reference_id, {"paths": {"knowhow_root": reference_root}})
    pg, pr = column_policy(dataset_id, cols, g), column_policy(reference_id, cols, r)
    label = {r["dictionary"]["label"]["col"]}
    leak = lambda p: set(p["drop"]["leakage"]) - label          # 只用于切分的 meta 字段不算泄漏，单独统计
    lg, lr = leak(pg), leak(pr)
    mg, mr = set(pg["drop"]["meta"]), set(pr["drop"]["meta"])
    fg, fr = g["dictionary"]["fields"], r["dictionary"]["fields"]
    avail = lambda f, fs, m: "meta" if f in m else fs[f]["availability"]
    common = [c for c in fr if c in fg and fr[c].get("availability")]
    dis = [{"field": c, "generated": avail(c, fg, mg), "reference": avail(c, fr, mr)}
           for c in common if avail(c, fg, mg) != avail(c, fr, mr)]
    return {"leak_ref": sorted(lr), "leak_found": sorted(lg & lr), "leak_missed": sorted(lr - lg), "leak_extra": sorted(lg - lr),
            "quarantine_ref": pr["quarantine"], "quarantine_found": sorted(set(pg["quarantine"]) & set(pr["quarantine"])),
            "quarantine_missed": sorted(set(pr["quarantine"]) - set(pg["quarantine"]) - lg),
            "meta_ref": sorted(mr), "meta_missed": sorted(mr - mg - lg),
            "availability_agree": round(1 - len(dis) / len(common), 4) if common else None,
            "n_common_fields": len(common), "disagreements": dis}
