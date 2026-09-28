"""仓库自带的酒店小样本（data_sample/，许可见其中的 NOTICE.md）：让 clone 下来不下载数据、不配 key 也能跑通整个流程。"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = ROOT / "data_sample" / "hotel_bookings" / "hotel_bookings_sample.csv"


def sample_art() -> Path:
    """样本运行的产物目录：默认 artifacts_sample/；环境变量 DATAMINER_SAMPLE_ART 可改（测试用它写到临时目录）。"""
    return Path(os.environ.get("DATAMINER_SAMPLE_ART") or ROOT / "artifacts_sample")


def use_sample(cfg: dict, art: Path | None = None) -> dict:
    """原地改 cfg：hotel_bookings 读样本；产物、报告、memory 都写到 art，不影响全量数据的结果。"""
    art = Path(art or sample_art())
    cfg["datasets"]["hotel_bookings"]["tables"]["main"] = str(SAMPLE_CSV)
    cfg["paths"].update(artifacts_root=str(art), reports_root=str(art / "reports"))
    cfg["memory"]["db"] = str(art / "memory.db")
    # 低保真度默认抽 10%，是按几十万行以上的数据定的。样本的训练集只有约 4,200 行，10% 只剩约 420 行、验证集约 100 行，
    # 过拟合、收敛这类诊断基本是噪声（hotel_deepseek1 里两次 OVERFIT_GAP 和 NO_CONVERGE 都来自这里）。样本上改为 50%。
    cfg["fidelity"]["low"]["sample_frac"] = 0.5
    return cfg
