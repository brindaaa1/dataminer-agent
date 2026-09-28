"""读取项目根目录的 .env（KEY=VALUE，# 开头为注释）。不覆盖已存在的环境变量。不依赖 python-dotenv。"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path | None = None) -> list[str]:
    """返回本次新设置的变量名（不返回值，避免 key 出现在日志里）。"""
    p = path or ROOT / ".env"
    if not p.exists():
        return []
    set_ = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip().removeprefix("export ").strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v
            set_.append(k)
    return set_
