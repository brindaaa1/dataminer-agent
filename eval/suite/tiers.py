"""评测档位：eval/tiers.yaml。"""
from pathlib import Path

import yaml

TIERS = Path(__file__).resolve().parents[1] / "tiers.yaml"


def load_tier(name: str) -> dict:
    return yaml.safe_load(TIERS.read_text())[name]
