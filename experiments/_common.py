"""Shared CLI helpers for the experiment scripts."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.data import build_dataset  # noqa: E402
from src.experiment import fast_config  # noqa: E402

RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"


def parse_and_load():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--fast", action="store_true", help="small budgets for a quick run")
    args = ap.parse_args()
    cfg = load_config(args.config, overrides=args.override)
    if args.fast:
        cfg = fast_config(cfg)
    bundle = build_dataset(cfg, write_outputs=False)
    return cfg, bundle
