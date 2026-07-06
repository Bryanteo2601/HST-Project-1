"""Overfitting mitigation: does Ledoit-Wolf shrinkage narrow the IS-OOS gap?

Runs the whole optimize->freeze->evaluate pipeline with sample covariance and with
Ledoit-Wolf, then compares the in-sample minus out-of-sample Sharpe gap per method
(averaged across seeds). This is a key insight section.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import build_dataset
from src.experiment import run_suite, save_table
from src.plotting import shrinkage_gap_bars


def _mean_gap_per_method(cfg, bundle) -> dict[str, float]:
    """Average IS-OOS Sharpe gap across seeds for each method."""
    from src.model import ProblemData
    from src.backtest import in_sample_vs_oos
    from src.experiment import run_multiseed
    p = ProblemData.from_segment(bundle.train, cfg)
    tickers = list(bundle.returns.columns)
    gaps = {}
    for name in ("SA", "GA", "ACO"):
        results, _, _ = run_multiseed(name, p, cfg, cfg.experiment.seeds)
        per_seed = [in_sample_vs_oos(r.best_w, bundle.train.returns, bundle.test_returns(),
                                     bundle.sectors, cfg, tickers)["gap"]["sharpe"] for r in results]
        gaps[name] = float(np.mean(per_seed))
    return gaps


def run(cfg, bundle, results_dir: Path, plots_dir: Path) -> pd.DataFrame:
    out = {}
    for estimator in ("sample", "ledoit_wolf"):
        c = deepcopy(cfg)
        c.covariance.estimator = estimator
        # rebuild bundle so split-local mu/Sigma use this estimator
        b = build_dataset(c, write_outputs=False)
        gaps = _mean_gap_per_method(c, b)
        for m, g in gaps.items():
            out.setdefault(m, {})[estimator] = g

    df = pd.DataFrame(out).T
    df["narrowed_by_shrinkage"] = df["sample"].abs() - df["ledoit_wolf"].abs()
    save_table(df.round(4), results_dir / "shrinkage_gap.csv")
    shrinkage_gap_bars(out, plots_dir / "shrinkage_gap.png")
    print(f"\n=== Shrinkage vs sample: mean IS-OOS Sharpe gap ===\n{df.round(4)}")
    print("(narrowed_by_shrinkage > 0 means Ledoit-Wolf reduced the |gap|)")
    return df


if __name__ == "__main__":
    from experiments._common import parse_and_load, RESULTS, PLOTS
    cfg, bundle = parse_and_load()
    run(cfg, bundle, RESULTS, PLOTS)
