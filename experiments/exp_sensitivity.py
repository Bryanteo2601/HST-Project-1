"""lambda/gamma sensitivity analysis, tuned on the VALIDATION split only.

LEAKAGE RULE: every configuration is scored on validation, never on test. The
grid result tells you how the operating point would be chosen before the single
look at the test set.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from src.model import ProblemData
from src.metaheuristics.aco import AntColony
from src import metrics as M
from src.experiment import save_table
from src.plotting import sensitivity_heatmap


def run(cfg, bundle, results_dir: Path, plots_dir: Path,
        lambdas=(1.0, 2.5, 5.0, 10.0), gammas=(0.0, 0.25, 0.5, 1.0, 2.0, 5.0)) -> pd.DataFrame:
    tickers = list(bundle.returns.columns)
    rows = []
    for lam in lambdas:
        for gam in gammas:
            c = deepcopy(cfg)
            c.objective.lambda_risk = lam
            c.objective.gamma_cost = gam
            p = ProblemData.from_segment(bundle.train, c)        # optimize on TRAIN
            r = AntColony(p, c, seed=c.seed).optimize()
            val_r = M.portfolio_daily_returns(r.best_w, bundle.validation.returns, tickers)
            rows.append({"lambda_risk": lam, "gamma_cost": gam,
                         "effective_cost_bps": round(gam * c.objective.c_i * 1e4, 2),
                         "val_sharpe": round(M.sharpe(val_r, c.objective.risk_free,
                                                      c.data.annualization), 3),
                         "val_ann_vol": round(M.annualized_vol(val_r, c.data.annualization), 3),
                         "n_holdings": int((r.best_w > 1e-6).sum())})
    df = pd.DataFrame(rows)
    save_table(df, results_dir / "sensitivity.csv")
    try:
        sensitivity_heatmap(df, plots_dir / "sensitivity_val_sharpe.png", value="val_sharpe")
    except Exception as e:  # pragma: no cover
        print(f"[sensitivity] heatmap skipped: {e}")
    best = df.loc[df["val_sharpe"].idxmax()]
    print(f"\n=== lambda/gamma sensitivity (VALIDATION-tuned) ===\n{df.to_string(index=False)}")
    print(f"\nBest on validation: lambda={best['lambda_risk']}, gamma={best['gamma_cost']} "
          f"(val_sharpe={best['val_sharpe']})  <- use this BEFORE touching test")
    return df


if __name__ == "__main__":
    from experiments._common import parse_and_load, RESULTS, PLOTS
    cfg, bundle = parse_and_load()
    run(cfg, bundle, RESULTS, PLOTS)
