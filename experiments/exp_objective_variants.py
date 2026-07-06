"""Objective variants A (three-term) vs B (Sharpe) vs C (L1-regularized QP).

A and B are searched by the metaheuristics (ACO shown); C is the convex L1 QP
swept over lambda_L1, with the operating point chosen on the VALIDATION split
(never test). Reports OOS metrics for each and the C cardinality/lambda trade-off.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from src.model import ProblemData
from src.qp import solve_l1_portfolio, effective_cardinality
from src.metaheuristics.aco import AntColony
from src.backtest import in_sample_vs_oos, problem_from_returns
from src import metrics as M
from src.experiment import save_table


def _oos_row(name, w, cfg, bundle, tickers):
    o = in_sample_vs_oos(w, bundle.train.returns, bundle.test_returns(),
                         bundle.sectors, cfg, tickers)
    oos = o["out_of_sample"]
    return {"objective": name, "is_sharpe": round(o["in_sample"]["sharpe"], 3),
            "oos_sharpe": round(oos["sharpe"], 3), "oos_ann_return": round(oos["ann_return"], 3),
            "oos_ann_vol": round(oos["ann_vol"], 3), "n_holdings": oos["n_holdings"],
            "gap_sharpe": round(o["gap"]["sharpe"], 3)}


def run(cfg, bundle, results_dir: Path, plots_dir: Path) -> pd.DataFrame:
    tickers = list(bundle.returns.columns)
    rows = []

    # Variants A and B via ACO
    for variant in ("A", "B"):
        c = deepcopy(cfg); c.objective.variant = variant
        p = ProblemData.from_segment(bundle.train, c)
        r = AntColony(p, c, seed=c.seed).optimize()
        rows.append(_oos_row(f"{variant} (ACO)", r.best_w, c, bundle, tickers))

    # Variant C: sweep lambda_L1, choose on validation by Sharpe near target cardinality
    c = deepcopy(cfg); c.objective.variant = "C"
    p_tr = ProblemData.from_segment(bundle.train, c)
    p_val = problem_from_returns(bundle.validation.returns, bundle.sectors, c, "val")
    sweep = []
    for lam in [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]:
        res = solve_l1_portfolio(p_tr, c, lambda_L1=lam)
        if not res.feasible:
            continue
        val_sharpe = M.sharpe(M.portfolio_daily_returns(res.w, bundle.validation.returns, tickers),
                              c.objective.risk_free, c.data.annualization)
        sweep.append({"lambda_L1": lam, "eff_cardinality": effective_cardinality(res.w),
                      "val_sharpe": round(val_sharpe, 3), "w": res.w})
    sweep_df = pd.DataFrame([{k: v for k, v in s.items() if k != "w"} for s in sweep])
    save_table(sweep_df.set_index("lambda_L1"), results_dir / "variant_C_sweep.csv")

    best_c = max(sweep, key=lambda s: s["val_sharpe"])  # selection on validation only
    rows.append(_oos_row(f"C (L1, lam={best_c['lambda_L1']})", best_c["w"], c, bundle, tickers))

    df = pd.DataFrame(rows).set_index("objective")
    save_table(df, results_dir / "objective_variants.csv")
    print(f"\n=== Objective variants (OOS) ===\n{df}")
    print(f"\nVariant C lambda_L1 sweep (cardinality control):\n{sweep_df.to_string(index=False)}")
    return df


if __name__ == "__main__":
    from experiments._common import parse_and_load, RESULTS, PLOTS
    cfg, bundle = parse_and_load()
    run(cfg, bundle, RESULTS, PLOTS)
