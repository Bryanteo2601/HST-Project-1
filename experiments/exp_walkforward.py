"""Walk-forward rolling rebalance for each method: does any edge persist OOS?"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.model import ProblemData
from src.backtest import walk_forward
from src import metrics as M
from src.experiment import METHODS, save_table


def _solver(name, cfg):
    def solve_fn(seg):
        p = ProblemData.from_segment(seg, cfg)
        return METHODS[name](p, cfg, seed=cfg.seed).optimize().best_w
    return solve_fn


def run(cfg, bundle, results_dir: Path, plots_dir: Path) -> pd.DataFrame:
    rows, curves = [], {}
    for name in METHODS:
        wf = walk_forward(bundle, _solver(name, cfg), cfg)
        if wf.daily_returns.empty:
            continue
        s = wf.summary
        rows.append({"method": name, "oos_sharpe": round(s["sharpe"], 3),
                     "oos_ann_return": round(s["ann_return"], 3), "oos_ann_vol": round(s["ann_vol"], 3),
                     "oos_max_drawdown": round(s["max_drawdown"], 3),
                     "n_rebalances": s["n_rebalances"], "avg_turnover": round(s["avg_turnover"], 3),
                     "total_cost": round(s["total_transaction_cost"], 4)})
        curves[name] = wf.equity

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, eq in curves.items():
        ax.plot(eq.index, eq.values, lw=2, label=name)
    ax.set_xlabel("date"); ax.set_ylabel("growth of $1 (walk-forward, net of cost)")
    ax.set_title("Walk-forward equity curves"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); (plots_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "walkforward_equity.png", dpi=130); plt.close(fig)

    df = pd.DataFrame(rows).set_index("method")
    save_table(df, results_dir / "walkforward.csv")
    print(f"\n=== Walk-forward (rolling rebalance, net of cost) ===\n{df}")
    return df


if __name__ == "__main__":
    from experiments._common import parse_and_load, RESULTS, PLOTS
    cfg, bundle = parse_and_load()
    run(cfg, bundle, RESULTS, PLOTS)
