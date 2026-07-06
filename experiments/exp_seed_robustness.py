"""Multi-seed robustness: mean +/- std of best fitness and OOS Sharpe per method.

This isolates algorithm stochasticity (kept conceptually separate from estimation
overfitting): the same problem, different random seeds.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.model import ProblemData
from src.backtest import in_sample_vs_oos
from src.experiment import METHODS, run_multiseed, save_table


def run(cfg, bundle, results_dir: Path, plots_dir: Path) -> pd.DataFrame:
    p = ProblemData.from_segment(bundle.train, cfg)
    tickers = list(bundle.returns.columns)
    rows = []
    for name in METHODS:
        results, agg, _ = run_multiseed(name, p, cfg, cfg.experiment.seeds)
        oos_sharpes = [in_sample_vs_oos(r.best_w, bundle.train.returns, bundle.test_returns(),
                                        bundle.sectors, cfg, tickers)["out_of_sample"]["sharpe"]
                       for r in results]
        rows.append({"method": name,
                     "fitness_mean": round(agg["best_fitness_mean"], 4),
                     "fitness_std": round(agg["best_fitness_std"], 4),
                     "fitness_max": round(agg["best_fitness_max"], 4),
                     "oos_sharpe_mean": round(float(np.mean(oos_sharpes)), 3),
                     "oos_sharpe_std": round(float(np.std(oos_sharpes)), 3),
                     "runtime_mean_s": round(agg["runtime_mean_s"], 2)})
    df = pd.DataFrame(rows).set_index("method")
    save_table(df, results_dir / "seed_robustness.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df.index, df["fitness_mean"], yerr=df["fitness_std"], capsize=6, alpha=0.85)
    ax.set_ylabel("best fitness (mean ± std)")
    ax.set_title(f"Seed robustness over {len(cfg.experiment.seeds)} seeds"); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "seed_robustness.png", dpi=130); plt.close(fig)

    print(f"\n=== Seed robustness (mean ± std) ===\n{df}")
    return df


if __name__ == "__main__":
    from experiments._common import parse_and_load, RESULTS, PLOTS
    cfg, bundle = parse_and_load()
    run(cfg, bundle, RESULTS, PLOTS)
