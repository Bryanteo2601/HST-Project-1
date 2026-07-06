"""Gap-to-optimal on the main problem (N=50, K=10): exact MIQP vs metaheuristics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.model import ProblemData
from src.benchmarks import equal_weight, exact_miqp, gap_to_optimal
from src.experiment import METHODS, save_table


def run(cfg, bundle, results_dir: Path, plots_dir: Path) -> pd.DataFrame:
    # Validate on the full main problem: Gurobi certifies the optimum in ~0.05s,
    # so no need for a reduced instance, and the numbers stay consistent with the
    # rest of the report (N=50, K from the config).
    p = ProblemData.from_segment(bundle.train, cfg)
    miqp = exact_miqp(p, cfg)

    rows = []
    if miqp.available:
        opt = miqp.fitness
        rows.append({"method": "MIQP(optimal)", "fitness": round(opt, 5), "gap_%": 0.0,
                     "solver": miqp.solver, "runtime_s": round(miqp.runtime_s, 3)})
    else:
        opt = None
        print(f"[gap_to_optimal] exact MIQP unavailable: {miqp.message}")

    for name, Opt in METHODS.items():
        r = Opt(p, cfg, seed=cfg.seed).optimize()
        gap = gap_to_optimal(r.best_fitness, opt) * 100 if opt is not None else np.nan
        rows.append({"method": name, "fitness": round(r.best_fitness, 5),
                     "gap_%": round(gap, 4), "solver": "-", "runtime_s": np.nan})

    port = equal_weight(p)
    gap = gap_to_optimal(port.fitness, opt) * 100 if opt is not None else np.nan
    rows.append({"method": "EqualWeight", "fitness": round(port.fitness, 5),
                 "gap_%": round(gap, 4), "solver": "-", "runtime_s": np.nan})

    df = pd.DataFrame(rows).set_index("method")
    save_table(df, results_dir / "gap_to_optimal.csv")
    print(f"\n=== Gap-to-optimal (N={p.n}, K={p.K}) ===\n{df}")
    return df


if __name__ == "__main__":
    from experiments._common import parse_and_load, RESULTS, PLOTS
    cfg, bundle = parse_and_load()
    run(cfg, bundle, RESULTS, PLOTS)
