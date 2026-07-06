"""Reproduce the full experiment suite end to end.

Usage:
    python run_all.py                          # full run (Ledoit-Wolf, variant A)
    python run_all.py --fast                   # small budgets, quick smoke run
    python run_all.py --override config/objective_B.yaml
    python run_all.py --only core gap          # run a subset of stages

Stages:
    data  -> build/cache prices, returns, mu/Sigma per split, CSV + Excel exports
    core  -> SA/GA/ACO (+ benchmarks) on TRAIN, freeze, evaluate OOS; master table
             + convergence / equity / risk-return / IS-OOS gap / ACO pheromone plots
    gap   -> exact MIQP gap-to-optimal on a small instance
    seed  -> multi-seed robustness (mean ± std)
    sens  -> lambda/gamma sensitivity (validation-tuned)
    obj   -> objective variants A vs B vs C
    wf    -> walk-forward rolling rebalance
    shrink-> sample vs Ledoit-Wolf overfitting-gap comparison
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.data import build_dataset  # noqa: E402
from src.experiment import fast_config, master_table, run_suite, save_table  # noqa: E402
from src import plotting as P  # noqa: E402
from experiments import (  # noqa: E402
    exp_gap_to_optimal, exp_objective_variants, exp_seed_robustness,
    exp_sensitivity, exp_shrinkage_gap, exp_walkforward,
)

STAGES = ["core", "gap", "seed", "sens", "obj", "wf", "shrink"]


def run_core(cfg, bundle, results: Path, plots: Path) -> None:
    suite = run_suite(bundle, cfg)
    mt = master_table(suite)
    save_table(mt, results / "master_table.csv")
    print(f"\n=== MASTER RESULTS TABLE (variant {cfg.objective.variant}, "
          f"{cfg.covariance.estimator}) ===\n{mt}\n")

    P.convergence_curves(suite, plots / "convergence.png")
    P.equity_curves(suite, plots / "oos_equity.png")
    P.risk_return_scatter(suite, plots / "risk_return.png")
    P.is_oos_gap_bars(suite, plots / "is_oos_gap.png")
    aco_best = suite["methods"]["ACO"]["best"]
    P.pheromone_heatmap(aco_best, suite["tickers"], plots / "aco_pheromone.png")
    P.selection_frequency(aco_best, suite["tickers"], plots / "aco_selection.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--only", nargs="+", choices=STAGES, default=STAGES)
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    if args.fast:
        cfg = fast_config(cfg)

    results = ROOT / cfg.experiment.out_dir
    plots = ROOT / cfg.experiment.out_dir.replace("results", "plots")
    results.mkdir(parents=True, exist_ok=True); plots.mkdir(parents=True, exist_ok=True)
    cfg.save(results / "resolved_config.yaml")

    print(f"[data] building dataset (covariance={cfg.covariance.estimator}, "
          f"objective={cfg.objective.variant}, seeds={cfg.experiment.seeds}) ...")
    bundle = build_dataset(cfg, write_outputs=True)

    stages = {
        "core": lambda: run_core(cfg, bundle, results, plots),
        "gap":  lambda: exp_gap_to_optimal.run(cfg, bundle, results, plots),
        "seed": lambda: exp_seed_robustness.run(cfg, bundle, results, plots),
        "sens": lambda: exp_sensitivity.run(cfg, bundle, results, plots),
        "obj":  lambda: exp_objective_variants.run(cfg, bundle, results, plots),
        "wf":   lambda: exp_walkforward.run(cfg, bundle, results, plots),
        "shrink": lambda: exp_shrinkage_gap.run(cfg, bundle, results, plots),
    }
    for stage in STAGES:
        if stage in args.only:
            print(f"\n########## stage: {stage} ##########")
            stages[stage]()

    print(f"\nDone. Tables -> {results}/  Figures -> {plots}/")


if __name__ == "__main__":
    main()
