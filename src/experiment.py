"""Orchestration shared by run_all.py and the experiments/ scripts.

Runs the metaheuristics (multi-seed) and benchmarks on a split, freezes each best
portfolio, evaluates it out-of-sample, and assembles the master results table.
Keeping this here means every experiment uses the same leakage-safe path: optimize
on TRAIN, evaluate on TEST via src.backtest.
"""
from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.data import DataBundle
from src.model import ProblemData
from src.metaheuristics.sa import SimulatedAnnealing
from src.metaheuristics.ga import GeneticAlgorithm
from src.metaheuristics.aco import AntColony
from src.metaheuristics.tabu import TabuSearch
from src.benchmarks import equal_weight, greedy_ratio, random_search
from src.backtest import in_sample_vs_oos
from src import metrics as M

# SA is the main method; Tabu is its memory-based refinement; GA/ACO are comparisons.
METHODS = {"SA": SimulatedAnnealing, "Tabu": TabuSearch,
           "GA": GeneticAlgorithm, "ACO": AntColony}


# --------------------------------------------------------------------------- #
# Config presets
# --------------------------------------------------------------------------- #
def fast_config(cfg: Config) -> Config:
    """Shrink iteration budgets for smoke tests / quick reproductions."""
    c = deepcopy(cfg)
    c.sa.iters_per_temp = 10; c.sa.cooling = 0.8; c.sa.t_min = 0.05
    c.ga.pop = 20; c.ga.generations = 15
    c.aco.ants = 10; c.aco.iters = 15
    c.experiment.seeds = [0, 1]
    return c


# --------------------------------------------------------------------------- #
# Method runs
# --------------------------------------------------------------------------- #
def run_method(name: str, p: ProblemData, cfg: Config, seed: int):
    t0 = time.time()
    res = METHODS[name](p, cfg, seed=seed).optimize()
    res.meta["runtime_s"] = time.time() - t0
    res.meta["seed"] = seed
    return res


def run_multiseed(name: str, p: ProblemData, cfg: Config, seeds: list[int]):
    results = [run_method(name, p, cfg, s) for s in seeds]
    fits = np.array([r.best_fitness for r in results], dtype=float)
    rts = np.array([r.meta["runtime_s"] for r in results], dtype=float)
    nev = np.array([r.n_evals for r in results], dtype=float)
    agg = {
        "method": name,
        "best_fitness_mean": float(np.mean(fits)),
        "best_fitness_std": float(np.std(fits)),
        "best_fitness_max": float(np.max(fits)),
        "runtime_mean_s": float(np.mean(rts)),
        "n_evals_mean": float(np.mean(nev)),
    }
    best = max(results, key=lambda r: r.best_fitness)
    return results, agg, best


# --------------------------------------------------------------------------- #
# Full suite on a bundle
# --------------------------------------------------------------------------- #
def run_suite(bundle: DataBundle, cfg: Config, methods: list[str] | None = None) -> dict:
    """Optimize each method (multi-seed) + benchmarks on TRAIN; evaluate OOS on TEST."""
    methods = methods or list(METHODS)
    tickers = list(bundle.returns.columns)
    p_train = ProblemData.from_segment(bundle.train, cfg)
    train_r, test_r = bundle.train.returns, bundle.test_returns()

    suite: dict = {"tickers": tickers, "methods": {}, "benchmarks": {},
                   "covariance": cfg.covariance.estimator, "variant": cfg.objective.variant}

    for name in methods:
        results, agg, best = run_multiseed(name, p_train, cfg, cfg.experiment.seeds)
        oos = in_sample_vs_oos(best.best_w, train_r, test_r, bundle.sectors, cfg, tickers)
        suite["methods"][name] = {
            "results": results, "agg": agg, "best": best, "oos": oos,
            "oos_returns": M.portfolio_daily_returns(best.best_w, test_r, tickers),
        }

    bench_ports = {
        "EqualWeight": equal_weight(p_train),
        "Greedy": greedy_ratio(p_train, cfg),
        "Random": random_search(p_train, cfg, n_samples=200, seed=cfg.seed),
    }
    for name, port in bench_ports.items():
        oos = in_sample_vs_oos(port.w, train_r, test_r, bundle.sectors, cfg, tickers)
        suite["benchmarks"][name] = {
            "portfolio": port, "oos": oos,
            "oos_returns": M.portfolio_daily_returns(port.w, test_r, tickers),
        }
    return suite


# --------------------------------------------------------------------------- #
# Master results table
# --------------------------------------------------------------------------- #
def master_table(suite: dict) -> pd.DataFrame:
    rows = []
    for name, d in suite["methods"].items():
        oos = d["oos"]["out_of_sample"]
        rows.append({
            "method": name, "type": "metaheuristic",
            "best_fitness_mean": d["agg"]["best_fitness_mean"],
            "best_fitness_std": d["agg"]["best_fitness_std"],
            "is_sharpe": d["oos"]["in_sample"]["sharpe"],
            "oos_sharpe": oos["sharpe"], "oos_ann_return": oos["ann_return"],
            "oos_ann_vol": oos["ann_vol"], "oos_max_drawdown": oos["max_drawdown"],
            "gap_sharpe": d["oos"]["gap"]["sharpe"],
            "n_holdings": oos["n_holdings"], "runtime_mean_s": d["agg"]["runtime_mean_s"],
            "n_evals_mean": d["agg"]["n_evals_mean"],
        })
    for name, d in suite["benchmarks"].items():
        oos = d["oos"]["out_of_sample"]
        rows.append({
            "method": name, "type": "benchmark",
            "best_fitness_mean": d["portfolio"].fitness, "best_fitness_std": 0.0,
            "is_sharpe": d["oos"]["in_sample"]["sharpe"],
            "oos_sharpe": oos["sharpe"], "oos_ann_return": oos["ann_return"],
            "oos_ann_vol": oos["ann_vol"], "oos_max_drawdown": oos["max_drawdown"],
            "gap_sharpe": d["oos"]["gap"]["sharpe"],
            "n_holdings": oos["n_holdings"], "runtime_mean_s": np.nan, "n_evals_mean": np.nan,
        })
    return pd.DataFrame(rows).set_index("method").round(4)


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if is_dataclass(o) and not isinstance(o, type):
        return asdict(o)
    return str(o)


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_jsonable)


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
