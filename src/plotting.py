"""All figure generation, written to plots/ (headless Agg backend).

Each function takes assembled results (see src.experiment) plus an output path and
saves one figure: convergence curves, OOS equity curves, risk-return scatter, the
IS-vs-OOS overfitting bars (with and without shrinkage), and the ACO pheromone
heatmap / selection-frequency chart.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import metrics as M  # noqa: E402


def _save(fig, path: str | Path) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


def convergence_curves(suite: dict, path: str | Path) -> Path:
    """Best-so-far objective value vs run progress, mean +/- std across seeds per method."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, d in suite["methods"].items():
        hists = [np.asarray(r.history, dtype=float) for r in d["results"] if len(r.history)]
        if not hists:
            continue
        L = min(len(h) for h in hists)
        H = np.vstack([h[:L] for h in hists])
        x = np.linspace(0, 1, L)
        mean, std = H.mean(0), H.std(0)
        ax.plot(x, mean, label=f"{name} (best={d['agg']['best_fitness_max']:.3f})", lw=2)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15)
    ax.set_xlabel("run progress (fraction)"); ax.set_ylabel("best-so-far objective value")
    ax.set_title("Convergence (mean ± std across seeds)"); ax.legend(); ax.grid(alpha=0.3)
    return _save(fig, path)


def equity_curves(suite: dict, path: str | Path) -> Path:
    """Out-of-sample cumulative equity for each method and benchmark."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, d in suite["methods"].items():
        eq = M.equity_curve(d["oos_returns"])
        ax.plot(eq.index, eq.values, lw=2, label=f"{name}")
    for name, d in suite["benchmarks"].items():
        eq = M.equity_curve(d["oos_returns"])
        ax.plot(eq.index, eq.values, lw=1.2, ls="--", alpha=0.8, label=f"{name}")
    ax.set_xlabel("date"); ax.set_ylabel("growth of $1 (OOS)")
    ax.set_title("Out-of-sample equity curves"); ax.legend(ncol=2, fontsize=8); ax.grid(alpha=0.3)
    return _save(fig, path)


def risk_return_scatter(suite: dict, path: str | Path) -> Path:
    """OOS annualized risk vs return for methods and benchmarks."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for group, marker in (("methods", "o"), ("benchmarks", "s")):
        for name, d in suite[group].items():
            oos = d["oos"]["out_of_sample"]
            ax.scatter(oos["ann_vol"], oos["ann_return"], marker=marker, s=90)
            ax.annotate(name, (oos["ann_vol"], oos["ann_return"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("OOS annualized volatility"); ax.set_ylabel("OOS annualized return")
    ax.set_title("Risk-return (out-of-sample)"); ax.grid(alpha=0.3)
    return _save(fig, path)


def is_oos_gap_bars(suite: dict, path: str | Path) -> Path:
    """Grouped bars: in-sample vs out-of-sample Sharpe per method (the gap)."""
    names = list(suite["methods"]) + list(suite["benchmarks"])
    is_s, oos_s = [], []
    for n in names:
        d = suite["methods"].get(n) or suite["benchmarks"][n]
        is_s.append(d["oos"]["in_sample"]["sharpe"]); oos_s.append(d["oos"]["out_of_sample"]["sharpe"])
    x = np.arange(len(names)); w = 0.4
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, is_s, w, label="in-sample")
    ax.bar(x + w / 2, oos_s, w, label="out-of-sample")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Sharpe"); ax.set_title("In-sample vs out-of-sample Sharpe (overfitting gap)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    return _save(fig, path)


def shrinkage_gap_bars(gaps: dict, path: str | Path) -> Path:
    """Compare the IS-OOS Sharpe gap with sample vs Ledoit-Wolf covariance.

    ``gaps`` maps method -> {"sample": gap, "ledoit_wolf": gap}.
    """
    names = list(gaps)
    samp = [gaps[n]["sample"] for n in names]
    lw = [gaps[n]["ledoit_wolf"] for n in names]
    x = np.arange(len(names)); w = 0.4
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, samp, w, label="sample cov")
    ax.bar(x + w / 2, lw, w, label="Ledoit-Wolf")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("IS - OOS Sharpe gap"); ax.axhline(0, color="k", lw=0.8)
    ax.set_title("Does shrinkage narrow the overfitting gap?"); ax.legend(); ax.grid(alpha=0.3, axis="y")
    return _save(fig, path)


def pheromone_heatmap(aco_result, tickers: list[str], path: str | Path, top: int = 25) -> Path:
    """Heatmap of pheromone over iterations for the most-selected stocks."""
    hist = np.asarray(aco_result.meta["pheromone_history"])  # (iters, n)
    counts = np.asarray(aco_result.meta["selection_counts"])
    order = np.argsort(counts)[::-1][:top]
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(hist[:, order].T, aspect="auto", cmap="viridis", origin="lower")
    ax.set_yticks(range(len(order))); ax.set_yticklabels([tickers[i] for i in order], fontsize=7)
    ax.set_xlabel("ACO iteration"); ax.set_ylabel("stock (most-selected at top)")
    ax.set_title("ACO pheromone evolution"); fig.colorbar(im, ax=ax, label="pheromone τ")
    return _save(fig, path)


def selection_frequency(aco_result, tickers: list[str], path: str | Path, top: int = 20) -> Path:
    counts = np.asarray(aco_result.meta["selection_counts"])
    order = np.argsort(counts)[::-1][:top]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([tickers[i] for i in order], counts[order])
    ax.set_ylabel("times selected by an ant"); ax.set_title("Most-persistently-selected stocks (ACO)")
    ax.tick_params(axis="x", rotation=60); ax.grid(alpha=0.3, axis="y")
    return _save(fig, path)


def sensitivity_heatmap(grid: pd.DataFrame, path: str | Path, value: str = "val_sharpe") -> Path:
    """Heatmap of a metric over a lambda x gamma grid (tuned on validation)."""
    piv = grid.pivot(index="lambda_risk", columns="gamma_cost", values=value)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(piv.values, aspect="auto", cmap="magma", origin="lower")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"{c:g}" for c in piv.columns], rotation=45)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([f"{i:g}" for i in piv.index])
    ax.set_xlabel("gamma_cost"); ax.set_ylabel("lambda_risk")
    ax.set_title(f"Validation {value} sensitivity"); fig.colorbar(im, ax=ax, label=value)
    return _save(fig, path)
