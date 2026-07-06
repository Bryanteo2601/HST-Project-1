"""Out-of-sample evaluation and walk-forward backtesting (the leakage boundary).

This module is the ONLY place TEST-window returns are consumed. ``evaluate_oos``
takes an already-FROZEN portfolio (optimized on train) and reports metrics on
held-out returns. ``in_sample_vs_oos`` reports the IS-vs-OOS gap (realized Sharpe
and model fitness) explicitly -- that gap IS the overfitting. ``walk_forward``
rolls estimate -> hold -> re-optimize -> rebalance, charging transaction cost on
each rebalance, to test whether any edge persists across many OOS windows.

Estimating mu/Sigma here (e.g. for OOS fitness) is evaluation, not optimization,
so it never feeds an optimizer -- the leakage rule still holds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.config import Config
from src.data import DataBundle, Segment, build_segment, walk_forward_windows
from src.model import ProblemData, fitness
from src import metrics as M


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def problem_from_returns(returns: pd.DataFrame, sectors: pd.Series, cfg: Config,
                         name: str = "eval") -> ProblemData:
    """Build a ProblemData from a return window (evaluation-side, ungated)."""
    seg = build_segment(name, returns, sectors.reindex(returns.columns), cfg)
    return ProblemData.from_segment(seg, cfg)


# --------------------------------------------------------------------------- #
# Single-split OOS evaluation
# --------------------------------------------------------------------------- #
def evaluate_oos(w: np.ndarray, test_returns: pd.DataFrame, sectors: pd.Series,
                 cfg: Config, tickers: list[str], w_prev: np.ndarray | None = None) -> dict:
    """Metrics for a frozen portfolio on the held-out TEST window."""
    r = M.portfolio_daily_returns(w, test_returns, tickers)
    return M.summary(r, w=w, w_prev=w_prev, sectors=sectors, tickers=tickers,
                     risk_free=cfg.objective.risk_free,
                     cost_bps=cfg.backtest.rebalance_cost_bps,
                     periods=cfg.data.annualization)


def in_sample_vs_oos(w: np.ndarray, train_returns: pd.DataFrame, test_returns: pd.DataFrame,
                     sectors: pd.Series, cfg: Config, tickers: list[str],
                     w_prev: np.ndarray | None = None) -> dict:
    """IS vs OOS comparison + the overfitting gap (realized Sharpe and model fitness)."""
    periods = cfg.data.annualization
    rf = cfg.objective.risk_free

    r_is = M.portfolio_daily_returns(w, train_returns, tickers)
    r_oos = M.portfolio_daily_returns(w, test_returns, tickers)

    p_is = problem_from_returns(train_returns, sectors, cfg, "is")
    p_oos = problem_from_returns(test_returns, sectors, cfg, "oos")
    fit_is = fitness(w, p_is)
    fit_oos = fitness(w, p_oos)

    is_metrics = M.summary(r_is, w=w, sectors=sectors, tickers=tickers,
                           risk_free=rf, periods=periods)
    oos_metrics = M.summary(r_oos, w=w, w_prev=w_prev, sectors=sectors, tickers=tickers,
                            risk_free=rf, cost_bps=cfg.backtest.rebalance_cost_bps,
                            periods=periods)
    return {
        "in_sample": {**is_metrics, "fitness": float(fit_is)},
        "out_of_sample": {**oos_metrics, "fitness": float(fit_oos)},
        "gap": {
            "sharpe": is_metrics["sharpe"] - oos_metrics["sharpe"],
            "ann_return": is_metrics["ann_return"] - oos_metrics["ann_return"],
            "fitness": float(fit_is - fit_oos),
        },
    }


# --------------------------------------------------------------------------- #
# Walk-forward rolling rebalance
# --------------------------------------------------------------------------- #
@dataclass
class WalkForwardResult:
    daily_returns: pd.Series             # stitched OOS returns, net of rebalance cost
    equity: pd.Series
    rebalances: list[dict]
    summary: dict
    weights_history: list = field(default_factory=list)


def walk_forward(bundle: DataBundle, solve_fn: Callable[[Segment], np.ndarray],
                 cfg: Config) -> WalkForwardResult:
    """Roll through walk-forward windows, re-optimizing and charging costs.

    ``solve_fn`` maps a TRAIN Segment to a weight vector over ``bundle`` tickers.
    Each window's OOS returns are charged a one-off rebalance cost (turnover *
    cost_bps) at its first day, then stitched into a single net equity curve.
    """
    tickers = list(bundle.returns.columns)
    sectors = bundle.sectors
    bps = cfg.backtest.rebalance_cost_bps

    chunks: list[pd.Series] = []
    rebalances: list[dict] = []
    weights_history: list = []
    w_prev: np.ndarray | None = None

    for tr_lo, tr_hi, te_lo, te_hi in walk_forward_windows(cfg):
        train_r = bundle.window_returns(tr_lo, tr_hi)
        test_r = bundle.window_returns(te_lo, te_hi)
        if train_r.empty or test_r.empty:
            continue

        seg = build_segment("wf_train", train_r, sectors.reindex(train_r.columns), cfg)
        w = np.asarray(solve_fn(seg), dtype=float)

        r = M.portfolio_daily_returns(w, test_r, tickers).copy()
        tno = M.turnover(w, w_prev)
        cost = M.transaction_cost(w, w_prev, bps) if cfg.backtest.charge_on_rebalance else 0.0
        if len(r) and cost:                         # charge cost at the first OOS day
            r.iloc[0] = (1.0 + r.iloc[0]) * (1.0 - cost) - 1.0

        chunks.append(r)
        weights_history.append((te_lo, w))
        rebalances.append({
            "rebalance_date": te_lo, "train": [tr_lo, tr_hi], "test": [te_lo, te_hi],
            "turnover": tno, "cost": cost, "n_holdings": M.n_holdings(w),
            "subset": [tickers[i] for i in np.where(w > 1e-6)[0]],
        })
        w_prev = w

    if not chunks:
        empty = pd.Series(dtype=float)
        return WalkForwardResult(empty, empty, [], {}, [])

    daily = pd.concat(chunks)
    daily = daily[~daily.index.duplicated(keep="first")].sort_index()
    summary = M.summary(daily, risk_free=cfg.objective.risk_free, periods=cfg.data.annualization)
    summary["total_transaction_cost"] = float(sum(rb["cost"] for rb in rebalances))
    summary["avg_turnover"] = float(np.mean([rb["turnover"] for rb in rebalances]))
    summary["n_rebalances"] = len(rebalances)
    return WalkForwardResult(daily, M.equity_curve(daily), rebalances, summary, weights_history)
