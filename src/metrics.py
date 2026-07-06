"""Performance metrics, computed identically in-sample and out-of-sample.

Everything is derived from a daily portfolio-return series (fixed weights, daily
rebalanced to target -- the standard simplification). Provides annualized return
(geometric CAGR), annualized volatility, annualized Sharpe (sqrt(252) scaling,
configurable r_f), maximum drawdown, turnover, transaction cost, holdings count,
and sector allocation. ``summary`` bundles them for the master results table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9


def portfolio_daily_returns(w: np.ndarray, asset_returns: pd.DataFrame,
                            tickers: list[str]) -> pd.Series:
    """Daily portfolio returns r_p,t = sum_i w_i r_i,t over the given window."""
    R = asset_returns.reindex(columns=tickers).to_numpy(dtype=float)
    return pd.Series(R @ np.asarray(w, dtype=float), index=asset_returns.index, name="port")


def equity_curve(r: pd.Series) -> pd.Series:
    return (1.0 + r).cumprod()


def annualized_return(r: pd.Series, periods: int = 252) -> float:
    """Geometric CAGR from the realized daily returns."""
    n = len(r)
    if n == 0:
        return 0.0
    total = float((1.0 + r).prod())
    if total <= 0:
        return -1.0
    return total ** (periods / n) - 1.0


def annualized_vol(r: pd.Series, periods: int = 252) -> float:
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=1) * np.sqrt(periods))


def sharpe(r: pd.Series, risk_free: float = 0.0, periods: int = 252) -> float:
    """Annualized Sharpe with sqrt(periods) scaling; r_f is an annual rate."""
    if len(r) < 2:
        return 0.0
    excess = r - risk_free / periods
    sd = excess.std(ddof=1)
    if sd < EPS:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods))


def max_drawdown(r: pd.Series) -> float:
    """Most negative peak-to-trough drawdown (<= 0)."""
    if len(r) == 0:
        return 0.0
    eq = equity_curve(r)
    return float((eq / eq.cummax() - 1.0).min())


def turnover(w_new: np.ndarray, w_old: np.ndarray | None) -> float:
    """One-way-ish total turnover sum_i |w_new_i - w_old_i| (w_old=None -> from cash)."""
    w_new = np.asarray(w_new, dtype=float)
    w_old = np.zeros_like(w_new) if w_old is None else np.asarray(w_old, dtype=float)
    return float(np.abs(w_new - w_old).sum())


def transaction_cost(w_new: np.ndarray, w_old: np.ndarray | None, cost_bps: float) -> float:
    return cost_bps / 1e4 * turnover(w_new, w_old)


def n_holdings(w: np.ndarray, tol: float = EPS) -> int:
    return int((np.abs(np.asarray(w)) > tol).sum())


def sector_allocation(w: np.ndarray, sectors: pd.Series, tickers: list[str]) -> pd.Series:
    s = pd.Series(np.asarray(w, dtype=float), index=tickers)
    return s.groupby(sectors.reindex(tickers)).sum().sort_values(ascending=False)


def summary(r: pd.Series, *, w: np.ndarray | None = None, w_prev: np.ndarray | None = None,
            sectors: pd.Series | None = None, tickers: list[str] | None = None,
            risk_free: float = 0.0, cost_bps: float = 0.0, periods: int = 252) -> dict:
    """One-stop metric bundle for a return series (+ optional weight context)."""
    out = {
        "ann_return": annualized_return(r, periods),
        "ann_vol": annualized_vol(r, periods),
        "sharpe": sharpe(r, risk_free, periods),
        "max_drawdown": max_drawdown(r),
        "n_periods": int(len(r)),
    }
    if w is not None:
        out["n_holdings"] = n_holdings(w)
        out["turnover"] = turnover(w, w_prev)
        out["transaction_cost"] = transaction_cost(w, w_prev, cost_bps)
        if sectors is not None and tickers is not None:
            out["sector_allocation"] = sector_allocation(w, sectors, tickers).to_dict()
    return out
