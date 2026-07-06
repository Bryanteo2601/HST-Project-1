"""Tests for metrics.py and the leakage-boundary backtest."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src import metrics as M  # noqa: E402
from src.backtest import evaluate_oos, in_sample_vs_oos, walk_forward  # noqa: E402
from src.data import TICKER_SECTOR, make_bundle  # noqa: E402


def _prices(tickers, start="2019-01-01", n=1400, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n)
    steps = rng.normal(0.0004, 0.012, size=(n, len(tickers)))
    df = pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=dates, columns=tickers)
    df.index.name = "date"
    return df


# ----- metrics ------------------------------------------------------------- #
def test_annualized_metrics_on_known_series():
    r = pd.Series([0.01, -0.005, 0.02, 0.0, 0.015] * 50)
    assert M.annualized_vol(r) == pytest.approx(r.std(ddof=1) * np.sqrt(252))
    assert M.sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    assert -1.0 <= M.max_drawdown(r) <= 0.0


def test_max_drawdown_detects_decline():
    r = pd.Series([0.1, 0.1, -0.5, -0.2, 0.1])   # big drop in the middle
    assert M.max_drawdown(r) < -0.5


def test_turnover_and_cost():
    w_new = np.array([0.5, 0.5, 0.0])
    w_old = np.array([0.0, 0.5, 0.5])
    assert M.turnover(w_new, w_old) == pytest.approx(1.0)
    assert M.transaction_cost(w_new, w_old, cost_bps=10) == pytest.approx(10 / 1e4 * 1.0)
    assert M.turnover(w_new, None) == pytest.approx(1.0)  # from cash


def test_sector_allocation_sums_to_weight():
    tickers = ["A", "B", "C"]
    sectors = pd.Series({"A": "Tech", "B": "Tech", "C": "Fin"})
    alloc = M.sector_allocation(np.array([0.3, 0.2, 0.5]), sectors, tickers)
    assert alloc["Tech"] == pytest.approx(0.5)
    assert alloc.sum() == pytest.approx(1.0)


# ----- backtest fixtures --------------------------------------------------- #
@pytest.fixture
def setup():
    tickers = list(TICKER_SECTOR.keys())[:10]
    prices = _prices(tickers, n=1400)
    sectors = pd.Series({t: TICKER_SECTOR[t] for t in tickers}, name="sector")
    cfg = Config()
    cfg.data.start = str(prices.index.min().date())
    cfg.data.end = str(prices.index.max().date())
    mid = prices.index[int(len(prices) * 0.6)].date()
    v = prices.index[int(len(prices) * 0.8)].date()
    cfg.split.train = [str(prices.index.min().date()), str(mid)]
    cfg.split.validation = [str(mid), str(v)]
    cfg.split.test = [str(v), str(prices.index.max().date())]
    cfg.constraints.K = 5
    cfg.validate()
    bundle = make_bundle(prices, sectors, cfg)
    return bundle, cfg, tickers


def test_evaluate_oos_runs(setup):
    bundle, cfg, tickers = setup
    w = np.zeros(len(tickers)); w[[0, 2, 4]] = [0.4, 0.3, 0.3]
    res = evaluate_oos(w, bundle.test_returns(), bundle.sectors, cfg, tickers)
    assert set(["ann_return", "ann_vol", "sharpe", "max_drawdown", "n_holdings"]).issubset(res)
    assert res["n_holdings"] == 3


def test_in_sample_vs_oos_reports_gap(setup):
    bundle, cfg, tickers = setup
    w = np.zeros(len(tickers)); w[[0, 1, 2]] = [0.34, 0.33, 0.33]
    out = in_sample_vs_oos(w, bundle.train.returns, bundle.test_returns(),
                           bundle.sectors, cfg, tickers)
    assert {"in_sample", "out_of_sample", "gap"} <= set(out)
    assert "fitness" in out["in_sample"] and "fitness" in out["gap"]
    # gap is exactly IS minus OOS
    assert out["gap"]["sharpe"] == pytest.approx(
        out["in_sample"]["sharpe"] - out["out_of_sample"]["sharpe"])


def test_walk_forward_stitches_and_charges_cost(setup):
    bundle, cfg, tickers = setup
    cfg.split.walkforward.train_months = 12
    cfg.split.walkforward.test_months = 3
    cfg.split.walkforward.step_months = 3

    def solve_fn(seg):
        # simple feasible portfolio: equal-weight the K highest-mu names
        idx = np.argsort(seg.mu.to_numpy())[::-1][: cfg.constraints.K]
        w = np.zeros(len(tickers)); w[idx] = 1.0 / cfg.constraints.K
        return w

    res = walk_forward(bundle, solve_fn, cfg)
    assert len(res.rebalances) >= 2
    assert res.rebalances[0]["turnover"] == pytest.approx(1.0)   # first rebalance from cash
    assert len(res.daily_returns) > 0
    assert res.summary["n_rebalances"] == len(res.rebalances)
    assert res.summary["total_transaction_cost"] >= 0.0
    # equity curve is consistent with the stitched returns
    assert res.equity.iloc[-1] == pytest.approx(float((1 + res.daily_returns).prod()))
