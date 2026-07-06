"""Smoke test: the orchestration layer, master table, and plotting on a tiny run."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.data import TICKER_SECTOR, make_bundle  # noqa: E402
from src.experiment import fast_config, master_table, run_suite  # noqa: E402
from src import plotting as P  # noqa: E402


def _prices(tickers, start="2019-01-01", n=1300, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n)
    df = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, (n, len(tickers))), 0)),
                      index=dates, columns=tickers)
    df.index.name = "date"
    return df


@pytest.fixture
def tiny():
    tickers = list(TICKER_SECTOR.keys())[:12]
    prices = _prices(tickers)
    sectors = pd.Series({t: TICKER_SECTOR[t] for t in tickers}, name="sector")
    cfg = Config()
    idx = prices.index
    cfg.data.start = str(idx.min().date()); cfg.data.end = str(idx.max().date())
    cfg.split.train = [str(idx.min().date()), str(idx[int(.6 * len(idx))].date())]
    cfg.split.validation = [str(idx[int(.6 * len(idx))].date()), str(idx[int(.8 * len(idx))].date())]
    cfg.split.test = [str(idx[int(.8 * len(idx))].date()), str(idx.max().date())]
    cfg.constraints.K = 5
    cfg = fast_config(cfg.validate())
    return make_bundle(prices, sectors, cfg), cfg


def test_run_suite_and_master_table(tiny):
    bundle, cfg = tiny
    suite = run_suite(bundle, cfg, methods=["ACO"])   # one method to stay fast
    assert "ACO" in suite["methods"]
    assert {"EqualWeight", "Greedy", "Random"} <= set(suite["benchmarks"])
    mt = master_table(suite)
    assert "oos_sharpe" in mt.columns and "gap_sharpe" in mt.columns
    assert mt.loc["ACO", "n_holdings"] <= cfg.constraints.K


def test_core_plots_write_files(tiny, tmp_path):
    bundle, cfg = tiny
    suite = run_suite(bundle, cfg, methods=["ACO"])
    P.convergence_curves(suite, tmp_path / "conv.png")
    P.equity_curves(suite, tmp_path / "eq.png")
    P.risk_return_scatter(suite, tmp_path / "rr.png")
    P.is_oos_gap_bars(suite, tmp_path / "gap.png")
    P.pheromone_heatmap(suite["methods"]["ACO"]["best"], suite["tickers"], tmp_path / "pher.png")
    for f in ["conv.png", "eq.png", "rr.png", "gap.png", "pher.png"]:
        assert (tmp_path / f).exists() and (tmp_path / f).stat().st_size > 0
