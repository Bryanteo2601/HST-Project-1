"""Offline tests for covariance and the data pipeline (no network)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.covariance import estimate_covariance  # noqa: E402
from src.data import (  # noqa: E402
    DataBundle, TICKER_SECTOR, _slice, annualized_mu, build_dataset,
    compute_returns, make_bundle, walk_forward_windows,
)


def _synthetic_prices(tickers, start="2021-06-01", n=900, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n)
    steps = rng.normal(0.0004, 0.012, size=(n, len(tickers)))
    prices = 100 * np.exp(np.cumsum(steps, axis=0))
    df = pd.DataFrame(prices, index=dates, columns=tickers)
    df.index.name = "date"
    return df


@pytest.fixture
def cfg():
    c = Config()
    c.data.start = "2021-06-01"
    c.data.end = "2024-06-30"
    c.split.train = ["2021-06-01", "2023-06-30"]
    c.split.validation = ["2023-06-30", "2023-12-31"]
    c.split.test = ["2023-12-31", "2024-06-30"]
    return c.validate()


@pytest.fixture
def bundle(cfg):
    tickers = list(TICKER_SECTOR.keys())[:8]
    prices = _synthetic_prices(tickers)
    sectors = pd.Series({t: TICKER_SECTOR[t] for t in tickers}, name="sector")
    return make_bundle(prices, sectors, cfg)


# ----- covariance ---------------------------------------------------------- #
@pytest.mark.parametrize("method", ["sample", "ledoit_wolf"])
def test_covariance_shape_and_symmetry(method):
    tickers = ["A", "B", "C", "D"]
    rets = compute_returns(_synthetic_prices(tickers, n=400))
    res = estimate_covariance(rets, method=method, annualization=252)
    assert res.sigma.shape == (4, 4)
    assert np.allclose(res.sigma.values, res.sigma.values.T)
    assert np.all(res.vol.values > 0)
    if method == "ledoit_wolf":
        assert 0.0 <= res.shrinkage <= 1.0
    else:
        assert res.shrinkage is None


def test_ledoit_wolf_is_psd():
    tickers = [f"T{i}" for i in range(12)]
    rets = compute_returns(_synthetic_prices(tickers, n=300))
    res = estimate_covariance(rets, method="ledoit_wolf")
    eig = np.linalg.eigvalsh(res.sigma.values)
    assert eig.min() > -1e-10  # positive semidefinite


def test_estimate_covariance_rejects_unknown():
    rets = compute_returns(_synthetic_prices(["A", "B"], n=50))
    with pytest.raises(ValueError):
        estimate_covariance(rets, method="bogus")


# ----- splits / bundle ----------------------------------------------------- #
def test_slices_are_disjoint_and_cover(bundle, cfg):
    tr = bundle.train.returns.index
    va = bundle.validation.returns.index
    te = bundle.test_returns().index
    assert tr.max() < pd.Timestamp(cfg.split.validation[0])
    assert va.max() < pd.Timestamp(cfg.split.test[0])
    assert len(tr.intersection(va)) == 0
    assert len(va.intersection(te)) == 0


def test_test_segment_is_gated(bundle):
    with pytest.raises(PermissionError):
        bundle.segment("test")
    # but realized test returns are reachable for evaluation
    assert not bundle.test_returns().empty


def test_segment_mu_sigma_consistent(bundle):
    seg = bundle.train
    assert seg.mu.shape[0] == len(seg.tickers)
    assert seg.sigma.shape == (len(seg.tickers), len(seg.tickers))
    # annualized mu matches manual computation
    manual = bundle.train.returns.mean() * 252
    assert np.allclose(seg.mu.values, manual.values)


def test_walk_forward_windows_are_ordered(cfg):
    wins = list(walk_forward_windows(cfg))
    assert len(wins) >= 1
    for tr_lo, tr_hi, te_lo, te_hi in wins:
        assert tr_lo < tr_hi == te_lo < te_hi


def test_build_dataset_offline(monkeypatch, cfg, tmp_path):
    """build_dataset must work fully offline when prices are already cached."""
    tickers = list(TICKER_SECTOR.keys())
    prices = _synthetic_prices(tickers)
    monkeypatch.setattr("src.data.load_prices", lambda c, t, force_refresh=False: prices[t])
    cfg.data.processed_dir = str(tmp_path / "processed")
    cfg.data.excel_path = str(tmp_path / "wb.xlsx")
    bundle = build_dataset(cfg, write_outputs=True)
    assert isinstance(bundle, DataBundle)
    assert (tmp_path / "processed" / "prices.csv").exists()
    assert (tmp_path / "processed" / "returns.csv").exists()
    assert (tmp_path / "processed" / "mu_sigma_sector.csv").exists()
    assert (tmp_path / "wb.xlsx").exists()
