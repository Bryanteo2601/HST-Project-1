"""Data pipeline: pull, cache, clean, returns, mu/Sigma, splits, exports.

Central object: :class:`DataBundle`. It exposes ``train`` and ``validation`` as
:class:`Segment` objects (each carrying cleaned realized returns plus split-local
``mu``/``Sigma`` estimates) for optimization and hyperparameter tuning. The TEST
window is deliberately NOT given an optimization Segment: it is reachable only as
raw realized returns via :meth:`DataBundle.test_returns`, consumed by
``src/backtest.py``. Because no ``mu``/``Sigma`` is ever estimated on test, there
is simply nothing for an optimizer to fit there -- the leakage rule is enforced by
construction, not by discipline.

Exports for the course deliverable:
  - parquet cache of raw prices            (data/raw/)        -> offline reruns
  - cleaned CSVs: prices, returns, mu/sigma/sector (data/processed/)
  - Excel workbook with the same sheets    (data/portfolio_data.xlsx)
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from src.config import Config
from src.covariance import CovarianceResult, estimate_covariance


# --------------------------------------------------------------------------- #
# Default universe: ~50 liquid S&P 500 large-caps + hardcoded GICS sectors.
# --------------------------------------------------------------------------- #
TICKER_SECTOR: dict[str, str] = {
    # Information Technology
    "AAPL": "Information Technology", "MSFT": "Information Technology",
    "NVDA": "Information Technology", "AVGO": "Information Technology",
    "ORCL": "Information Technology", "CRM": "Information Technology",
    "ADBE": "Information Technology", "CSCO": "Information Technology",
    "ACN": "Information Technology", "AMD": "Information Technology",
    # Communication Services
    "GOOGL": "Communication Services", "META": "Communication Services",
    "NFLX": "Communication Services", "DIS": "Communication Services",
    "CMCSA": "Communication Services", "T": "Communication Services",
    "VZ": "Communication Services",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "LOW": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary",
    # Consumer Staples
    "PG": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples",
    "COST": "Consumer Staples", "WMT": "Consumer Staples",
    # Health Care
    "UNH": "Health Care", "JNJ": "Health Care", "LLY": "Health Care",
    "MRK": "Health Care", "ABBV": "Health Care", "PFE": "Health Care",
    "TMO": "Health Care",
    # Financials
    "BRK-B": "Financials", "JPM": "Financials", "V": "Financials",
    "MA": "Financials", "BAC": "Financials", "WFC": "Financials",
    # Energy
    "XOM": "Energy", "CVX": "Energy",
    # Industrials
    "CAT": "Industrials", "BA": "Industrials", "HON": "Industrials",
    "GE": "Industrials",
    # Utilities / Materials
    "NEE": "Utilities", "LIN": "Materials",
}
DEFAULT_TICKERS: list[str] = list(TICKER_SECTOR.keys())


# --------------------------------------------------------------------------- #
# Segment / DataBundle
# --------------------------------------------------------------------------- #
@dataclass
class Segment:
    """A split-local view used for optimization or tuning (train / validation)."""
    name: str
    returns: pd.DataFrame          # cleaned daily simple returns (dates x tickers)
    mu: pd.Series                  # annualized expected returns, indexed by ticker
    cov: CovarianceResult          # annualized covariance + metadata
    sectors: pd.Series             # ticker -> GICS sector

    @property
    def tickers(self) -> list[str]:
        return list(self.returns.columns)

    @property
    def sigma(self) -> pd.DataFrame:
        return self.cov.sigma


@dataclass
class DataBundle:
    """Full cleaned dataset plus leakage-guarded split access."""
    cfg: Config
    prices: pd.DataFrame           # cleaned adjusted prices (dates x tickers)
    returns: pd.DataFrame          # cleaned daily simple returns (full history)
    sectors: pd.Series             # ticker -> GICS sector
    _segments: dict[str, Segment] = field(default_factory=dict)

    # ----- optimization-facing access (train / validation only) ------------- #
    def segment(self, name: str) -> Segment:
        if name == "test":
            raise PermissionError(
                "test window has no optimization Segment by design; evaluate frozen "
                "portfolios via src/backtest.py using DataBundle.test_returns()"
            )
        if name not in self._segments:
            raise KeyError(f"unknown segment {name!r}; expected 'train' or 'validation'")
        return self._segments[name]

    @property
    def train(self) -> Segment:
        return self.segment("train")

    @property
    def validation(self) -> Segment:
        return self.segment("validation")

    # ----- evaluation-only access (the single test door) -------------------- #
    def test_returns(self) -> pd.DataFrame:
        """Realized TEST-window returns for OOS evaluation. No mu/Sigma here."""
        lo, hi = self.cfg.split.test
        return _slice(self.returns, lo, hi)

    def window_returns(self, start: str, end: str) -> pd.DataFrame:
        """Arbitrary [start, end) realized-return slice (used by walk-forward)."""
        return _slice(self.returns, start, end)


# --------------------------------------------------------------------------- #
# Pure helpers (network-free, unit-testable)
# --------------------------------------------------------------------------- #
def _slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Half-open [start, end) slice so adjacent splits never share a row."""
    idx = pd.to_datetime(df.index)
    mask = (idx >= pd.Timestamp(start)) & (idx < pd.Timestamp(end))
    return df.loc[mask]


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns from adjusted prices; drops the first NaN row."""
    return prices.pct_change().dropna(how="all")


def annualized_mu(returns: pd.DataFrame, annualization: int = 252) -> pd.Series:
    """Annualized expected return = mean daily simple return * annualization."""
    return returns.mean() * annualization


def clean_prices(prices: pd.DataFrame, max_nan_frac: float = 0.05) -> pd.DataFrame:
    """Drop tickers with too many gaps, forward/back fill the rest, drop dead rows."""
    prices = prices.sort_index()
    keep = prices.columns[prices.isna().mean() <= max_nan_frac]
    prices = prices[keep].ffill().bfill().dropna(axis=0, how="any")
    return prices


def build_segment(name: str, returns: pd.DataFrame, sectors: pd.Series, cfg: Config) -> Segment:
    mu = annualized_mu(returns, cfg.data.annualization)
    cov = estimate_covariance(returns, cfg.covariance.estimator, cfg.data.annualization)
    return Segment(name=name, returns=returns, mu=mu, cov=cov, sectors=sectors)


def make_bundle(prices: pd.DataFrame, sectors: pd.Series, cfg: Config) -> DataBundle:
    """Assemble a DataBundle from cleaned prices (no network)."""
    returns = compute_returns(prices)
    segs: dict[str, Segment] = {}
    for name in ("train", "validation"):
        lo, hi = getattr(cfg.split, name)
        r = _slice(returns, lo, hi)
        if r.empty:
            raise ValueError(f"{name} split [{lo}, {hi}) has no return rows")
        segs[name] = build_segment(name, r, sectors, cfg)
    return DataBundle(cfg=cfg, prices=prices, returns=returns, sectors=sectors, _segments=segs)


def walk_forward_windows(cfg: Config) -> Iterator[tuple[str, str, str, str]]:
    """Yield (train_start, train_end, test_start, test_end) for rolling rebalance."""
    wf = cfg.split.walkforward
    start = pd.Timestamp(cfg.data.start)
    end = pd.Timestamp(cfg.data.end)
    tr_lo = start
    while True:
        tr_hi = tr_lo + pd.DateOffset(months=wf.train_months)
        te_hi = tr_hi + pd.DateOffset(months=wf.test_months)
        if te_hi > end:
            break
        yield (tr_lo.strftime("%Y-%m-%d"), tr_hi.strftime("%Y-%m-%d"),
               tr_hi.strftime("%Y-%m-%d"), te_hi.strftime("%Y-%m-%d"))
        tr_lo = tr_lo + pd.DateOffset(months=wf.step_months)


# --------------------------------------------------------------------------- #
# Network / IO: pull + cache + exports
# --------------------------------------------------------------------------- #
def _cache_path(cfg: Config, tickers: list[str]) -> Path:
    key = "|".join([*sorted(tickers), cfg.data.start, cfg.data.end, cfg.data.price_field])
    h = hashlib.sha1(key.encode()).hexdigest()[:10]
    return Path(cfg.data.cache_dir) / f"prices_{cfg.data.start}_{cfg.data.end}_{h}.parquet"


def _extract_field(raw: pd.DataFrame, field_name: str, tickers: list[str]) -> pd.DataFrame:
    """Pull the requested price field out of a yfinance frame, robust to layout."""
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0)
        chosen = field_name if field_name in set(lvl0) else ("Close" if "Close" in set(lvl0) else lvl0[0])
        out = raw[chosen].copy()
    else:  # single ticker -> flat columns
        col = field_name if field_name in raw.columns else "Close"
        out = raw[[col]].copy()
        out.columns = tickers[:1]
    return out.reindex(columns=[t for t in tickers if t in out.columns])


def load_prices(cfg: Config, tickers: list[str], force_refresh: bool = False) -> pd.DataFrame:
    """Load cleaned adjusted prices, using the parquet cache when available."""
    cache = _cache_path(cfg, tickers)
    if cache.exists() and not force_refresh:
        return pd.read_parquet(cache)

    import yfinance as yf  # imported lazily so offline/cached runs need no network

    raw = yf.download(
        tickers, start=cfg.data.start, end=cfg.data.end,
        auto_adjust=False, progress=False, group_by="column",
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data; check connectivity/tickers")

    prices = _extract_field(raw, cfg.data.price_field, tickers)
    prices = clean_prices(prices)
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"

    cache.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(cache)
    return prices


# --------------------------------------------------------------------------- #
# Scalable S&P 500 universe (top-N by liquidity)
# --------------------------------------------------------------------------- #
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def sp500_constituents(cfg: Config) -> pd.Series:
    """Ticker -> GICS sector for current S&P 500 members (Wikipedia, cached).

    Falls back to the built-in 50-name dict if the fetch fails (offline runs).
    """
    cache = Path(cfg.data.cache_dir) / "sp500_constituents.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        return pd.Series(df["sector"].values, index=df["ticker"].values, name="sector")
    try:
        import io
        import requests  # bundled with yfinance; uses certifi so macOS SSL works
        html = requests.get(SP500_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
        tables = pd.read_html(io.StringIO(html))
        t = tables[0]
        sym = t["Symbol"].astype(str).str.replace(".", "-", regex=False)
        s = pd.Series(t["GICS Sector"].values, index=sym.values, name="sector")
        s = s[~s.index.duplicated()]
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"ticker": s.index, "sector": s.values}).to_csv(cache, index=False)
        return s
    except Exception as e:  # pragma: no cover - network dependent
        print(f"[data] S&P500 fetch failed ({e}); falling back to built-in 50-name universe")
        return pd.Series(TICKER_SECTOR, name="sector")


def load_panel(cfg: Config, tickers: list[str], force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Adjusted close + volume for ``tickers`` (cached), used for liquidity ranking."""
    key = "|".join([*sorted(tickers), cfg.data.start, cfg.data.end])
    h = hashlib.sha1(key.encode()).hexdigest()[:10]
    base = Path(cfg.data.cache_dir) / f"panel_{cfg.data.start}_{cfg.data.end}_{h}"
    pf, vf = base.with_suffix(".price.parquet"), base.with_suffix(".vol.parquet")
    if pf.exists() and vf.exists() and not force_refresh:
        return {"price": pd.read_parquet(pf), "volume": pd.read_parquet(vf)}

    import yfinance as yf

    raw = yf.download(tickers, start=cfg.data.start, end=cfg.data.end,
                      auto_adjust=False, progress=False, group_by="column")
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data for the S&P500 panel")
    price = _extract_field(raw, cfg.data.price_field, tickers)
    volume = _extract_field(raw, "Volume", tickers)
    price.index = pd.to_datetime(price.index); price.index.name = "date"
    volume.index = pd.to_datetime(volume.index); volume.index.name = "date"
    base.parent.mkdir(parents=True, exist_ok=True)
    price.to_parquet(pf); volume.to_parquet(vf)
    return {"price": price, "volume": volume}


def build_universe(cfg: Config, n: int, force_refresh: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Top-``n`` most-liquid S&P 500 names: returns (clean prices, sectors)."""
    const = sp500_constituents(cfg)
    panel = load_panel(cfg, list(const.index), force_refresh=force_refresh)
    price = clean_prices(panel["price"])                       # keep names with full history
    vol = panel["volume"].reindex(columns=price.columns).ffill().bfill()
    dollar_vol = (price * vol).median().sort_values(ascending=False)
    top = list(dollar_vol.index[:n])
    return price[top], const.reindex(top).fillna("Unknown")


def export_csvs(bundle: DataBundle, cfg: Config) -> dict[str, Path]:
    """Write cleaned prices, returns, and mu/sigma/sector tables to CSV."""
    out = Path(cfg.data.processed_dir)
    out.mkdir(parents=True, exist_ok=True)

    full_cov = estimate_covariance(bundle.returns, cfg.covariance.estimator, cfg.data.annualization)
    mu = annualized_mu(bundle.returns, cfg.data.annualization)
    summary = pd.DataFrame({
        "mu_annual": mu,
        "vol_annual": full_cov.vol,
        "sector": bundle.sectors.reindex(mu.index),
    })

    paths = {
        "prices": out / "prices.csv",
        "returns": out / "returns.csv",
        "mu_sigma_sector": out / "mu_sigma_sector.csv",
        "covariance": out / "covariance.csv",
    }
    bundle.prices.to_csv(paths["prices"])
    bundle.returns.to_csv(paths["returns"])
    summary.to_csv(paths["mu_sigma_sector"], index_label="ticker")
    full_cov.sigma.to_csv(paths["covariance"], index_label="ticker")
    return paths


def export_excel(bundle: DataBundle, cfg: Config) -> Path:
    """Write the required Excel workbook (prices, returns, mu/sigma/sector, covariance)."""
    path = Path(cfg.data.excel_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    full_cov = estimate_covariance(bundle.returns, cfg.covariance.estimator, cfg.data.annualization)
    mu = annualized_mu(bundle.returns, cfg.data.annualization)
    summary = pd.DataFrame({
        "mu_annual": mu,
        "vol_annual": full_cov.vol,
        "sector": bundle.sectors.reindex(mu.index),
    })
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        bundle.prices.to_excel(xl, sheet_name="prices")
        bundle.returns.to_excel(xl, sheet_name="returns")
        summary.to_excel(xl, sheet_name="mu_sigma_sector", index_label="ticker")
        full_cov.sigma.to_excel(xl, sheet_name="covariance", index_label="ticker")
    return path


def build_dataset(cfg: Config, force_refresh: bool = False, write_outputs: bool = True) -> DataBundle:
    """End-to-end: resolve universe -> load/cache prices -> bundle -> exports."""
    if cfg.data.tickers:
        tickers = list(cfg.data.tickers)
        sectors = pd.Series({t: TICKER_SECTOR.get(t, "Unknown") for t in tickers}, name="sector")
        prices = load_prices(cfg, tickers, force_refresh=force_refresh)
    elif cfg.data.universe_size:
        prices, sectors = build_universe(cfg, cfg.data.universe_size, force_refresh=force_refresh)
    else:
        tickers = list(DEFAULT_TICKERS)
        sectors = pd.Series(TICKER_SECTOR, name="sector")
        prices = load_prices(cfg, tickers, force_refresh=force_refresh)

    sectors = sectors.reindex(prices.columns).fillna("Unknown")  # align to surviving tickers
    bundle = make_bundle(prices, sectors, cfg)

    if write_outputs:
        export_csvs(bundle, cfg)
        export_excel(bundle, cfg)
    return bundle
