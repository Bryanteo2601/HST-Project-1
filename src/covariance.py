"""Covariance estimators behind a single flag.

``estimate_covariance(returns, method, annualization)`` returns an annualized
covariance matrix for ``method in {"sample", "ledoit_wolf"}``. Ledoit-Wolf uses
``sklearn.covariance.LedoitWolf`` and is the project's main overfitting-mitigation
lever: running the whole pipeline with each method is the shrinkage experiment.

Estimation is done on *daily* simple returns; the result is annualized by the
``annualization`` factor (trading days per year). Tickers are preserved on the
returned DataFrame's index/columns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


@dataclass
class CovarianceResult:
    """Annualized covariance plus metadata for reporting."""
    sigma: pd.DataFrame          # annualized covariance matrix (tickers x tickers)
    method: str                  # "sample" | "ledoit_wolf"
    shrinkage: Optional[float]   # LW shrinkage intensity in [0, 1]; None for sample

    @property
    def vol(self) -> pd.Series:
        """Annualized volatility (sqrt of the diagonal), indexed by ticker."""
        return pd.Series(np.sqrt(np.diag(self.sigma.values)), index=self.sigma.index, name="vol")


def estimate_covariance(
    returns: pd.DataFrame,
    method: str = "ledoit_wolf",
    annualization: int = 252,
) -> CovarianceResult:
    """Estimate an annualized covariance matrix from daily simple returns.

    Parameters
    ----------
    returns : DataFrame
        Daily simple returns, rows = dates, columns = tickers.
    method : {"sample", "ledoit_wolf"}
    annualization : int
        Trading days per year used to scale the daily covariance.
    """
    if method not in {"sample", "ledoit_wolf"}:
        raise ValueError(f"unknown covariance estimator: {method!r}")
    if returns.isnull().values.any():
        raise ValueError("returns contain NaNs; clean before estimating covariance")

    tickers = list(returns.columns)
    X = returns.to_numpy(dtype=float)

    if method == "sample":
        # ddof=1 sample covariance, then annualize.
        daily = np.cov(X, rowvar=False, ddof=1)
        shrinkage: Optional[float] = None
    else:
        lw = LedoitWolf().fit(X)          # LedoitWolf centers internally
        daily = lw.covariance_
        shrinkage = float(lw.shrinkage_)

    sigma = pd.DataFrame(daily * annualization, index=tickers, columns=tickers)
    # enforce exact symmetry (guards against tiny numerical asymmetry downstream)
    sigma = (sigma + sigma.T) / 2.0
    return CovarianceResult(sigma=sigma, method=method, shrinkage=shrinkage)
