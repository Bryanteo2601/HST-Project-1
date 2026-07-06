"""Portfolio representation, objective variants, feasibility and repair.

The decision is encoded as a full-length weight vector ``w`` (length N over the
universe) whose support is the selection ``z`` (``z_i = 1`` iff ``w_i > 0``). All
optimizers maximize the SAME ``fitness`` on training data:

  Variant A (default): mu^T w - lambda w^T Sigma w - gamma sum_i c_i |w_i - w0_i|
  Variant B:           Sharpe = (mu - rf)^T w / sqrt(w^T Sigma w)
  Variant C:           mu^T w - lambda w^T Sigma w - lambda_L1 ||w||_1  (no z; see qp.py)

:class:`ProblemData` bundles everything an optimizer or the inner QP needs and is
built once per split via :meth:`ProblemData.from_segment`. Feasibility/repair here
serve the NATIVE SA/GA weight path; the inner QP (qp.py) produces feasible weights
directly for a fixed subset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from src.config import Config
from src.data import Segment

EPS = 1e-9


@dataclass
class ProblemData:
    tickers: list[str]
    mu: np.ndarray                       # (N,) annualized expected returns
    sigma: np.ndarray                    # (N, N) annualized covariance
    sectors: np.ndarray                  # (N,) GICS sector label per ticker
    sector_masks: dict[str, np.ndarray]  # sector -> boolean mask over tickers
    # constraints
    K: int
    u: np.ndarray                        # (N,) per-name upper bound
    L: float                             # per-sector cap
    fully_invested: bool
    # objective
    variant: str
    lambda_risk: float
    gamma_cost: float
    c: np.ndarray                        # (N,) per-name transaction-cost coeff
    w0: np.ndarray                       # (N,) incumbent weights for |w - w0|
    risk_free: float
    lambda_L1: float

    @property
    def n(self) -> int:
        return len(self.tickers)

    def restrict(self, idx, K: Optional[int] = None) -> "ProblemData":
        """Return a sub-instance over ticker indices ``idx`` (for universe scaling)."""
        idx = np.asarray(idx, dtype=int)
        sectors = self.sectors[idx]
        return ProblemData(
            tickers=[self.tickers[i] for i in idx], mu=self.mu[idx],
            sigma=self.sigma[np.ix_(idx, idx)], sectors=sectors,
            sector_masks={s: (sectors == s) for s in np.unique(sectors)},
            K=self.K if K is None else K, u=self.u[idx], L=self.L,
            fully_invested=self.fully_invested, variant=self.variant,
            lambda_risk=self.lambda_risk, gamma_cost=self.gamma_cost, c=self.c[idx],
            w0=np.full(len(idx), 1.0 / len(idx)), risk_free=self.risk_free,
            lambda_L1=self.lambda_L1,
        )

    @classmethod
    def from_segment(cls, seg: Segment, cfg: Config) -> "ProblemData":
        tickers = seg.tickers
        n = len(tickers)
        mu = seg.mu.reindex(tickers).to_numpy(dtype=float)
        sigma = seg.sigma.reindex(index=tickers, columns=tickers).to_numpy(dtype=float)
        sectors = seg.sectors.reindex(tickers).to_numpy()
        masks = {s: (sectors == s) for s in sorted(set(sectors))}

        if cfg.objective.w0 == "equal":
            w0 = np.full(n, 1.0 / n)
        else:  # "zero" -> start from cash
            w0 = np.zeros(n)

        return cls(
            tickers=tickers, mu=mu, sigma=sigma, sectors=sectors, sector_masks=masks,
            K=cfg.constraints.K,
            u=np.full(n, cfg.constraints.u_i),
            L=cfg.constraints.L_s,
            fully_invested=cfg.constraints.fully_invested,
            variant=cfg.objective.variant,
            lambda_risk=cfg.objective.lambda_risk,
            gamma_cost=cfg.objective.gamma_cost,
            c=np.full(n, cfg.objective.c_i),
            w0=w0,
            risk_free=cfg.objective.risk_free,
            lambda_L1=cfg.objective.lambda_L1,
        )


@dataclass
class Portfolio:
    """A candidate solution: full-length weights over the universe."""
    w: np.ndarray
    tickers: list[str]
    fitness: Optional[float] = None
    meta: dict = field(default_factory=dict)

    @property
    def selected(self) -> np.ndarray:
        return np.where(self.w > EPS)[0]

    @property
    def n_holdings(self) -> int:
        return int((self.w > EPS).sum())

    def weights_series(self):
        import pandas as pd
        return pd.Series(self.w, index=self.tickers, name="weight")


# --------------------------------------------------------------------------- #
# Fitness (the quantity every optimizer maximizes)
# --------------------------------------------------------------------------- #
def fitness(w: np.ndarray, p: ProblemData, variant: Optional[str] = None) -> float:
    variant = variant or p.variant
    ret = float(p.mu @ w)
    risk = float(w @ p.sigma @ w)
    if variant == "A":
        tc = float(p.gamma_cost * (p.c * np.abs(w - p.w0)).sum())
        return ret - p.lambda_risk * risk - tc
    if variant == "B":
        vol = np.sqrt(max(risk, 0.0))
        if vol < EPS:
            return -np.inf
        return (ret - p.risk_free) / vol
    if variant == "C":
        l1 = float(p.lambda_L1 * np.abs(w).sum())
        return ret - p.lambda_risk * risk - l1
    raise ValueError(f"unknown objective variant: {variant!r}")


# --------------------------------------------------------------------------- #
# Feasibility / constraint violations (cardinality problem; variants A & B)
# --------------------------------------------------------------------------- #
def violations(w: np.ndarray, p: ProblemData) -> dict[str, float]:
    """Magnitude of each constraint violation (0 == satisfied)."""
    v = {}
    v["budget"] = abs(float(w.sum()) - 1.0) if p.fully_invested else 0.0
    v["negativity"] = float(np.maximum(0.0, -w).sum())
    v["upper"] = float(np.maximum(0.0, w - p.u).sum())
    n_sel = int((w > EPS).sum())
    v["cardinality"] = float(max(0, n_sel - p.K))
    sec = 0.0
    for mask in p.sector_masks.values():
        sec += max(0.0, float(w[mask].sum()) - p.L)
    v["sector"] = sec
    return v


def is_feasible(w: np.ndarray, p: ProblemData, tol: float = 1e-6) -> bool:
    return all(val <= tol for val in violations(w, p).values())


def penalty(w: np.ndarray, p: ProblemData, weight: float) -> float:
    return weight * float(sum(violations(w, p).values()))


def penalized_fitness(w: np.ndarray, p: ProblemData, weight: float,
                      variant: Optional[str] = None) -> float:
    """Fitness minus a penalty for any residual infeasibility (penalty mode)."""
    return fitness(w, p, variant) - penalty(w, p, weight)


# --------------------------------------------------------------------------- #
# Repair (native SA/GA path): coerce arbitrary weights to a feasible portfolio
# --------------------------------------------------------------------------- #
def repair(w: np.ndarray, p: ProblemData, subset: Optional[Sequence[int]] = None,
           max_iter: int = 100) -> np.ndarray:
    """Project weights onto the feasible set (cardinality, box, sector, budget).

    If ``subset`` is given, the support is fixed to those indices; otherwise the
    top-K names by weight are retained. Iterates clip -> sector-scale -> renormalize
    until stable. Best-effort: residual infeasibility (rare, subset-specific) is
    left to the penalty term.
    """
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)

    if subset is not None:
        keep = np.zeros(p.n, dtype=bool)
        keep[np.asarray(subset, dtype=int)] = True
    else:
        keep = np.zeros(p.n, dtype=bool)
        order = np.argsort(w)[::-1]
        keep[order[: p.K]] = True
    w[~keep] = 0.0

    if w.sum() <= EPS:  # degenerate: seed equal weight on the kept names
        w[keep] = 1.0

    for _ in range(max_iter):
        w = np.minimum(w, p.u)                       # box upper
        for mask in p.sector_masks.values():         # sector caps
            tot = w[mask].sum()
            if tot > p.L + EPS:
                w[mask] *= p.L / tot
        s = w.sum()
        if s <= EPS:
            break
        if p.fully_invested:
            w = w / s                                # budget
        if is_feasible(w, p, tol=1e-6):
            break
    return w
