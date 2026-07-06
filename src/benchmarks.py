"""Baseline and reference portfolios.

Heuristic baselines (each returns a :class:`Portfolio`):
  - ``equal_weight``      : classic 1/N over the whole universe (naive diversification).
  - ``random_search``     : sample many K-subsets, weight via the shared inner QP.
  - ``greedy_ratio``      : top-K by return/risk (mu/sigma), weighted via the inner QP.

Exact reference:
  - ``exact_miqp``        : cardinality-constrained MIQP (variant A) for SMALL
    instances, giving the optimal fitness for gap-to-optimal. Prefers Gurobi
    (academic license) and falls back to a cvxpy MIQP backend (SCIP); if no MIQP
    solver is installed it degrades gracefully with a clear message.

Use ``make_small_problem`` to carve a small ProblemData (e.g. N=15, K=5) for the
exact reference.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cvxpy as cp
import numpy as np

from src.config import Config
from src.model import Portfolio, ProblemData, fitness, repair
from src.qp import solve_weights

_MIQP_PREFERENCE = ["GUROBI", "CPLEX", "MOSEK", "SCIP", "XPRESS"]


# --------------------------------------------------------------------------- #
# Heuristic baselines
# --------------------------------------------------------------------------- #
def equal_weight(p: ProblemData) -> Portfolio:
    """Naive 1/N over the entire universe (a reference; ignores cardinality K)."""
    w = np.full(p.n, 1.0 / p.n)
    return Portfolio(w=w, tickers=p.tickers, fitness=fitness(w, p),
                     meta={"name": "equal_weight", "cardinality_respected": p.n <= p.K})


def _weight_subset(subset, p: ProblemData, cfg: Config, use_qp: bool) -> tuple[np.ndarray, float]:
    if use_qp:
        res = solve_weights(sorted(subset), p, cfg)
        return res.w, res.fitness
    w = repair(np.isin(np.arange(p.n), list(subset)).astype(float), p, subset=list(subset))
    return w, fitness(w, p)


def greedy_ratio(p: ProblemData, cfg: Config, use_qp: bool = True) -> Portfolio:
    """Pick K names by mu/sigma, seeding enough distinct sectors to stay feasible.

    A pure top-K-by-ratio pick can be sector-concentrated; with a sector cap L the
    subset then cannot be fully invested. We first take the best-ratio name from
    each new sector until ceil(1/L) sectors are represented, then fill the rest by
    global ratio -- guaranteeing the subset admits a feasible fully-invested QP
    whenever the instance itself does.
    """
    import math

    sigma_i = np.sqrt(np.clip(np.diag(p.sigma), 1e-12, None))
    order = np.argsort(p.mu / sigma_i)[::-1].tolist()
    # Each distinct-sector name contributes min(u, L) = u of investable budget, so
    # full investment needs ceil(1/u) distinct sectors (tighter than ceil(1/L)).
    u_min = float(np.min(p.u))
    need_sectors = min(math.ceil(1.0 / u_min), p.K)

    selected: list[int] = []
    seen: set = set()
    for i in order:                       # pass 1: diversify across sectors
        if len(seen) >= need_sectors or len(selected) >= p.K:
            break
        if p.sectors[i] not in seen:
            selected.append(i); seen.add(p.sectors[i])
    for i in order:                       # pass 2: fill remaining by ratio
        if len(selected) >= p.K:
            break
        if i not in selected:
            selected.append(i)

    subset = sorted(selected)
    w, f = _weight_subset(subset, p, cfg, use_qp)
    return Portfolio(w=w, tickers=p.tickers, fitness=f,
                     meta={"name": "greedy_ratio", "subset": subset})


def random_search(p: ProblemData, cfg: Config, n_samples: int = 200,
                  seed: int | None = None, use_qp: bool = True) -> Portfolio:
    """Sample ``n_samples`` random K-subsets; return the best. Stores the full
    fitness distribution (for the risk-return scatter) in ``meta``."""
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    best, samples = None, []
    for _ in range(n_samples):
        subset = sorted(rng.choice(p.n, size=min(p.K, p.n), replace=False).tolist())
        w, f = _weight_subset(subset, p, cfg, use_qp)
        samples.append((subset, f))
        if best is None or f > best.fitness:
            best = Portfolio(w=w, tickers=p.tickers, fitness=f,
                             meta={"name": "random_search", "subset": subset})
    best.meta["samples"] = samples
    best.meta["n_samples"] = n_samples
    return best


# --------------------------------------------------------------------------- #
# Exact MIQP reference (variant A, small instances)
# --------------------------------------------------------------------------- #
@dataclass
class MIQPResult:
    portfolio: Optional[Portfolio]
    fitness: float
    solver: str
    status: str
    runtime_s: float
    available: bool
    message: str = ""
    proven_optimal: bool = True   # False when a time limit left only an incumbent


def _miqp_solver(cfg: Config) -> Optional[str]:
    installed = set(cp.installed_solvers())
    pref = cfg.inner_qp.solver
    if pref and pref != "auto" and pref in installed and pref in _MIQP_PREFERENCE:
        return pref
    for s in _MIQP_PREFERENCE:
        if s in installed:
            return s
    return None


def make_small_problem(p: ProblemData, n: int = 15, K: int = 5,
                       seed: int = 0, top_by_ratio: bool = True) -> ProblemData:
    """Carve a small sub-instance for the exact reference (default N=15, K=5)."""
    if top_by_ratio:
        sigma_i = np.sqrt(np.clip(np.diag(p.sigma), 1e-12, None))
        idx = np.argsort(p.mu / sigma_i)[::-1][:n]
    else:
        idx = np.random.default_rng(seed).choice(p.n, size=n, replace=False)
    idx = np.sort(idx)
    sectors = p.sectors[idx]
    return ProblemData(
        tickers=[p.tickers[i] for i in idx], mu=p.mu[idx], sigma=p.sigma[np.ix_(idx, idx)],
        sectors=sectors, sector_masks={s: (sectors == s) for s in np.unique(sectors)},
        K=K, u=p.u[idx], L=p.L, fully_invested=p.fully_invested, variant=p.variant,
        lambda_risk=p.lambda_risk, gamma_cost=p.gamma_cost, c=p.c[idx],
        w0=np.full(len(idx), 1.0 / len(idx)), risk_free=p.risk_free, lambda_L1=p.lambda_L1,
    )


def exact_miqp(p: ProblemData, cfg: Config, time_limit: int = 120) -> MIQPResult:
    """Solve the cardinality-constrained MIQP exactly (variant A only)."""
    if p.variant != "A":
        return MIQPResult(None, -np.inf, "", "unsupported_variant", 0.0, False,
                          message=f"exact MIQP is defined for variant A, not {p.variant}")
    solver = _miqp_solver(cfg)
    if solver is None:
        return MIQPResult(None, -np.inf, "", "no_miqp_solver", 0.0, False,
                          message="No MIQP solver installed (tried Gurobi/CPLEX/MOSEK/SCIP/"
                                  "XPRESS). Install gurobipy or pyscipopt for gap-to-optimal.")

    n = p.n
    w = cp.Variable(n)
    z = cp.Variable(n, boolean=True)
    buy = cp.Variable(n, nonneg=True)
    sell = cp.Variable(n, nonneg=True)

    obj = (p.mu @ w
           - p.lambda_risk * cp.quad_form(w, cp.psd_wrap(p.sigma))
           - p.gamma_cost * (p.c @ (buy + sell)))
    cons = [w == p.w0 + buy - sell, w >= 0, w <= cp.multiply(p.u, z), cp.sum(z) <= p.K]
    if p.fully_invested:
        cons.append(cp.sum(w) == 1)
    for mask in p.sector_masks.values():
        cons.append(cp.sum(w[mask]) <= p.L)

    prob = cp.Problem(cp.Maximize(obj), cons)
    kw = {"TimeLimit": time_limit} if solver == "GUROBI" else {}
    t0 = time.time()
    try:
        prob.solve(solver=solver, **kw)
    except Exception as e:  # pragma: no cover - solver-environment dependent
        return MIQPResult(None, -np.inf, solver, "solver_error", time.time() - t0, False,
                          message=str(e))
    dt = time.time() - t0

    # Keep an incumbent even if a time limit prevented a proven-optimal certificate.
    if w.value is None:
        return MIQPResult(None, -np.inf, solver, prob.status or "no_solution", dt, False,
                          message=f"solver returned status {prob.status}")
    proven = prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    wv = np.clip(np.asarray(w.value).ravel(), 0.0, None)
    port = Portfolio(w=wv, tickers=p.tickers, fitness=fitness(wv, p),
                     meta={"name": "exact_miqp", "solver": solver, "status": prob.status})
    msg = "" if proven else f"time limit hit ({time_limit}s); incumbent only, not proven optimal"
    return MIQPResult(port, float(fitness(wv, p)), solver, prob.status, dt, True,
                      message=msg, proven_optimal=proven)


def gap_to_optimal(heuristic_fitness: float, optimal_fitness: float) -> float:
    """Relative optimality gap in [0, inf); 0 == matched the optimum."""
    denom = abs(optimal_fitness) if abs(optimal_fitness) > 1e-12 else 1.0
    return (optimal_fitness - heuristic_fitness) / denom
