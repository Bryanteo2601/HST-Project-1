"""Shared inner QP: optimal weights for a FIXED subset (cvxpy).

This single convex solver is used by ACO (always) and optionally by SA/GA, so the
three methods become apples-to-apples -- they differ only in how they search the
discrete subset space, never in how weights are assigned.

- ``solve_weights(subset, p, cfg)``  : variants A and B for a fixed subset S.
    * Variant A linearizes the transaction term |w_i - w0_i| via a BUY/SELL split
      (w = w0 + buy - sell, buy,sell >= 0), keeping the problem a convex QP.
    * Variant B (Sharpe) uses the Schaible homogenization (variables y, kappa) so
      box and sector caps stay linear; weights recovered as w = y / kappa.
- ``solve_l1_portfolio(p, cfg)``      : variant C, full universe, shorting allowed,
    lambda_L1 * ||w||_1 penalty inducing sparsity (no cardinality / no z).

Reported fitness is always recomputed with ``model.fitness`` on the full-length
weight vector, guaranteeing the QP path and the native path use identical
accounting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cvxpy as cp
import numpy as np

from src.config import Config
from src.model import ProblemData, fitness

_PREFERENCE = ["CLARABEL", "OSQP", "ECOS", "SCS"]


@dataclass
class QPResult:
    w: np.ndarray            # full-length weights over the universe (zeros off-subset)
    fitness: float
    status: str
    feasible: bool
    solver: str


def _solver_chain(cfg: Config) -> list[str]:
    installed = set(cp.installed_solvers())
    chain = [s for s in _PREFERENCE if s in installed]
    pref = cfg.inner_qp.solver
    if pref and pref != "auto" and pref in installed:
        chain = [pref] + [s for s in chain if s != pref]
    return chain or list(installed)


def _solve(prob: cp.Problem, cfg: Config) -> tuple[str, str]:
    """Try solvers in order; return (status, solver_name). Never raises on failure."""
    last_status = "no_solver"
    for s in _solver_chain(cfg):
        try:
            prob.solve(solver=s, verbose=False)
        except (cp.error.SolverError, Exception):
            continue
        if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return prob.status, s
        last_status = prob.status or last_status
    return last_status, ""


def _sector_constraints(weight_expr, p: ProblemData, idx: np.ndarray, rhs):
    """sum of weights within each sector (restricted to idx) <= rhs."""
    cons = []
    sub_sectors = p.sectors[idx]
    for s in np.unique(sub_sectors):
        mask = sub_sectors == s
        if mask.any():
            cons.append(cp.sum(weight_expr[mask]) <= rhs)
    return cons


# --------------------------------------------------------------------------- #
# Variants A & B: fixed subset
# --------------------------------------------------------------------------- #
def solve_weights(subset: Sequence[int], p: ProblemData, cfg: Config,
                  variant: Optional[str] = None) -> QPResult:
    variant = variant or p.variant
    idx = np.asarray(sorted(set(int(i) for i in subset)), dtype=int)
    if idx.size == 0:
        return QPResult(np.zeros(p.n), -np.inf, "empty_subset", False, "")
    if idx.size > p.K:
        raise ValueError(f"subset size {idx.size} exceeds cardinality K={p.K}")

    mu_s = p.mu[idx]
    Sig_s = cp.psd_wrap(p.sigma[np.ix_(idx, idx)])
    u_s = p.u[idx]

    if variant == "A":
        return _solve_A(idx, mu_s, Sig_s, u_s, p, cfg)
    if variant == "B":
        return _solve_B(idx, mu_s, Sig_s, u_s, p, cfg)
    raise ValueError(f"solve_weights supports variants A/B, got {variant!r}")


def _solve_A(idx, mu_s, Sig_s, u_s, p: ProblemData, cfg: Config) -> QPResult:
    m = idx.size
    w0_s = p.w0[idx]
    c_s = p.c[idx]
    buy = cp.Variable(m, nonneg=True)
    sell = cp.Variable(m, nonneg=True)
    w = w0_s + buy - sell                       # buy/sell split

    obj = (mu_s @ w
           - p.lambda_risk * cp.quad_form(w, Sig_s)
           - p.gamma_cost * (c_s @ (buy + sell)))
    cons = [w >= 0, w <= u_s]
    if p.fully_invested:
        cons.append(cp.sum(w) == 1)
    cons += _sector_constraints(w, p, idx, p.L)

    prob = cp.Problem(cp.Maximize(obj), cons)
    status, solver = _solve(prob, cfg)
    return _finalize(idx, w, p, cfg, status, solver, "A")


def _solve_B(idx, mu_s, Sig_s, u_s, p: ProblemData, cfg: Config) -> QPResult:
    m = idx.size
    mu_excess = mu_s - p.risk_free
    if np.max(mu_excess) <= 0:                  # cannot normalize numerator > 0
        return QPResult(np.zeros(p.n), -np.inf, "no_positive_excess", False, "")

    y = cp.Variable(m, nonneg=True)
    kappa = cp.Variable(nonneg=True)
    cons = [mu_excess @ y == 1, cp.sum(y) == kappa, y <= cp.multiply(u_s, kappa)]
    cons += _sector_constraints(y, p, idx, p.L * kappa)

    prob = cp.Problem(cp.Minimize(cp.quad_form(y, Sig_s)), cons)
    status, solver = _solve(prob, cfg)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or kappa.value is None or kappa.value < 1e-9:
        return QPResult(np.zeros(p.n), -np.inf, status, False, solver)

    w_full = np.zeros(p.n)
    w_full[idx] = np.asarray(y.value).ravel() / float(kappa.value)
    fit = fitness(w_full, p, "B")
    return QPResult(w_full, fit, status, True, solver)


# --------------------------------------------------------------------------- #
# Variant C: L1-regularized sparse portfolio (no cardinality, shorting allowed)
# --------------------------------------------------------------------------- #
def solve_l1_portfolio(p: ProblemData, cfg: Config,
                       lambda_L1: Optional[float] = None) -> QPResult:
    lam1 = p.lambda_L1 if lambda_L1 is None else lambda_L1
    w = cp.Variable(p.n)
    obj = (p.mu @ w
           - p.lambda_risk * cp.quad_form(w, cp.psd_wrap(p.sigma))
           - lam1 * cp.norm1(w))
    cons = [cp.sum(w) == 1, w <= p.u, w >= -p.u]   # shorting within [-u, u] makes L1 bite
    for s, mask in p.sector_masks.items():
        cons.append(cp.sum(w[mask]) <= p.L)

    prob = cp.Problem(cp.Maximize(obj), cons)
    status, solver = _solve(prob, cfg)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or w.value is None:
        return QPResult(np.zeros(p.n), -np.inf, status, False, solver)

    w_full = np.asarray(w.value).ravel()
    w_full[np.abs(w_full) < 1e-5] = 0.0
    fit = fitness(w_full, p, "C")
    return QPResult(w_full, fit, status, True, solver)


def effective_cardinality(w: np.ndarray, tol: float = 1e-4) -> int:
    return int((np.abs(w) > tol).sum())


# --------------------------------------------------------------------------- #
def _finalize(idx, w_expr, p: ProblemData, cfg: Config, status, solver, variant) -> QPResult:
    if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or w_expr.value is None:
        return QPResult(np.zeros(p.n), -np.inf, status, False, solver)
    w_full = np.zeros(p.n)
    w_full[idx] = np.clip(np.asarray(w_expr.value).ravel(), 0.0, None)
    fit = fitness(w_full, p, variant)
    return QPResult(w_full, fit, status, True, solver)
