"""Tests for fitness/feasibility/repair (model.py) and the inner QP (qp.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.model import (  # noqa: E402
    ProblemData, fitness, is_feasible, penalized_fitness, repair, violations,
)
from src.qp import (  # noqa: E402
    effective_cardinality, solve_l1_portfolio, solve_weights,
)


def make_problem(variant="A", n=6, seed=0, K=3) -> ProblemData:
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.05, 0.30, size=n)
    A = rng.normal(size=(n, n))
    sigma = (A @ A.T) * 0.02 + np.eye(n) * 0.05
    sectors = np.array(["Tech", "Tech", "Fin", "Fin", "Energy", "Energy"][:n])
    masks = {s: (sectors == s) for s in np.unique(sectors)}
    return ProblemData(
        tickers=[f"S{i}" for i in range(n)], mu=mu, sigma=sigma, sectors=sectors,
        sector_masks=masks, K=K, u=np.full(n, 0.5), L=0.7, fully_invested=True,
        variant=variant, lambda_risk=5.0, gamma_cost=0.001, c=np.full(n, 0.001),
        w0=np.full(n, 1.0 / n), risk_free=0.0, lambda_L1=0.01,
    )


@pytest.fixture
def cfg():
    return Config()


# ----- fitness ------------------------------------------------------------- #
def test_fitness_variant_A_matches_formula():
    p = make_problem("A")
    w = np.zeros(p.n); w[[0, 1, 2]] = [0.4, 0.3, 0.3]
    expected = (p.mu @ w - p.lambda_risk * (w @ p.sigma @ w)
                - p.gamma_cost * (p.c * np.abs(w - p.w0)).sum())
    assert fitness(w, p, "A") == pytest.approx(expected)


def test_fitness_variant_B_is_sharpe():
    p = make_problem("B")
    w = np.zeros(p.n); w[[0, 1]] = [0.5, 0.5]
    vol = np.sqrt(w @ p.sigma @ w)
    assert fitness(w, p, "B") == pytest.approx((p.mu @ w) / vol)


def test_fitness_zero_vol_guard():
    p = make_problem("B")
    assert fitness(np.zeros(p.n), p, "B") == -np.inf


# ----- feasibility / repair ------------------------------------------------ #
def test_repair_produces_feasible_portfolio():
    p = make_problem("A")
    rng = np.random.default_rng(1)
    raw = rng.uniform(-0.5, 1.0, size=p.n)   # infeasible: negatives, wrong sum, too many names
    w = repair(raw, p)
    assert is_feasible(w, p)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-9).all() and (w <= p.u + 1e-9).all()
    assert (w > 1e-9).sum() <= p.K


def test_repair_respects_fixed_subset():
    p = make_problem("A")
    w = repair(np.ones(p.n), p, subset=[1, 3, 4])
    assert set(np.where(w > 1e-9)[0]).issubset({1, 3, 4})
    assert is_feasible(w, p)


def test_penalty_zero_when_feasible():
    p = make_problem("A")
    w = repair(np.ones(p.n), p, subset=[0, 1, 2])
    assert sum(violations(w, p).values()) == pytest.approx(0.0, abs=1e-6)
    assert penalized_fitness(w, p, 100.0) == pytest.approx(fitness(w, p))


# ----- inner QP: variant A ------------------------------------------------- #
def test_qp_A_feasible_and_optimal(cfg):
    p = make_problem("A")
    subset = [0, 2, 4]
    res = solve_weights(subset, p, cfg, "A")
    assert res.feasible
    assert is_feasible(res.w, p)
    assert set(np.where(res.w > 1e-6)[0]).issubset(set(subset))
    # QP must do at least as well as a repaired equal-weight on the same subset
    eq = repair(np.where(np.isin(np.arange(p.n), subset), 1.0, 0.0), p, subset=subset)
    assert res.fitness >= fitness(eq, p, "A") - 1e-6
    # reported fitness equals recomputed fitness on the returned vector
    assert res.fitness == pytest.approx(fitness(res.w, p, "A"), abs=1e-6)


def test_qp_A_rejects_oversized_subset(cfg):
    p = make_problem("A", K=3)
    with pytest.raises(ValueError):
        solve_weights([0, 1, 2, 3], p, cfg, "A")


# ----- inner QP: variant B (Sharpe) ---------------------------------------- #
def test_qp_B_feasible_and_beats_equal_weight(cfg):
    p = make_problem("B")
    subset = [0, 1, 2]
    res = solve_weights(subset, p, cfg, "B")
    assert res.feasible
    assert is_feasible(res.w, p)
    eq = repair(np.where(np.isin(np.arange(p.n), subset), 1.0, 0.0), p, subset=subset)
    assert res.fitness >= fitness(eq, p, "B") - 1e-6


# ----- inner QP: variant C (L1 sparsity) ----------------------------------- #
def test_l1_more_penalty_means_more_sparsity(cfg):
    p = make_problem("C", n=6)
    lo = solve_l1_portfolio(p, cfg, lambda_L1=0.0)
    hi = solve_l1_portfolio(p, cfg, lambda_L1=0.5)
    assert lo.feasible and hi.feasible
    assert lo.w.sum() == pytest.approx(1.0, abs=1e-4)
    assert effective_cardinality(hi.w) <= effective_cardinality(lo.w)
