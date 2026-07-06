"""Tests for heuristic baselines and the exact MIQP reference."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.model import ProblemData, is_feasible  # noqa: E402
from src.benchmarks import (  # noqa: E402
    equal_weight, exact_miqp, gap_to_optimal, greedy_ratio, make_small_problem,
    random_search,
)


def make_problem(n=8, K=4, seed=0, variant="A") -> ProblemData:
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.05, 0.35, size=n)
    A = rng.normal(size=(n, n))
    sigma = (A @ A.T) * 0.02 + np.eye(n) * 0.05
    sectors = np.array(["Tech", "Tech", "Fin", "Fin", "Energy", "Energy", "Health", "Health"][:n])
    masks = {s: (sectors == s) for s in np.unique(sectors)}
    return ProblemData(
        tickers=[f"S{i}" for i in range(n)], mu=mu, sigma=sigma, sectors=sectors,
        sector_masks=masks, K=K, u=np.full(n, 0.5), L=0.7, fully_invested=True,
        variant=variant, lambda_risk=3.0, gamma_cost=0.001, c=np.full(n, 0.001),
        w0=np.full(n, 1.0 / n), risk_free=0.0, lambda_L1=0.01,
    )


@pytest.fixture
def cfg():
    return Config()


def test_equal_weight(cfg):
    p = make_problem()
    port = equal_weight(p)
    assert port.w.sum() == pytest.approx(1.0)
    assert np.allclose(port.w, 1.0 / p.n)
    assert np.isfinite(port.fitness)


def test_greedy_ratio_feasible(cfg):
    p = make_problem()
    port = greedy_ratio(p, cfg)
    assert is_feasible(port.w, p)
    assert (port.w > 1e-9).sum() <= p.K


def test_random_search_returns_best_and_samples(cfg):
    p = make_problem()
    port = random_search(p, cfg, n_samples=50, seed=1)
    assert is_feasible(port.w, p)
    assert len(port.meta["samples"]) == 50
    assert port.fitness >= max(f for _, f in port.meta["samples"]) - 1e-9


def test_make_small_problem_shapes():
    p = make_problem(n=8, K=4)
    sm = make_small_problem(p, n=5, K=3)
    assert sm.n == 5 and sm.K == 3
    assert sm.sigma.shape == (5, 5)


# ----- exact MIQP ---------------------------------------------------------- #
def test_exact_miqp_is_optimal_and_feasible(cfg):
    p = make_problem(n=8, K=4)
    res = exact_miqp(p, cfg)
    if not res.available:
        pytest.skip(res.message)
    assert is_feasible(res.portfolio.w, p)
    assert (res.portfolio.w > 1e-9).sum() <= p.K
    # exhaustive random search with QP weights cannot beat the MIQP optimum
    best_random = random_search(p, cfg, n_samples=200, seed=0)
    assert best_random.fitness <= res.fitness + 1e-4
    # and on this tiny instance it should essentially match it
    assert gap_to_optimal(best_random.fitness, res.fitness) < 1e-3


def test_exact_miqp_rejects_non_A_variant(cfg):
    p = make_problem(variant="B")
    res = exact_miqp(p, cfg)
    assert not res.available
    assert "variant A" in res.message


def test_graceful_when_no_miqp_solver(cfg, monkeypatch):
    p = make_problem()
    monkeypatch.setattr("src.benchmarks._miqp_solver", lambda c: None)
    res = exact_miqp(p, cfg)
    assert not res.available
    assert "No MIQP solver" in res.message


def test_gap_to_optimal_sign():
    assert gap_to_optimal(1.0, 1.0) == pytest.approx(0.0)
    assert gap_to_optimal(0.5, 1.0) == pytest.approx(0.5)
    assert gap_to_optimal(1.0, 0.5) < 0  # heuristic beat reference -> negative gap
