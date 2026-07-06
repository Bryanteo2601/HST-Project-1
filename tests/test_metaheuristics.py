"""Tests for SA / GA / ACO: feasibility, convergence, determinism, apples-to-apples."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.model import ProblemData, is_feasible  # noqa: E402
from src.metaheuristics.sa import SimulatedAnnealing  # noqa: E402
from src.metaheuristics.ga import GeneticAlgorithm  # noqa: E402
from src.metaheuristics.aco import AntColony  # noqa: E402
from src.metaheuristics.tabu import TabuSearch  # noqa: E402


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


def small_cfg(**inner) -> Config:
    c = Config()
    c.sa.t_init = 0.5; c.sa.t_min = 0.05; c.sa.cooling = 0.8; c.sa.iters_per_temp = 15
    c.ga.pop = 20; c.ga.generations = 15
    c.aco.ants = 10; c.aco.iters = 15
    c.tabu.iters = 60; c.tabu.tenure = 4; c.tabu.neighbors = 50; c.tabu.patience = 30
    c.inner_qp.use_for_sa = inner.get("sa", False)
    c.inner_qp.use_for_ga = inner.get("ga", True)
    return c.validate()


# ----- SA ------------------------------------------------------------------ #
@pytest.mark.parametrize("use_qp", [False, True])
def test_sa_returns_feasible_and_improves(use_qp):
    p = make_problem()
    cfg = small_cfg(sa=use_qp)
    res = SimulatedAnnealing(p, cfg, seed=1).optimize()
    assert is_feasible(res.best_w, p)
    assert res.best_w.sum() == pytest.approx(1.0, abs=1e-6)
    assert (res.best_w > 1e-9).sum() <= p.K
    assert res.history[-1] >= res.history[0] - 1e-9      # best-so-far is non-decreasing
    assert all(res.history[i] <= res.history[i + 1] + 1e-12 for i in range(len(res.history) - 1))


def test_sa_is_deterministic_given_seed():
    p = make_problem()
    cfg = small_cfg(sa=True)
    a = SimulatedAnnealing(p, cfg, seed=7).optimize()
    b = SimulatedAnnealing(p, cfg, seed=7).optimize()
    assert a.best_fitness == pytest.approx(b.best_fitness)
    assert a.best_subset == b.best_subset


# ----- GA ------------------------------------------------------------------ #
def test_ga_returns_feasible_and_monotone_best():
    p = make_problem()
    res = GeneticAlgorithm(p, small_cfg(ga=True), seed=2).optimize()
    assert is_feasible(res.best_w, p)
    assert (res.best_w > 1e-9).sum() <= p.K
    assert all(res.history[i] <= res.history[i + 1] + 1e-12 for i in range(len(res.history) - 1))


def test_ga_elitism_never_loses_best():
    p = make_problem()
    res = GeneticAlgorithm(p, small_cfg(ga=True), seed=3).optimize()
    assert res.best_fitness >= max(res.curve) - 1e-9


# ----- ACO ----------------------------------------------------------------- #
def test_aco_feasible_and_pheromone_artifacts():
    p = make_problem()
    res = AntColony(p, small_cfg(), seed=4).optimize()
    assert is_feasible(res.best_w, p)
    assert len(res.best_subset) <= p.K
    tau = res.meta["pheromone"]
    assert tau.shape == (p.n,)
    assert tau.min() >= small_cfg().aco.tau_min - 1e-9
    assert tau.max() <= small_cfg().aco.tau_max + 1e-9
    assert res.meta["pheromone_history"].shape[0] == small_cfg().aco.iters
    assert res.meta["selection_counts"].sum() > 0


def test_aco_concentrates_pheromone_on_winners():
    """The globally-best subset should end with above-average pheromone."""
    p = make_problem()
    res = AntColony(p, small_cfg(), seed=5).optimize()
    tau = res.meta["pheromone"]
    assert tau[res.best_subset].mean() >= tau.mean()


# ----- Tabu ---------------------------------------------------------------- #
def test_tabu_returns_feasible_and_monotone_best():
    p = make_problem()
    res = TabuSearch(p, small_cfg(), seed=1).optimize()
    assert is_feasible(res.best_w, p)
    assert (res.best_w > 1e-9).sum() <= p.K
    assert all(res.history[i] <= res.history[i + 1] + 1e-12 for i in range(len(res.history) - 1))


def test_tabu_is_deterministic_given_seed():
    p = make_problem()
    a = TabuSearch(p, small_cfg(), seed=5).optimize()
    b = TabuSearch(p, small_cfg(), seed=5).optimize()
    assert a.best_fitness == pytest.approx(b.best_fitness)
    assert a.best_subset == b.best_subset


def test_tabu_finds_global_optimum_small():
    """On a small instance Tabu must reach the exhaustively-verified global optimum."""
    from itertools import combinations
    from src.metaheuristics.base import SubsetEvaluator
    p = make_problem(n=8, K=3, seed=1)
    cfg = small_cfg()
    ev = SubsetEvaluator(p, cfg, use_qp=True)
    opt = max(ev.eval(c)[1] for c in combinations(range(p.n), p.K))
    res = TabuSearch(p, cfg, seed=0).optimize()
    assert res.best_fitness >= opt - 1e-4


# ----- apples-to-apples ---------------------------------------------------- #
def test_all_methods_run_in_qp_mode():
    p = make_problem()
    cfg = small_cfg(sa=True, ga=True)
    for Opt in (SimulatedAnnealing, TabuSearch, GeneticAlgorithm, AntColony):
        res = Opt(p, cfg, seed=0).optimize()
        assert is_feasible(res.best_w, p)
        assert np.isfinite(res.best_fitness)
