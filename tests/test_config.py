"""Smoke tests for the configuration backbone."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Config, load_config  # noqa: E402


def test_default_loads_and_validates():
    cfg = load_config(ROOT / "config" / "default.yaml")
    assert cfg.constraints.K == 10
    assert cfg.covariance.estimator in {"sample", "ledoit_wolf"}
    assert cfg.objective.variant == "A"


@pytest.mark.parametrize("variant,letter", [("A", "A"), ("B", "B"), ("C", "C")])
def test_objective_overrides_merge(variant, letter):
    cfg = load_config(
        ROOT / "config" / "default.yaml",
        overrides=[ROOT / "config" / f"objective_{variant}.yaml"],
    )
    assert cfg.objective.variant == letter
    # untouched keys survive the merge
    assert cfg.constraints.K == 10
    assert cfg.seed == 42


def test_roundtrip_dict():
    cfg = load_config(ROOT / "config" / "default.yaml")
    cfg2 = Config.from_dict(cfg.to_dict()).validate()
    assert cfg2.to_dict() == cfg.to_dict()


def test_validation_rejects_overlapping_splits():
    bad = Config()
    bad.split.test = ["2023-01-01", "2024-01-01"]  # starts before validation ends
    with pytest.raises(AssertionError):
        bad.validate()


def test_validation_rejects_infeasible_cardinality():
    bad = Config()
    bad.constraints.K = 3
    bad.constraints.u_i = 0.20  # 3 * 0.20 = 0.6 < 1 -> cannot be fully invested
    with pytest.raises(AssertionError):
        bad.validate()
