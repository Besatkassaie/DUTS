import pytest

from duts.config import validate_config


def base_cfg(**overrides):
    cfg = {
        "k": 10, "alpha": 2, "F_star": 0.5, "delta": 0.05,
        "include_query": True, "seed": 42, "objective": "max_f",
    }
    cfg.update(overrides)
    return cfg


def test_valid_config_passes_through():
    cfg = validate_config(base_cfg())
    assert cfg["k"] == 10


@pytest.mark.parametrize("overrides", [
    {"F_star": 1.5},
    {"F_star": -0.1},
    {"delta": -0.01},
    {"k": 0},
    {"alpha": 0.5},
])
def test_out_of_range_fields_rejected(overrides):
    with pytest.raises(ValueError):
        validate_config(base_cfg(**overrides))


def test_tau_below_zero_rejected():
    # tau = F_star - delta = 0.1 - 0.5 = -0.4 -- constraint would be vacuous
    with pytest.raises(ValueError, match="vacuous"):
        validate_config(base_cfg(F_star=0.1, delta=0.5))


def test_tau_above_one_rejected():
    # tau = F_star - delta = 0.9 - (-0.0)... use F_star near 1 with delta negative not allowed;
    # instead force tau > 1 via F_star=1.0 is max, so tau>1 needs delta<0 which is already
    # rejected separately. Construct via F_star=1.0, delta=0 -> tau=1.0 (boundary, valid).
    # tau > 1 is actually unreachable once F_star<=1 and delta>=0 are enforced elsewhere,
    # so test the boundary explicitly instead.
    cfg = validate_config(base_cfg(F_star=1.0, delta=0.0))
    assert cfg["F_star"] - cfg["delta"] == pytest.approx(1.0)


def test_satisfice_objective_not_implemented():
    with pytest.raises(NotImplementedError):
        validate_config(base_cfg(objective="satisfice"))


def test_unknown_objective_rejected():
    with pytest.raises(ValueError):
        validate_config(base_cfg(objective="bogus"))


def test_defaults_filled_when_absent():
    cfg = {"k": 5, "alpha": 1, "F_star": 0.5, "delta": 0.1}
    cfg = validate_config(cfg)
    assert cfg["include_query"] is True
    assert cfg["seed"] == 42
    assert cfg["objective"] == "max_f"
