"""Environment assertions and config validation (PLAN.md Phase 0)."""
import json
import sys
from typing import Dict

MIN_PYTHON = (3, 8)
MIN_SCIPY = (1, 9)


def assert_environment() -> None:
    """Fail fast if the interpreter or scipy don't meet the plan's stated
    requirements, rather than surfacing as a cryptic error inside milp()."""
    if sys.version_info[:2] < MIN_PYTHON:
        raise RuntimeError(
            f"DUTS requires Python >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}, "
            f"got {sys.version.split()[0]}"
        )
    import scipy

    scipy_version = tuple(int(p) for p in scipy.__version__.split(".")[:2])
    if scipy_version < MIN_SCIPY:
        raise RuntimeError(
            f"DUTS requires scipy >= {MIN_SCIPY[0]}.{MIN_SCIPY[1]} "
            f"(for scipy.optimize.milp), got {scipy.__version__}"
        )
    from scipy.optimize import milp  # noqa: F401


def validate_config(cfg: Dict) -> Dict:
    """Validate a loaded config dict in place; returns it with defaults filled.

    Catches two failure modes early rather than as an empty result later
    (PLAN.md Phase 0): tau = F_star - delta outside [0,1] makes Stage 2's
    constraint either vacuous (tau < 0) or unsatisfiable (tau > 1).
    """
    F_star = cfg["F_star"]
    delta = cfg["delta"]
    k = cfg["k"]
    alpha = cfg["alpha"]

    if not (0.0 <= F_star <= 1.0):
        raise ValueError(f"F_star must be in [0,1], got {F_star}")
    if delta < 0:
        raise ValueError(f"delta must be >= 0, got {delta}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if alpha < 1:
        raise ValueError(f"alpha must be >= 1, got {alpha}")

    tau = F_star - delta
    if tau < 0.0:
        raise ValueError(
            f"tau = F_star - delta = {tau} < 0: Stage 2's distributional "
            f"constraint is vacuous (every subset satisfies it)"
        )
    if tau > 1.0:
        raise ValueError(
            f"tau = F_star - delta = {tau} > 1: Stage 2's distributional "
            f"constraint is unsatisfiable (no subset can satisfy it)"
        )

    cfg.setdefault("include_query", True)
    cfg.setdefault("seed", 42)
    cfg.setdefault("objective", "max_f")
    if cfg["objective"] not in ("max_f", "satisfice"):
        raise ValueError(
            f"objective must be 'max_f' or 'satisfice', got {cfg['objective']!r}"
        )
    if cfg["objective"] == "satisfice":
        # PLAN.md C3: "diversity subject to F >= tau" has no defined objective
        # function yet. Fail loudly rather than silently falling back to max_f.
        raise NotImplementedError(
            "objective='satisfice' has no defined objective function yet "
            "(PLAN.md C3 open question) and is not implemented in this build"
        )
    return cfg


def load_config(path: str) -> Dict:
    with open(path) as f:
        cfg = json.load(f)
    return validate_config(cfg)
