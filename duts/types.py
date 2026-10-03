"""Core types for the DUTS two-stage optimization core (PLAN.md §4).

Both stages are pure functions of ``List[CandidateStats]`` plus a
``(N_Q, n_Q)`` query offset -- no embeddings, no index, no Starmie. That is
what makes the core independently testable (PLAN.md §2, §6).
"""
from typing import Dict, List, NamedTuple, Optional


class CandidateStats(NamedTuple):
    table: str    # id
    N: int        # |{t in T_i : t.V_D in M}| -- exact integer (C2)
    n: int        # |T_i|
    U: float      # U_{V_D}(Q,T_i) -- given, not computed here


class QuerySpec(NamedTuple):
    N_Q: int              # matching tuples in the query table
    n_Q: int               # |query table|
    k: int                 # required result size
    alpha: float            # pool-expansion factor, alpha >= 1
    F_star: float            # target proportion
    delta: float              # max allowable deviation, delta >= 0
    include_query: bool = True   # C1: does Q enter the stage objectives?


class StageResult(NamedTuple):
    """Generic outcome of a stage-level optimization.

    ``objective_value`` is stage-specific: Stage 1 -> the achieved
    distributional proportion F(T(S u Q)); Stage 2 -> sum(U_i) over the
    selected subset. ``iterations`` is the Dinkelbach iteration count for
    Stage 1, and 0 for Stage 2 (a single ILP solve, not iterative).
    """
    selected: List[CandidateStats]
    feasible: bool
    objective_value: Optional[float]
    iterations: int
    info: Optional[Dict] = None


class Telemetry(NamedTuple):
    """Per-query telemetry (PLAN.md Phase 4)."""
    n_D: int
    n_P: int
    F_P: Optional[float]
    F_R: Optional[float]
    delta_R: Optional[float]
    sum_U: Optional[float]
    dinkelbach_iterations_certify: int
    dinkelbach_iterations_stage1: int
    F_max_k: Optional[float]
    certify_time_s: Optional[float]
    ilp_time_s: Optional[float]
    lp_precheck_ran: bool
    lp_feasible: Optional[bool]
    lp_time_s: Optional[float]
    effective_alpha: float
    feasible: bool
    infeasibility_cause: Optional[str]  # None | "insufficient_candidates" | "intrinsic"
                                         # | "alpha_induced" | "stage2_infeasible" (certify=False)


class PipelineResult(NamedTuple):
    selected: List[CandidateStats]   # R; empty list if infeasible
    pool: List[CandidateStats]        # P; empty list if certify or |D|<k short-circuited
    telemetry: Telemetry
