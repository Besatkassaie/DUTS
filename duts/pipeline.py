"""The §4.1 driver: [certify (C4) ->] Stage 1 -> Stage 2, with telemetry.

Degenerate cardinalities (PLAN.md C5):
  |D| < k             -> empty result, no stages run ("insufficient_candidates").
  k <= |D| < alpha*k  -> P is clamped to D (effective_alpha = |D|/k logged);
                         Stage 1 is skipped since P == D already.

The C4 certificate (`certify` param, default False for now) is a diagnostic,
not a requirement -- Stage 2 is exact either way. With it off, a Stage 2
failure is reported as "stage2_infeasible" rather than the more specific
"intrinsic"/"alpha_induced" split, since without running certify_feasible
first we genuinely don't know which of the two it is.
"""
import time
from typing import Dict, List

from . import stage1_dinkelbach, stage2_ilp
from .stats import F as F_ratio
from .stats import aggregate, delta as delta_of, drop_empty
from .types import CandidateStats, PipelineResult, QuerySpec, Telemetry


def _telemetry(**overrides) -> Telemetry:
    """Build a Telemetry with sensible defaults, overridden by whatever the
    caller actually computed at that point in the pipeline."""
    defaults: Dict = dict(
        n_D=0, n_P=0, F_P=None, F_R=None, delta_R=None, sum_U=None,
        dinkelbach_iterations_certify=0, dinkelbach_iterations_stage1=0,
        F_max_k=None, certify_time_s=None, ilp_time_s=None,
        lp_precheck_ran=False, lp_feasible=None, lp_time_s=None,
        effective_alpha=None, feasible=False, infeasibility_cause=None,
    )
    defaults.update(overrides)
    return Telemetry(**defaults)


def run(D: List[CandidateStats], query: QuerySpec, certify: bool = False) -> PipelineResult:
    """[certify (C4) ->] Stage 1 (S5) -> Stage 2 (S6.1), over `D`.

    `certify=False` (default, for now) skips the C4 feasibility certificate
    entirely -- no extra Dinkelbach run at cardinality k over all of D. Stage
    1 and Stage 2 still run and are still exact; the only thing lost is the
    intrinsic-vs-alpha_induced diagnostic on a `∅` result (see module
    docstring). Pass `certify=True` to restore the full PLAN.md §4.1 driver.
    """
    tau = query.F_star - query.delta
    D = drop_empty(D)
    N_Q, n_Q, k, alpha = query.N_Q, query.n_Q, query.k, query.alpha
    include_query = query.include_query

    if len(D) < k:
        return PipelineResult(selected=[], pool=[], telemetry=_telemetry(
            n_D=len(D), effective_alpha=alpha, infeasibility_cause="insufficient_candidates",
        ))

    f_max_k, n_iter_cert, certify_time_s = None, 0, None
    if certify:
        t0 = time.perf_counter()
        feasible_cert, f_max_k, n_iter_cert = stage1_dinkelbach.certify_feasible(
            D, k, tau, N_Q, n_Q, include_query
        )
        certify_time_s = time.perf_counter() - t0

        if not feasible_cert:
            return PipelineResult(selected=[], pool=[], telemetry=_telemetry(
                n_D=len(D), F_max_k=f_max_k, dinkelbach_iterations_certify=n_iter_cert,
                certify_time_s=certify_time_s, effective_alpha=alpha,
                infeasibility_cause="intrinsic",
            ))

    pool_size = int(alpha * k)
    effective_alpha = alpha
    n_iter_stage1 = 0
    if pool_size >= len(D):
        # C5: |D| < alpha*k -- clamp P to D rather than padding; Stage 1 is a
        # no-op since the "top-(alpha*k)" selection over D is just D itself.
        P = D
        effective_alpha = len(D) / k
        F_P = F_ratio(*aggregate(P, N_Q, n_Q, include_query))
    else:
        stage1_result = stage1_dinkelbach.solve(D, pool_size, N_Q, n_Q, include_query)
        P = stage1_result.selected
        F_P = stage1_result.objective_value
        n_iter_stage1 = stage1_result.iterations

    t1 = time.perf_counter()
    stage2_result = stage2_ilp.solve(P, k, tau, N_Q, n_Q, include_query)
    ilp_time_s = time.perf_counter() - t1
    info = stage2_result.info or {}

    if not stage2_result.feasible:
        # With certify=True and the certificate having already confirmed a
        # feasible k-subset exists somewhere in D, a Stage 2 failure on P is
        # necessarily alpha_induced. With certify=False we never ran that
        # check, so we can't tell alpha_induced from intrinsic -- report
        # stage2_infeasible instead of guessing.
        cause = "alpha_induced" if certify else "stage2_infeasible"
        return PipelineResult(selected=[], pool=P, telemetry=_telemetry(
            n_D=len(D), n_P=len(P), F_P=F_P, F_max_k=f_max_k,
            dinkelbach_iterations_certify=n_iter_cert,
            dinkelbach_iterations_stage1=n_iter_stage1,
            certify_time_s=certify_time_s, ilp_time_s=ilp_time_s,
            lp_precheck_ran=info.get("lp_precheck_ran", False),
            lp_feasible=info.get("lp_feasible"), lp_time_s=info.get("lp_time_s"),
            effective_alpha=effective_alpha, infeasibility_cause=cause,
        ))

    R = stage2_result.selected
    F_R = F_ratio(*aggregate(R, N_Q, n_Q, include_query))
    telemetry = _telemetry(
        n_D=len(D), n_P=len(P), F_P=F_P, F_R=F_R,
        delta_R=delta_of(query.F_star, F_R), sum_U=stage2_result.objective_value,
        F_max_k=f_max_k, dinkelbach_iterations_certify=n_iter_cert,
        dinkelbach_iterations_stage1=n_iter_stage1,
        certify_time_s=certify_time_s, ilp_time_s=ilp_time_s,
        lp_precheck_ran=info.get("lp_precheck_ran", False),
        lp_feasible=info.get("lp_feasible"), lp_time_s=info.get("lp_time_s"),
        effective_alpha=effective_alpha, feasible=True, infeasibility_cause=None,
    )
    return PipelineResult(selected=R, pool=P, telemetry=telemetry)
