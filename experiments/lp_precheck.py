"""Experiment 2 -- LP pre-check savings (PLAN-integration.md §4 Phase E; fills
§6.1's ``Section ??`` placeholder).

Direct ILP vs. LP-relaxation-first, replayed over the ``Stage2Instance``s
harvested by ``experiments/alpha_sweep.py`` -- i.e. every ``(query, k, alpha)``
condition across the WHOLE sweep that actually reached Stage 2, not one fixed
slice. Reusing the already-scored pool means this experiment repeats none of
retrieval, Stage 1, or unionability scoring (the expensive parts); it isolates
exactly ``duts.stage2_ilp.solve``'s own cost with ``lp_precheck=True`` vs.
``False`` on the same input.

Also checks the NOTES.md/CLAUDE.md invariant that the two settings must never
disagree on feasibility for this constraint shape (LP/ILP equivalence, one
cardinality equality + one inequality) -- a violation here would be a real
regression in ``duts/stage2_ilp.py``, worth surfacing loudly rather than
averaging away.
"""
import random
import time
from typing import List

from duts import stage2_ilp
from duts.types import CandidateStats

from .alpha_sweep import Stage2Instance
from .schema import ResultRow


def run_lp_precheck_experiment(
    instances: List[Stage2Instance],
    adapters: dict,
    seed: int = 42,
    experiment: str = "lp_precheck",
    verbose: bool = True,
) -> List[ResultRow]:
    """For every harvested ``Stage2Instance``, call ``stage2_ilp.solve`` twice
    (``lp_precheck=True`` then ``False``) on the identical pool ``P``, and
    emit one ``ResultRow`` per variant (two rows per instance, distinguished
    by ``lp_precheck``). Raises ``AssertionError`` immediately if the two
    variants disagree on feasibility -- see module docstring."""
    rows: List[ResultRow] = []
    disagreements = []
    for i, inst in enumerate(instances):
        task = inst.task
        tau = task.F_star - task.delta
        variant_results = {}
        for lp_precheck in (True, False):
            t0 = time.perf_counter()
            stage2_result = stage2_ilp.solve(
                inst.P, task.k, tau, inst.N_Q, inst.n_Q, task.include_query,
                lp_precheck=lp_precheck,
            )
            wall = time.perf_counter() - t0
            variant_results[lp_precheck] = stage2_result

            info = stage2_result.info or {}
            row = ResultRow(
                experiment=experiment, config_id=inst.config_id, seed=seed,
                q_table=task.q_table, attr=str(task.attr), M_size=len(task.M),
                k=task.k, alpha=task.alpha, F_star=task.F_star, delta=task.delta,
                include_query=task.include_query, top_n=task.top_n,
                adapter_synopsis=adapters.get("synopsis", ""),
                adapter_semantic=adapters.get("semantic", ""),
                adapter_overlap=adapters.get("overlap", ""),
                adapter_unionability=adapters.get("unionability", ""),
                lp_precheck=lp_precheck,
                n_sem=None, n_ovl=None, n_pair=None,
                feasible=bool(stage2_result.feasible),
                infeasibility_cause=None if stage2_result.feasible else "stage2_infeasible",
                error=None,
                n_D=0, n_P=len(inst.P),
                n_R=len(stage2_result.selected) if stage2_result.feasible else 0,
                F_P=None, F_R=None, delta_R=None,
                sum_U=stage2_result.objective_value,
                n_unionability_computations=0,  # reused pool -- no NEW scoring in this experiment
                dinkelbach_iterations_stage1=0, retrieval_time_s=None, stage1_time_s=None,
                effective_alpha=task.alpha,
                ilp_time_s=wall,
                lp_precheck_ran=bool(info.get("lp_precheck_ran", False)),
                lp_feasible=info.get("lp_feasible"),
                lp_time_s=info.get("lp_time_s"),
                wall_time_s=wall,
            )
            rows.append(row)

        if variant_results[True].feasible != variant_results[False].feasible:
            disagreements.append(inst)

    if disagreements:
        raise AssertionError(
            "LP pre-check and direct ILP disagreed on feasibility for {} instance(s) -- "
            "this contradicts the verified LP/ILP equivalence for Stage 2's constraint shape "
            "(CLAUDE.md Invariants); first offender: q_table={!r} config_id={!r}".format(
                len(disagreements), disagreements[0].task.q_table, disagreements[0].config_id,
            )
        )

    if verbose:
        n_instances = len(instances)
        n_lp_infeasible = sum(
            1 for r in rows if r.lp_precheck and r.lp_precheck_ran and not r.lp_feasible
        )
        print("  {} Stage2 instances replayed, {} rows; "
              "LP pre-check found {} of them LP-infeasible (ILP skipped entirely)".format(
                  n_instances, len(rows), n_lp_infeasible))
    return rows


def run_lp_precheck_synthetic_demo(
    seed: int = 42,
    n_instances: int = 60,
    pool_size: int = 400,
    experiment: str = "lp_precheck_synthetic",
    verbose: bool = True,
) -> List[ResultRow]:
    """Supplementary, synthetic (NOT real-data) demonstration of LP pre-check
    savings at a pool scale where branch-and-bound has real work to do.

    **Why this exists.** The real santos benchmark's retrieved pools top out
    around ~10 candidates (PLAN-integration.md Phase B/D numbers) -- far too
    small for HiGHS's branch-and-bound to cost anything measurable, so
    ``run_lp_precheck_experiment`` on real-data instances (see
    ``experiments/results/lp_precheck.csv``) found ZERO Stage-2-infeasible
    pools among the whole alpha sweep and therefore cannot demonstrate a
    timing difference -- an honest finding about this dataset's operating
    range, not a bug (see this module's caller in ``cli.py`` and the report
    this experiment feeds). To still give §6.1's ``Section ??`` a real
    number, this function builds ``pool_size``-candidate synthetic pools
    (an order of magnitude above anything santos retrieves) with a mix of
    feasible and genuinely infeasible instances -- constructed the same way
    ``tests/test_stage2.py``'s property tests do (random ``N``/``n``/``U``
    plus a tight/loose ``tau``) -- and times ``stage2_ilp.solve`` both ways
    on each. Labelled a distinct ``experiment`` name and never merged into
    the real-data CSVs, so a reader cannot mistake it for a santos number.
    """
    rng = random.Random(seed)
    rows: List[ResultRow] = []
    k = max(3, pool_size // 20)
    for i in range(n_instances):
        pool = [
            CandidateStats(
                table="s{}".format(j), N=rng.randint(0, 20), n=rng.randint(1, 20),
                U=rng.random() * 10,
            )
            for j in range(pool_size)
        ]
        # Alternate feasible (tau trivially satisfiable) and infeasible
        # (tau set above any achievable ratio) instances, so both code paths
        # -- "LP proves infeasible, ILP skipped" and "LP passes, ILP runs" --
        # are actually exercised, not just one of them.
        tau = -100.0 if i % 2 == 0 else 1e6
        for lp_precheck in (True, False):
            t0 = time.perf_counter()
            result = stage2_ilp.solve(pool, k, tau, 0, 0, False, lp_precheck=lp_precheck)
            wall = time.perf_counter() - t0
            info = result.info or {}
            rows.append(ResultRow(
                experiment=experiment, config_id="pool_size={}_k={}".format(pool_size, k),
                seed=seed, q_table="synthetic_{}".format(i), attr="-1", M_size=0,
                k=k, alpha=float(pool_size) / k, F_star=0.0, delta=0.0,
                include_query=False, top_n=0,
                adapter_synopsis="synthetic", adapter_semantic="synthetic",
                adapter_overlap="synthetic", adapter_unionability="synthetic",
                lp_precheck=lp_precheck, n_sem=None, n_ovl=None, n_pair=None,
                feasible=bool(result.feasible),
                infeasibility_cause=None if result.feasible else "stage2_infeasible",
                error=None, n_D=0, n_P=pool_size,
                n_R=len(result.selected) if result.feasible else 0,
                F_P=None, F_R=None, delta_R=None, sum_U=result.objective_value,
                n_unionability_computations=0, dinkelbach_iterations_stage1=0,
                retrieval_time_s=None, stage1_time_s=None,
                effective_alpha=float(pool_size) / k,
                ilp_time_s=wall,
                lp_precheck_ran=bool(info.get("lp_precheck_ran", False)),
                lp_feasible=info.get("lp_feasible"), lp_time_s=info.get("lp_time_s"),
                wall_time_s=wall,
            ))

    if verbose:
        infeasible_rows = [r for r in rows if not r.feasible]
        if infeasible_rows:
            mean_true = sum(r.ilp_time_s for r in infeasible_rows if r.lp_precheck) / (len(infeasible_rows) // 2)
            mean_false = sum(r.ilp_time_s for r in infeasible_rows if not r.lp_precheck) / (len(infeasible_rows) // 2)
            print("  synthetic infeasible instances (pool_size={}, k={}): "
                  "mean stage2 time with LP pre-check={:.5f}s, without={:.5f}s".format(
                      pool_size, k, mean_true, mean_false))
    return rows
