"""Phase E: the experiment harness (PLAN-integration.md §4 Phase E).

Sweeps ``dutsx.runner.run_query`` over the real santos benchmark and writes a
tidy one-row-per-``(query, config)`` results table. See ``experiments/cli.py``
for the reproducible entry point and ``experiments/schema.py`` for the column
contract every driver in this package writes.

Depends on ``dutsx/`` (never on ``duts/`` directly, except for the handful of
``duts.stats``/``duts.stage2_ilp`` calls the LP pre-check driver needs to
reuse an already-scored pool -- see ``experiments/lp_precheck.py``).
"""
