"""``dutsx`` -- everything that touches real data (PLAN-integration.md).

``duts/`` is a pure function of ``List[CandidateStats]`` plus a
``(N_Q, n_Q)`` query offset; it stays untouched by this package. ``dutsx/``
holds the adapters that read CSVs, load ``.pkl`` synopses, and (in later
phases) call an ANN index or a unionability scorer -- everything behind the
``typing.Protocol`` seams in ``dutsx/ports.py``. The dependency arrow is
``dutsx -> duts``, never the reverse.
"""
