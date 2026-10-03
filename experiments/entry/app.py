"""``ExperimentApp``: the single entry point behind ``python main.py``.

    collect configuration  ->  validate values  ->  check prerequisites
        -> (something missing: explain, collect build parameters, build, ask for a rerun)
        -> run the system  ->  results summary  ->  execution environment

Exit codes: 0 success; 2 invalid or missing configuration; 3 prerequisites were built (rerun
the printed command); 4 a prerequisite is missing and was not built (input data the program
cannot create, or building was declined); 130 interrupted.
"""
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

from . import config, prereqs, report, systems
from .config import ConfigError, Prompter

EXIT_OK, EXIT_CONFIG, EXIT_BUILT, EXIT_MISSING, EXIT_INTERRUPTED = 0, 2, 3, 4, 130


class ExperimentApp(object):

    def __init__(self, argv: Optional[List[str]] = None, stdin=None, interactive: Optional[bool] = None):
        self.args = config.build_parser().parse_args(argv)
        stream = stdin or sys.stdin
        if interactive is None:
            interactive = not self.args.non_interactive and stream.isatty()
        self.prompter = Prompter(interactive, stream_in=stream)

    # -- top level -------------------------------------------------------------------------

    def run(self) -> int:
        try:
            return self._run()
        except KeyboardInterrupt:
            print("\ninterrupted")
            return EXIT_INTERRUPTED

    def _run(self) -> int:
        print(report.RULE)
        print("DUTS experiment runner")
        print(report.RULE)
        try:
            if self.prompter.interactive:
                print("Enter a value, or press Enter to accept the [suggestion].")
            values = config.collect(self.args, self.prompter,
                                    use_defaults=self.args.defaults or not self.prompter.interactive)
        except ConfigError as e:
            print("\nconfiguration error: {}".format(e))
            print("run `python main.py --help` for every option")
            return EXIT_CONFIG
        errors = config.validate(values)
        if errors:
            print("\ninvalid configuration:")
            for e in errors:
                print("  - " + e)
            return EXIT_CONFIG
        self.values = values
        self._print_config(values)

        hnsw_m = prereqs.find_index_m(values, self.args.hnsw_m) if values["system"] == "duts" else None
        reqs = prereqs.check(values, hnsw_m or 32)
        self._print_checklist(reqs)
        fatal = [r for r in reqs if not r.ok and r.fatal]
        if fatal:
            print("\nCannot run: the following inputs are missing or invalid and cannot be generated:")
            for r in fatal:
                print("  - {}: {}\n      fix: {}".format(r.label, r.detail, r.how_to_fix))
            return EXIT_MISSING
        missing = [r for r in reqs if not r.ok]
        if missing:
            return self._build_missing(missing, values)

        env = report.collect_environment(values)
        print("\nrunning {} on {} ...".format(values["system"], values["dataset"]))
        t0 = time.perf_counter()
        try:
            out = systems.run(values, hnsw_m)
        except prereqs.StaleIndexError as e:
            print("\n{}".format(e))
            stale = prereqs.Requirement("duts_index", "DUTS HNSW index", False, str(e), False)
            return self._build_missing([stale], values)
        total_s = time.perf_counter() - t0
        if not out.outcomes:
            print("\nno queries were evaluated: none of the rows in {} has both a query table and an "
                  "embedding".format(values["protected_csv"]))
            return EXIT_CONFIG
        summary = report.summarize(out, total_s)
        files = report.write_outputs(values, out, summary, env)
        report.print_summary(values, summary, files)
        report.print_environment(env)
        return EXIT_OK

    # -- prerequisites ---------------------------------------------------------------------

    def _build_missing(self, missing: List[prereqs.Requirement], values: Dict[str, Any]) -> int:
        print("\nThe following prerequisites must be built before {} can run:".format(values["system"]))
        what = {
            "embeddings": "Starmie column embeddings for every datalake and query table "
                          "(model inference; uses a GPU if one is visible)",
            "metadata": "the value-distribution synopsis (one histogram per column, read from every CSV)",
            "duts_index": "the DUTS HNSW index over the categorical columns' embeddings",
        }
        for r in missing:
            print("  - {}: {}\n      {}".format(r.label, r.detail, what.get(r.key, "")))
        if not (self.args.build_missing or self.prompter.interactive):
            print("\nNot building without confirmation. Rerun with --build-missing to build them:")
            print("  " + config.rerun_command(values) + " --build-missing")
            return EXIT_MISSING
        try:
            bp = prereqs.collect_build_params(missing, values, self.args, self.prompter)
        except (ConfigError, FileNotFoundError) as e:
            print("\ncannot build: {}".format(e))
            return EXIT_MISSING
        if not self.args.build_missing and not self.prompter.confirm("\nBuild them now?"):
            print("Nothing built. Rerun once the prerequisites exist.")
            return EXIT_MISSING
        t0 = time.perf_counter()
        try:
            written = prereqs.build(missing, values, bp)
        except Exception as e:  # noqa: BLE001 -- report which build failed, with the traceback
            traceback.print_exc()
            print("\nbuilding prerequisites failed: {}: {}".format(type(e).__name__, e))
            return EXIT_MISSING
        print("\nBuilt in {:.0f}s:".format(time.perf_counter() - t0))
        for w in written:
            print("  " + w)
        cmd = config.rerun_command(values)
        if "duts_index" in {r.key for r in missing} and bp.hnsw_m != 32:
            cmd += " --hnsw-m {}".format(bp.hnsw_m)
        print("\nPrerequisites are ready. Rerun the experiment with:\n  " + cmd)
        return EXIT_BUILT

    # -- printing --------------------------------------------------------------------------

    @staticmethod
    def _print_config(values: Dict[str, Any]) -> None:
        print("\nconfiguration")
        for p in config.PARAMS:
            if p.name in values and p.applies_to(values["system"]):
                print("  {:<16}{}".format(p.name, values[p.name]))
        if values.get("limit_queries"):
            print("  {:<16}{}".format("limit_queries", values["limit_queries"]))

    @staticmethod
    def _print_checklist(reqs: List[prereqs.Requirement]) -> None:
        print("\nprerequisites")
        for r in reqs:
            print("  [{}] {:<44} {}".format("ok" if r.ok else "--", r.label, r.detail))


def main(argv: Optional[List[str]] = None) -> int:
    return ExperimentApp(argv).run()
