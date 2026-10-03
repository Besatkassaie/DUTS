"""Terminal reports: the results summary at the end of a run and the execution environment.

Metric convention (the one ``experiments/fair3-report.md`` and ``baselines-report.md`` use):
precision and recall are means over ALL evaluated queries, and a query without a feasible
result counts 0. Precision/recall of whatever was returned, regardless of feasibility, is shown
as a second line (it is what the unconstrained ``starmie`` baseline is usually judged by).
"""
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional

from .systems import QueryOutcome, RunOutput

RULE = "=" * 72


def _mean(xs: List[Optional[float]]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarize(out: RunOutput, total_s: float) -> Dict[str, Any]:
    o = out.outcomes
    n = len(o)
    feas = [q for q in o if q.feasible]
    returned = [q for q in o if q.n_returned > 0]
    query_s = sum(q.runtime_s for q in o)
    causes: Dict[str, int] = {}
    for q in o:
        if not q.feasible:
            causes[q.cause or "unknown"] = causes.get(q.cause or "unknown", 0) + 1
    return {
        "n_queries": n,
        "n_skipped": out.n_skipped,
        "n_feasible": len(feas),
        "pct_feasible": 100.0 * len(feas) / n if n else 0.0,
        "infeasible_by_cause": causes,
        "precision": _mean([(q.precision or 0.0) if q.feasible else 0.0 for q in o]),
        "recall": _mean([(q.recall or 0.0) if q.feasible else 0.0 for q in o]),
        "ideal_recall": _mean([q.ideal_recall for q in o]),
        "precision_as_returned": _mean([q.precision for q in returned]),
        "recall_as_returned": _mean([q.recall for q in returned]),
        "n_returned_any": len(returned),
        "mean_F_R_feasible": _mean([q.F_R for q in feas]),
        "mean_sum_U_feasible": _mean([q.sum_U for q in feas]),
        "setup_s": out.setup_s,
        "query_time_s": query_s,
        "avg_query_s": query_s / n if n else 0.0,
        "total_s": total_s,
    }


def _f(x: Optional[float], digits: int = 4) -> str:
    return "n/a" if x is None else "{:.{}f}".format(x, digits)


def print_summary(v: Dict[str, Any], s: Dict[str, Any], files: List[str]) -> None:
    print("\n" + RULE)
    print("RESULTS  system={}  dataset={}".format(v["system"], v["dataset"]))
    print(RULE)
    params = ["k", "f_star", "delta", "sigma"] + (
        ["alpha", "top_n", "theta_cat"] if v["system"] == "duts" else ["n_columns", "workers"])
    print("  parameters        " + "  ".join("{}={}".format(p, v[p]) for p in params)
          + "  (tau={:.3f})".format(v["f_star"] - v["delta"]))
    print("  queries           {} evaluated{}".format(
        s["n_queries"], ", {} skipped (not in query dir / no embedding)".format(s["n_skipped"])
        if s["n_skipped"] else ""))
    print("  feasible          {} / {}  ({:.1f}%)".format(s["n_feasible"], s["n_queries"], s["pct_feasible"]))
    for cause, n in sorted(s["infeasible_by_cause"].items()):
        print("    infeasible: {:<28} {}".format(cause, n))
    print("  precision         {}   (mean over all queries; infeasible = 0)".format(_f(s["precision"])))
    print("  recall            {}   (ideal recall@k = {})".format(_f(s["recall"]), _f(s["ideal_recall"])))
    print("  as returned       precision {}  recall {}  ({} queries returned tables)".format(
        _f(s["precision_as_returned"]), _f(s["recall_as_returned"]), s["n_returned_any"]))
    print("  mean F_R          {}   mean sum U {}   (feasible queries)".format(
        _f(s["mean_F_R_feasible"], 3), _f(s["mean_sum_U_feasible"], 2)))
    print("  runtime           total {:.2f}s  (setup {:.2f}s, queries {:.2f}s, imports/other {:.2f}s)".format(
        s["total_s"], s["setup_s"], s["query_time_s"],
        max(0.0, s["total_s"] - s["setup_s"] - s["query_time_s"])))
    print("  avg per query     {:.4f}s".format(s["avg_query_s"]))
    for f in files:
        print("  wrote             {}".format(f))


def write_outputs(v: Dict[str, Any], out: RunOutput, s: Dict[str, Any], env: Dict[str, Any]) -> List[str]:
    os.makedirs(v["output_dir"], exist_ok=True)
    tag = "{}_{}_k{}".format(v["dataset"], v["system"], v["k"])
    if v["system"] == "duts":
        tag += "_a{:g}".format(v["alpha"])
    per_query = os.path.join(v["output_dir"], tag + "_queries.csv")
    with open(per_query, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(QueryOutcome._fields)
        for q in out.outcomes:
            w.writerow(q)
    summary = os.path.join(v["output_dir"], tag + "_summary.json")
    with open(summary, "w") as f:
        json.dump({"config": v, "summary": s, "details": out.details, "environment": env},
                  f, indent=2, default=str)
    return [per_query, summary]


# -- environment ---------------------------------------------------------------------------

PACKAGES = ("numpy", "scipy", "pandas", "hnswlib", "networkx", "munkres", "pyarrow",
            "torch", "transformers", "tokenizers", "scikit-learn", "mlflow", "tqdm")


def _pkg_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:  # noqa: BLE001 -- not installed
        return None


def _cmd(args: List[str], stderr: bool = False) -> Optional[str]:
    if not shutil.which(args[0]):
        return None
    try:
        r = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15,
                           universal_newlines=True)
    except Exception:  # noqa: BLE001
        return None
    text = (r.stderr if stderr else r.stdout).strip()
    return text or None


def _git_rev(path: str) -> Optional[str]:
    if not os.path.isdir(os.path.join(path, ".git")):
        return None
    rev = _cmd(["git", "-C", path, "rev-parse", "--short", "HEAD"])
    dirty = _cmd(["git", "-C", path, "status", "--porcelain", "--untracked-files=no"])
    return None if rev is None else rev + ("-dirty" if dirty else "")


def collect_environment(v: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    from .config import REPO_ROOT
    env: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        # the interpreter's own env, not the shell's activated one (they differ when the env's
        # python is invoked by path)
        "conda_env": (os.path.basename(sys.prefix)
                      if os.path.basename(os.path.dirname(sys.prefix)) == "envs"
                      else os.environ.get("CONDA_DEFAULT_ENV")),
        "packages": {p: _pkg_version(p) for p in PACKAGES},
        "duts_git": _git_rev(REPO_ROOT),
    }
    if v and v.get("starmie_root"):
        env["starmie_git"] = _git_rev(v["starmie_root"])
    scipy_v = env["packages"].get("scipy")
    env["milp_solver"] = "HiGHS via scipy.optimize.milp (bundled with SciPy {})".format(scipy_v) if scipy_v else None
    env["ann_library"] = "hnswlib {}".format(env["packages"].get("hnswlib"))
    java = _cmd(["java", "-version"], stderr=True)
    env["java"] = java.splitlines()[0] if java else None
    env["java_note"] = "not used by this code"
    env.update(_cuda_info())
    return env


def _cuda_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"cuda_torch": None, "cuda_available": False, "gpus": []}
    try:
        import torch
        info["cuda_torch"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001 -- torch is only needed to generate embeddings
        pass
    smi = _cmd(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    info["nvidia_smi"] = smi.splitlines() if smi else None
    return info


def print_environment(env: Dict[str, Any]) -> None:
    print("\n" + RULE)
    print("EXECUTION ENVIRONMENT")
    print(RULE)
    print("  python            {} ({}) {}".format(env["python"], env["implementation"], env["python_executable"]))
    print("  conda env         {}".format(env["conda_env"]))
    print("  platform          {}  [{}; {} CPUs]".format(env["platform"], env["machine"], env["cpu_count"]))
    print("  DUTS revision     {}".format(env.get("duts_git") or "n/a (not a git checkout)"))
    if "starmie_git" in env:
        print("  Starmie revision  {}".format(env["starmie_git"] or "n/a (not a git checkout)"))
    print("  packages")
    for name, ver in env["packages"].items():
        print("    {:<16}{}".format(name, ver or "not installed"))
    print("  MILP solver       {}".format(env["milp_solver"] or "n/a"))
    print("  ANN index         {}".format(env["ann_library"]))
    print("  Java/JDK          {}  ({})".format(env["java"] or "not found", env["java_note"]))
    gpu = ", ".join(env["gpus"]) if env["gpus"] else "none visible to torch"
    print("  CUDA (torch)      {}  available={}  cuDNN={}".format(
        env["cuda_torch"] or "n/a", env["cuda_available"], env.get("cudnn") or "n/a"))
    print("  GPUs              {}".format(gpu))
    if env.get("nvidia_smi"):
        for line in env["nvidia_smi"]:
            print("    nvidia-smi      {}".format(line))
    print("  (CUDA/GPU are used only to generate embeddings; DUTS and the baselines run on CPU)")
