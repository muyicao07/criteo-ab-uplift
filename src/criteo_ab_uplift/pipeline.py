"""One entry point for the fixed, complete public-data analysis."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import threading
import time
import traceback

import psutil

from .data import EXPECTED_ROWS, download_data, prepare_data
from .adjustment import lin_adjustment
from .statistics import analyze_groups
from .simulations import run_simulations
from .uplift import FIXED_PARAMETERS, _ensure_identity, run_uplift


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def source_fingerprint(root: Path) -> str:
    paths = list((root / "src" / "criteo_ab_uplift").glob("*.py")) + list((root / "sql").glob("*.sql"))
    paths += [root / name for name in ["pyproject.toml", "requirements-lock.txt", "run.sh", "Run_Project.command", "LICENSE"]]
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def git_provenance(root: Path, fingerprint: str) -> dict:
    release = root / "RELEASE.json"
    record = json.loads(release.read_text()) if release.exists() else {}
    info = {"commit": None, "source": "unversioned", "working_tree_dirty": None,
            "release_source_matches": record.get("source_sha256") == fingerprint}
    # Do not accidentally identify a parent project's unrelated Git commit.
    if (root / ".git").exists():
        try:
            info.update(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                        source="local_repository", working_tree_dirty=bool(subprocess.check_output(
                            ["git", "status", "--porcelain"], cwd=root, text=True).strip()))
        except (OSError, subprocess.CalledProcessError):
            pass
    if info["commit"] is None and info["release_source_matches"]:
        info.update(commit=record.get("commit"), source="verified_packaged_source")
    return info


class MemoryMonitor:
    """Sample process-tree RSS; shared pages can be counted more than once."""
    def __init__(self):
        self.peak = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        process = psutil.Process()
        while not self.stop_event.is_set():
            try:
                total = process.memory_info().rss
                for child in process.children(recursive=True):
                    try:
                        total += child.memory_info().rss
                    except psutil.Error:
                        pass
                self.peak = max(self.peak, total)
            except psutil.Error:
                pass
            self.stop_event.wait(.25)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join()


def hardware():
    cpu = platform.processor()
    if platform.system() == "Darwin":
        try:
            cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return {"os": platform.platform(), "architecture": platform.machine(), "cpu": cpu,
            "physical_cores": psutil.cpu_count(logical=False), "logical_cores": psutil.cpu_count(),
            "ram_bytes": psutil.virtual_memory().total, "python": platform.python_version()}


def reusable_uplift(cache: Path, output: Path, seed: int, threads: int) -> dict | None:
    paths = [output / name for name in ["uplift_results.json", "uplift_targeting.csv", "uplift_qini.csv",
                                       "uplift_curves.csv", "uplift_bootstrap.npz"]]
    if not all(path.is_file() for path in paths):
        return None
    # Published results may survive removal of the local data/model cache.
    # Without its original identity there is no checkpoint to validate or reuse.
    if not (cache / "uplift" / "identity.json").is_file():
        return None
    result = json.loads(paths[0].read_text())
    parameters = {**FIXED_PARAMETERS, "random_state": seed, "n_jobs": threads}
    identity = _ensure_identity(cache / "uplift", cache / "X.npy", cache / "labels.npy", parameters, seed)
    expected = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if (result.get("checkpoint_identity_sha256") != expected or result.get("parameters") != parameters
            or result.get("n_total") != EXPECTED_ROWS or result.get("bootstrap", {}).get("repetitions") != 200
            or result.get("split", {}).get("seed") != seed):
        raise RuntimeError("Saved uplift results do not match this run's data and fixed configuration.")
    return result


def run(root: Path, output: Path, threads: int):
    from .reporting import write_tables, write_figures, render_reports, execute_notebooks

    root, output = root.resolve(), output.resolve()
    output.relative_to(root / "results")  # Keep report links and packaging inside this project.
    output.mkdir(parents=True, exist_ok=True)
    cache = root / "data" / "processed"
    seed = 2026
    configuration = {
        "seed": seed, "expected_rows": EXPECTED_ROWS, "train_fraction": .7,
        "stratify": ["treatment", "visit"], "primary_outcome": "visit", "secondary_outcome": "conversion",
        "primary_model": "x_learner", "bootstrap_repetitions": 200, "bootstrap_ci": "95% percentile",
        "budgets": [.1, .3, .5], "threads": threads, "lightgbm": FIXED_PARAMETERS,
        "outcome_objective": "binary", "effect_objective": "regression_l2",
        "aa_repetitions": 1000, "power_repetitions": 500, "power_total_n": [10000, 100000, 1000000],
        "peeking_repetitions": 1000, "peeking_increment_per_arm": 10000, "peeking_looks": 50,
        "coverage_repetitions": 1000, "coverage_per_arm_n": [500, 2000, 10000],
        "posterior_draws_full": 100000, "posterior_draws_coverage": 2000,
    }
    fingerprint = source_fingerprint(root)
    manifest = {
        "status": "running", "started_utc": utc_now(), "configuration": configuration,
        "git": git_provenance(root, fingerprint), "source_sha256": fingerprint,
        "package_versions": {d.metadata["Name"]: d.version for d in distributions() if d.metadata["Name"]},
        "seeds": {"analysis": seed, "simulation_root": seed, "split": seed, "lightgbm": seed,
                  "targeting_tie_break": seed, "uplift_row_bootstrap": seed + 1},
        "hardware": hardware(), "stage_timings": {}, "data_sha256": None,
        "memory_measurement_scope": "This pipeline invocation only; process-tree RSS sampled every 0.25s. "
                                    "Shared pages can be counted twice. Reused earlier computations are excluded.",
    }
    save_json(output / "configuration.json", configuration)
    started = time.monotonic()

    def stage(name, function):
        print(f"\n{name}", flush=True)
        t0 = time.monotonic()
        result = function()
        manifest["stage_timings"][name] = {"wall_seconds_this_invocation": time.monotonic() - t0}
        save_json(output / "manifest.json", manifest)
        return result

    with MemoryMonitor() as memory:
        try:
            csv_path = stage("Verify or download official dataset", lambda: download_data(root / "data" / "criteo-research-uplift-v2.1.csv.gz"))
            cached_data = (cache / "metadata.json").exists()
            audit = stage("Audit and prepare all rows", lambda: prepare_data(csv_path, cache, seed=seed, threads=threads,
                                                                            expected_rows=EXPECTED_ROWS))
            manifest["stage_timings"]["Audit and prepare all rows"]["cache_existed_before_run"] = cached_data
            manifest["data_sha256"] = audit["data_sha256"]
            # Published files use repository-relative paths rather than a local user directory.
            published_audit = {**audit, "data_path": csv_path.relative_to(root).as_posix(), "cache_dir": cache.relative_to(root).as_posix()}
            save_json(output / "data_audit.json", published_audit)
            ab = stage("Frequentist, Bayesian and exposure IV", lambda: analyze_groups(audit["groups"], seed=seed))
            save_json(output / "ab_results.json", ab)
            lin = stage("Full-data Lin adjustment with HC2", lambda: lin_adjustment(cache, audit))
            save_json(output / "regression_adjustment.json", lin)
            sims = stage("Finite-population experiment simulations", lambda: run_simulations(audit["groups"], seed=seed,
                                                                                           joint_counts=audit["joint_counts"]))
            save_json(output / "simulations.json", sims)
            previous = reusable_uplift(cache, output, seed, threads)
            model_cache_existed = (cache / "uplift" / "mu1.txt").exists()
            uplift = stage("Full-data uplift and paired bootstrap", lambda: previous if previous is not None else
                           run_uplift(cache, output, seed=seed, threads=threads, bootstrap_reps=200))
            model_timing = manifest["stage_timings"]["Full-data uplift and paired bootstrap"]
            model_timing.update(reused_completed_result=previous is not None,
                                model_cache_existed_before_invocation=model_cache_existed,
                                saved_uplift_computation_seconds=uplift["elapsed_seconds"])
            save_json(output / "manifest.json", manifest)
            stage("Generate tables and figures", lambda: (write_tables(output), write_figures(output)))
            stage("Generate English reports", lambda: render_reports(root, output))
            stage("Execute five results-only notebooks", lambda: execute_notebooks(root, output))
            manifest.update(status="complete", completed_utc=utc_now(), elapsed_seconds=time.monotonic() - started,
                            peak_sampled_process_tree_rss_bytes=memory.peak)
            manifest["output_files"] = {}
            for path in sorted(output.rglob("*")):
                if path.is_file() and path.name not in {"manifest.json", "run.log"}:
                    manifest["output_files"][path.relative_to(output).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            save_json(output / "manifest.json", manifest)
            (root / "results" / "LATEST.txt").write_text(output.relative_to(root).as_posix() + "\n")
            print(f"\nComplete. Results: {output.relative_to(root)}\nREADME.md and CONCLUSION.md are ready.", flush=True)
        except BaseException as exc:
            manifest.update(status="failed", stopped_utc=utc_now(), elapsed_seconds=time.monotonic() - started,
                            peak_sampled_process_tree_rss_bytes=memory.peak, error=str(exc), traceback=traceback.format_exc())
            save_json(output / "manifest.json", manifest)
            raise


def main():
    parser = argparse.ArgumentParser(description="Run the full-data Criteo analysis.")
    parser.add_argument("command", nargs="?", choices=["run"], default="run")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, help="Results folder inside this project's results/; defaults to a new timestamp.")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    root = Path(__file__).resolve().parents[2]
    output = args.output or root / "results" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    if not output.is_absolute():
        output = root / output
    os.environ.setdefault("MPLCONFIGDIR", str(root / "results" / ".mpl-cache"))
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(root / "results" / ".jupyter"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(root, output, args.threads)
