"""
IPOP-CMA-ES Baseline: Implementation and Benchmarking
======================================================

This module implements and benchmarks the IPOP-CMA-ES restart strategy
(Auger & Hansen, 2005) as a baseline, using the modcma library and the
DACBench BBOB benchmark suite.

IPOP doubles the population size lambda after each restart, which allows
the algorithm to escape local optima by broadening the search.
The default ipop_factor=2, doubling lambda on each restart until
mu reaches the cap of 512.

References
----------
- Auger, A. & Hansen, N. (2005). A restart CMA evolution strategy with
  increasing population size. IEEE CEC 2005.
- Hansen, N. (2016). The CMA evolution strategy: A tutorial.
- Hansen, N. et al. (2021). COCO: A platform for comparing continuous
  optimizers in a black-box setting.
"""

from __future__ import annotations

import csv
import json
import os
import time
import warnings
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from modcma import ModularCMAES, Parameters
from dacbench.envs.env_utils.toy_functions import IOHFunction
from dacbench.benchmarks.cma_benchmark import FUNCTION_NAMES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Total function evaluations budget per run (per instance).
# A generous budget is needed for multimodal BBOB functions where restarts
# are critical (Auger & Hansen, 2005).
DEFAULT_BUDGET = 100_000

# Default problem dimension matching the DACBench instance sets.
DEFAULT_DIM = 10

# Number of independent runs per instance (for variance estimation).
N_RUNS = 5

# IPOP population doubling factor (Auger & Hansen, 2005).
IPOP_FACTOR = 2

# modcma config-array indices for the modules we care about:
#   (active, elitist, orthogonal, sequential, threshold_convergence,
#    step_size_adaptation, mirrored, base_sampler, weights_option,
#    local_restart, bound_correction)
# local_restart options: (None, 'restart', 'IPOP', 'BIPOP', 'STOP')
#  => index 0 = None (no restart), index 2 = IPOP
IDX_NO_RESTART = 0
IDX_IPOP = 2

# Standard BBOB test functions (subset representative of function groups).
# Covers unimodal separable, unimodal non-separable, and multimodal groups
# as characterised by Hansen et al. (2021) in the COCO benchmarking platform.
BENCHMARK_FUNCTIONS = [
    "Sphere",           # f1  – unimodal, separable
    "Ellipsoid",        # f5  – unimodal, ill-conditioned
    "Rosenbrock",       # f15 – unimodal, non-separable
    "Rastrigin",        # f13 – multimodal, separable
    "RastriginRotated", # f14 – multimodal, rotated
    "Schwefel",         # f19 – multimodal, weak global structure
    "GriewankRosenbrock",  # f9  – multimodal, funnel
    "LunacekBiRastrigin",  # f12 – multimodal, deceptive
]

# Instance IDs to test (one representative instance per function).
INSTANCE_IDS = [1, 2, 3]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """Records the outcome of a single CMA-ES run."""
    func_name: str
    fid: int
    iid: int
    run: int
    strategy: str          # 'IPOP' or 'CMA-ES'
    dim: int
    budget: int
    fopt: float            # Best function value found (f - f*)
    used_budget: int       # Function evaluations consumed
    n_restarts: int        # Number of restarts performed
    final_lambda: int      # Population size at termination
    runtime_s: float       # Wall-clock time in seconds
    converged: bool        # Whether fopt < 1e-8 (COCO success criterion)


@dataclass
class AggregatedResult:
    """Aggregated statistics across multiple runs for one (func, iid, strategy)."""
    func_name: str
    fid: int
    iid: int
    strategy: str
    dim: int
    budget: int
    n_runs: int
    mean_fopt: float
    std_fopt: float
    median_fopt: float
    mean_used_budget: float
    mean_n_restarts: float
    mean_final_lambda: float
    success_rate: float    # Fraction of runs where converged=True
    mean_runtime_s: float


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def _build_config_array(local_restart_idx: int) -> np.ndarray:
    """
    Build a modcma config array with all defaults except local_restart.

    The config array encodes the modular CMA-ES configuration as integer
    indices into each module's option list (Hansen, 2016).  All entries are
    set to 0 (the first / default option) except local_restart.

    Parameters
    ----------
    local_restart_idx : int
        Index into local_restart options:
        0 = None, 1 = 'restart', 2 = 'IPOP', 3 = 'BIPOP'.

    Returns
    -------
    np.ndarray
        Integer array of length len(Parameters.__modules__).
    """
    n_modules = len(Parameters.__modules__)
    cfg = np.zeros(n_modules, dtype=int)
    # local_restart is the 10th module (0-indexed: position 9)
    local_restart_pos = list(Parameters.__modules__).index("local_restart")
    cfg[local_restart_pos] = local_restart_idx
    return cfg


def run_single(
    func_name: str,
    iid: int,
    strategy: str,
    dim: int = DEFAULT_DIM,
    budget: int = DEFAULT_BUDGET,
    run_idx: int = 0,
    seed: Optional[int] = None,
) -> RunResult:
    """
    Execute one complete CMA-ES run with the specified restart strategy.

    For IPOP (Auger & Hansen, 2005), the population size lambda is doubled
    after each restart up to a maximum mu of 512, giving the algorithm
    successive opportunities to locate the global optimum.

    Parameters
    ----------
    func_name : str
        Name of the BBOB function (must be in FUNCTION_NAMES).
    iid : int
        Instance ID for the BBOB function.
    strategy : str
        Either 'IPOP' or 'CMA-ES'.
    dim : int
        Problem dimensionality.
    budget : int
        Maximum number of function evaluations.
    run_idx : int
        Index of this run (for labelling).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    RunResult
    """
    if seed is not None:
        np.random.seed(seed + run_idx)

    func = IOHFunction(function_name=func_name, dim=dim, iid=iid)

    if strategy == "IPOP":
        local_restart_idx = IDX_IPOP
    else:  # plain CMA-ES, no restarts
        local_restart_idx = IDX_NO_RESTART

    cfg = _build_config_array(local_restart_idx)
    params = Parameters.from_config_array(dim, cfg)
    params.budget = budget

    t0 = time.perf_counter()
    es = ModularCMAES(func, parameters=params)
    while es.step():
        pass
    elapsed = time.perf_counter() - t0

    fopt = float(es.parameters.fopt)
    used = int(es.parameters.used_budget)
    n_restarts = max(0, len(es.parameters.restarts) - 1)  # first entry is t=0
    final_lambda = int(es.parameters.lambda_)

    return RunResult(
        func_name=func_name,
        fid=FUNCTION_NAMES[func_name],
        iid=iid,
        run=run_idx,
        strategy=strategy,
        dim=dim,
        budget=budget,
        fopt=fopt,
        used_budget=used,
        n_restarts=n_restarts,
        final_lambda=final_lambda,
        runtime_s=elapsed,
        converged=(fopt < 1e-8),
    )


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    func_names: List[str] = BENCHMARK_FUNCTIONS,
    instance_ids: List[int] = INSTANCE_IDS,
    strategies: List[str] = ("IPOP", "CMA-ES"),
    dim: int = DEFAULT_DIM,
    budget: int = DEFAULT_BUDGET,
    n_runs: int = N_RUNS,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[List[RunResult], List[AggregatedResult]]:
    """
    Run the full benchmark comparing IPOP-CMA-ES with plain CMA-ES.

    Each (function, instance, strategy) combination is evaluated n_runs times
    to obtain statistical estimates of algorithm performance, following the
    COCO benchmarking methodology described by Hansen et al. (2021).

    Parameters
    ----------
    func_names : list of str
        BBOB function names to benchmark.
    instance_ids : list of int
        BBOB instance IDs to benchmark.
    strategies : list of str
        Restart strategies to compare.
    dim : int
        Problem dimensionality.
    budget : int
        Function evaluation budget per run.
    n_runs : int
        Number of independent runs per configuration.
    seed : int
        Base random seed.
    verbose : bool
        Print progress if True.

    Returns
    -------
    all_results : list of RunResult
        Every individual run.
    aggregated : list of AggregatedResult
        Summary statistics grouped by (func, iid, strategy).
    """
    all_results: List[RunResult] = []
    total = len(func_names) * len(instance_ids) * len(strategies) * n_runs

    if verbose:
        print(f"{'='*70}")
        print(f"IPOP-CMA-ES Benchmark  |  dim={dim}  budget={budget}  runs={n_runs}")
        print(f"Functions : {len(func_names)}   Instances : {len(instance_ids)}   "
              f"Strategies : {len(strategies)}")
        print(f"Total runs: {total}")
        print(f"{'='*70}")

    count = 0
    for func_name in func_names:
        for iid in instance_ids:
            for strategy in strategies:
                run_fopt = []
                for r in range(n_runs):
                    result = run_single(
                        func_name=func_name,
                        iid=iid,
                        strategy=strategy,
                        dim=dim,
                        budget=budget,
                        run_idx=r,
                        seed=seed,
                    )
                    all_results.append(result)
                    run_fopt.append(result.fopt)
                    count += 1

                if verbose:
                    mean_f = np.mean(run_fopt)
                    print(
                        f"[{count:4d}/{total}]  {func_name:<22s}  iid={iid}  "
                        f"{strategy:<7s}  mean_fopt={mean_f:.3e}"
                    )

    # Aggregate
    aggregated = _aggregate_results(all_results)
    return all_results, aggregated


def _aggregate_results(results: List[RunResult]) -> List[AggregatedResult]:
    """Group RunResult list by (func_name, iid, strategy) and compute stats."""
    from collections import defaultdict
    groups: Dict[Tuple, List[RunResult]] = defaultdict(list)
    for r in results:
        groups[(r.func_name, r.iid, r.strategy)].append(r)

    aggregated = []
    for (func_name, iid, strategy), group in sorted(groups.items()):
        fopts = np.array([g.fopt for g in group])
        budgets = np.array([g.used_budget for g in group])
        restarts = np.array([g.n_restarts for g in group])
        lambdas = np.array([g.final_lambda for g in group])
        runtimes = np.array([g.runtime_s for g in group])
        converged = np.array([g.converged for g in group])

        aggregated.append(AggregatedResult(
            func_name=func_name,
            fid=group[0].fid,
            iid=iid,
            strategy=strategy,
            dim=group[0].dim,
            budget=group[0].budget,
            n_runs=len(group),
            mean_fopt=float(np.mean(fopts)),
            std_fopt=float(np.std(fopts)),
            median_fopt=float(np.median(fopts)),
            mean_used_budget=float(np.mean(budgets)),
            mean_n_restarts=float(np.mean(restarts)),
            mean_final_lambda=float(np.mean(lambdas)),
            success_rate=float(np.mean(converged)),
            mean_runtime_s=float(np.mean(runtimes)),
        ))
    return aggregated


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------

def save_results(
    all_results: List[RunResult],
    aggregated: List[AggregatedResult],
    out_dir: str = "results",
) -> None:
    """
    Save raw and aggregated results to CSV files.

    Parameters
    ----------
    all_results : list of RunResult
    aggregated : list of AggregatedResult
    out_dir : str
        Output directory (created if it does not exist).
    """
    os.makedirs(out_dir, exist_ok=True)

    raw_path = os.path.join(out_dir, "ipop_raw_results.csv")
    agg_path = os.path.join(out_dir, "ipop_aggregated_results.csv")

    # Raw
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(all_results[0]).keys()))
        writer.writeheader()
        for r in all_results:
            writer.writerow(asdict(r))

    # Aggregated
    with open(agg_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(aggregated[0]).keys()))
        writer.writeheader()
        for r in aggregated:
            writer.writerow(asdict(r))

    print(f"\nResults saved to {out_dir}/")
    print(f"  Raw    : {raw_path}")
    print(f"  Summary: {agg_path}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(aggregated: List[AggregatedResult]) -> None:
    """Print a formatted comparison table to stdout."""
    print(f"\n{'='*90}")
    print(f"{'BENCHMARK SUMMARY':^90}")
    print(f"{'='*90}")
    print(
        f"{'Function':<22} {'iid':>3} {'Strategy':>7} "
        f"{'mean_fopt':>12} {'std_fopt':>12} "
        f"{'success%':>9} {'mean_restarts':>14} {'mean_lambda':>12}"
    )
    print(f"{'-'*90}")

    prev_key = None
    for r in aggregated:
        key = (r.func_name, r.iid)
        if key != prev_key and prev_key is not None:
            print(f"{'':90}")
        prev_key = key

        print(
            f"{r.func_name:<22} {r.iid:>3} {r.strategy:>7} "
            f"{r.mean_fopt:>12.3e} {r.std_fopt:>12.3e} "
            f"{r.success_rate*100:>8.1f}% {r.mean_n_restarts:>14.1f} "
            f"{r.mean_final_lambda:>12.1f}"
        )
    print(f"{'='*90}")

    # Per-strategy aggregate
    ipop_rows = [r for r in aggregated if r.strategy == "IPOP"]
    cmaes_rows = [r for r in aggregated if r.strategy == "CMA-ES"]

    def _avg(rows, attr):
        return np.mean([getattr(r, attr) for r in rows]) if rows else float("nan")

    print(f"\n{'OVERALL AVERAGES':^90}")
    print(f"{'Strategy':<10} {'mean_fopt':>14} {'success_rate':>14} "
          f"{'mean_restarts':>15} {'mean_lambda':>12}")
    print(f"{'-'*65}")
    for name, rows in [("IPOP", ipop_rows), ("CMA-ES", cmaes_rows)]:
        print(
            f"{name:<10} {_avg(rows,'mean_fopt'):>14.3e} "
            f"{_avg(rows,'success_rate')*100:>13.1f}% "
            f"{_avg(rows,'mean_n_restarts'):>15.1f} "
            f"{_avg(rows,'mean_final_lambda'):>12.1f}"
        )
    print(f"{'='*90}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the IPOP-CMA-ES benchmarking pipeline."""
    all_results, aggregated = run_benchmark(
        func_names=BENCHMARK_FUNCTIONS,
        instance_ids=INSTANCE_IDS,
        strategies=["IPOP", "CMA-ES"],
        dim=DEFAULT_DIM,
        budget=DEFAULT_BUDGET,
        n_runs=N_RUNS,
        seed=42,
        verbose=True,
    )

    print_summary(aggregated)
    save_results(all_results, aggregated, out_dir="results")


if __name__ == "__main__":
    main()