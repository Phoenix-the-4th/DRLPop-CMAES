"""
benchmark.py
============
Orchestrates the full IPOP-CMA-ES benchmarking pipeline.

This script is the single entry point for running and saving experiments.
It is deliberately kept free of CMA-ES mechanics and controller logic —
those live in cmaes_runner.py and population_controller.py respectively.

Pipeline
--------
1.  Build a list of (function, instance) pairs from the DACBench BBOB
    instance sets.
2.  For each configuration (function × instance × controller × repeat):
      a.  Instantiate the appropriate PopulationController.
      b.  Wrap it in a CMAESRunner.
      c.  Run the optimisation and collect a RunResult.
3.  Aggregate results across repeats.
4.  Save raw and aggregated results to CSV.
5.  Print a formatted summary table.

Controllers benchmarked
-----------------------
-   IPOPController   : IPOP-CMA-ES (Auger & Hansen, 2005)
-   Baseline CMA-ES  : same runner with enable_restarts=False

Adding a new controller requires only:
    1.  Implementing PopulationController.select_lambda in a new subclass.
    2.  Adding an entry to CONTROLLER_REGISTRY below.

References
----------
Auger, A. & Hansen, N. (2005). A restart CMA evolution strategy with
    increasing population size. IEEE CEC 2005.
Hansen, N. et al. (2021). COCO: A platform for comparing continuous
    optimizers in a black-box setting. Optimization Methods and Software.
Hansen, N. (2016). The CMA evolution strategy: A tutorial.
    arXiv:1604.00772.
"""

from __future__ import annotations

import csv
import os
import time
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from dacbench.envs.env_utils.toy_functions import IOHFunction
from dacbench.benchmarks.cma_benchmark import FUNCTION_NAMES

from population_controller import IPOPController, PopulationController
from cmaes_runner import CMAESRunner, RunResult


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

# Total function evaluations per run.
# A budget of 1e5 * dim is typical for BBOB comparisons (Hansen et al., 2021).
DEFAULT_BUDGET: int = 100_000

# Problem dimensionality (must match the instance-set CSVs).
DEFAULT_DIM: int = 10

# Independent repeats per (function × instance × controller).
N_RUNS: int = 5

# Random seed for reproducibility.
BASE_SEED: int = 42

# BBOB functions to benchmark.
# Drawn from all five groups defined by Hansen et al. (2021) in COCO:
#   separable unimodal, non-separable unimodal, unimodal with high
#   conditioning, multimodal with adequate global structure, multimodal
#   with weak global structure.
BENCHMARK_FUNCTIONS: List[str] = [
    # --- Unimodal separable ---
    "Sphere",              # f1
    # --- Unimodal, moderate conditioning ---
    "Rosenbrock",          # f15
    # --- Unimodal, high conditioning ---
    "Ellipsoid",           # f5
    # --- Multimodal, adequate global structure ---
    "Rastrigin",           # f13
    "RastriginRotated",    # f14
    "GriewankRosenbrock",  # f9
    # --- Multimodal, weak global structure / deceptive ---
    "Schwefel",            # f19
    "LunacekBiRastrigin",  # f12
]

# BBOB instance IDs (different rotations / translations of each function).
INSTANCE_IDS: List[int] = [1, 2, 3]

# ---------------------------------------------------------------------------
# Controller registry
# ---------------------------------------------------------------------------
# Each entry is (label, factory_fn, enable_restarts).
# factory_fn() must return a fresh PopulationController.
# enable_restarts=False disables restart detection in the runner (plain CMA-ES).
#
# To add a DRL controller, append:
#   ("DRL", lambda: MyDRLController(model_path="..."), True)

CONTROLLER_REGISTRY: List[Tuple[str, Callable[[], PopulationController], bool]] = [
    (
        "IPOP",
        lambda: IPOPController(ipop_factor=2, max_lambda=512),
        True,   # restarts enabled
    ),
    (
        "CMA-ES",
        lambda: IPOPController(),   # controller irrelevant; restarts disabled
        False,  # no restarts – plain single-run CMA-ES
    ),
]


# ---------------------------------------------------------------------------
# Aggregated result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AggregatedResult:
    """
    Summary statistics for one (function, instance, controller) combination.

    Attributes mirror the RunResult fields but are averaged / summarised
    across N_RUNS independent repeats.
    """
    func_name: str
    fid: int
    iid: int
    controller: str
    dim: int
    budget: int
    n_runs: int
    # --- performance ---
    mean_fopt: float
    std_fopt: float
    median_fopt: float
    best_fopt: float
    worst_fopt: float
    # --- convergence ---
    success_rate: float       # fraction of runs with fopt < convergence_target
    # --- resource usage ---
    mean_used_budget: float
    std_used_budget: float
    # --- restart statistics ---
    mean_n_restarts: float
    mean_initial_lambda: float
    mean_final_lambda: float
    # --- timing ---
    mean_runtime_s: float


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    func_names: List[str] = BENCHMARK_FUNCTIONS,
    instance_ids: List[int] = INSTANCE_IDS,
    controller_registry: List[Tuple[str, Callable, bool]] = CONTROLLER_REGISTRY,
    dim: int = DEFAULT_DIM,
    budget: int = DEFAULT_BUDGET,
    n_runs: int = N_RUNS,
    seed: int = BASE_SEED,
    convergence_target: float = 1e-8,
    sigma0: float = 2.0,
    verbose: bool = True,
) -> Tuple[List[RunResult], List[AggregatedResult]]:
    """
    Run the full benchmark and return raw and aggregated results.

    Parameters
    ----------
    func_names : list of str
    instance_ids : list of int
    controller_registry : list of (label, factory, enable_restarts)
    dim : int
    budget : int
    n_runs : int
    seed : int
    convergence_target : float
    sigma0 : float
    verbose : bool

    Returns
    -------
    all_results : list of RunResult
    aggregated : list of AggregatedResult
    """
    all_results: List[RunResult] = []

    total_runs = (
        len(func_names) * len(instance_ids) * len(controller_registry) * n_runs
    )

    if verbose:
        _print_header(total_runs, dim, budget, n_runs)

    count = 0
    for func_name in func_names:
        fid = FUNCTION_NAMES[func_name]

        for iid in instance_ids:
            func = IOHFunction(function_name=func_name, dim=dim, iid=iid)

            for label, factory, enable_restarts in controller_registry:

                phase_fopt_list: List[float] = []

                for r in range(n_runs):
                    np.random.seed(seed + count)

                    controller = factory()
                    runner = CMAESRunner(
                        controller=controller,
                        dim=dim,
                        budget=budget,
                        sigma0=sigma0,
                        convergence_target=convergence_target,
                        enable_restarts=enable_restarts,
                    )

                    result = runner.run(
                        fitness_func=func,
                        func_name=func_name,
                        fid=fid,
                        iid=iid,
                        run_index=r,
                        label=label,
                    )
                    all_results.append(result)
                    phase_fopt_list.append(result.fopt)
                    count += 1

                if verbose:
                    _print_progress(
                        count, total_runs,
                        func_name, iid, label,
                        np.mean(phase_fopt_list),
                    )

    aggregated = _aggregate(all_results)
    return all_results, aggregated


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(results: List[RunResult]) -> List[AggregatedResult]:
    """Group results by (func_name, iid, controller_label) and compute stats."""
    groups: Dict[Tuple, List[RunResult]] = defaultdict(list)
    for r in results:
        groups[(r.func_name, r.iid, r.controller_label)].append(r)

    aggregated: List[AggregatedResult] = []
    for (func_name, iid, ctrl_label), group in sorted(groups.items()):
        fopts       = np.array([g.fopt for g in group])
        budgets     = np.array([g.used_budget for g in group])
        restarts    = np.array([g.n_restarts for g in group])
        runtimes    = np.array([g.runtime_s for g in group])
        converged   = np.array([g.converged for g in group], dtype=float)

        init_lambdas = np.array(
            [g.lambda_history[0] for g in group if g.lambda_history]
        )
        final_lambdas = np.array(
            [g.lambda_history[-1] for g in group if g.lambda_history]
        )

        aggregated.append(AggregatedResult(
            func_name=func_name,
            fid=group[0].fid,
            iid=iid,
            controller=ctrl_label,
            dim=group[0].dim,
            budget=group[0].budget,
            n_runs=len(group),
            mean_fopt=float(np.mean(fopts)),
            std_fopt=float(np.std(fopts)),
            median_fopt=float(np.median(fopts)),
            best_fopt=float(np.min(fopts)),
            worst_fopt=float(np.max(fopts)),
            success_rate=float(np.mean(converged)),
            mean_used_budget=float(np.mean(budgets)),
            std_used_budget=float(np.std(budgets)),
            mean_n_restarts=float(np.mean(restarts)),
            mean_initial_lambda=float(np.mean(init_lambdas)) if len(init_lambdas) else 0.0,
            mean_final_lambda=float(np.mean(final_lambdas)) if len(final_lambdas) else 0.0,
            mean_runtime_s=float(np.mean(runtimes)),
        ))
    return aggregated


# ---------------------------------------------------------------------------
# Saving
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
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- raw results -------------------------------------------------------
    raw_path = os.path.join(out_dir, "raw_results.csv")
    raw_rows = []
    for r in all_results:
        row = {
            "func_name": r.func_name,
            "fid": r.fid,
            "iid": r.iid,
            "run_index": r.run_index,
            "controller": r.controller_label,
            "controller_class": r.controller_name.split("(")[0],
            "dim": r.dim,
            "budget": r.budget,
            "fopt": r.fopt,
            "used_budget": r.used_budget,
            "n_restarts": r.n_restarts,
            "lambda_history": str(r.lambda_history),
            "runtime_s": r.runtime_s,
            "converged": r.converged,
        }
        raw_rows.append(row)

    pd.DataFrame(raw_rows).to_csv(raw_path, index=False)

    # --- aggregated results ------------------------------------------------
    agg_path = os.path.join(out_dir, "aggregated_results.csv")
    agg_rows = [asdict(a) for a in aggregated]
    pd.DataFrame(agg_rows).to_csv(agg_path, index=False)

    print(f"\nResults saved to '{out_dir}/'")
    print(f"  Raw        : {raw_path}  ({len(raw_rows)} rows)")
    print(f"  Aggregated : {agg_path}  ({len(agg_rows)} rows)")


# ---------------------------------------------------------------------------
# Console printing helpers
# ---------------------------------------------------------------------------

def _print_header(total: int, dim: int, budget: int, n_runs: int) -> None:
    w = 80
    print("=" * w)
    print(f"{'IPOP-CMA-ES BENCHMARK':^{w}}")
    print("=" * w)
    print(f"  Dimension : {dim}   Budget : {budget:,}   Runs/config : {n_runs}")
    print(f"  Total runs: {total:,}")
    print("=" * w)


def _print_progress(
    count: int,
    total: int,
    func_name: str,
    iid: int,
    label: str,
    mean_fopt: float,
) -> None:
    pct = 100 * count / total
    print(
        f"  [{count:5d}/{total}] {pct:5.1f}%  "
        f"{func_name:<22s}  iid={iid}  {label:<8s}  "
        f"mean_fopt={mean_fopt:.3e}"
    )


def print_summary(aggregated: List[AggregatedResult]) -> None:
    """Print a formatted comparison table to stdout."""
    w = 100
    print(f"\n{'=' * w}")
    print(f"{'BENCHMARK SUMMARY':^{w}}")
    print(f"{'=' * w}")
    print(
        f"{'Function':<22} {'iid':>3} {'Controller':>10} "
        f"{'mean_fopt':>12} {'std_fopt':>12} "
        f"{'success%':>9} {'restarts':>9} {'λ_final':>9}"
    )
    print(f"{'-' * w}")

    prev_func = None
    for r in aggregated:
        if r.func_name != prev_func and prev_func is not None:
            print()
        prev_func = r.func_name
        print(
            f"{r.func_name:<22} {r.iid:>3} {r.controller:>10} "
            f"{r.mean_fopt:>12.3e} {r.std_fopt:>12.3e} "
            f"{r.success_rate * 100:>8.1f}% {r.mean_n_restarts:>9.1f} "
            f"{r.mean_final_lambda:>9.1f}"
        )

    print(f"{'=' * w}")

    # Per-controller overall averages
    print(f"\n{'OVERALL AVERAGES':^{w}}")
    print(
        f"{'Controller':<12} {'mean_fopt':>14} {'success%':>10} "
        f"{'mean_restarts':>15} {'mean_λ_final':>14}"
    )
    print(f"{'-' * 65}")

    controllers = sorted({r.controller for r in aggregated})
    for ctrl in controllers:
        rows = [r for r in aggregated if r.controller == ctrl]
        print(
            f"{ctrl:<12} {np.mean([r.mean_fopt for r in rows]):>14.3e} "
            f"{np.mean([r.success_rate for r in rows]) * 100:>9.1f}% "
            f"{np.mean([r.mean_n_restarts for r in rows]):>15.1f} "
            f"{np.mean([r.mean_final_lambda for r in rows]):>14.1f}"
        )
    print(f"{'=' * w}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the benchmark pipeline end-to-end."""
    all_results, aggregated = run_benchmark(
        func_names=BENCHMARK_FUNCTIONS,
        instance_ids=INSTANCE_IDS,
        controller_registry=CONTROLLER_REGISTRY,
        dim=DEFAULT_DIM,
        budget=DEFAULT_BUDGET,
        n_runs=N_RUNS,
        seed=BASE_SEED,
        convergence_target=1e-8,
        sigma0=2.0,
        verbose=True,
    )

    print_summary(aggregated)
    save_results(all_results, aggregated, out_dir="results")


if __name__ == "__main__":
    main()