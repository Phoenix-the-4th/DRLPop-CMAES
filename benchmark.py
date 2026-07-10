from __future__ import annotations
 
import argparse
import os
import shutil
import warnings
from typing import Callable, Dict, List, Optional, Tuple
 
import ioh
import numpy as np
import pandas as pd
 
warnings.filterwarnings("ignore")
 
from controllers import Default, IPOP, PopulationController
from cmaes import CMAES, BaseResult
 
 
# ===========================================================================
# Benchmark defaults
# ===========================================================================
 
# ALL_FIDS: List[int] = list(range(1, 25))   # all 24 BBOB functions
ALL_FIDS: List[int] = list(range(1, 3))   # all 24 BBOB functions
DEFAULT_IIDS: List[int] = [1, 2, 3, 4, 5]
DEFAULT_DIMS: List[int] = [5, 10, 20]
DEFAULT_N_RUNS: int = 5
DEFAULT_BUDGET: int = 100000
CONVERGENCE_TARGET: float = 1e-8
SIGMA0: float = 2.0
BASE_SEED: int = 42
 
 
# ===========================================================================
# Controller registry
# ===========================================================================
 
CONTROLLER_REGISTRY: List[Tuple[str, Callable[[], PopulationController], bool]] = [Default(), IPOP()]
 
 
# ===========================================================================
# IOH function ID -> name map
# ===========================================================================
 
def _fid_name_map() -> Dict[int, str]:
    return ioh.get_problem(1, dimension=2, instance=1).problems
 
FID_NAME: Dict[int, str] = _fid_name_map()
 
 
# ===========================================================================
# Per-controller benchmark runner
# ===========================================================================
 
def _run_controller(
    label: str,
    factory: Callable[[], PopulationController],
    enable_restarts: bool,
    fids: List[int],
    iids: List[int],
    dims: List[int],
    n_runs: int,
    budget: int,
    sigma0: float,
    convergence_target: float,
    out_root: str,
    seed: int,
    verbose: bool,
) -> List:
    """
    Run all (fid, iid, dim) combinations for one controller.
 
    A single ioh.logger.Analyzer is created for this controller and covers
    the full (fids × iids × dims) sweep via ioh.suite.BBOB.  The suite
    iterates problems in (fid, iid, dim) order; n_runs independent repeats
    per problem are executed by calling prob.reset() between runs.
 
    The IOH Analyzer accumulates all runs automatically into the correct
    .dat files, producing the standard multi-run IOHprofiler format.
    """
    suite = ioh.suite.BBOB(fids, iids, dims)
 
    logger = ioh.logger.Analyzer(
        root=out_root,
        folder_name=label,
        algorithm_name=label,
    )
    suite.attach_logger(logger)
 
    all_results: List = []
    total_probs = len(fids) * len(iids) * len(dims)
    prob_idx = 0
 
    for prob in suite:
        dim = prob.meta_data.n_variables
        fid = prob.meta_data.problem_id
        iid = prob.meta_data.instance
        func_name = prob.meta_data.name
        prob_idx += 1
 
        run_fopt = []
        for r in range(n_runs):
            np.random.seed(seed + fid * 10_000 + iid * 100 + dim + r)
 
            controller = factory
            runner = CMAES(
                controller=controller,
            )
            result = runner.run(prob, "BBOB", run_index=r)
            all_results.append(result)
            run_fopt.append(result.fopt_best)
 
        if verbose:
            mean_f = np.mean(run_fopt)
            # sr = np.mean([r.converged for r in all_results[-n_runs:]])
            print(
                f"  [{prob_idx:4d}/{total_probs}]  {label:<14}  "
                f"f{fid:02d} ({func_name:<22})  "
                f"iid={iid}  d={dim:3d}  "
                # f"mean_fopt={mean_f:.3e}  sr={sr*100:.0f}%"
            )
 
    logger.close()
    return all_results
 
 
# ===========================================================================
# Orchestrator
# ===========================================================================
 
def run_benchmark(
    fids: List[int] = ALL_FIDS,
    iids: List[int] = DEFAULT_IIDS,
    dims: List[int] = DEFAULT_DIMS,
    n_runs: int = DEFAULT_N_RUNS,
    budget: int = DEFAULT_BUDGET,
    sigma0: float = SIGMA0,
    convergence_target: float = CONVERGENCE_TARGET,
    out_root: str = "ioh_results",
    controller_registry: Optional[List[Tuple]] = None,
    seed: int = BASE_SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the full benchmark across all controllers and return a summary DataFrame.
 
    IOH logs are written to out_root/<controller_label>/.
    A flat summary CSV is written to out_root/summary.csv.
 
    Parameters
    ----------
    fids : list of int      BBOB function IDs (1-24).
    iids : list of int      Instance IDs.
    dims : list of int      Dimensionalities.
    n_runs : int            Independent repeats per (fid, iid, dim).
    budget : int            Max function evaluations per run.
    sigma0 : float          Initial step size.
    convergence_target : float
    out_root : str          Root directory for all output.
    controller_registry : list or None  Uses module default if None.
    seed : int              Base random seed.
    verbose : bool
 
    Returns
    -------
    pd.DataFrame
        One row per run.  Columns: controller, fid, func_name, iid, dim,
        run_index, fopt, used_budget, n_restarts, final_lambda, n_phases,
        lambda_history, runtime_s, converged.
    """
    if controller_registry is None:
        controller_registry = CONTROLLER_REGISTRY
 
    os.makedirs(out_root, exist_ok=True)
 
    n_configs = len(fids) * len(iids) * len(dims)
    total_runs = n_configs * len(controller_registry) * n_runs
 
    if verbose:
        print("=" * 75)
        print(f"  BBOB Benchmark")
        print(f"  Controllers : {[c for c in controller_registry]}")
        print(f"  Functions   : {len(fids)} (fids {min(fids)}-{max(fids)})")
        print(f"  Instances   : {iids}")
        print(f"  Dimensions  : {dims}")
        print(f"  Runs/config : {n_runs}  |  Budget: {budget:,}")
        print(f"  Total runs  : {total_runs:,}")
        print(f"  Output      : {os.path.abspath(out_root)}")
        print("=" * 75)
 
    all_results: List = []
 
    for factory in controller_registry:
        if verbose:
            print(f"\n--- {factory.__repr__()} ---")
        results = _run_controller(
            label=factory.__repr__(),
            factory=factory,
            enable_restarts=factory.restart,
            fids=fids,
            iids=iids,
            dims=dims,
            n_runs=n_runs,
            budget=budget,
            sigma0=sigma0,
            convergence_target=convergence_target,
            out_root=out_root,
            seed=seed,
            verbose=verbose,
        )
        all_results.extend(results)
 
    df = _to_dataframe(all_results)
    csv_path = os.path.join(out_root, "summary.csv")
    df.to_csv(csv_path, index=False)
 
    if verbose:
        print(f"\nSummary CSV: {csv_path}")
        _print_summary(df)
 
    return df
 
 
# ===========================================================================
# Helpers
# ===========================================================================
 
def _to_dataframe(results: List) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "controller":     r.controller_name,
            "fid":            r.fid,
            "func_name":      FID_NAME.get(r.fid, f"f{r.fid}"),
            "iid":            r.iid,
            "dim":            r.dim,
            "run_index":      r.run_index,
            "fopt":           r.fopt,
            "used_budget":    r.used_budget,
            "n_restarts":     r.n_restarts,
            "final_lambda":   r.lambda_history[-1] if r.lambda_history else None,
            "n_phases":       len(r.lambda_history),
            "lambda_history": str(r.lambda_history),
            "runtime_s":      r.runtime_s,
            "converged":      r.converged,
        })
    return pd.DataFrame(rows)
 
 
def _print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 75)
    print("OVERALL SUMMARY (averaged across all fids, iids, dims, runs)")
    print("=" * 75)
    summary = (
        df.groupby("controller")
        .agg(
            mean_fopt=("fopt", "mean"),
            success_rate=("converged", "mean"),
            mean_restarts=("n_restarts", "mean"),
            mean_budget_used=("used_budget", "mean"),
            total_runs=("fopt", "count"),
        )
        .reset_index()
    )
    summary["success_rate"] = (summary["success_rate"] * 100).map("{:.1f}%".format)
    summary["mean_fopt"] = summary["mean_fopt"].map("{:.3e}".format)
    summary["mean_budget_used"] = summary["mean_budget_used"].map("{:,.0f}".format)
    summary["mean_restarts"] = summary["mean_restarts"].map("{:.2f}".format)
    print(summary.to_string(index=False))
    print("=" * 75)
 
 
# ===========================================================================
# CLI
# ===========================================================================
 
def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BBOB benchmark: compare CMA-ES population controllers."
    )
    p.add_argument("--fids",   nargs="+", type=int, default=ALL_FIDS,
                   help="BBOB function IDs 1-24 (default: all 24)")
    p.add_argument("--iids",   nargs="+", type=int, default=DEFAULT_IIDS,
                   help="Instance IDs (default: 1 2 3 4 5)")
    p.add_argument("--dims",   nargs="+", type=int, default=DEFAULT_DIMS,
                   help="Dimensions (default: 5 10 20)")
    p.add_argument("--runs",   type=int, default=DEFAULT_N_RUNS,
                   help="Runs per config (default: 5)")
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                   help="Max evaluations per run (default: 100,000)")
    p.add_argument("--out",    default="ioh_results",
                   help="Output directory (default: ioh_results/)")
    p.add_argument("--seed",   type=int, default=BASE_SEED)
    return p.parse_args()
 
 
def main() -> None:
    args = _parse()
    run_benchmark(
        fids=args.fids,
        iids=args.iids,
        dims=args.dims,
        n_runs=args.runs,
        budget=args.budget,
        out_root=args.out,
        seed=args.seed,
        verbose=True,
    )
 
 
if __name__ == "__main__":
    main()
 