"""
plot_results.py
===============
Generates publication-quality comparison figures for the dissertation:
    "Deep Reinforcement Learning for Dynamic Configuration of
     Population Size Parameter in CMA-ES"

All plots compare IPOP-CMA-ES (Auger & Hansen, 2005) against the plain
CMA-ES baseline across the BBOB benchmark suite (Hansen et al., 2021).

Usage
-----
    python plot_results.py --csv path/to/aggregated_results.csv

Outputs (written to ./figures/)
--------------------------------
Fig 1  mean_fopt_heatmap.pdf
    Mean best-found function value (f - f*) per function × controller,
    averaged over instances.  Log-scale colour.  Allows rapid visual
    comparison of solution quality across the full function suite.

Fig 2  success_rate_bar.pdf
    Success rate (fraction of runs achieving f - f* < 1e-8) per function,
    grouped by controller.  Directly compares convergence reliability.

Fig 3  budget_efficiency.pdf
    Mean used budget versus mean fopt scatter (log–log).  Points are
    colour-coded by function group.  Illustrates the trade-off between
    evaluation cost and quality.

Fig 4  restart_profile.pdf
    Mean number of restarts and mean final population size for IPOP,
    broken down by function.  Shows the IPOP growth behaviour described
    by Auger & Hansen (2005).

Fig 5  fopt_boxplot.pdf
    Box-plot of mean_fopt distributions (across instances) per function,
    side-by-side for both controllers, log-scale.  Shows variance as well
    as central tendency.

Fig 6  success_rate_radar.pdf
    Radar (spider) chart of success rates across all eight functions for
    both controllers.  Gives an at-a-glance multi-dimensional profile.

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

import argparse
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Aesthetic constants
# ---------------------------------------------------------------------------

# Dissertation-appropriate palette: muted, print-safe, colourblind-friendly.
# IPOP is teal; CMA-ES baseline is warm red-orange.  Both read in greyscale.
PALETTE: Dict[str, str] = {
    "IPOP":   "#2A7B9B",   # teal
    "CMA-ES": "#D95F02",   # warm orange
}

# BBOB function groups (Hansen et al., 2021)
FUNC_GROUPS: Dict[str, str] = {
    "Sphere":             "Unimodal separable",
    "Ellipsoid":          "Unimodal separable",
    "Rosenbrock":         "Unimodal non-sep.",
    "Rastrigin":          "Multimodal",
    "RastriginRotated":   "Multimodal",
    "GriewankRosenbrock": "Multimodal",
    "Schwefel":           "Multimodal weak",
    "LunacekBiRastrigin": "Multimodal weak",
}

GROUP_COLOURS: Dict[str, str] = {
    "Unimodal separable": "#4dac26",
    "Unimodal non-sep.":  "#b8e186",
    "Multimodal":         "#f1b6da",
    "Multimodal weak":    "#d01c8b",
}

# Ordered function display names (short labels for axes)
FUNC_ORDER: List[str] = [
    "Sphere", "Ellipsoid", "Rosenbrock",
    "Rastrigin", "RastriginRotated", "GriewankRosenbrock",
    "Schwefel", "LunacekBiRastrigin",
]
FUNC_LABELS: Dict[str, str] = {
    "Sphere":             "Sphere",
    "Ellipsoid":          "Ellipsoid",
    "Rosenbrock":         "Rosenbrock",
    "Rastrigin":          "Rastrigin",
    "RastriginRotated":   "Rastrigin\n(Rotated)",
    "GriewankRosenbrock": "Griewank-\nRosenbrock",
    "Schwefel":           "Schwefel",
    "LunacekBiRastrigin": "Lunacek Bi-\nRastrigin",
}

CONTROLLERS: List[str] = ["IPOP", "CMA-ES"]

# Global figure style
mpl.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["DejaVu Serif"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#e0e0e0",
    "grid.linewidth":    0.6,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load and validate the aggregated results CSV.

    Takes absolute values of fopt columns so that functions where the
    optimum shifts the raw values negative are handled uniformly
    (we always plot distance to optimum).

    Parameters
    ----------
    csv_path : str

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(csv_path)

    required = {
        "func_name", "controller", "iid",
        "mean_fopt", "std_fopt", "median_fopt",
        "best_fopt", "worst_fopt",
        "success_rate", "mean_used_budget", "std_used_budget",
        "mean_n_restarts", "mean_initial_lambda", "mean_final_lambda",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # Distance to optimum is always non-negative
    for col in ["mean_fopt", "std_fopt", "median_fopt", "best_fopt", "worst_fopt"]:
        df[col] = df[col].abs()

    # Floor to avoid log(0)
    for col in ["mean_fopt", "median_fopt", "best_fopt", "worst_fopt"]:
        df[col] = df[col].clip(lower=1e-12)

    df["func_group"] = df["func_name"].map(FUNC_GROUPS).fillna("Other")
    return df


def _instance_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Average numeric columns over instance IDs for each (func, controller)."""
    numeric_cols = [
        "mean_fopt", "std_fopt", "median_fopt",
        "best_fopt", "worst_fopt",
        "success_rate", "mean_used_budget", "std_used_budget",
        "mean_n_restarts", "mean_initial_lambda", "mean_final_lambda",
        "mean_runtime_s",
    ]
    return (
        df.groupby(["func_name", "controller"])[numeric_cols]
        .mean()
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Figure 1 – Mean fopt heatmap
# ---------------------------------------------------------------------------

def plot_fopt_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Heatmap of mean best-found function value (log scale) for each
    function × controller combination, averaged over instances.

    Allows rapid identification of which functions benefit most from the
    IPOP restart strategy (Auger & Hansen, 2005).
    """
    agg = _instance_mean(df)

    # Pivot: rows = functions, columns = controllers
    pivot = agg.pivot(index="func_name", columns="controller", values="mean_fopt")
    pivot = pivot.reindex(index=[f for f in FUNC_ORDER if f in pivot.index])
    pivot = pivot.reindex(columns=CONTROLLERS)
    log_vals = np.log10(pivot.values.astype(float) + 1e-12)

    fig, ax = plt.subplots(figsize=(5.5, 5.0))

    vmin, vmax = np.nanmin(log_vals), np.nanmax(log_vals)
    im = ax.imshow(log_vals, aspect="auto", cmap="RdYlGn_r",
                   vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(CONTROLLERS)))
    ax.set_xticklabels(CONTROLLERS, fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([FUNC_LABELS.get(f, f) for f in pivot.index], fontsize=9)
    ax.set_xlabel("Controller", fontsize=10)
    ax.set_title(
        "Mean Best Function Value $f - f^*$ (log$_{10}$)\naveraged over instances",
        fontsize=10, pad=10,
    )

    # Annotate cells
    for i in range(log_vals.shape[0]):
        for j in range(log_vals.shape[1]):
            raw = pivot.values[i, j]
            txt = f"{raw:.2e}" if not np.isnan(raw) else "—"
            colour = "white" if log_vals[i, j] > (vmin + 0.6 * (vmax - vmin)) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                    color=colour)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("$\\log_{10}(f - f^*)$", fontsize=9)

    ax.set_frame_on(False)
    fig.tight_layout()
    _save(fig, out_dir, "fig1_mean_fopt_heatmap")


# ---------------------------------------------------------------------------
# Figure 2 – Success rate bar chart
# ---------------------------------------------------------------------------

def plot_success_rate_bar(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Grouped bar chart of success rate (fraction of runs with f - f* < 1e-8)
    per function for each controller.

    Success rate is the primary performance metric used in the COCO framework
    (Hansen et al., 2021).
    """
    agg = _instance_mean(df)
    funcs = [f for f in FUNC_ORDER if f in agg["func_name"].values]
    n = len(funcs)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(9.0, 4.2))

    for i, ctrl in enumerate(CONTROLLERS):
        sub = agg[agg["controller"] == ctrl].set_index("func_name")
        vals = [sub.loc[f, "success_rate"] * 100 if f in sub.index else 0.0
                for f in funcs]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width,
                      label=ctrl, color=PALETTE[ctrl], alpha=0.88,
                      edgecolor="white", linewidth=0.5)
        # Value labels on bars
        for bar, v in zip(bars, vals):
            if v > 3:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.8, f"{v:.0f}%",
                        ha="center", va="bottom", fontsize=7, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([FUNC_LABELS.get(f, f) for f in funcs], fontsize=8.5)
    ax.set_ylabel("Success Rate (%)", fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_title(
        "Success Rate per Function  ($f - f^* < 10^{-8}$)",
        fontsize=10, pad=10,
    )
    ax.legend(frameon=False)

    # Function group background bands
    _add_group_bands(ax, funcs, y0=0, y1=110, alpha=0.06)

    fig.tight_layout()
    _save(fig, out_dir, "fig2_success_rate_bar")


# ---------------------------------------------------------------------------
# Figure 3 – Budget efficiency scatter
# ---------------------------------------------------------------------------

def plot_budget_efficiency(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Scatter plot of mean used budget vs mean fopt (both log scale),
    with points coloured by function group.

    Reveals whether IPOP's improved solution quality comes at a
    disproportionate evaluation cost.
    """
    agg = _instance_mean(df)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    markers = {"IPOP": "o", "CMA-ES": "s"}
    group_seen = set()

    for _, row in agg.iterrows():
        ctrl = row["controller"]
        fname = row["func_name"]
        group = FUNC_GROUPS.get(fname, "Other")
        gc = GROUP_COLOURS.get(group, "#888888")

        label_group = group if group not in group_seen else None
        group_seen.add(group)

        ax.scatter(
            row["mean_used_budget"], row["mean_fopt"],
            marker=markers[ctrl], color=gc,
            s=70, alpha=0.85, zorder=3,
            label=label_group,
        )
        # Annotate with short function name
        ax.annotate(
            fname[:5], (row["mean_used_budget"], row["mean_fopt"]),
            textcoords="offset points", xytext=(5, 3),
            fontsize=6.5, color="#444444",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean Used Budget (function evaluations)", fontsize=10)
    ax.set_ylabel("Mean $f - f^*$", fontsize=10)
    ax.set_title("Budget vs. Solution Quality", fontsize=10, pad=10)

    # Manual legend: markers for controllers
    ctrl_handles = [
        Line2D([0], [0], marker=m, color="grey", linestyle="None",
               markersize=7, label=c)
        for c, m in markers.items()
    ]
    group_handles = [
        Line2D([0], [0], marker="o", color=GROUP_COLOURS[g],
               linestyle="None", markersize=7, label=g)
        for g in GROUP_COLOURS
        if g in group_seen
    ]
    leg1 = ax.legend(handles=ctrl_handles, loc="upper left",
                     frameon=False, title="Controller", title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=group_handles, loc="lower right",
              frameon=False, title="Function group", title_fontsize=8)

    fig.tight_layout()
    _save(fig, out_dir, "fig3_budget_efficiency")


# ---------------------------------------------------------------------------
# Figure 4 – IPOP restart profile
# ---------------------------------------------------------------------------

def plot_restart_profile(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Dual-axis bar/line chart for IPOP only.

    Left axis: mean number of restarts (bars).
    Right axis: mean final population size lambda (line+markers).

    Visualises the population growth dynamics described by Auger &
    Hansen (2005): lambda doubles with each restart up to max_lambda.
    """
    agg = _instance_mean(df)
    ipop = agg[agg["controller"] == "IPOP"].copy()
    ipop = ipop[ipop["func_name"].isin(FUNC_ORDER)].copy()
    ipop["_order"] = ipop["func_name"].map({f: i for i, f in enumerate(FUNC_ORDER)})
    ipop = ipop.sort_values("_order")
    funcs = ipop["func_name"].tolist()

    x = np.arange(len(funcs))

    fig, ax1 = plt.subplots(figsize=(9.0, 4.5))
    ax2 = ax1.twinx()

    bars = ax1.bar(x, ipop["mean_n_restarts"].values, color=PALETTE["IPOP"],
                   alpha=0.75, width=0.55, label="Mean restarts", zorder=3)
    ax1.set_ylabel("Mean Number of Restarts", color=PALETTE["IPOP"], fontsize=10)
    ax1.tick_params(axis="y", labelcolor=PALETTE["IPOP"])
    ax1.set_ylim(0, ipop["mean_n_restarts"].max() * 1.4)

    line = ax2.plot(x, ipop["mean_final_lambda"].values,
                    color="#E5A020", marker="D", linewidth=1.8,
                    markersize=7, label="Mean final $\\lambda$", zorder=4)
    ax2.set_ylabel("Mean Final Population Size $\\lambda$",
                   color="#E5A020", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="#E5A020")
    ax2.set_ylim(0, ipop["mean_final_lambda"].max() * 1.4)

    ax1.set_xticks(x)
    ax1.set_xticklabels([FUNC_LABELS.get(f, f) for f in funcs], fontsize=8.5)
    ax1.set_title(
        "IPOP Restart Profile: Restarts and Final Population Size per Function",
        fontsize=10, pad=10,
    )
    ax1.grid(axis="y", color="#e0e0e0", linewidth=0.6, zorder=0)
    ax1.set_axisbelow(True)

    # Combined legend
    handles = bars.patches[:1] + line
    labels  = ["Mean restarts", "Mean final $\\lambda$"]
    ax1.legend(handles, labels, loc="upper left", frameon=False)

    # Annotate restart counts above bars
    for bar, v in zip(bars, ipop["mean_n_restarts"].values):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.1, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=8, color=PALETTE["IPOP"])

    fig.tight_layout()
    _save(fig, out_dir, "fig4_restart_profile")


# ---------------------------------------------------------------------------
# Figure 5 – fopt box-plot
# ---------------------------------------------------------------------------

def plot_fopt_boxplot(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Box-plot of mean_fopt across instances for each (function, controller),
    log-scale y-axis.

    Shows both central tendency and spread (variance across instances),
    complementing the heatmap in Figure 1.
    """
    # Build per-function, per-controller distributions across instances
    funcs = [f for f in FUNC_ORDER if f in df["func_name"].unique()]
    n = len(funcs)
    x = np.arange(n)
    width = 0.32

    fig, ax = plt.subplots(figsize=(10.0, 4.8))

    for i, ctrl in enumerate(CONTROLLERS):
        sub = df[df["controller"] == ctrl]
        # Collect fopt values per function across instances
        data = [
            sub[sub["func_name"] == f]["mean_fopt"].values
            for f in funcs
        ]
        positions = x + (i - 0.5) * width
        bp = ax.boxplot(
            data, positions=positions, widths=width * 0.85,
            patch_artist=True,
            boxprops=dict(facecolor=PALETTE[ctrl], alpha=0.7, linewidth=0.8),
            medianprops=dict(color="white", linewidth=1.5),
            whiskerprops=dict(color=PALETTE[ctrl], linewidth=0.9),
            capprops=dict(color=PALETTE[ctrl], linewidth=0.9),
            flierprops=dict(marker="x", color=PALETTE[ctrl],
                            markersize=4, linewidth=0.7),
            zorder=3,
        )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([FUNC_LABELS.get(f, f) for f in funcs], fontsize=8.5)
    ax.set_ylabel("$f - f^*$  (log scale)", fontsize=10)
    ax.set_title(
        "Distribution of Mean Best Function Value Across Instances",
        fontsize=10, pad=10,
    )

    legend_handles = [
        mpl.patches.Patch(facecolor=PALETTE[c], alpha=0.7, label=c)
        for c in CONTROLLERS
    ]
    ax.legend(handles=legend_handles, frameon=False)
    _add_group_bands(ax, funcs, y0=ax.get_ylim()[0],
                     y1=ax.get_ylim()[1], alpha=0.06)
    fig.tight_layout()
    _save(fig, out_dir, "fig5_fopt_boxplot")


# ---------------------------------------------------------------------------
# Figure 6 – Radar / spider chart
# ---------------------------------------------------------------------------

def plot_success_rate_radar(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Radar chart of success rates across all benchmark functions for each
    controller.

    Provides a compact multi-dimensional view of algorithm performance,
    useful for summarising dissertation results.
    """
    agg = _instance_mean(df)
    funcs = [f for f in FUNC_ORDER if f in agg["func_name"].values]
    n = len(funcs)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(6.0, 6.0),
                           subplot_kw=dict(polar=True))

    for ctrl in CONTROLLERS:
        sub = agg[agg["controller"] == ctrl].set_index("func_name")
        vals = [
            sub.loc[f, "success_rate"] * 100 if f in sub.index else 0.0
            for f in funcs
        ]
        vals += vals[:1]
        ax.plot(angles, vals, "-o", color=PALETTE[ctrl],
                linewidth=2.0, markersize=5, label=ctrl)
        ax.fill(angles, vals, color=PALETTE[ctrl], alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([FUNC_LABELS.get(f, f) for f in funcs], fontsize=8.5)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=7.5)
    ax.set_ylim(0, 100)
    ax.set_title(
        "Success Rate Profile  ($f - f^* < 10^{-8}$)",
        fontsize=10, pad=20,
    )
    ax.legend(loc="lower right", bbox_to_anchor=(1.3, -0.05), frameon=False)

    fig.tight_layout()
    _save(fig, out_dir, "fig6_success_rate_radar")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_group_bands(
    ax: plt.Axes,
    funcs: List[str],
    y0: float,
    y1: float,
    alpha: float = 0.08,
) -> None:
    """Shade alternating function-group regions behind a bar/box chart."""
    groups = [FUNC_GROUPS.get(f, "Other") for f in funcs]
    prev_group, start = groups[0], 0
    for i, g in enumerate(groups[1:] + [None], start=1):
        if g != prev_group:
            colour = GROUP_COLOURS.get(prev_group, "#cccccc")
            ax.axvspan(start - 0.5, i - 0.5, ymin=0, ymax=1,
                       color=colour, alpha=alpha, zorder=0)
            prev_group, start = g, i


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    """Save figure to PDF and PNG."""
    for ext in ("pdf", "png"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {stem}.pdf / .png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate dissertation comparison plots for IPOP vs CMA-ES."
    )
    p.add_argument(
        "--csv",
        default="results/aggregated_results.csv",
        help="Path to the aggregated_results.csv produced by benchmark.py",
    )
    p.add_argument(
        "--out",
        default="figures",
        help="Output directory for figures (default: ./figures/)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {args.csv}")
    df = load_data(args.csv)
    print(f"  {len(df)} rows  |  controllers: {df['controller'].unique().tolist()}")
    print(f"  functions: {df['func_name'].unique().tolist()}")
    print(f"\nGenerating figures → {out_dir}/\n")

    plot_fopt_heatmap(df, out_dir)
    plot_success_rate_bar(df, out_dir)
    plot_budget_efficiency(df, out_dir)
    plot_restart_profile(df, out_dir)
    plot_fopt_boxplot(df, out_dir)
    plot_success_rate_radar(df, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()