from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_METRIC_COLUMNS = [
    "ts",
    "epoch",
    "alpha",
    "spectral_loss_coeff",
    "F_EV",
    "R_EV",
    "BPI",
    "linear_probe_top1",
    "knn_top1",
    "seed",
    "save_note",
    "version",
]
NUMERIC_COLUMNS = [
    "alpha",
    "spectral_loss_coeff",
    "F_EV",
    "R_EV",
    "BPI",
    "linear_probe_top1",
    "knn_top1",
    "seed",
    "epoch",
    "ts",
]
TREND_SPECS = [
    ("F_EV", "Forward EV vs. Measured α", "Forward Explained Variance (%)", "#2c7fb8"),
    ("R_EV", "Reverse EV vs. Measured α", "Reverse Explained Variance (%)", "#1f1f1f"),
    ("linear_probe_top1", "Linear Probe vs. Measured α", "Linear Probe Top-1 Accuracy (%)", "#8c4c2f"),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    add = ap.add_argument
    add("--roots", type=Path, nargs="+", required=True, help="Directories containing start_*/logs/simclr_baseline.csv trees.")
    add("--out-dir", type=Path, default=Path("figures/neuro_alignment_tradeoffs"), help="Where to write figures and summary CSVs.")
    add(
        "--baseline-roots",
        type=Path,
        nargs="*",
        default=None,
        help="Optional separate root(s) containing baseline runs. These runs are force-labeled baseline.",
    )
    add(
        "--alpha-bin-edges",
        type=float,
        nargs="*",
        default=None,
        help="Optional explicit measured-alpha bin edges. If omitted, quantile bins are used for the summary CSV.",
    )
    add("--n-alpha-bins", type=int, default=5, help="Number of quantile bins when explicit alpha-bin edges are not supplied.")
    add("--smooth-bandwidth", type=float, default=None, help="Bandwidth for measured-alpha smoothing. Defaults to a range-based heuristic.")
    add("--trend-grid-points", type=int, default=200, help="Number of x-grid points used for smoothed trends.")
    add("--bootstrap-samples", type=int, default=400, help="Number of bootstrap resamples for confidence ribbons.")
    add("--bootstrap-seed", type=int, default=0, help="Random seed for bootstrap ribbons.")
    add("--alpha-min", type=float, default=None, help="Optional lower measured-alpha filter.")
    add("--alpha-max", type=float, default=None, help="Optional upper measured-alpha filter.")
    add("--baseline-coeff-max", type=float, default=1e-8, help="Runs with spectral_loss_coeff <= this value are treated as baseline.")
    add("--run-row-policy", choices=["last_informative", "last_nonzero_fev", "final_row"], default="last_informative", help="How to choose the single run-level row from each CSV.")
    add("--figure-dpi", type=int, default=300, help="PNG export DPI.")
    add("--no-pdf", action="store_true", help="Skip PDF export and only write PNG files.")
    return ap.parse_args()


def get_berlin_cmap():
    try:
        import cmcrameri.cm as cmc

        return cmc.berlin, "berlin"
    except Exception as exc:
        warnings.warn(
            "cmcrameri is not installed, so berlin is unavailable. "
            "Falling back to matplotlib's RdBu_r. "
            f"Install cmcrameri for exact berlin support. Details: {exc}"
        )
        return mpl.colormaps["RdBu_r"], "RdBu_r"


def set_paper_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.5,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def find_log_csvs(roots: Iterable[Path]) -> list[Path]:
    csvs: list[Path] = []
    for root in roots:
        if not root.exists():
            warnings.warn(f"Skipping missing root: {root}")
            continue
        csvs.extend(sorted(root.rglob("logs/simclr_baseline.csv")))
    return csvs


def coerce_numeric(df: pd.DataFrame, columns: Iterable[str] = NUMERIC_COLUMNS) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def select_run_row(df: pd.DataFrame, policy: str) -> pd.Series | None:
    if df.empty:
        return None
    work = coerce_numeric(df)
    if policy == "final_row":
        return work.iloc[-1]
    if policy == "last_nonzero_fev":
        return work.loc[work["F_EV"].fillna(0) != 0].iloc[-1] if "F_EV" in work and (work["F_EV"].fillna(0) != 0).any() else None
    required = {"alpha", "F_EV", "R_EV", "linear_probe_top1"}
    if not required.issubset(work.columns):
        return None
    mask = work[list(required)].notna().all(axis=1) & work["F_EV"].fillna(0).ne(0) & work["R_EV"].fillna(0).ne(0)
    informative = work.loc[mask]
    return informative.iloc[-1] if not informative.empty else None


def summarize_run(csv_path: Path, policy: str, force_baseline: bool = False) -> dict | None:
    try:
        row = select_run_row(pd.read_csv(csv_path), policy)
    except Exception as exc:
        warnings.warn(f"Failed to read {csv_path}: {exc}")
        return None
    if row is None:
        warnings.warn(f"No informative row found in {csv_path}")
        return None
    run_dir = csv_path.parent.parent
    record = {k: row.get(k, np.nan) for k in DEFAULT_METRIC_COLUMNS if k in row.index}
    record.update(csv_path=str(csv_path), run_dir=str(run_dir), run_name=run_dir.name, force_baseline=bool(force_baseline))
    return record


def load_runs(roots: Iterable[Path], policy: str, force_baseline: bool = False) -> pd.DataFrame:
    rows = [record for csv_path in find_log_csvs(roots) if (record := summarize_run(csv_path, policy, force_baseline)) is not None]
    if not rows:
        raise RuntimeError("No valid simclr_baseline.csv run summaries were found.")
    df = coerce_numeric(pd.DataFrame(rows))
    if "ts" in df.columns:
        df["datetime"] = pd.to_datetime(df["ts"], unit="s", utc=True, errors="coerce")
    return df


def apply_alpha_filter(df: pd.DataFrame, alpha_min: float | None, alpha_max: float | None) -> pd.DataFrame:
    out = df.copy()
    if alpha_min is not None:
        out = out.loc[out["alpha"] >= alpha_min]
    if alpha_max is not None:
        out = out.loc[out["alpha"] <= alpha_max]
    return out.reset_index(drop=True)


def pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    work = df[[x_col, y_col]].dropna().sort_values([x_col, y_col], ascending=[False, False])
    if work.empty:
        return work
    frontier, best_y = [], -np.inf
    for x_val, y_val in work.to_numpy():
        if y_val > best_y:
            frontier.append((x_val, y_val))
            best_y = y_val
    return pd.DataFrame(frontier, columns=[x_col, y_col]).sort_values(x_col)


def compute_bin_edges(alpha: pd.Series, explicit_edges: list[float] | None, n_bins: int) -> np.ndarray:
    if explicit_edges:
        edges = np.unique(np.asarray(explicit_edges, dtype=float))
        if len(edges) < 2:
            raise ValueError("Need at least two explicit alpha-bin edges.")
        return edges
    edges = np.unique(np.quantile(alpha.dropna().to_numpy(), np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        raise ValueError("Measured alpha has too few unique values for quantile bins. Supply explicit alpha-bin edges instead.")
    return edges


def label_bin_interval(interval: pd.Interval, n: int) -> str:
    return f"{interval.left:.2f}–{interval.right:.2f}\n(n={n})"


def mean_ci(values: pd.Series, confidence: float = 0.95) -> tuple[float, float, float]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, mean, mean
    lo, hi = stats.t.interval(confidence, df=arr.size - 1, loc=mean, scale=stats.sem(arr, nan_policy="omit"))
    return mean, float(lo), float(hi)


def baseline_summary(values: pd.Series) -> tuple[float, float, float] | None:
    mean, lo, hi = mean_ci(values)
    return None if np.isnan(mean) else (mean, lo, hi)


def make_binned_summary(df: pd.DataFrame, alpha_bin_edges: np.ndarray) -> pd.DataFrame:
    work = df.assign(alpha_bin=pd.cut(df["alpha"], bins=alpha_bin_edges, include_lowest=True)).dropna(subset=["alpha_bin"])
    rows = []
    for alpha_bin, group in work.groupby("alpha_bin", observed=True):
        row = {
            "alpha_bin": alpha_bin,
            "alpha_bin_label": label_bin_interval(alpha_bin, len(group)),
            "n": len(group),
            "alpha_mean": float(group["alpha"].mean()),
        }
        for metric in [spec[0] for spec in TREND_SPECS]:
            mean, lo, hi = mean_ci(group[metric])
            row |= {f"{metric}_mean": mean, f"{metric}_ci_lo": lo, f"{metric}_ci_hi": hi}
        rows.append(row)
    return pd.DataFrame(rows).sort_values("alpha_mean").reset_index(drop=True)


def default_smooth_bandwidth(alpha: pd.Series) -> float:
    a = alpha.dropna().to_numpy(dtype=float)
    return 0.1 if a.size < 2 else max(0.08, 0.18 * float(np.max(a) - np.min(a)))


def kernel_smooth(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray, bandwidth: float) -> np.ndarray:
    weights = np.exp(-0.5 * ((x[:, None] - x_grid[None, :]) / bandwidth) ** 2)
    denom = weights.sum(axis=0)
    return np.divide((weights * y[:, None]).sum(axis=0), denom, out=np.full_like(x_grid, np.nan, dtype=float), where=denom > 0)


def bootstrap_smooth_ci(x: pd.Series, y: pd.Series, bandwidth: float, n_grid: int, n_boot: int, seed: int) -> pd.DataFrame:
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(data) < 3:
        raise ValueError("Need at least 3 non-missing points to compute a smooth trend.")
    x_vals, y_vals = data["x"].to_numpy(dtype=float), data["y"].to_numpy(dtype=float)
    x_grid = np.linspace(np.min(x_vals), np.max(x_vals), n_grid)
    y_hat = kernel_smooth(x_vals, y_vals, x_grid, bandwidth)
    rng, n = np.random.default_rng(seed), len(data)
    boot_curves = np.empty((n_boot, n_grid), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_curves[b] = kernel_smooth(x_vals[idx], y_vals[idx], x_grid, bandwidth)
    return pd.DataFrame(
        {
            "alpha_grid": x_grid,
            "y_hat": y_hat,
            "ci_lo": np.nanpercentile(boot_curves, 2.5, axis=0),
            "ci_hi": np.nanpercentile(boot_curves, 97.5, axis=0),
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, dpi: int, no_pdf: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    if not no_pdf:
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")


def add_baseline_band(ax: plt.Axes, values: pd.Series, line_color: str = "#7f7f7f", band_color: str = "#bcbcbc") -> None:
    stats_tuple = baseline_summary(values)
    if stats_tuple is None:
        return
    mean, lo, hi = stats_tuple
    ax.axhline(mean, color=line_color, linewidth=1.2, linestyle="--", zorder=0)
    if hi > lo:
        ax.axhspan(lo, hi, color=band_color, alpha=0.18, zorder=0)


def plot_raw_plus_smooth(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    color: str,
    bandwidth: float,
    n_grid: int,
    n_boot: int,
    seed: int,
    baseline_values: pd.Series | None = None,
) -> None:
    ax.scatter(df["alpha"], df[metric], color="#2d2d2d", s=34, alpha=0.55, edgecolors="none", zorder=2)
    trend = bootstrap_smooth_ci(df["alpha"], df[metric], bandwidth=bandwidth, n_grid=n_grid, n_boot=n_boot, seed=seed)
    ax.fill_between(trend["alpha_grid"], trend["ci_lo"], trend["ci_hi"], color=color, alpha=0.18, zorder=1)
    ax.plot(trend["alpha_grid"], trend["y_hat"], color=color, linewidth=2.0, zorder=3)
    if baseline_values is not None:
        add_baseline_band(ax, baseline_values)


def plot_forward_vs_reverse(
    df: pd.DataFrame,
    baseline_mask: pd.Series,
    out_dir: Path,
    cmap,
    dpi: int,
    no_pdf: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sc = ax.scatter(
        df["F_EV"],
        df["R_EV"],
        c=df["alpha"],
        cmap=cmap,
        norm=mpl.colors.Normalize(vmin=df["alpha"].min(), vmax=df["alpha"].max()),
        s=54,
        alpha=0.92,
        edgecolors="white",
        linewidths=0.7,
        zorder=3,
    )
    frontier = pareto_frontier(df, "F_EV", "R_EV")
    if not frontier.empty:
        ax.plot(frontier["F_EV"], frontier["R_EV"], color="#3a3a3a", linewidth=1.3, linestyle="--", label="Pareto frontier", zorder=2)
        ax.scatter(frontier["F_EV"], frontier["R_EV"], s=42, color="#3a3a3a", edgecolors="white", linewidths=0.5, zorder=3)
    if baseline_mask.any():
        base = df.loc[baseline_mask]
        ax.scatter(base["F_EV"], base["R_EV"], facecolors="none", edgecolors="black", s=90, linewidths=1.2, label=f"Baseline runs (n={len(base)})", zorder=4)
        ax.scatter([base["F_EV"].mean()], [base["R_EV"].mean()], marker="*", s=300, color="gold", edgecolors="black", linewidths=1.0, label="Baseline mean", zorder=5)
    fig.colorbar(sc, ax=ax, pad=0.02).set_label("Measured α")
    ax.set(xlabel="Forward Explained Variance (%)", ylabel="Reverse Explained Variance (%)", title="Forward–Reverse Alignment Tradeoff")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=True, loc="best")
    save_figure(fig, out_dir, "forward_vs_reverse_by_alpha", dpi, no_pdf)
    plt.close(fig)


def plot_linear_probe_vs_alpha(
    df: pd.DataFrame,
    baseline_mask: pd.Series,
    out_dir: Path,
    bandwidth: float,
    n_grid: int,
    n_boot: int,
    seed: int,
    dpi: int,
    no_pdf: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.scatter(df["alpha"], df["linear_probe_top1"], color="#2d2d2d", s=54, alpha=0.78, edgecolors="white", linewidths=0.7, zorder=3)
    if baseline_mask.any():
        base = df.loc[baseline_mask]
        ax.scatter(base["alpha"], base["linear_probe_top1"], facecolors="none", edgecolors="black", s=90, linewidths=1.2, label=f"Baseline runs (n={len(base)})", zorder=4)
        ax.scatter([base["alpha"].mean()], [base["linear_probe_top1"].mean()], marker="*", s=300, color="gold", edgecolors="black", linewidths=1.0, label="Baseline mean", zorder=5)
        add_baseline_band(ax, base["linear_probe_top1"], line_color="#7f7f7f")
    trend = bootstrap_smooth_ci(df["alpha"], df["linear_probe_top1"], bandwidth=bandwidth, n_grid=n_grid, n_boot=n_boot, seed=seed)
    ax.fill_between(trend["alpha_grid"], trend["ci_lo"], trend["ci_hi"], color="#7a4a34", alpha=0.18, zorder=1)
    ax.plot(trend["alpha_grid"], trend["y_hat"], color="#7a4a34", linewidth=2.0, zorder=2, label="Smoothed trend")
    ax.set(xlabel="Measured α", ylabel="Linear Probe Top-1 Accuracy (%)", title="Linear Probe Accuracy vs. Measured α")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=True, loc="best")
    save_figure(fig, out_dir, "linear_probe_vs_alpha", dpi, no_pdf)
    plt.close(fig)


def plot_alpha_metric_trends(
    df: pd.DataFrame,
    baseline_mask: pd.Series,
    out_dir: Path,
    bandwidth: float,
    n_grid: int,
    n_boot: int,
    seed: int,
    dpi: int,
    no_pdf: bool,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6), sharex=True)
    base_df = df.loc[baseline_mask]
    for ax, (metric, title, ylabel, color) in zip(axes, TREND_SPECS):
        plot_raw_plus_smooth(
            ax,
            df,
            metric=metric,
            color=color,
            bandwidth=bandwidth,
            n_grid=n_grid,
            n_boot=n_boot,
            seed=seed,
            baseline_values=None if base_df.empty else base_df[metric],
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Measured α")
    fig.suptitle("Metric Trends Across Measured α (raw runs + smoothed trend + 95% bootstrap CI)", y=1.02)
    fig.tight_layout()
    save_figure(fig, out_dir, "alpha_metric_trend_summary", dpi, no_pdf)
    plt.close(fig)


def prepare_dataframe(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [load_runs(args.roots, args.run_row_policy, force_baseline=False)]
    if args.baseline_roots:
        frames.append(load_runs(args.baseline_roots, args.run_row_policy, force_baseline=True))
    df = apply_alpha_filter(pd.concat(frames, ignore_index=True), args.alpha_min, args.alpha_max)
    if df.empty:
        raise RuntimeError("No runs remain after alpha filtering.")
    df = df.sort_values(["alpha", "F_EV", "R_EV"]).reset_index(drop=True)
    df["is_baseline"] = df["force_baseline"].fillna(False).astype(bool) | df["spectral_loss_coeff"].fillna(np.inf).le(args.baseline_coeff_max)
    summary_df = make_binned_summary(df, compute_bin_edges(df["alpha"], args.alpha_bin_edges, args.n_alpha_bins))
    return df, summary_df


def main() -> None:
    args = parse_args()
    set_paper_style()
    df, summary_df = prepare_dataframe(args)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "run_level_metrics.csv", index=False)
    summary_df.to_csv(out_dir / "alpha_binned_summary.csv", index=False)

    cmap, cmap_name = get_berlin_cmap()
    bandwidth = args.smooth_bandwidth or default_smooth_bandwidth(df["alpha"])
    print(f"Using colormap: {cmap_name}")
    print(f"Loaded {len(df)} runs from {len(args.roots)} root(s).")
    print(f"Baseline runs detected: {int(df['is_baseline'].sum())}")
    print(f"Using smoothing bandwidth: {bandwidth:.4f}")

    plot_forward_vs_reverse(df, df["is_baseline"], out_dir, cmap, args.figure_dpi, args.no_pdf)
    plot_linear_probe_vs_alpha(df, df["is_baseline"], out_dir, bandwidth, args.trend_grid_points, args.bootstrap_samples, args.bootstrap_seed, args.figure_dpi, args.no_pdf)
    plot_alpha_metric_trends(df, df["is_baseline"], out_dir, bandwidth, args.trend_grid_points, args.bootstrap_samples, args.bootstrap_seed, args.figure_dpi, args.no_pdf)


if __name__ == "__main__":
    main()
