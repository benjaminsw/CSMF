# =============================================================================
# Version: DIAG-REORG-PlotUtils-v1.0 | Abbr: PU
# Description: Shared plot primitives for CSMF diagnostic scripts.
#              Extracted from EXP-SANITY v1.1, FI-DIAG v1.5, SC-DIAG v1.1.
#              Called by SA-DIAG, SB-DIAG, SC-DIAG. All functions are
#              stateless — no shared state. Each function saves its own figure
#              via save_figure() and returns bool (True=success). All errors
#              are logged; functions never raise. Callers decide whether a
#              False return is fatal.
# Changelog:
#   v1.0 (2026-04-04): Initial implementation — 10 plot primitives extracted
#                      from existing diagnostic scripts; save_figure() is the
#                      single save path for all functions; plot_epoch_lines
#                      uses "val" key prefix convention for dashed styling;
#                      plot_reconstruction_grid handles both multi-expert
#                      (rows=samples, cols=y|x_clean|x̂_k) and 2-row layouts;
#                      plot_pairwise_scatter and plot_reconstruction_snapshots
#                      added as additional functions beyond the core 7
# Dependencies: matplotlib, numpy, torch (for tensor squeeze/numpy conversion)
# =============================================================================

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

logger = logging.getLogger(__name__)

# Colour palette used across B-vs-C comparisons (consistent with SC-DIAG v1.1)
_COLOR_B    = "#5DA5DA"
_COLOR_C    = "#FAA43A"
_COLOR_CMAP = "Set2"

# FI status colour map (consistent with FI-DIAG v1.5)
_FI_STATUS_COLORS = {
    "alive":      "#4CAF50",
    "warn_low":   "#FF9800",
    "fatal_dead": "#F44336",
    "no_data":    "#9E9E9E",
}


# =============================================================================
# 1. save_figure — single save path for all plot functions
# =============================================================================

def save_figure(fig, output_path: str) -> bool:
    """
    Save fig to output_path at 150 dpi, then close it.

    Args:
        fig         : matplotlib Figure object.
        output_path : Full path including filename (e.g. /results/plot.png).

    Returns:
        True on success, False on failure (error logged).
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"PU | saved: {output_path}")
        return True
    except Exception as e:
        logger.error(f"PU | save_figure failed for {output_path}: {e}")
        return False
    finally:
        plt.close(fig)


# =============================================================================
# 2. plot_epoch_lines — generic multi-line epoch plot
# Source: EXP-SANITY _plot_nll_over_epochs, _plot_inv_err_over_epochs;
#         SC-DIAG _plot_loss_curves, _plot_gate_weights_over_epochs,
#                 _plot_residual_over_epochs;
#         FI-DIAG _plot_fi_per_epoch
# =============================================================================

def plot_epoch_lines(
    data_dict: Dict[str, List[float]],
    output_path: str,
    title: str,
    xlabel: str = "Epoch",
    ylabel: str = "",
    hlines: Optional[List[Tuple[float, str, str]]] = None,
    log_scale: bool = False,
    nan_safe: bool = True,
) -> bool:
    """
    Multi-line epoch plot. Each key in data_dict becomes one line.

    Styling convention (applied automatically):
        - Keys containing "val" → dashed line with square markers
        - All other keys       → solid line with circle markers

    Args:
        data_dict   : {label: [val_epoch_0, val_epoch_1, ...]}
        output_path : Full save path.
        title       : Figure title.
        xlabel      : X-axis label (default: "Epoch").
        ylabel      : Y-axis label.
        hlines      : Optional horizontal reference lines as
                      [(y_value, label, color), ...].
        log_scale   : If True, y-axis is log scale.
        nan_safe    : If True, NaN values are skipped in plotting
                      (epoch alignment preserved via masked array).

    Returns:
        True on success, False on failure.

    Source: EXP-SANITY _plot_nll_over_epochs / _plot_inv_err_over_epochs,
            SC-DIAG _plot_loss_curves / _plot_gate_weights_over_epochs /
            _plot_residual_over_epochs, FI-DIAG _plot_fi_per_epoch.
    """
    try:
        if not data_dict:
            logger.warning(f"PU | plot_epoch_lines: empty data_dict for {output_path}")
            return False

        # Filter out completely empty series
        non_empty = {k: v for k, v in data_dict.items() if v}
        if not non_empty:
            logger.warning(
                f"PU | plot_epoch_lines: all series empty for {output_path}"
            )
            return False

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = plt.cm.tab10.colors

        for idx, (label, values) in enumerate(non_empty.items()):
            color      = colors[idx % len(colors)]
            is_val     = "val" in label.lower()
            linestyle  = "--" if is_val else "-"
            marker     = "s"  if is_val else "o"
            epochs     = list(range(1, len(values) + 1))

            if nan_safe:
                vals_arr = np.array(values, dtype=float)
                mask     = np.isfinite(vals_arr)
                if not mask.any():
                    logger.warning(
                        f"PU | plot_epoch_lines: all NaN for '{label}' — skipping line"
                    )
                    continue
                ax.plot(
                    np.array(epochs)[mask],
                    vals_arr[mask],
                    label=label, color=color,
                    linestyle=linestyle, marker=marker, markersize=3, linewidth=1.5,
                )
            else:
                ax.plot(
                    epochs, values,
                    label=label, color=color,
                    linestyle=linestyle, marker=marker, markersize=3, linewidth=1.5,
                )

        if hlines:
            for y_val, h_label, h_color in hlines:
                ax.axhline(
                    y=y_val, color=h_color, linestyle=":", linewidth=1.2,
                    alpha=0.7, label=h_label,
                )

        if log_scale:
            ax.set_yscale("log")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return save_figure(fig, output_path)

    except Exception as e:
        logger.error(f"PU | plot_epoch_lines failed for {output_path}: {e}")
        plt.close("all")
        return False


# =============================================================================
# 3. plot_histogram — generic histogram with optional N(0,1) reference
# Source: EXP-SANITY _plot_z_histograms (z distribution);
#         SC-DIAG _plot_residual_dist (residual distribution)
# =============================================================================

def plot_histogram(
    data: np.ndarray,
    output_path: str,
    title: str,
    xlabel: str = "",
    ylabel: str = "Density",
    ref_gaussian: bool = False,
    vline_mean: bool = True,
    bins: int = 60,
    xlim: Optional[Tuple[float, float]] = None,
) -> bool:
    """
    Single-panel histogram with optional N(0,1) reference overlay.

    Args:
        data         : 1-D numpy array of values.
        output_path  : Full save path.
        title        : Figure title.
        xlabel       : X-axis label.
        ylabel       : Y-axis label (default: "Density").
        ref_gaussian : If True, overlay N(0,1) PDF (for z histograms).
        vline_mean   : If True, draw vertical line at mean.
        bins         : Number of histogram bins.
        xlim         : Optional (xmin, xmax) axis limits.

    Returns:
        True on success, False on failure.

    Source: EXP-SANITY _plot_z_histograms (ref_gaussian=True),
            SC-DIAG _plot_residual_dist (ref_gaussian=False, vline_mean=True).
    """
    try:
        data = np.asarray(data, dtype=float)
        data = data[np.isfinite(data)]

        if len(data) == 0:
            logger.warning(
                f"PU | plot_histogram: no finite data for {output_path}"
            )
            return False

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(data, bins=bins, density=True, alpha=0.7,
                edgecolor="black", linewidth=0.4, color="#4C72B0")

        if ref_gaussian:
            x_ref = np.linspace(-4, 4, 300)
            y_ref = np.exp(-x_ref ** 2 / 2) / np.sqrt(2 * np.pi)
            ax.plot(x_ref, y_ref, "k--", linewidth=1.5, label="N(0,1)")
            ax.legend(fontsize=8)

        if vline_mean:
            mu = data.mean()
            ax.axvline(
                x=mu, color="red", linestyle="--", linewidth=1.2,
                label=f"Mean={mu:.4f}",
            )
            if not ref_gaussian:
                ax.legend(fontsize=8)

        if xlim is not None:
            ax.set_xlim(*xlim)

        stats_str = f"μ={data.mean():.3f}  σ={data.std():.3f}  n={len(data)}"
        ax.set_xlabel(f"{xlabel}\n{stats_str}" if xlabel else stats_str)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return save_figure(fig, output_path)

    except Exception as e:
        logger.error(f"PU | plot_histogram failed for {output_path}: {e}")
        plt.close("all")
        return False


# =============================================================================
# 4. plot_scatter — single-panel scatter (annotated or dense)
# Source: FI-DIAG _plot_fi_vs_nll_scatter (annotated, per-expert)
# =============================================================================

def plot_scatter(
    x: np.ndarray,
    y: np.ndarray,
    output_path: str,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    point_labels: Optional[List[str]] = None,
    diagonal: bool = False,
    alpha: float = 0.15,
    s: float = 8,
) -> bool:
    """
    Single-panel scatter plot.

    Two modes:
        - Annotated (point_labels provided): few points, each labelled.
          Used for FI vs NLL (one point per expert).
        - Dense (point_labels=None): many unlabelled points.
          Used for pairwise NLL (one point per val sample) — use
          plot_pairwise_scatter for multi-panel pairwise layouts.

    Args:
        x            : 1-D array, x-axis values.
        y            : 1-D array, y-axis values (same length as x).
        output_path  : Full save path.
        title        : Figure / panel title.
        xlabel       : X-axis label.
        ylabel       : Y-axis label.
        point_labels : If provided, annotate each point with the label string.
        diagonal     : If True, draw identity line y=x (for pairwise NLL).
        alpha        : Point transparency (dense mode).
        s            : Point size (dense mode).

    Returns:
        True on success, False on failure.

    Source: FI-DIAG _plot_fi_vs_nll_scatter (annotated mode).
    """
    try:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if len(x) == 0 or len(y) == 0:
            logger.warning(f"PU | plot_scatter: empty arrays for {output_path}")
            return False

        if len(x) != len(y):
            logger.error(
                f"PU | plot_scatter: x/y length mismatch ({len(x)} vs {len(y)}) "
                f"for {output_path}"
            )
            return False

        annotated = point_labels is not None
        fig_size  = (6, 5) if annotated else (5, 5)
        fig, ax   = plt.subplots(figsize=fig_size)

        if annotated:
            # One point per expert — larger markers, labelled
            for i, (xi, yi) in enumerate(zip(x, y)):
                if not (np.isfinite(xi) and np.isfinite(yi)):
                    logger.warning(
                        f"PU | plot_scatter: non-finite point {i} "
                        f"({point_labels[i] if i < len(point_labels) else i}) — skipping"
                    )
                    continue
                ax.scatter(xi, yi, s=80, zorder=5)
                label = point_labels[i] if i < len(point_labels) else str(i)
                ax.annotate(
                    label, (xi, yi),
                    textcoords="offset points", xytext=(6, 4), fontsize=9,
                )
        else:
            # Dense scatter — filter non-finite
            mask = np.isfinite(x) & np.isfinite(y)
            ax.scatter(
                x[mask], y[mask],
                alpha=alpha, s=s, edgecolors="none", color="#4C72B0",
            )

        if diagonal:
            finite_vals = np.concatenate([x[np.isfinite(x)], y[np.isfinite(y)]])
            if len(finite_vals) > 0:
                p5, p95 = np.percentile(finite_vals, [5, 95])
                ax.plot([p5, p95], [p5, p95], "r--", alpha=0.5, linewidth=1)

        if not annotated and len(x[np.isfinite(x)]) > 1 and len(y[np.isfinite(y)]) > 1:
            xi_f = x[np.isfinite(x) & np.isfinite(y)]
            yi_f = y[np.isfinite(x) & np.isfinite(y)]
            if np.std(xi_f) > 1e-8 and np.std(yi_f) > 1e-8:
                corr = np.corrcoef(xi_f, yi_f)[0, 1]
                ax.set_title(f"{title}\nρ = {corr:.3f}")
            else:
                ax.set_title(title)
        else:
            ax.set_title(title)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return save_figure(fig, output_path)

    except Exception as e:
        logger.error(f"PU | plot_scatter failed for {output_path}: {e}")
        plt.close("all")
        return False


# =============================================================================
# 5. plot_pairwise_scatter — multi-panel pairwise scatter (one panel per pair)
# Source: EXP-SANITY _plot_pairwise_nll_scatter;
#         SC-DIAG _plot_pairwise_nll_comparison;
#         FI-DIAG _plot_fi_expert_scatter
# =============================================================================

def plot_pairwise_scatter(
    pairs_data: List[Dict[str, Any]],
    output_path: str,
    suptitle: str = "",
    n_rows: int = 1,
) -> bool:
    """
    Multi-panel scatter figure, one panel per pair.

    Args:
        pairs_data  : List of panel dicts, each with keys:
                          "x"      : np.ndarray (1-D)
                          "y"      : np.ndarray (1-D)
                          "xlabel" : str
                          "ylabel" : str
                          "title"  : str
        output_path : Full save path.
        suptitle    : Figure-level super-title.
        n_rows      : Number of subplot rows (default 1). Columns are
                      inferred as ceil(n_panels / n_rows).

    Returns:
        True on success, False on failure.

    Source: EXP-SANITY _plot_pairwise_nll_scatter,
            SC-DIAG _plot_pairwise_nll_comparison,
            FI-DIAG _plot_fi_expert_scatter.
    """
    try:
        n_panels = len(pairs_data)
        if n_panels == 0:
            logger.warning(
                f"PU | plot_pairwise_scatter: no panels provided for {output_path}"
            )
            return False

        import math
        n_cols   = math.ceil(n_panels / n_rows)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(5 * n_cols, 5 * n_rows),
            squeeze=False,
        )
        axes_flat = axes.flatten()

        for idx, panel in enumerate(pairs_data):
            ax     = axes_flat[idx]
            x      = np.asarray(panel.get("x", []), dtype=float)
            y      = np.asarray(panel.get("y", []), dtype=float)
            xlabel = panel.get("xlabel", "")
            ylabel = panel.get("ylabel", "")
            ptitle = panel.get("title", "")

            if len(x) == 0 or len(y) == 0:
                ax.set_title(f"{ptitle}\n(no data)")
                continue

            n_min = min(len(x), len(y))
            x, y  = x[:n_min], y[:n_min]
            mask  = np.isfinite(x) & np.isfinite(y)

            if not mask.any():
                ax.set_title(f"{ptitle}\n(no finite data)")
                continue

            ax.scatter(x[mask], y[mask], alpha=0.15, s=8, edgecolors="none",
                       color="#4C72B0")

            # Identity / diagonal line
            finite_all = np.concatenate([x[mask], y[mask]])
            if len(finite_all) > 0:
                p5, p95 = np.percentile(finite_all, [5, 95])
                ax.plot([p5, p95], [p5, p95], "r--", alpha=0.5, linewidth=1)

            # Pearson correlation
            if np.std(x[mask]) > 1e-8 and np.std(y[mask]) > 1e-8:
                corr = np.corrcoef(x[mask], y[mask])[0, 1]
                ax.set_title(f"{ptitle}\nρ = {corr:.3f}")
            else:
                ax.set_title(ptitle)

            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide unused axes
        for idx in range(n_panels, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        if suptitle:
            fig.suptitle(suptitle, fontsize=13)

        plt.tight_layout()
        return save_figure(fig, output_path)

    except Exception as e:
        logger.error(f"PU | plot_pairwise_scatter failed for {output_path}: {e}")
        plt.close("all")
        return False


# =============================================================================
# 6. plot_reconstruction_grid — encode→decode per-expert grid
# Source: EXP-SANITY _plot_reconstruction_grid (multi-expert layout);
#         SC-DIAG _plot_reconstruction_grid (2-row layout)
# =============================================================================

def plot_reconstruction_grid(
    y: Any,
    output_path: str,
    title: str,
    x_clean: Optional[Any] = None,
    x_hat: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Reconstruction grid figure.

    Two layout modes, selected automatically:

    Multi-expert mode (x_hat is a dict with ≥1 entry):
        rows = n_samples
        cols = [y (degraded)] [x_clean (optional)] [x̂ per expert ...]
        Used by SA-DIAG P4.

    2-row mode (x_hat is None or empty dict):
        row 0 = degraded y,  row 1 = reconstruction
        Used by SC-DIAG P5 (mixture reconstruction).

    Args:
        y           : Tensor or ndarray (n, C, H, W) — degraded input.
        output_path : Full save path.
        title       : Figure super-title.
        x_clean     : Optional Tensor (n, C, H, W) — clean ground truth.
        x_hat       : Dict {expert_name: Tensor (n, C, H, W)} for multi-expert,
                      or single Tensor (n, C, H, W) as {"_": tensor} for 2-row.

    Returns:
        True on success, False on failure.

    Source: EXP-SANITY v1.1 _plot_reconstruction_grid (encode→decode),
            SC-DIAG v1.1 _plot_reconstruction_grid (2-row mixture).
    """
    try:
        import torch

        def _to_np(t):
            if t is None:
                return None
            if hasattr(t, "numpy"):
                return t.detach().cpu().numpy()
            return np.asarray(t)

        y_np      = _to_np(y)
        xc_np     = _to_np(x_clean)
        xhat_dict = x_hat or {}

        if y_np is None or len(y_np) == 0:
            logger.warning(f"PU | plot_reconstruction_grid: no y data for {output_path}")
            return False

        n = y_np.shape[0]
        multi_expert = bool(xhat_dict)
        expert_names = list(xhat_dict.keys()) if multi_expert else []

        if multi_expert:
            # --- Multi-expert: rows=samples, cols=[y, x_clean?, *x̂_k] ---
            col_labels = ["y (degraded)"]
            col_arrays = [y_np]
            if xc_np is not None:
                col_labels.append("x (clean)")
                col_arrays.append(xc_np)
            for name in expert_names:
                xh = _to_np(xhat_dict[name])
                col_labels.append(f"{name}\nx̂")
                col_arrays.append(xh)

            n_cols = len(col_labels)
            fig, axes = plt.subplots(
                n, n_cols, figsize=(2.5 * n_cols, 2.5 * n), squeeze=False
            )

            for i in range(n):
                for c, arr in enumerate(col_arrays):
                    ax = axes[i, c]
                    if arr is None or i >= len(arr):
                        ax.axis("off")
                        continue
                    img = arr[i].squeeze()
                    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                    ax.axis("off")
                    if i == 0:
                        ax.set_title(col_labels[c], fontsize=8)

        else:
            # --- 2-row: row 0 = y, row 1 = x̂ (single reconstruction) ---
            # x̂ comes from the first value of xhat_dict if present, else None
            xh_np = None
            if xhat_dict:
                xh_np = _to_np(list(xhat_dict.values())[0])

            fig, axes = plt.subplots(
                2, n, figsize=(2 * n, 4), squeeze=False
            )
            for i in range(n):
                img_y = y_np[i].squeeze()
                axes[0, i].imshow(img_y, cmap="gray", vmin=0, vmax=1)
                axes[0, i].axis("off")
                if i == 0:
                    axes[0, i].set_title("y (degraded)", fontsize=8, loc="left")

                if xh_np is not None and i < len(xh_np):
                    img_xh = xh_np[i].squeeze()
                    axes[1, i].imshow(img_xh, cmap="gray", vmin=0, vmax=1)
                else:
                    axes[1, i].axis("off")
                axes[1, i].axis("off")
                if i == 0:
                    axes[1, i].set_title("x̂", fontsize=8, loc="left")

        fig.suptitle(title, fontsize=11)
        plt.tight_layout()
        return save_figure(fig, output_path)

    except Exception as e:
        logger.error(f"PU | plot_reconstruction_grid failed for {output_path}: {e}")
        plt.close("all")
        return False


# =============================================================================
# 7. plot_reconstruction_snapshots — training dynamics grid over epochs
# Source: SC-DIAG v1.1 _plot_recon_snapshots
# =============================================================================

def plot_reconstruction_snapshots(
    snapshots: List[Dict[str, Any]],
    output_path: str,
    title: str = "Reconstruction Quality Over Epochs",
    n_cols: int = 8,
) -> bool:
    """
    Epoch-over-time reconstruction grid.

    Layout: rows = epoch_snapshot × 2 (y row / x̂ row), cols = samples.

    Args:
        snapshots   : List of dicts, one per epoch checkpoint:
                          {"epoch": int,
                           "y":     Tensor (B, 1, H, W),
                           "x_hat": Tensor (B, 1, H, W)}
        output_path : Full save path.
        title       : Figure super-title.
        n_cols      : Number of sample columns (default 8).

    Returns:
        True on success, False on failure.

    Source: SC-DIAG v1.1 _plot_recon_snapshots — logic preserved exactly.
    """
    try:
        if not snapshots:
            logger.warning(
                f"PU | plot_reconstruction_snapshots: no snapshots for {output_path}"
            )
            return False

        def _to_np(t):
            if hasattr(t, "numpy"):
                return t.detach().cpu().numpy()
            return np.asarray(t) if t is not None else None

        n_snaps  = len(snapshots)
        n_rows   = n_snaps * 2  # y + x̂ per snapshot
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(2 * n_cols, 4 * n_snaps),
            squeeze=False,
        )

        for snap_idx, snap in enumerate(snapshots):
            epoch_label = snap.get("epoch", "?")
            y_imgs      = _to_np(snap.get("y"))
            xhat_imgs   = _to_np(snap.get("x_hat"))

            row_y    = snap_idx * 2
            row_xhat = snap_idx * 2 + 1

            if y_imgs is None or xhat_imgs is None:
                logger.error(
                    f"PU | plot_reconstruction_snapshots: snapshot epoch={epoch_label} "
                    f"missing 'y' or 'x_hat'"
                )
                for c in range(n_cols):
                    axes[row_y,    c].axis("off")
                    axes[row_xhat, c].axis("off")
                continue

            n_show = min(n_cols, y_imgs.shape[0])
            for c in range(n_cols):
                if c < n_show:
                    axes[row_y, c].imshow(
                        y_imgs[c].squeeze(), cmap="gray", vmin=0, vmax=1
                    )
                    axes[row_xhat, c].imshow(
                        xhat_imgs[c].squeeze(), cmap="gray", vmin=0, vmax=1
                    )
                axes[row_y,    c].axis("off")
                axes[row_xhat, c].axis("off")

            axes[row_y,    0].set_ylabel(f"Ep {epoch_label}\ny",  fontsize=8)
            axes[row_xhat, 0].set_ylabel("x̂", fontsize=8)

        fig.suptitle(title, fontsize=12)
        plt.tight_layout()
        return save_figure(fig, output_path)

    except Exception as e:
        logger.error(
            f"PU | plot_reconstruction_snapshots failed for {output_path}: {e}"
        )
        plt.close("all")
        return False


# =============================================================================
# 8. plot_expert_bars — per-expert bar chart
# Source: EXP-SANITY _plot_nll_rank_histogram;
#         SC-DIAG _plot_per_expert_nll;
#         FI-DIAG _plot_fi_scalar_per_expert (status-coloured variant)
# =============================================================================

def plot_expert_bars(
    data_dict: Dict[str, float],
    output_path: str,
    title: str,
    ylabel: str = "",
    hline: Optional[Tuple[float, str]] = None,
    color_by_status: Optional[Dict[str, str]] = None,
    value_labels: bool = True,
    ylim_bottom: float = 0.0,
    bar_width: float = 0.6,
) -> bool:
    """
    Per-expert bar chart with optional status colouring and value labels.

    Args:
        data_dict        : {expert_name: scalar_value} — bar heights.
        output_path      : Full save path.
        title            : Figure title.
        ylabel           : Y-axis label.
        hline            : Optional (y_value, label) horizontal reference line.
        color_by_status  : Optional {expert_name: status_str} where status_str is
                           one of "alive", "warn_low", "fatal_dead", "no_data".
                           Uses _FI_STATUS_COLORS palette. If None, uses Set2.
        value_labels     : If True, annotate each bar with its value.
        ylim_bottom      : Y-axis lower bound (default 0.0).
        bar_width        : Bar width (default 0.6).

    Returns:
        True on success, False on failure.

    Source: EXP-SANITY _plot_nll_rank_histogram,
            SC-DIAG _plot_per_expert_nll,
            FI-DIAG _plot_fi_scalar_per_expert (status colours).
    """
    try:
        if not data_dict:
            logger.warning(
                f"PU | plot_expert_bars: empty data_dict for {output_path}"
            )
            return False

        names  = list(data_dict.keys())
        values = [data_dict[n] for n in names]
        K      = len(names)

        if color_by_status is not None:
            colors = [
                _FI_STATUS_COLORS.get(color_by_status.get(n, "no_data"), "#9E9E9E")
                for n in names
            ]
            # Add legend patches for status colours
            legend_handles = [
                Patch(facecolor=c, label=l)
                for l, c in _FI_STATUS_COLORS.items()
                if any(color_by_status.get(n) == l for n in names)
            ]
        else:
            cmap   = plt.cm.get_cmap(_COLOR_CMAP, K)
            colors = [cmap(i) for i in range(K)]
            legend_handles = []

        fig, ax = plt.subplots(figsize=(max(5, K * 1.5), 4))
        bars = ax.bar(
            range(K), values, color=colors,
            edgecolor="black", linewidth=0.5,
            width=bar_width,
        )

        if value_labels:
            for bar, v in zip(bars, values):
                if np.isfinite(v):
                    offset = max(abs(v) * 0.02, 0.005)
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + offset,
                        f"{v:.3g}",
                        ha="center", va="bottom", fontsize=9,
                    )

        if hline is not None:
            h_val, h_label = hline
            ax.axhline(
                y=h_val, color="red", linestyle="--", linewidth=1.5,
                label=h_label,
            )
            ax.legend(fontsize=8)

        if legend_handles:
            ax.legend(handles=legend_handles, fontsize=8, loc="upper right")

        ax.set_xticks(range(K))
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(bottom=ylim_bottom)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        return save_figure(fig, output_path)

    except Exception as e:
        logger.error(f"PU | plot_expert_bars failed for {output_path}: {e}")
        plt.close("all")
        return False


# =============================================================================
# 9. plot_comparison_bars — Stage B vs C grouped bar chart
# Source: SC-DIAG _plot_nll_comparison, _plot_gate_weights_comparison,
#                  _plot_neff_comparison
# =============================================================================

def plot_comparison_bars(
    b_dict: Dict[str, float],
    c_dict: Dict[str, float],
    output_path: str,
    title: str,
    ylabel: str = "",
    hline: Optional[Tuple[float, str]] = None,
    value_fmt: str = ".3g",
) -> bool:
    """
    Side-by-side grouped bar chart comparing Stage B vs Stage C scalars.

    Args:
        b_dict      : {label: float} — Stage B values.
        c_dict      : {label: float} — Stage C values (same keys as b_dict).
        output_path : Full save path.
        title       : Figure title.
        ylabel      : Y-axis label.
        hline       : Optional (y_value, label) horizontal reference line.
        value_fmt   : Python format spec for bar value annotations (default ".3g").

    Returns:
        True on success, False on failure.

    Source: SC-DIAG _plot_nll_comparison, _plot_gate_weights_comparison,
            _plot_neff_comparison — palette and layout preserved.
    """
    try:
        if not b_dict or not c_dict:
            logger.warning(
                f"PU | plot_comparison_bars: empty dicts for {output_path}"
            )
            return False

        # Align keys: use union, sorted by b_dict order
        labels = list(b_dict.keys())
        b_vals = [b_dict.get(l, float("nan")) for l in labels]
        c_vals = [c_dict.get(l, float("nan")) for l in labels]

        x      = np.arange(len(labels))
        width  = 0.35
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 4))

        bars_b = ax.bar(
            x - width / 2, b_vals, width,
            label="Stage B", color=_COLOR_B,
            edgecolor="black", linewidth=0.5,
        )
        bars_c = ax.bar(
            x + width / 2, c_vals, width,
            label="Stage C", color=_COLOR_C,
            edgecolor="black", linewidth=0.5,
        )

        for bar, v in zip(bars_b, b_vals):
            if np.isfinite(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + abs(bar.get_height()) * 0.02 + 0.005,
                    format(v, value_fmt),
                    ha="center", va="bottom", fontsize=7,
                )
        for bar, v in zip(bars_c, c_vals):
            if np.isfinite(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + abs(bar.get_height()) * 0.02 + 0.005,
                    format(v, value_fmt),
                    ha="center", va="bottom", fontsize=7,
                )

        if hline is not None:
            h_val, h_label = hline
            ax.axhline(
                y=h_val, color="red", linestyle=":", linewidth=1.2,
                alpha=0.7, label=h_label,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        return save_figure(fig, output_path)

    except Exception as e:
        logger.error(f"PU | plot_comparison_bars failed for {output_path}: {e}")
        plt.close("all")
        return False


# =============================================================================
# 10. plot_residual_boxplot — Stage B vs C physics residual boxplot
# Source: SC-DIAG _plot_residual_comparison
# =============================================================================

def plot_residual_boxplot(
    b_residuals: np.ndarray,
    c_residuals: np.ndarray,
    output_path: str,
    title: str = "Stage B vs C — Physics Residual",
    ylabel: str = "‖Ax̂ − y‖²",
) -> bool:
    """
    Side-by-side boxplot comparing per-sample physics residuals for Stage B vs C.

    Args:
        b_residuals : 1-D array of per-sample ‖Ax̂-y‖² values from Stage B.
        c_residuals : 1-D array of per-sample ‖Ax̂-y‖² values from Stage C.
        output_path : Full save path.
        title       : Figure title.
        ylabel      : Y-axis label.

    Returns:
        True on success, False on failure.

    Source: SC-DIAG v1.1 _plot_residual_comparison — boxplot layout preserved.
    """
    try:
        b_arr = np.asarray(b_residuals, dtype=float)
        c_arr = np.asarray(c_residuals, dtype=float)

        b_arr = b_arr[np.isfinite(b_arr)]
        c_arr = c_arr[np.isfinite(c_arr)]

        if len(b_arr) == 0 or len(c_arr) == 0:
            logger.warning(
                f"PU | plot_residual_boxplot: insufficient data for {output_path} "
                f"(B={len(b_arr)}, C={len(c_arr)})"
            )
            return False

        fig, ax = plt.subplots(figsize=(5, 4))
        bp = ax.boxplot(
            [b_arr, c_arr],
            labels=["Stage B", "Stage C"],
            patch_artist=True,
        )
        for patch, color in zip(bp["boxes"], [_COLOR_B, _COLOR_C]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Annotate means
        for pos, arr, label in zip([1, 2], [b_arr, c_arr], ["Stage B", "Stage C"]):
            ax.text(
                pos, ax.get_ylim()[1] * 0.98,
                f"μ={arr.mean():.4f}",
                ha="center", va="top", fontsize=8,
            )

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        return save_figure(fig, output_path)

    except Exception as e:
        logger.error(f"PU | plot_residual_boxplot failed for {output_path}: {e}")
        plt.close("all")
        return False
