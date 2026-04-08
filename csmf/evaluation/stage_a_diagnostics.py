# =============================================================================
# Version: DIAG-REORG-StageADiag-v1.1 | Abbr: SA-DIAG
# Description: Stage A diagnostic runner — expert quality after per-expert
#              pretraining, before gate training. Merges EXP-SANITY v1.1 and
#              FI-DIAG v1.5. Delegates metric collection to MU v1.0 and
#              plotting to PU v1.0. Saves stage_a_summary.json for downstream
#              analysis and B-vs-C comparison. All plots are non-fatal; a
#              failed plot is logged and skipped. Fatal conditions are limited
#              to: epoch_logs required keys missing AND all MU collectors fail.
# Changelog:
#   v1.1 (2026-04-07): [DIAG-OUTPUT] Wired into TRAIN-MAIN v2.7 — SA-DIAG now
#                      called from train_csmf.py after Stage A; output directed
#                      to run_dir/stage_a_diagnostics/; P7 (FI over epochs) and
#                      P8 (FI vs NLL scatter) now active for all runs; no code
#                      changes — version bump tracks integration milestone
#   v1.0 (2026-04-04): Initial implementation — 8 plots (P1-P8) merging
#                      EXP-SANITY v1.1 (P1-P6) and FI-DIAG v1.5 (P7-P8);
#                      FI multi-batch aggregation via local _collect_fi_option_a()
#                      using MU.compute_fi_option_a_batch; verdict thresholds
#                      preserved from FI-DIAG (RATIO_FATAL=0.05, RATIO_WARN=0.15,
#                      INV_ERR_FATAL=5e-3); stage_a_summary.json includes all
#                      epoch arrays for downstream analysis; P3 z-histogram uses
#                      combined multi-panel figure via PU.save_figure
# Dependencies: LS v1.0, MU v1.0, PU v1.0, CSMF-MAIN v1.3.23+, torch, numpy
# Deprecates: EXP-SANITY v1.1 (retained for backward compat), FI-DIAG v1.5
#             (standalone CLI still functional; run_fi_diagnostics() kept)
# =============================================================================

import datetime
import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#from log_schema import validate_stage_a_logs, available_optional_keys_a
from .log_schema import validate_stage_a_logs, available_optional_keys_a
from .metric_utils import (
    collect_per_expert_nll,
    collect_latent_stats,
    collect_invertibility,
    collect_reconstruction_batch,
    compute_fi_option_a_batch,
)
from .plot_utils import (
    plot_epoch_lines,
    plot_pairwise_scatter,
    plot_reconstruction_grid,
    plot_expert_bars,
    plot_scatter,
    save_figure,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FI thresholds — preserved from FI-DIAG v1.5
# ---------------------------------------------------------------------------
_RATIO_FATAL   = 0.05
_RATIO_WARN    = 0.15
_INV_ERR_WARN  = 1e-4
_INV_ERR_FATAL = 5e-3   # raised in FI-DIAG v1.4 — logit preprocessing noise


# =============================================================================
# Main entry point
# =============================================================================

def run(
    csmf_model,
    val_loader,
    device: torch.device,
    epoch_logs: Dict[str, Dict[str, list]],
    output_dir: str,
    expert_names: Optional[List[str]] = None,
    fwd_model=None,
    n_fi_batches: int = 5,
    max_val_batches: int = 20,
) -> Dict[str, Any]:
    """
    Run Stage A diagnostics after per-expert pretraining.

    Replaces calling run_expert_sanity() + run_fi_diagnostics() separately.
    Generates 8 plots and saves stage_a_summary.json.

    Args:
        csmf_model      : CSMF model with trained (frozen) experts.
        val_loader      : Validation DataLoader yielding (x_clean, y_deg).
        device          : Compute device.
        epoch_logs      : Dict from train_stage_A():
                          {expert_name: {train_nll, val_nll, inv_err, fi_a}}.
        output_dir      : Directory for plots + JSON (created if absent).
        expert_names    : Optional explicit list. If None, derived from model.
        fwd_model       : Forward model A — only needed if fwd_model residual
                          checks are required (currently unused in SA-DIAG v1.0).
        n_fi_batches    : Val batches for final FI Option A aggregation (default 5).
        max_val_batches : Val batches for MU collectors (default 20).

    Returns:
        summary dict (also saved to stage_a_summary.json).
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(
        f"SA-DIAG | Starting Stage A diagnostics | output_dir={output_dir}"
    )

    # Derive expert names from model if not provided
    if expert_names is None:
        expert_names = [type(e).__name__ for e in csmf_model.experts]
    K = len(expert_names)

    csmf_model.eval()

    # ------------------------------------------------------------------
    # Step 1: Validate epoch_logs (LS)
    # ------------------------------------------------------------------
    logs_ok, missing_keys = validate_stage_a_logs(epoch_logs, expert_names)
    if not logs_ok:
        logger.error(
            f"SA-DIAG | epoch_logs validation failed — missing: {missing_keys}. "
            f"Epoch-based plots (P1, P2, P7) will be skipped."
        )

    # ------------------------------------------------------------------
    # Step 2: Collect val-set metrics (MU)
    # ------------------------------------------------------------------
    logger.info("SA-DIAG | Collecting val-set metrics via MU...")

    nll_metrics   = collect_per_expert_nll(
        csmf_model, val_loader, device, max_batches=max_val_batches
    )
    latent_stats  = collect_latent_stats(
        csmf_model, val_loader, device, max_batches=max_val_batches
    )
    inv_metrics   = collect_invertibility(
        csmf_model, val_loader, device, max_batches=max_val_batches
    )
    recon_batch   = collect_reconstruction_batch(
        csmf_model, val_loader, device, n_samples=8
    )

    if nll_metrics is None:
        logger.error(
            "SA-DIAG | collect_per_expert_nll returned None — "
            "P5, P6, P8 will be skipped."
        )
    if latent_stats is None:
        logger.error(
            "SA-DIAG | collect_latent_stats returned None — P3 will be skipped."
        )
    if inv_metrics is None:
        logger.error(
            "SA-DIAG | collect_invertibility returned None — "
            "invertibility summary unavailable."
        )
    if recon_batch is None:
        logger.error(
            "SA-DIAG | collect_reconstruction_batch returned None — P4 will be skipped."
        )

    # ------------------------------------------------------------------
    # Step 3: FI Option A — multi-batch aggregation
    # ------------------------------------------------------------------
    logger.info(
        f"SA-DIAG | Computing FI Option A over {n_fi_batches} val batches..."
    )
    fi_summary = _collect_fi_option_a(
        csmf_model, val_loader, device, expert_names, n_fi_batches
    )

    # ------------------------------------------------------------------
    # Step 4: Plots (all non-fatal)
    # ------------------------------------------------------------------

    # P1: NLL over epochs
    _plot_p1_nll_epochs(epoch_logs, expert_names, logs_ok, output_dir)

    # P2: Invertibility over epochs
    _plot_p2_inv_epochs(epoch_logs, expert_names, logs_ok, output_dir)

    # P3: Latent z histograms (combined multi-panel)
    _plot_p3_latent_z(latent_stats, expert_names, output_dir)

    # P4: Reconstruction grid (encode→decode per expert)
    _plot_p4_recon_grid(recon_batch, expert_names, output_dir)

    # P5: Pairwise NLL scatter
    _plot_p5_pairwise_nll(nll_metrics, expert_names, K, output_dir)

    # P6: NLL rank histogram (winner counts)
    _plot_p6_nll_rank(nll_metrics, expert_names, K, output_dir)

    # P7: FI over epochs
    _plot_p7_fi_epochs(epoch_logs, expert_names, logs_ok, output_dir)

    # P8: FI vs NLL scatter (one point per expert, annotated)
    _plot_p8_fi_vs_nll(fi_summary, nll_metrics, expert_names, output_dir)

    # ------------------------------------------------------------------
    # Step 5: Build and save stage_a_summary.json
    # ------------------------------------------------------------------
    summary = _build_summary(
        expert_names   = expert_names,
        epoch_logs     = epoch_logs,
        nll_metrics    = nll_metrics,
        latent_stats   = latent_stats,
        inv_metrics    = inv_metrics,
        fi_summary     = fi_summary,
    )

    json_path = os.path.join(output_dir, "stage_a_summary.json")
    try:
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"SA-DIAG | Summary saved: {json_path}")
    except Exception as e:
        logger.error(f"SA-DIAG | Failed to save stage_a_summary.json: {e}")

    csmf_model.train()
    logger.info(
        f"SA-DIAG | Complete | ready_for_stage_B="
        f"{summary.get('fi_summary', {}).get('verdict', {}).get('ready_for_stage_B', 'unknown')}"
    )
    return summary


# =============================================================================
# Local FI aggregation helper
# =============================================================================

def _collect_fi_option_a(
    csmf_model,
    val_loader,
    device: torch.device,
    expert_names: List[str],
    n_batches: int,
) -> Dict[str, Any]:
    """
    Run FI Option A over n_batches val batches per expert.
    Returns per-expert mean/std and relative ratio verdict.

    Uses MU.compute_fi_option_a_batch() per batch per expert.
    Thresholds: RATIO_FATAL=0.05, RATIO_WARN=0.15 (FI-DIAG v1.5).
    """
    K              = len(expert_names)
    fi_per_batch   = {k: [] for k in range(K)}
    batches_done   = 0
    val_iter       = iter(val_loader)

    for _ in range(n_batches):
        try:
            x_clean, y_deg = next(val_iter)
        except StopIteration:
            logger.warning(
                f"SA-DIAG | _collect_fi_option_a: val_loader exhausted "
                f"after {batches_done} batches"
            )
            break
        x_clean = x_clean.to(device)
        y_deg   = y_deg.to(device)

        for k, expert in enumerate(csmf_model.experts):
            fi_val = compute_fi_option_a_batch(csmf_model, expert, k, x_clean, y_deg)
            if np.isfinite(fi_val):
                fi_per_batch[k].append(fi_val)
            else:
                logger.warning(
                    f"SA-DIAG | FI Option A: non-finite | expert={k} "
                    f"({expert_names[k]}) batch={batches_done}"
                )

            # Restore expert grad after FI call (frozen by compute_fi_option_a_batch)
            for p in expert.parameters():
                p.requires_grad_(False)

        batches_done += 1

    if batches_done == 0:
        logger.error("SA-DIAG | _collect_fi_option_a: no val batches collected")
        return {}

    # Aggregate per-expert
    fi_results: Dict[str, Any] = {}
    fi_means   = []

    for k in range(K):
        name = expert_names[k]
        vals = fi_per_batch[k]
        if not vals:
            logger.error(
                f"SA-DIAG | FI Option A: no data for expert={k} ({name})"
            )
            fi_results[name] = {
                "F_k_mean": None, "F_k_std": None,
                "F_ratio_to_best": None, "status": "no_data",
            }
            fi_means.append(0.0)
        else:
            arr = np.array(vals)
            fi_results[name] = {
                "F_k_mean": round(float(arr.mean()), 6),
                "F_k_std":  round(float(arr.std()),  6),
                "F_ratio_to_best": None,  # filled below
                "status": None,           # filled below
            }
            fi_means.append(float(arr.mean()))

    # Relative ratios + status
    best_fk = max(fi_means) if fi_means else 1.0
    if best_fk <= 0.0:
        logger.warning("SA-DIAG | best FI <= 0 — all experts may be inactive")
        best_fk = 1.0

    dead, weak = [], []
    for k in range(K):
        name  = expert_names[k]
        entry = fi_results[name]
        if entry["F_k_mean"] is None:
            continue
        ratio = fi_means[k] / best_fk
        entry["F_ratio_to_best"] = round(ratio, 4)
        if ratio < _RATIO_FATAL:
            entry["status"] = "fatal_dead"
            dead.append(name)
            logger.error(
                f"SA-DIAG | FI FATAL | expert={k} ({name}) | "
                f"ratio={ratio:.4f} < {_RATIO_FATAL} — expert dead. "
                f"Note: logit preprocessing compresses gradient magnitudes — "
                f"verify before concluding collapse."
            )
        elif ratio < _RATIO_WARN:
            entry["status"] = "warn_low"
            weak.append(name)
            logger.warning(
                f"SA-DIAG | FI WARN | expert={k} ({name}) | "
                f"ratio={ratio:.4f} < {_RATIO_WARN}"
            )
        else:
            entry["status"] = "alive"
            logger.info(
                f"SA-DIAG | FI OK | expert={k} ({name}) | ratio={ratio:.4f}"
            )

    # 2-signal verdict (FI + invertibility only — NLL gap separate via nll_metrics)
    fi_ok  = not dead
    notes  = []
    if dead:
        notes.append(f"Dead experts (FI): {dead}")
    if weak:
        notes.append(f"Weak experts (FI): {weak}")
    if not notes:
        notes.append("FI checks passed")

    verdict = {
        "fi_ratio_pass":     fi_ok,
        "ready_for_stage_B": fi_ok,
        "notes":             " | ".join(notes),
    }

    return {
        "per_expert": fi_results,
        "fi_means":   fi_means,
        "n_batches":  batches_done,
        "verdict":    verdict,
    }


# =============================================================================
# Plot helpers (all non-fatal — log error and return on failure)
# =============================================================================

def _plot_p1_nll_epochs(
    epoch_logs: Dict, expert_names: List[str], logs_ok: bool, output_dir: str
) -> None:
    """P1: Per-expert train + val NLL over epochs."""
    if not logs_ok:
        logger.warning("SA-DIAG | P1 skipped — epoch_logs validation failed")
        return
    try:
        data_dict: Dict[str, List[float]] = {}
        for name in expert_names:
            logs = epoch_logs.get(name, {})
            if logs.get("train_nll"):
                data_dict[name] = logs["train_nll"]
            if logs.get("val_nll"):
                data_dict[f"val {name}"] = logs["val_nll"]

        if not data_dict:
            logger.warning("SA-DIAG | P1: no NLL data in epoch_logs")
            return

        plot_epoch_lines(
            data_dict   = data_dict,
            output_path = os.path.join(output_dir, "P1_nll_epochs.png"),
            title       = "Stage A — Per-Expert NLL Over Epochs",
            ylabel      = "NLL",
        )
    except Exception as e:
        logger.error(f"SA-DIAG | P1 failed: {e}")


def _plot_p2_inv_epochs(
    epoch_logs: Dict, expert_names: List[str], logs_ok: bool, output_dir: str
) -> None:
    """P2: Per-expert invertibility error over epochs."""
    if not logs_ok:
        logger.warning("SA-DIAG | P2 skipped — epoch_logs validation failed")
        return
    try:
        data_dict: Dict[str, List[float]] = {}
        for name in expert_names:
            inv = epoch_logs.get(name, {}).get("inv_err", [])
            if inv:
                data_dict[name] = inv

        if not data_dict:
            logger.warning("SA-DIAG | P2: no inv_err data — skipping")
            return

        plot_epoch_lines(
            data_dict   = data_dict,
            output_path = os.path.join(output_dir, "P2_inv_epochs.png"),
            title       = "Stage A — Per-Expert Invertibility Error Over Epochs",
            ylabel      = "‖f⁻¹(f(x)) − x‖ (mean abs)",
            hlines      = [(_INV_ERR_FATAL, f"Fatal threshold ({_INV_ERR_FATAL:.0e})", "red")],
            log_scale   = True,
        )
    except Exception as e:
        logger.error(f"SA-DIAG | P2 failed: {e}")


def _plot_p3_latent_z(
    latent_stats: Optional[Dict], expert_names: List[str], output_dir: str
) -> None:
    """
    P3: Latent z histograms — combined multi-panel, one panel per expert.
    Uses N(0,1) reference overlay. Calls PU.save_figure.
    """
    if latent_stats is None:
        logger.warning("SA-DIAG | P3 skipped — latent_stats collection failed")
        return
    try:
        K        = len(expert_names)
        ref_x    = np.linspace(-4, 4, 300)
        ref_y    = np.exp(-ref_x ** 2 / 2) / np.sqrt(2 * np.pi)
        fig, axes = plt.subplots(1, K, figsize=(5 * K, 4), squeeze=False)

        for k in range(K):
            ax   = axes[0, k]
            name = expert_names[k]
            stats = latent_stats.get(k)

            if stats is None or stats["z_all"].numel() == 0:
                ax.set_title(f"{name}\n(no data)")
                continue

            z_all  = stats["z_all"].numpy()          # (N, D)
            n_dims = min(5, z_all.shape[1])
            for d in range(n_dims):
                ax.hist(
                    z_all[:, d], bins=60, density=True,
                    alpha=0.3, label=f"dim {d}",
                )
            ax.plot(ref_x, ref_y, "k--", linewidth=1.5, label="N(0,1)")
            ax.set_xlim(-4, 4)
            ax.set_title(
                f"{name}\nμ={stats['z_mean']:.3f}  σ={stats['z_std']:.3f}",
                fontsize=9,
            )
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Stage A — Latent z Distributions Per Expert", fontsize=13)
        plt.tight_layout()
        save_figure(fig, os.path.join(output_dir, "P3_latent_z_hist.png"))

    except Exception as e:
        logger.error(f"SA-DIAG | P3 failed: {e}")
        plt.close("all")


def _plot_p4_recon_grid(
    recon_batch: Optional[Dict], expert_names: List[str], output_dir: str
) -> None:
    """P4: Per-expert encode→decode reconstruction grid."""
    if recon_batch is None:
        logger.warning("SA-DIAG | P4 skipped — recon_batch collection failed")
        return
    try:
        # Build x_hat dict with expert_names as keys
        raw_xhat = recon_batch.get("x_hat", {})
        xhat_named = {
            expert_names[k]: v
            for k, v in raw_xhat.items()
            if k < len(expert_names) and v is not None
        }

        plot_reconstruction_grid(
            y           = recon_batch.get("y"),
            output_path = os.path.join(output_dir, "P4_recon_grid.png"),
            title       = "Stage A — Per-Expert Reconstruction (encode→decode: z=f(x,h), x̂=f⁻¹(z,h))",
            x_clean     = recon_batch.get("x_clean"),
            x_hat       = xhat_named,
        )
    except Exception as e:
        logger.error(f"SA-DIAG | P4 failed: {e}")


def _plot_p5_pairwise_nll(
    nll_metrics: Optional[Dict], expert_names: List[str], K: int, output_dir: str
) -> None:
    """P5: Pairwise NLL scatter for all expert pairs."""
    if nll_metrics is None:
        logger.warning("SA-DIAG | P5 skipped — nll_metrics collection failed")
        return
    if K < 2:
        logger.warning("SA-DIAG | P5 skipped — need K >= 2 experts")
        return
    try:
        per_sample = nll_metrics.get("per_expert_nll_samples", {})
        pairs_data = []

        for i in range(K):
            for j in range(i + 1, K):
                nll_i = per_sample.get(i, torch.tensor([]))
                nll_j = per_sample.get(j, torch.tensor([]))
                if hasattr(nll_i, "numpy"):
                    nll_i = nll_i.numpy()
                if hasattr(nll_j, "numpy"):
                    nll_j = nll_j.numpy()
                pairs_data.append({
                    "x":      np.asarray(nll_i, dtype=float),
                    "y":      np.asarray(nll_j, dtype=float),
                    "xlabel": f"{expert_names[i]} NLL",
                    "ylabel": f"{expert_names[j]} NLL",
                    "title":  f"{expert_names[i]} vs {expert_names[j]}",
                })

        plot_pairwise_scatter(
            pairs_data  = pairs_data,
            output_path = os.path.join(output_dir, "P5_pairwise_nll.png"),
            suptitle    = "Stage A — Pairwise Expert NLL Scatter",
        )
    except Exception as e:
        logger.error(f"SA-DIAG | P5 failed: {e}")


def _plot_p6_nll_rank(
    nll_metrics: Optional[Dict], expert_names: List[str], K: int, output_dir: str
) -> None:
    """P6: NLL rank histogram — which expert wins per val sample."""
    if nll_metrics is None:
        logger.warning("SA-DIAG | P6 skipped — nll_metrics collection failed")
        return
    if K < 2:
        logger.warning("SA-DIAG | P6 skipped — need K >= 2 experts")
        return
    try:
        per_sample = nll_metrics.get("per_expert_nll_samples", {})
        nll_lists  = []
        min_n      = float("inf")

        for k in range(K):
            nll_k = per_sample.get(k, torch.tensor([]))
            if hasattr(nll_k, "numpy"):
                nll_k = nll_k.numpy()
            nll_k = np.asarray(nll_k, dtype=float)
            if len(nll_k) == 0:
                logger.warning(
                    f"SA-DIAG | P6: no NLL data for expert={k} ({expert_names[k]}) — skipping"
                )
                return
            nll_lists.append(nll_k)
            min_n = min(min_n, len(nll_k))

        nll_matrix = np.stack([nl[:int(min_n)] for nl in nll_lists], axis=1)  # (N, K)
        winners    = nll_matrix.argmin(axis=1)                                 # (N,)
        win_counts = {expert_names[k]: int((winners == k).sum()) for k in range(K)}
        total      = sum(win_counts.values())
        win_pct    = {n: round(c / total * 100, 2) for n, c in win_counts.items()}

        plot_expert_bars(
            data_dict   = win_pct,
            output_path = os.path.join(output_dir, "P6_nll_rank_hist.png"),
            title       = "Stage A — Expert NLL Rank (Best Expert Per Sample)",
            ylabel      = "Win Rate (%)",
        )
    except Exception as e:
        logger.error(f"SA-DIAG | P6 failed: {e}")


def _plot_p7_fi_epochs(
    epoch_logs: Dict, expert_names: List[str], logs_ok: bool, output_dir: str
) -> None:
    """P7: FI Option A per expert over epochs."""
    if not logs_ok:
        logger.warning("SA-DIAG | P7 skipped — epoch_logs validation failed")
        return
    try:
        data_dict: Dict[str, List[float]] = {}
        for name in expert_names:
            fi_vals = epoch_logs.get(name, {}).get("fi_a", [])
            if fi_vals:
                data_dict[name] = fi_vals

        if not data_dict:
            logger.warning(
                "SA-DIAG | P7: no fi_a data in epoch_logs — "
                "requires CSMF-MAIN v1.3.19+"
            )
            return

        plot_epoch_lines(
            data_dict   = data_dict,
            output_path = os.path.join(output_dir, "P7_fi_epochs.png"),
            title       = "Stage A — Fisher Information Option A Per Expert Over Epochs",
            ylabel      = "FI Option A (grad norm² / B)",
        )
    except Exception as e:
        logger.error(f"SA-DIAG | P7 failed: {e}")


def _plot_p8_fi_vs_nll(
    fi_summary: Dict, nll_metrics: Optional[Dict],
    expert_names: List[str], output_dir: str
) -> None:
    """P8: FI vs NLL scatter — one annotated point per expert."""
    if not fi_summary:
        logger.warning("SA-DIAG | P8 skipped — fi_summary empty")
        return
    if nll_metrics is None:
        logger.warning("SA-DIAG | P8 skipped — nll_metrics collection failed")
        return
    try:
        per_expert_fi  = fi_summary.get("per_expert", {})
        per_expert_nll = nll_metrics.get("per_expert_nll_mean", {})

        x_vals, y_vals, labels = [], [], []
        for name in expert_names:
            nll_val = per_expert_nll.get(name, float("nan"))
            fi_val  = (per_expert_fi.get(name) or {}).get("F_k_mean") or float("nan")
            x_vals.append(float(nll_val))
            y_vals.append(float(fi_val))
            labels.append(name)

        plot_scatter(
            x            = np.array(x_vals),
            y            = np.array(y_vals),
            output_path  = os.path.join(output_dir, "P8_fi_vs_nll.png"),
            title        = "Stage A — FI vs NLL Per Expert",
            xlabel       = "NLL_k (mean over val batches)",
            ylabel       = "FI_k Option A (mean grad norm² / B)",
            point_labels = labels,
        )
    except Exception as e:
        logger.error(f"SA-DIAG | P8 failed: {e}")


# =============================================================================
# JSON summary builder
# =============================================================================

def _build_summary(
    expert_names: List[str],
    epoch_logs:   Dict,
    nll_metrics:  Optional[Dict],
    latent_stats: Optional[Dict],
    inv_metrics:  Optional[Dict],
    fi_summary:   Dict,
) -> Dict[str, Any]:
    """
    Build the full stage_a_summary.json payload.
    All tensors converted to Python scalars / lists for JSON serialisation.
    """

    def _safe_list(x):
        """Convert a list of possibly non-serialisable values to Python floats."""
        if x is None:
            return []
        return [float(v) if np.isfinite(float(v)) else None for v in x]

    # Epoch arrays per expert
    epoch_logs_serial: Dict[str, Any] = {}
    for name in expert_names:
        logs = epoch_logs.get(name, {})
        epoch_logs_serial[name] = {
            "train_nll": _safe_list(logs.get("train_nll", [])),
            "val_nll":   _safe_list(logs.get("val_nll", [])),
            "inv_err":   _safe_list(logs.get("inv_err", [])),
            "fi_a":      _safe_list(logs.get("fi_a", [])),
        }

    # Val-set metrics
    val_metrics: Dict[str, Any] = {}

    if nll_metrics is not None:
        val_metrics["per_expert_nll_mean"] = {
            k: round(float(v), 6) if np.isfinite(float(v)) else None
            for k, v in nll_metrics["per_expert_nll_mean"].items()
        }
        val_metrics["mixture_nll_mean"] = round(float(nll_metrics["mixture_nll_mean"]), 6)
        # NLL rank win counts
        per_sample = nll_metrics.get("per_expert_nll_samples", {})
        K = len(expert_names)
        nll_lists = []
        min_n = float("inf")
        all_ok = True
        for k in range(K):
            arr = per_sample.get(k, torch.tensor([]))
            if hasattr(arr, "numpy"):
                arr = arr.numpy()
            arr = np.asarray(arr, dtype=float)
            if len(arr) == 0:
                all_ok = False
                break
            nll_lists.append(arr)
            min_n = min(min_n, len(arr))
        if all_ok and K >= 2:
            nll_matrix = np.stack([nl[:int(min_n)] for nl in nll_lists], axis=1)
            winners    = nll_matrix.argmin(axis=1)
            val_metrics["nll_rank_win_counts"] = {
                expert_names[k]: int((winners == k).sum()) for k in range(K)
            }
    else:
        val_metrics["per_expert_nll_mean"]  = None
        val_metrics["mixture_nll_mean"]     = None
        val_metrics["nll_rank_win_counts"]  = None

    if latent_stats is not None:
        val_metrics["latent_stats"] = {
            expert_names[k]: {
                "z_mean": round(float(s["z_mean"]), 4) if np.isfinite(float(s["z_mean"])) else None,
                "z_std":  round(float(s["z_std"]),  4) if np.isfinite(float(s["z_std"]))  else None,
            }
            for k, s in latent_stats.items()
            if k < len(expert_names)
        }
    else:
        val_metrics["latent_stats"] = None

    if inv_metrics is not None:
        val_metrics["invertibility"] = {
            expert_names[k]: {
                "log_det_std":      round(float(s["log_det_std"]), 6)
                                    if np.isfinite(float(s["log_det_std"])) else None,
                "log_det_collapse": s["log_det_collapse"],
            }
            for k, s in inv_metrics.items()
            if k < len(expert_names)
        }
    else:
        val_metrics["invertibility"] = None

    return {
        "metadata": {
            "timestamp":    datetime.datetime.now().isoformat(timespec="seconds"),
            "expert_names": expert_names,
            "diag_version": "SA-DIAG-v1.0",
        },
        "epoch_logs":  epoch_logs_serial,
        "val_metrics": val_metrics,
        "fi_summary": {
            "per_expert": fi_summary.get("per_expert", {}),
            "n_batches":  fi_summary.get("n_batches", 0),
            "verdict":    fi_summary.get("verdict", {}),
        },
    }
