# =============================================================================
# Version: DIAG-REORG-StageADiag-v1.8 | Abbr: SA-DIAG
# Description: Stage A diagnostic runner — expert quality after per-expert
#              pretraining, before gate training. Merges EXP-SANITY v1.1 and
#              FI-DIAG v1.5. Delegates metric collection to MU v1.0 and
#              plotting to PU v1.0. Saves stage_a_summary.json for downstream
#              analysis and B-vs-C comparison. All plots are non-fatal; a
#              failed plot is logged and skipped. Fatal conditions are limited
#              to: epoch_logs required keys missing AND all MU collectors fail.
# Changelog:
#   v1.8 (2026-04-18): [P4-4COL] Replace _plot_p4_recon_grid with per-expert
#                      4-row panel: Original/Degraded/Cycle/Generated. One PNG
#                      per expert (P4_recon_panel_<Name>.png). Imports
#                      plot_recon_panel_4col from PU v1.4. Falls back to legacy
#                      plot_reconstruction_grid if x_gen key absent from batch.
#   v1.7 (2026-04-17): [P9-3ROW] Update _plot_p9_recon_snapshots title to
#                      reflect 3-row layout (True / Enc→Dec / Cond. Prior).
#                      no structural change — PU v1.2 handles layout internally
#                      based on keys present in snapshot dict (CSMF-MAIN v2.6+)
#   v1.6 (2026-04-17): [PROX-T] Remove P8 (fi_vs_nll scatter — FI artifact,
#                      confirmed dead signal) and P11 (gap_hist — Stage A
#                      soft-competition diagnostic, not affected by PROX-T);
#                      add P_PROX1 (residual convergence per step), P_PROX2
#                      (residuals_by_T bar + NLL annotation), P_PROX3 (sample
#                      std pre/post); run() gains fwd_model_adj, T_values, lam
#                      args; collect_prox_diagnostics (MU v1.3) wired in Step 2;
#                      _build_summary gains prox_diagnostics key; all non-fatal;
#                      skipped gracefully when fwd_model or fwd_model_adj is None
#   v1.5 (2026-04-11): [LOGDET-DIAG] Add P12–P15 log-det diagnostic plots;
#                      call collect_logdet_decomposition() (MU v1.2) in run();
#                      P12=mean(log_det)/mean(log_p_z) bar, P13=log_det hist,
#                      P14=log_det vs log_p_z scatter, P15=temporal mean+std
#                      log_det over epochs from epoch_logs; _build_summary gains
#                      log_det_mean, log_det_per_dim, log_p_z_mean per expert;
#                      all new plots non-fatal; P15 skipped if key absent
#   v1.4 (2026-04-09): [PATCH-SA-SCW] Add gap_penalty to _build_summary()
#                      epoch_logs serialisation — List[float] per expert per
#                      epoch from CSMF-MAIN v1.5; consumed from LS v1.3
#                      _STAGE_A_OPTIONAL; diag_version bumped to SA-DIAG-v1.4
#   v1.3 (2026-04-09): [PATCH-SA-SCW] Add P10 soft-weight-over-epochs line plot
#                      and P11 NLL-gap histogram; P10 reads soft_weights from
#                      epoch_logs[name]["soft_weights"] (CSMF-MAIN v1.4+); P11
#                      computes gap [N,K] from nll_metrics per_expert_nll_samples
#                      at diagnostic time — no new logging required; both wired
#                      into run() and _build_summary(); skipped gracefully if
#                      soft_weights absent or nll_metrics None
#   v1.2 (2026-04-09): [RECON-SNAP] Add P9 per-expert reconstruction snapshots —
#                      iterates expert_names, reads epoch_logs[name]["recon_snapshots"],
#                      calls plot_reconstruction_snapshots once per expert; output
#                      P9_recon_snapshots_{name}.png; skipped gracefully if key absent
#                      or list empty. Requires CSMF-MAIN v1.3.30+.
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
    collect_logdet_decomposition,   # [LOGDET-DIAG] v1.5
    collect_prox_diagnostics,       # [PROX-T] v1.6
)
from .plot_utils import (
    plot_epoch_lines,
    plot_pairwise_scatter,
    plot_reconstruction_grid,
    plot_reconstruction_snapshots,
    plot_recon_panel_4col,
    plot_expert_bars,
    plot_scatter,
    save_figure,
    plot_prox_residual_convergence,  # [PROX-T] v1.6
    plot_prox_nll_scatter,           # [PROX-T] v1.6
    plot_prox_sample_spread,         # [PROX-T] v1.6
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
    fwd_model_adj=None,
    T_values: Optional[List[int]] = None,
    lam: float = 0.1,
    n_fi_batches: int = 5,
    max_val_batches: int = 20,
) -> Dict[str, Any]:
    """
    Run Stage A diagnostics after per-expert pretraining.

    Replaces calling run_expert_sanity() + run_fi_diagnostics() separately.
    Generates plots and saves stage_a_summary.json.

    Args:
        csmf_model      : CSMF model with trained (frozen) experts.
        val_loader      : Validation DataLoader yielding (x_clean, y_deg).
        device          : Compute device.
        epoch_logs      : Dict from train_stage_A():
                          {expert_name: {train_nll, val_nll, inv_err, fi_a}}.
        output_dir      : Directory for plots + JSON (created if absent).
        expert_names    : Optional explicit list. If None, derived from model.
        fwd_model       : Forward model A with .forward(x). Required for prox
                          diagnostics (P_PROX1–3). If None, prox plots skipped.
        fwd_model_adj   : Adjoint operator At with .adjoint(r) or callable.
                          Required for prox diagnostics. If None, prox plots skipped.
        T_values        : Proximal step counts to evaluate. Default [0,1,2,3].
        lam             : Proximal gradient step size (default 0.1).
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
    logdet_metrics = collect_logdet_decomposition(   # [LOGDET-DIAG] v1.5
        csmf_model, val_loader, device, max_batches=max_val_batches
    )

    # [PROX-T] v1.6 — collect proximal correction diagnostics
    _can_run_prox = (fwd_model is not None) and (fwd_model_adj is not None)
    if _can_run_prox:
        _A_fn  = fwd_model.forward if hasattr(fwd_model, "forward") else fwd_model
        _At_fn = (fwd_model_adj.adjoint
                  if hasattr(fwd_model_adj, "adjoint") else fwd_model_adj)
        prox_diagnostics = collect_prox_diagnostics(
            csmf_model, val_loader, _A_fn, _At_fn, device,
            T_values=T_values if T_values is not None else [0, 1, 2, 3],
            lam=lam, max_batches=max_val_batches,
        )
        if prox_diagnostics is None:
            logger.error(
                "SA-DIAG | collect_prox_diagnostics returned None — "
                "P_PROX1, P_PROX2, P_PROX3 will be skipped."
            )
    else:
        prox_diagnostics = None
        logger.warning(
            "SA-DIAG | fwd_model or fwd_model_adj is None — "
            "P_PROX1, P_PROX2, P_PROX3 skipped. "
            "Pass fwd_model=BlurDownsampleOperator(...) and "
            "fwd_model_adj=op (same instance) to enable prox diagnostics."
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
    if logdet_metrics is None:                          # [LOGDET-DIAG] v1.5
        logger.error(
            "SA-DIAG | collect_logdet_decomposition returned None — "
            "P12, P13, P14 will be skipped."
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

    # P8: REMOVED [PROX-T] v1.6 — FI vs NLL scatter retired (FI ratios are
    #     logit-preprocessing artifacts; no actionable signal post-PROX-T)

    # P9: Per-expert reconstruction snapshots over epochs
    _plot_p9_recon_snapshots(epoch_logs, expert_names, logs_ok, output_dir)

    # P10: Soft competition weights over epochs [PATCH-SA-SCW]
    _plot_p10_soft_weights_epoch(epoch_logs, expert_names, logs_ok, output_dir)

    # P11: REMOVED [PROX-T] v1.6 — NLL gap histogram retired (Stage A
    #      soft-competition diagnostic only; not affected by PROX-T sampling)

    # P12: mean(log_det) vs mean(log_p_z) bar chart per expert [LOGDET-DIAG]
    _plot_p12_logdet_bar(logdet_metrics, expert_names, output_dir)

    # P13: Histogram of log_det per expert [LOGDET-DIAG]
    _plot_p13_logdet_hist(logdet_metrics, expert_names, output_dir)

    # P14: Scatter log_det vs log_p_z per expert [LOGDET-DIAG]
    _plot_p14_logdet_scatter(logdet_metrics, expert_names, output_dir)

    # P15: Temporal mean + std(log_det) over epochs [LOGDET-DIAG]
    _plot_p15_logdet_temporal(epoch_logs, expert_names, logs_ok, output_dir)

    # P_PROX1: Residual convergence per step — ||Ax^(t)-y||² vs t [PROX-T]
    _plot_pprox1_residual_convergence(prox_diagnostics, output_dir)

    # P_PROX2: Residuals by T bar chart with NLL baseline annotation [PROX-T]
    _plot_pprox2_nll_scatter(prox_diagnostics, output_dir)

    # P_PROX3: Sample std before vs after proximal correction [PROX-T]
    _plot_pprox3_sample_spread(prox_diagnostics, output_dir)

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
        logdet_metrics = logdet_metrics,     # [LOGDET-DIAG] v1.5
        prox_diagnostics = prox_diagnostics, # [PROX-T] v1.6
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
    """P4: Per-expert 4-row reconstruction panel (Original/Degraded/Cycle/Generated).

    [P4-4COL] Generates one PNG per expert:
        P4_recon_panel_<ExpertName>.png
    Falls back to legacy plot_reconstruction_grid if x_gen is absent.
    """
    if recon_batch is None:
        logger.warning("SA-DIAG | P4 skipped — recon_batch collection failed")
        return

    y       = recon_batch.get("y")
    x_clean = recon_batch.get("x_clean")
    x_hat   = recon_batch.get("x_hat", {})
    x_gen   = recon_batch.get("x_gen", {})

    has_gen = bool(x_gen)

    for k, name in enumerate(expert_names):
        x_cycle = x_hat.get(k)
        x_gen_k = x_gen.get(k) if has_gen else None
        safe_name = name.replace("/", "_").replace(" ", "_")

        if has_gen:
            # [P4-4COL] Full 4-row panel
            plot_recon_panel_4col(
                x_clean     = x_clean,
                y           = y,
                x_cycle     = x_cycle,
                x_gen       = x_gen_k,
                expert_name = name,
                output_path = os.path.join(output_dir, f"P4_recon_panel_{safe_name}.png"),
                n_samples   = 8,
            )
        else:
            # Fallback: legacy 3-column grid (no Generated row)
            logger.warning(
                f"SA-DIAG | P4 | expert={name}: x_gen absent — using legacy grid"
            )
            try:
                plot_reconstruction_grid(
                    y           = y,
                    output_path = os.path.join(output_dir, f"P4_recon_grid_{safe_name}.png"),
                    title       = f"Stage A — {name} (encode→decode only)",
                    x_clean     = x_clean,
                    x_hat       = {name: x_cycle} if x_cycle is not None else {},
                )
            except Exception as e:
                logger.error(f"SA-DIAG | P4 legacy fallback failed for {name}: {e}")


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


# _plot_p8_fi_vs_nll — RETIRED [PROX-T] v1.6
# FI ratio flags were confirmed to be artifacts of logit preprocessing, not
# true expert collapse. Scatter of FI vs NLL added noise without actionable
# signal post-PROX-T. Removed to reduce diagnostic clutter.




def _plot_p9_recon_snapshots(
    epoch_logs: Dict, expert_names: List[str], logs_ok: bool, output_dir: str
) -> None:
    """P9: Per-expert reconstruction snapshots over Stage A epochs.
    3-row layout per snapshot (CSMF-MAIN v2.6+, PU v1.2+):
      Row 0 — Ground Truth | Row 1 — Encode→Decode | Row 2 — Cond. Prior
    Falls back to legacy 2-row layout for older checkpoints.
    """
    if not logs_ok:
        logger.warning("SA-DIAG | P9 skipped — epoch_logs not available")
        return
    for name in expert_names:
        try:
            snapshots = epoch_logs.get(name, {}).get("recon_snapshots", [])
            if not snapshots:
                logger.warning(f"SA-DIAG | P9: no snapshots for expert '{name}' — skipping")
                continue
            safe_name = name.replace(" ", "_")
            plot_reconstruction_snapshots(
                snapshots   = snapshots,
                output_path = os.path.join(output_dir, f"P9_recon_snapshots_{safe_name}.png"),
                title       = (
                    f"Stage A — {name} Reconstruction Over Epochs\n"
                    "Row 0: Ground Truth | Row 1: Encode→Decode | Row 2: Cond. Prior"
                ),
            )
        except Exception as e:
            logger.error(f"SA-DIAG | P9 failed for expert '{name}': {e}")

def _build_summary(
    expert_names:    List[str],
    epoch_logs:      Dict,
    nll_metrics:     Optional[Dict],
    latent_stats:    Optional[Dict],
    inv_metrics:     Optional[Dict],
    fi_summary:      Dict,
    logdet_metrics:  Optional[Dict] = None,     # [LOGDET-DIAG] v1.5
    prox_diagnostics: Optional[Dict] = None,    # [PROX-T] v1.6
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
            "train_nll":     _safe_list(logs.get("train_nll", [])),
            "val_nll":       _safe_list(logs.get("val_nll", [])),
            "inv_err":       _safe_list(logs.get("inv_err", [])),
            "fi_a":          _safe_list(logs.get("fi_a", [])),
            "soft_weights":  _safe_list(logs.get("soft_weights", [])),  # [PATCH-SA-SCW]
            "gap_penalty":   _safe_list(logs.get("gap_penalty", [])),   # [PATCH-SA-SCW v1.5]
            "log_det_mean":  _safe_list(logs.get("log_det_mean", [])),  # [LOGDET-DIAG] v1.5
            "log_det_std":   _safe_list(logs.get("log_det_std",  [])),  # [LOGDET-DIAG] v1.5
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

    # [LOGDET-DIAG] v1.5 — per-expert log_det / log_p_z summary from val set
    if logdet_metrics is not None:
        logdet_summary: Dict[str, Any] = {}
        for name, ld_data in logdet_metrics.items():
            ld  = ld_data["log_det"]
            lp  = ld_data["log_p_z"]
            D   = ld_data["D"]
            ld_mean = float(ld.mean().item())
            lp_mean = float(lp.mean().item())
            logdet_summary[name] = {
                "log_det_mean":    round(ld_mean, 6) if np.isfinite(ld_mean) else None,
                "log_det_per_dim": round(ld_mean / D, 6) if D > 0 and np.isfinite(ld_mean) else None,
                "log_p_z_mean":    round(lp_mean, 6) if np.isfinite(lp_mean) else None,
            }
        val_metrics["logdet_decomposition"] = logdet_summary
    else:
        val_metrics["logdet_decomposition"] = None

    return {
        "metadata": {
            "timestamp":    datetime.datetime.now().isoformat(timespec="seconds"),
            "expert_names": expert_names,
            "diag_version": "SA-DIAG-v1.6",   # [PROX-T] bumped from v1.5
        },
        "epoch_logs":  epoch_logs_serial,
        "val_metrics": val_metrics,
        "fi_summary": {
            "per_expert": fi_summary.get("per_expert", {}),
            "n_batches":  fi_summary.get("n_batches", 0),
            "verdict":    fi_summary.get("verdict", {}),
        },
        "prox_diagnostics": prox_diagnostics,   # [PROX-T] v1.6 — None if skipped
    }


# =============================================================================
# P10: Soft competition weights over epochs [PATCH-SA-SCW]
# =============================================================================

def _plot_p10_soft_weights_epoch(
    epoch_logs:   Dict[str, Any],
    expert_names: List[str],
    logs_ok:      bool,
    output_dir:   str,
) -> bool:
    """
    P10: Line plot of mean soft competition weight w_k per expert over training epochs.

    One line per expert. If all lines stay flat and equal (~1/K), tau_A may need
    tuning or NLL gaps are too large for the current scale. If one expert dominates
    from the start, competition is not working.

    Reads: epoch_logs[name]["soft_weights"] — List[float], one value per epoch.
    Requires CSMF-MAIN v1.4+ (PATCH-SA-SCW).
    Skipped gracefully if key absent for all experts.
    """
    try:
        if not logs_ok:
            logger.warning("SA-DIAG | P10: epoch_logs invalid — skipping soft weights plot")
            return False

        # Check at least one expert has soft_weights
        has_weights = any(
            "soft_weights" in epoch_logs.get(name, {}) and
            len(epoch_logs[name]["soft_weights"]) > 0
            for name in expert_names
        )
        if not has_weights:
            logger.warning(
                "SA-DIAG | P10: soft_weights absent in all experts — "
                "requires CSMF-MAIN v1.4+; skipping"
            )
            return False

        fig, ax = plt.subplots(figsize=(8, 4))
        plotted = False

        for name in expert_names:
            sw = epoch_logs.get(name, {}).get("soft_weights", [])
            if not sw:
                logger.warning(f"SA-DIAG | P10: soft_weights empty for '{name}' — skipping")
                continue
            ax.plot(range(1, len(sw) + 1), sw, label=name, linewidth=1.5)
            plotted = True

        if not plotted:
            logger.error("SA-DIAG | P10: no valid soft_weights series to plot")
            plt.close(fig)
            return False

        K = len(expert_names)
        ax.axhline(1.0 / K, color="gray", linestyle="--", linewidth=0.8,
                   label=f"Uniform (1/{K}={1/K:.3f})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean competition weight $w_k$")
        ax.set_title("Stage A — Soft Competition Weights per Expert [PATCH-SA-SCW]")
        ax.legend(fontsize=8)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)

        out_path = os.path.join(output_dir, "P10_soft_weights_epoch.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"SA-DIAG | P10 saved: {out_path}")
        return True

    except Exception as e:
        logger.error(f"SA-DIAG | P10 failed: {e}")
        try:
            plt.close("all")
        except Exception:
            pass
        return False


# _plot_p11_gap_hist — RETIRED [PROX-T] v1.6
# NLL gap histogram was a Stage A soft-competition diagnostic only.
# Not affected by PROX-T sampling; removed to reduce diagnostic clutter.


# =============================================================================
# P12–P15: Log-det decomposition diagnostics [LOGDET-DIAG] v1.5
# =============================================================================

def _plot_p12_logdet_bar(
    logdet_metrics: Optional[Dict],
    expert_names:   List[str],
    output_dir:     str,
) -> None:
    """P12: Grouped bar — mean(log_det) vs mean(log_p_z) per expert.
    Pinpoints which component drives RealNVP's NLL advantage."""
    if logdet_metrics is None:
        logger.warning("SA-DIAG | P12 skipped — logdet_metrics unavailable")
        return
    try:
        names   = [n for n in expert_names if n in logdet_metrics]
        if not names:
            logger.warning("SA-DIAG | P12 skipped — no experts in logdet_metrics")
            return
        ld_means = [logdet_metrics[n]["log_det"].mean().item() for n in names]
        lp_means = [logdet_metrics[n]["log_p_z"].mean().item() for n in names]

        x   = np.arange(len(names))
        w   = 0.35
        fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.5), 5))
        ax.bar(x - w / 2, ld_means, w, label="mean(log_det)", color="steelblue")
        ax.bar(x + w / 2, lp_means, w, label="mean(log_p(z))", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylabel("Value")
        ax.set_title("Stage A — NLL Decomposition: log_det vs log_p(z) per Expert")
        ax.legend()
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        fig.tight_layout()
        save_figure(fig, os.path.join(output_dir, "P12_logdet_bar.png"))
    except Exception as e:
        logger.error(f"SA-DIAG | P12 failed: {e}")


def _plot_p13_logdet_hist(
    logdet_metrics: Optional[Dict],
    expert_names:   List[str],
    output_dir:     str,
) -> None:
    """P13: Histogram of log_det distribution per expert.
    RealNVP -> shifted; NSF -> heavy tails; NICE -> near zero."""
    if logdet_metrics is None:
        logger.warning("SA-DIAG | P13 skipped — logdet_metrics unavailable")
        return
    try:
        names = [n for n in expert_names if n in logdet_metrics]
        if not names:
            logger.warning("SA-DIAG | P13 skipped — no experts in logdet_metrics")
            return
        K   = len(names)
        fig, axes = plt.subplots(1, K, figsize=(4 * K, 4), squeeze=False)
        for i, name in enumerate(names):
            ld  = logdet_metrics[name]["log_det"].numpy()
            ax  = axes[0, i]
            ax.hist(ld, bins=40, color="steelblue", edgecolor="none", alpha=0.8)
            ax.axvline(float(ld.mean()), color="red", linestyle="--", label=f"mean={ld.mean():.2f}")
            ax.set_title(name)
            ax.set_xlabel("log|det J|")
            ax.set_ylabel("Count" if i == 0 else "")
            ax.legend(fontsize=8)
        fig.suptitle("Stage A — log|det J| Distribution per Expert")
        fig.tight_layout()
        save_figure(fig, os.path.join(output_dir, "P13_logdet_hist.png"))
    except Exception as e:
        logger.error(f"SA-DIAG | P13 failed: {e}")


def _plot_p14_logdet_scatter(
    logdet_metrics: Optional[Dict],
    expert_names:   List[str],
    output_dir:     str,
) -> None:
    """P14: Scatter log_det (x) vs log_p(z) (y) per expert.
    Separates geometry (log_det) from distribution fit (latent Gaussian)."""
    if logdet_metrics is None:
        logger.warning("SA-DIAG | P14 skipped — logdet_metrics unavailable")
        return
    try:
        names = [n for n in expert_names if n in logdet_metrics]
        if not names:
            logger.warning("SA-DIAG | P14 skipped — no experts in logdet_metrics")
            return
        K   = len(names)
        fig, axes = plt.subplots(1, K, figsize=(4 * K, 4), squeeze=False)
        for i, name in enumerate(names):
            ld = logdet_metrics[name]["log_det"].numpy()
            lp = logdet_metrics[name]["log_p_z"].numpy()
            ax = axes[0, i]
            # Subsample for speed if large
            n_plot = min(len(ld), 2000)
            idx    = np.random.choice(len(ld), n_plot, replace=False) if len(ld) > n_plot else np.arange(len(ld))
            ax.scatter(ld[idx], lp[idx], s=4, alpha=0.4, color="steelblue")
            ax.set_xlabel("log|det J|")
            ax.set_ylabel("log p(z)" if i == 0 else "")
            ax.set_title(name)
        fig.suptitle("Stage A — log|det J| vs log p(z) per Expert")
        fig.tight_layout()
        save_figure(fig, os.path.join(output_dir, "P14_logdet_scatter.png"))
    except Exception as e:
        logger.error(f"SA-DIAG | P14 failed: {e}")


def _plot_p15_logdet_temporal(
    epoch_logs:   Dict,
    expert_names: List[str],
    logs_ok:      bool,
    output_dir:   str,
) -> None:
    """P15: Temporal mean ± std of log_det over epochs per expert.
    Detects drift (exploitation) and instability (spikes)."""
    if not logs_ok:
        logger.warning("SA-DIAG | P15 skipped — epoch_logs validation failed")
        return
    try:
        has_data = False
        fig, ax  = plt.subplots(figsize=(8, 4))
        for name in expert_names:
            logs = epoch_logs.get(name, {})
            ld_mean = logs.get("log_det_mean", [])
            ld_std  = logs.get("log_det_std",  [])
            if not ld_mean:
                logger.warning(
                    f"SA-DIAG | P15: 'log_det_mean' absent for expert '{name}' — "
                    "requires CSMF-MAIN v1.7+; skipping this expert"
                )
                continue
            epochs = np.arange(1, len(ld_mean) + 1)
            mu     = np.array(ld_mean, dtype=float)
            sd     = np.array(ld_std,  dtype=float) if ld_std else np.zeros_like(mu)
            line,  = ax.plot(epochs, mu, label=name)
            ax.fill_between(epochs, mu - sd, mu + sd, alpha=0.2, color=line.get_color())
            has_data = True

        if not has_data:
            logger.warning("SA-DIAG | P15 skipped — no log_det_mean data in epoch_logs")
            plt.close(fig)
            return

        ax.set_xlabel("Epoch")
        ax.set_ylabel("mean(log|det J|) ± std")
        ax.set_title("Stage A — Temporal log|det J| Stability per Expert")
        ax.legend()
        fig.tight_layout()
        save_figure(fig, os.path.join(output_dir, "P15_logdet_temporal.png"))
    except Exception as e:
        logger.error(f"SA-DIAG | P15 failed: {e}")


# =============================================================================
# P_PROX1–3: Proximal correction diagnostics [PROX-T] v1.6
# All non-fatal — log error and return False on failure.
# =============================================================================

def _plot_pprox1_residual_convergence(
    prox_diagnostics: Optional[Dict[str, Any]],
    output_dir: str,
) -> bool:
    """
    P_PROX1: Per-step residual convergence ||Ax^(t)-y||² vs t for max(T_values).

    Confirms each prox step reduces the measurement residual.
    Skipped gracefully if prox_diagnostics is None.
    """
    if prox_diagnostics is None:
        logger.warning("SA-DIAG | P_PROX1 skipped — prox_diagnostics unavailable")
        return False
    try:
        residual_steps = prox_diagnostics.get("residual_steps", [])
        if not residual_steps or len(residual_steps) < 2:
            logger.warning(
                "SA-DIAG | P_PROX1 skipped — residual_steps has fewer than 2 points "
                "(need T>=1 in T_values)"
            )
            return False

        return plot_prox_residual_convergence(
            residual_steps = residual_steps,
            output_path    = os.path.join(output_dir, "P_PROX1_residual_convergence.png"),
            title          = "Stage A — Proximal Residual Convergence ||Ax^(t)-y||²",
        )
    except Exception as e:
        logger.error(f"SA-DIAG | P_PROX1 failed: {e}")
        return False


def _plot_pprox2_nll_scatter(
    prox_diagnostics: Optional[Dict[str, Any]],
    output_dir: str,
) -> bool:
    """
    P_PROX2: Mean residual at T=0,1,2,3 bar chart with NLL baseline annotation.

    Core WP1 M1 evidence: does T>0 reduce residual without wrecking NLL?
    Skipped gracefully if prox_diagnostics is None.
    """
    if prox_diagnostics is None:
        logger.warning("SA-DIAG | P_PROX2 skipped — prox_diagnostics unavailable")
        return False
    try:
        residuals_by_T = prox_diagnostics.get("residuals_by_T", {})
        nll_baseline   = prox_diagnostics.get("nll_baseline", float("nan"))

        if not residuals_by_T:
            logger.warning("SA-DIAG | P_PROX2 skipped — residuals_by_T empty")
            return False

        return plot_prox_nll_scatter(
            residuals_by_T = residuals_by_T,
            nll_baseline   = nll_baseline,
            output_path    = os.path.join(output_dir, "P_PROX2_residual_vs_T.png"),
            title          = "Stage A — Residual by T (WP1 M1 Evidence)",
        )
    except Exception as e:
        logger.error(f"SA-DIAG | P_PROX2 failed: {e}")
        return False


def _plot_pprox3_sample_spread(
    prox_diagnostics: Optional[Dict[str, Any]],
    output_dir: str,
) -> bool:
    """
    P_PROX3: Sample std before (T=0) vs after (T=max) prox correction.

    Guards against prox collapsing posterior diversity.
    Skipped gracefully if prox_diagnostics is None.
    """
    if prox_diagnostics is None:
        logger.warning("SA-DIAG | P_PROX3 skipped — prox_diagnostics unavailable")
        return False
    try:
        std_pre  = prox_diagnostics.get("sample_std_pre",  float("nan"))
        std_post = prox_diagnostics.get("sample_std_post", float("nan"))
        T_values = prox_diagnostics.get("T_values", [0])
        T_max    = max(T_values)

        if not np.isfinite(std_pre) or not np.isfinite(std_post):
            logger.error(
                f"SA-DIAG | P_PROX3: non-finite std values "
                f"(pre={std_pre}, post={std_post}) — skipping"
            )
            return False

        return plot_prox_sample_spread(
            sample_std_pre  = std_pre,
            sample_std_post = std_post,
            T_max           = T_max,
            output_path     = os.path.join(output_dir, "P_PROX3_sample_spread.png"),
            title           = f"Stage A — Sample Spread Before/After Prox (T={T_max})",
        )
    except Exception as e:
        logger.error(f"SA-DIAG | P_PROX3 failed: {e}")
        return False
