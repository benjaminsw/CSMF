# =============================================================================
# Version: DIAG-REORG-StageCDiag-v2.3 | Abbr: SC-DIAG
# Description: Stage C diagnostic runner — final system quality + B-vs-C
#              comparison. Refactored from v1.1: inline _collect_metrics() and
#              all _plot_*() functions replaced by MU v1.0 and PU v1.0 calls.
#              B-vs-C comparison first tries to load stage_b_summary.json
#              (written by SB-DIAG); falls back to loading the Stage B
#              checkpoint and running MU collectors if JSON not found.
#              Public API (run_stage_c_diagnostics signature) unchanged —
#              backward-compatible with train_csmf.py v2.7.
# Changelog:
#   v2.3 (2026-04-15): [SC-RECFIRST] Replace P_loss_components (NLL/SW2/cal) with
#                      P_loss_components_c1 (cons_c1, img_loss, alive_penalty) to match
#                      HYBRID v1.9 forward_stage_c() loss_dict keys; detection flag
#                      updated from nll_loss to cons_c1; old NLL keys absent from new
#                      Stage C — backward-compat: plots silently skipped if key absent.
#                      Add P_neff_c1 — inline matplotlib plot of neff_c1 over epochs
#                      with alive_penalty threshold (1.5) and collapse floor (1.0)
#                      dashed lines; reads neff_c1 from epoch_logs; non-fatal.
#   v2.2 (2026-04-07): [DIAG-OUTPUT] Add P_loss_components per-component
#                      hybrid loss plot (NLL, consistency, SW2 transport,
#                      calibration per epoch); gated on nll_loss key presence;
#                      requires CSMF-MAIN v1.3.29+; non-fatal if absent
#   v2.1 (2026-04-06): [ALIVE-PLOT] Add P3c alive experts vs epoch — counts
#                      experts with gate_weight >= _ALIVE_THRESHOLD (0.05) per
#                      epoch; step-line plot with K baseline; gated on
#                      has_gate_weights; mirrors SB-DIAG P6 for Stage C;
#                      _ALIVE_THRESHOLD constant added at module level
#   v2.0 (2026-04-04): [DIAG-REORG] Refactor — strip inline _collect_metrics()
#                      and _plot_*() functions; delegate to MU v1.0 and PU v1.0;
#                      B-vs-C loads stage_b_summary.json (SB-DIAG output) via
#                      _load_stage_b_data() with checkpoint-load fallback;
#                      saves stage_c_summary.json (richer than v1.1 JSON);
#                      adds optional stage_b_summary_path param; 7 core plots
#                      (P1-P7) + optional P3b gate-weights-over-epochs when
#                      epoch_logs["gate_weights"] present; LS validation added
#   v1.1 (2026-04-03): Added P3 gate weights/epochs, P4 residual/epochs,
#                      P5 recon grid, P6 recon snapshots (CSMF-v1.3.22+)
#   v1.0 (2026-03-29): Initial implementation — 8 plots + JSON summary
# Dependencies: LS v1.0, MU v1.0, PU v1.0, CSMF-MAIN v1.3.23+, torch, numpy
# =============================================================================

import copy
import datetime
import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .log_schema import validate_stage_c_logs, available_optional_keys_c
from .metric_utils import (
    collect_per_expert_nll,
    collect_gate_metrics,
    collect_reconstruction_metrics,
    collect_reconstruction_batch,
)
from .plot_utils import (
    plot_epoch_lines,
    plot_expert_bars,
    plot_histogram,
    plot_reconstruction_grid,
    plot_reconstruction_snapshots,
    plot_comparison_bars,
    plot_residual_boxplot,
)

logger = logging.getLogger(__name__)

# Gate weight threshold below which an expert is considered "dead"
_ALIVE_THRESHOLD = 0.05


# =============================================================================
# Public entry point — signature unchanged from v1.1 for backward compat
# =============================================================================

def run_stage_c_diagnostics(
    csmf_model,
    val_loader,
    fwd_model,
    epoch_logs:            Dict[str, list],
    ckpt_path_B:           str,
    expert_names:          List[str],
    output_dir:            str = "results/stage_c_diagnostics",
    max_val_batches:       int = 20,
    stage_b_summary_path:  Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run Stage C diagnostics: final system quality + B-vs-C comparison.

    Args:
        csmf_model           : CSMF model after Stage C training.
        val_loader           : Validation DataLoader yielding (x_clean, y_deg).
        fwd_model            : Forward model A (SRForwardModel).
        epoch_logs           : Dict from train_stage_C() — must satisfy
                               StageCEpochLogs contract (LS v1.0). Required
                               keys: train_loss, val_loss, neff.
        ckpt_path_B          : Path to Stage B checkpoint. Used as fallback
                               for B-vs-C if stage_b_summary.json not found.
        expert_names         : Short expert name list e.g. ["realnvp","nice","nsf"].
        output_dir           : Directory for plots + JSON (created if absent).
        max_val_batches      : Max val batches for MU collectors.
        stage_b_summary_path : Optional explicit path to stage_b_summary.json
                               (SB-DIAG output). If None, tries sibling dir
                               heuristic then falls back to checkpoint load.

    Returns:
        Summary dict (also saved to stage_c_summary.json).
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(
        f"SC-DIAG | Starting Stage C diagnostics | output_dir={output_dir}"
    )

    device = csmf_model.device
    csmf_model.eval()

    # ------------------------------------------------------------------
    # Step 1: Validate epoch_logs (LS)
    # ------------------------------------------------------------------
    logs_ok, missing_keys = validate_stage_c_logs(epoch_logs)
    if not logs_ok:
        logger.error(
            f"SC-DIAG | epoch_logs validation failed — missing: {missing_keys}. "
            f"Epoch-based plots (P1, P4) will be skipped."
        )

    optional_present  = available_optional_keys_c(epoch_logs)
    has_gate_weights  = "gate_weights"    in optional_present
    has_residual      = "residual"        in optional_present
    has_snapshots     = "recon_snapshots" in optional_present
    has_loss_comps    = "cons_c1"         in optional_present   # [v2.3] SC-RECFIRST keys

    # ------------------------------------------------------------------
    # Step 2: Collect Stage C val-set metrics (MU)
    # ------------------------------------------------------------------
    logger.info("SC-DIAG | Collecting Stage C val-set metrics via MU...")

    nll_c    = collect_per_expert_nll(
        csmf_model, val_loader, device, max_batches=max_val_batches
    )
    gate_c   = collect_gate_metrics(
        csmf_model, val_loader, device, max_batches=max_val_batches
    )
    resid_c  = collect_reconstruction_metrics(
        csmf_model, val_loader, fwd_model, device, max_batches=max_val_batches
    )
    recon_c  = collect_reconstruction_batch(
        csmf_model, val_loader, device, n_samples=8
    )

    if nll_c   is None: logger.error("SC-DIAG | collect_per_expert_nll returned None — P2 skipped.")
    if gate_c  is None: logger.error("SC-DIAG | collect_gate_metrics returned None — gate data unavailable.")
    if resid_c is None: logger.error("SC-DIAG | collect_reconstruction_metrics returned None — P3, P7d skipped.")
    if recon_c is None: logger.error("SC-DIAG | collect_reconstruction_batch returned None — P5 skipped.")

    # ------------------------------------------------------------------
    # Step 3: Load Stage B data for B-vs-C comparison
    # ------------------------------------------------------------------
    b_data = _load_stage_b_data(
        stage_b_summary_path = stage_b_summary_path,
        output_dir           = output_dir,
        ckpt_path_B          = ckpt_path_B,
        csmf_model           = csmf_model,
        val_loader           = val_loader,
        fwd_model            = fwd_model,
        device               = device,
        expert_names         = expert_names,
        max_val_batches      = max_val_batches,
    )

    # ------------------------------------------------------------------
    # Step 4: Plots — core 7 + optional P3b
    # ------------------------------------------------------------------

    # P1: Joint loss train/val over epochs
    _plot_p1_joint_loss(epoch_logs, logs_ok, output_dir)

    # P2: Per-expert NLL bars
    _plot_p2_per_expert_nll(nll_c, expert_names, output_dir)

    # P3: Residual distribution histogram
    _plot_p3_residual_dist(resid_c, output_dir)

    # P3b: Gate weights over epochs (optional, when epoch_logs["gate_weights"] present)
    if has_gate_weights:
        _plot_p3b_gate_weights_epochs(epoch_logs, expert_names, output_dir)

    # P3c: Alive experts vs epoch (optional, same gate on has_gate_weights)
    if has_gate_weights:
        _plot_p3c_alive_experts(epoch_logs, expert_names, len(expert_names), output_dir)

    # P4: Residual over epochs
    _plot_p4_residual_epochs(epoch_logs, has_residual, logs_ok, output_dir)

    # P5: Final reconstruction grid (encode→decode via MU)
    _plot_p5_recon_grid(recon_c, expert_names, output_dir)

    # P6: Reconstruction snapshots over epochs
    _plot_p6_recon_snapshots(epoch_logs, has_snapshots, output_dir)

    # P7: B vs C comparison (4 sub-plots)
    _plot_p7_b_vs_c(
        b_data       = b_data,
        nll_c        = nll_c,
        gate_c       = gate_c,
        resid_c      = resid_c,
        expert_names = expert_names,
        output_dir   = output_dir,
    )

    # P_loss_components_c1: SC-RECFIRST loss components over epochs [v2.3]
    _plot_p_loss_components(epoch_logs, has_loss_comps, output_dir)

    # P_neff_c1: Neff from reconstruction-first loss over epochs [v2.3]
    _plot_p_neff_c1(epoch_logs, output_dir)

    # ------------------------------------------------------------------
    # Step 5: Build and save stage_c_summary.json
    # ------------------------------------------------------------------
    summary = _build_summary(
        epoch_logs   = epoch_logs,
        expert_names = expert_names,
        nll_c        = nll_c,
        gate_c       = gate_c,
        resid_c      = resid_c,
        b_data       = b_data,
        logs_ok      = logs_ok,
    )

    json_path = os.path.join(output_dir, "stage_c_summary.json")
    try:
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"SC-DIAG | Summary saved: {json_path}")
    except Exception as e:
        logger.error(f"SC-DIAG | Failed to save stage_c_summary.json: {e}")

    csmf_model.train()
    logger.info("SC-DIAG | Stage C diagnostics complete.")
    return summary


# =============================================================================
# Stage B data loader — JSON-first, checkpoint fallback
# =============================================================================

def _load_stage_b_data(
    stage_b_summary_path: Optional[str],
    output_dir:           str,
    ckpt_path_B:          str,
    csmf_model,
    val_loader,
    fwd_model,
    device:               torch.device,
    expert_names:         List[str],
    max_val_batches:      int,
) -> Dict[str, Any]:
    """
    Load Stage B scalar metrics for B-vs-C comparison.

    Strategy (in order):
      1. Explicit stage_b_summary_path if provided.
      2. Sibling heuristic: <parent of output_dir>/stage_b_diagnostics/stage_b_summary.json
      3. Checkpoint fallback: load ckpt_path_B, run MU collectors, restore Stage C state.
      4. Return {} if all attempts fail.

    Returns dict with keys matching MU collector outputs (scalar values only):
        "per_expert_nll_mean", "mixture_nll_mean",
        "neff_mean", "gate_weights_mean",
        "residual_mean", "residual_std", "residuals_all" (optional, for boxplot)
    """
    # --- Attempt 1: explicit path ---
    if stage_b_summary_path and os.path.isfile(stage_b_summary_path):
        result = _parse_stage_b_json(stage_b_summary_path, expert_names)
        if result:
            logger.info(
                f"SC-DIAG | Stage B data loaded from explicit path: {stage_b_summary_path}"
            )
            return result

    # --- Attempt 2: sibling directory heuristic ---
    sibling_path = os.path.join(
        os.path.dirname(output_dir.rstrip("/")),
        "stage_b_diagnostics",
        "stage_b_summary.json",
    )
    if os.path.isfile(sibling_path):
        result = _parse_stage_b_json(sibling_path, expert_names)
        if result:
            logger.info(
                f"SC-DIAG | Stage B data loaded from sibling dir: {sibling_path}"
            )
            return result

    # --- Attempt 3: checkpoint fallback ---
    if not os.path.isfile(ckpt_path_B):
        logger.warning(
            f"SC-DIAG | Stage B checkpoint not found: {ckpt_path_B} — "
            f"B-vs-C comparison plots will be skipped."
        )
        return {}

    logger.info(
        f"SC-DIAG | stage_b_summary.json not found — "
        f"loading Stage B checkpoint for B-vs-C: {ckpt_path_B}"
    )

    state_c = copy.deepcopy(csmf_model.state_dict())
    try:
        payload_b = torch.load(ckpt_path_B, map_location="cpu")
        csmf_model.load_state_dict(payload_b["state_dict"])
        csmf_model.to(device)
        csmf_model.eval()
        logger.info("SC-DIAG | Stage B checkpoint loaded for metric collection.")

        nll_b   = collect_per_expert_nll(
            csmf_model, val_loader, device, max_batches=max_val_batches
        )
        gate_b  = collect_gate_metrics(
            csmf_model, val_loader, device, max_batches=max_val_batches
        )
        resid_b = collect_reconstruction_metrics(
            csmf_model, val_loader, fwd_model, device, max_batches=max_val_batches
        )

        result: Dict[str, Any] = {}
        if nll_b:
            result["per_expert_nll_mean"] = nll_b["per_expert_nll_mean"]
            result["mixture_nll_mean"]    = nll_b["mixture_nll_mean"]
        if gate_b:
            result["neff_mean"]          = gate_b["neff_mean"]
            result["gate_weights_mean"]  = gate_b["gate_weights_mean"]
        if resid_b:
            result["residual_mean"]  = resid_b["residual_mean"]
            result["residual_std"]   = resid_b["residual_std"]
            result["residuals_all"]  = resid_b["residuals_all"]

        return result

    except Exception as e:
        logger.error(f"SC-DIAG | Stage B checkpoint fallback failed: {e}")
        return {}
    finally:
        csmf_model.load_state_dict(state_c)
        csmf_model.to(device)
        logger.info("SC-DIAG | Stage C model state restored after Stage B collection.")


def _parse_stage_b_json(path: str, expert_names: List[str]) -> Dict[str, Any]:
    """
    Parse stage_b_summary.json (SB-DIAG output) into MU-compatible scalar dict.
    Returns {} on parse failure.
    """
    try:
        with open(path) as f:
            sb = json.load(f)

        final     = sb.get("final", {})
        meta      = sb.get("metadata", {})

        result: Dict[str, Any] = {}

        # Gate weights
        gw = final.get("gate_weights", {})
        if gw:
            result["gate_weights_mean"] = {k: float(v) for k, v in gw.items() if v is not None}

        # Neff
        neff_val = final.get("neff")
        if neff_val is not None:
            result["neff_mean"] = float(neff_val)

        # NLL — stage_b_summary.json does not store per_expert_nll_mean by default
        # (SB-DIAG is a pure epoch_logs consumer with no model access).
        # These will be absent; B-vs-C NLL panel will be skipped unless
        # checkpoint fallback was used. Log info so the user is aware.
        if "per_expert_nll_mean" not in result:
            logger.info(
                "SC-DIAG | stage_b_summary.json: per_expert_nll_mean absent — "
                "B-vs-C NLL panel will be skipped. "
                "Run SB-DIAG with model access or use checkpoint fallback."
            )

        return result

    except Exception as e:
        logger.error(f"SC-DIAG | Failed to parse stage_b_summary.json at {path}: {e}")
        return {}


# =============================================================================
# Plot helpers (all non-fatal)
# =============================================================================

def _plot_p1_joint_loss(epoch_logs: Dict, logs_ok: bool, output_dir: str) -> None:
    """P1: Stage C joint hybrid loss train/val over epochs."""
    if not logs_ok:
        logger.warning("SC-DIAG | P1 skipped — epoch_logs validation failed")
        return
    try:
        data_dict: Dict[str, list] = {}
        if epoch_logs.get("train_loss"):
            data_dict["Train Loss"] = epoch_logs["train_loss"]
        if epoch_logs.get("val_loss"):
            data_dict["val Loss"] = epoch_logs["val_loss"]  # "val" → dashed in PU
        if not data_dict:
            logger.warning("SC-DIAG | P1: no loss data in epoch_logs")
            return
        plot_epoch_lines(
            data_dict   = data_dict,
            output_path = os.path.join(output_dir, "P1_joint_loss.png"),
            title       = "Stage C — Joint Hybrid Loss Over Epochs",
            ylabel      = "Loss",
        )
    except Exception as e:
        logger.error(f"SC-DIAG | P1 failed: {e}")


def _plot_p2_per_expert_nll(
    nll_c: Optional[Dict], expert_names: List[str], output_dir: str
) -> None:
    """P2: Per-expert NLL bars after Stage C joint training."""
    if nll_c is None:
        logger.warning("SC-DIAG | P2 skipped — nll_c collection failed")
        return
    try:
        nll_means = nll_c.get("per_expert_nll_mean", {})
        if not nll_means:
            logger.warning("SC-DIAG | P2: per_expert_nll_mean empty")
            return
        mix_nll = nll_c.get("mixture_nll_mean", float("nan"))
        plot_expert_bars(
            data_dict   = nll_means,
            output_path = os.path.join(output_dir, "P2_per_expert_nll.png"),
            title       = "Stage C — Per-Expert NLL",
            ylabel      = "NLL",
            hline       = (mix_nll, f"Mixture NLL={mix_nll:.3f}"),
        )
    except Exception as e:
        logger.error(f"SC-DIAG | P2 failed: {e}")


def _plot_p3_residual_dist(resid_c: Optional[Dict], output_dir: str) -> None:
    """P3: Physics residual ‖Ax̂ - y‖² distribution histogram."""
    if resid_c is None:
        logger.warning("SC-DIAG | P3 skipped — resid_c collection failed")
        return
    try:
        residuals = resid_c.get("residuals_all")
        if residuals is None or len(residuals) == 0:
            logger.warning("SC-DIAG | P3: residuals_all empty")
            return
        if hasattr(residuals, "numpy"):
            residuals = residuals.numpy()
        plot_histogram(
            data        = np.asarray(residuals, dtype=float),
            output_path = os.path.join(output_dir, "P3_residual_dist.png"),
            title       = "Stage C — Physics Residual Distribution",
            xlabel      = "‖Ax̂ − y‖²",
            ylabel      = "Density",
            vline_mean  = True,
        )
    except Exception as e:
        logger.error(f"SC-DIAG | P3 failed: {e}")


def _plot_p3b_gate_weights_epochs(
    epoch_logs: Dict, expert_names: List[str], output_dir: str
) -> None:
    """
    P3b (optional): Per-expert mean gate weight over Stage C epochs.
    Only generated when epoch_logs["gate_weights"] is present (CSMF-MAIN v1.3.22+).
    Mirrors SB-DIAG P3 transpose logic (epoch-major → expert-major).
    """
    try:
        gw_epochs = epoch_logs.get("gate_weights", [])
        if not gw_epochs:
            return
        K_actual = len(gw_epochs[0]) if gw_epochs else 0
        if K_actual == 0:
            return
        K_plot    = min(len(expert_names), K_actual)
        uniform   = round(1.0 / K_plot, 4)
        data_dict = {
            (expert_names[k] if k < len(expert_names) else f"Expert {k}"):
            [float(gw_epochs[ep][k]) for ep in range(len(gw_epochs))]
            for k in range(K_plot)
        }
        plot_epoch_lines(
            data_dict   = data_dict,
            output_path = os.path.join(output_dir, "P3b_gate_weights_epochs.png"),
            title       = "Stage C — Gate Weights Over Epochs",
            ylabel      = "Mean Gate Weight",
            hlines      = [(uniform, f"Uniform 1/K={uniform}", "black")],
        )
    except Exception as e:
        logger.error(f"SC-DIAG | P3b failed: {e}")


def _plot_p3c_alive_experts(
    epoch_logs: Dict, expert_names: List[str], K: int, output_dir: str
) -> None:
    """
    P3c (optional): Alive experts vs epoch — number of experts with mean gate
    weight >= _ALIVE_THRESHOLD per epoch, plotted as a step line.

    Mirrors SB-DIAG P6 for Stage C. Shows exact epoch at which experts
    disappear during joint fine-tuning.
    """
    try:
        gw_epochs = epoch_logs.get("gate_weights", [])
        if not gw_epochs:
            return

        K_actual = len(gw_epochs[0]) if gw_epochs else 0
        if K_actual == 0:
            return

        K_plot = min(K, K_actual)
        alive_counts = []
        for gw in gw_epochs:
            w_arr = np.array(gw[:K_plot], dtype=float)
            alive_counts.append(int(np.sum(w_arr >= _ALIVE_THRESHOLD)))

        # Log first disappearance epoch
        for ep, cnt in enumerate(alive_counts):
            if cnt < K_plot:
                logger.warning(
                    f"SC-DIAG | P3c: first expert disappears at epoch {ep + 1} "
                    f"(alive={cnt}/{K_plot}, threshold={_ALIVE_THRESHOLD})"
                )
                break
        else:
            logger.info(
                f"SC-DIAG | P3c: all {K_plot} experts alive throughout Stage C "
                f"(threshold={_ALIVE_THRESHOLD})"
            )

        plot_epoch_lines(
            data_dict   = {"Alive experts": alive_counts},
            output_path = os.path.join(output_dir, "P3c_alive_experts.png"),
            title       = f"Stage C — Alive Experts vs Epoch (threshold={_ALIVE_THRESHOLD})",
            ylabel      = "# Alive Experts",
            hlines      = [(K_plot, f"K={K_plot} (all alive)", "green")],
            drawstyle   = "steps-post",
        )
    except Exception as e:
        logger.error(f"SC-DIAG | P3c failed: {e}")


def _plot_p4_residual_epochs(
    epoch_logs: Dict, has_residual: bool, logs_ok: bool, output_dir: str
) -> None:
    """P4: Physics residual ‖Ax̂ - y‖² over Stage C epochs."""
    if not logs_ok:
        logger.warning("SC-DIAG | P4 skipped — epoch_logs validation failed")
        return
    if not has_residual:
        logger.warning(
            "SC-DIAG | P4 skipped — 'residual' key absent from epoch_logs. "
            "Requires CSMF-MAIN v1.3.22+."
        )
        return
    try:
        residuals = epoch_logs.get("residual", [])
        if not residuals:
            logger.warning("SC-DIAG | P4: residual list empty")
            return
        plot_epoch_lines(
            data_dict   = {"‖Ax̂ − y‖²": residuals},
            output_path = os.path.join(output_dir, "P4_residual_epochs.png"),
            title       = "Stage C — Measurement Residual Over Epochs",
            ylabel      = "‖Ax̂ − y‖²",
            nan_safe    = True,
        )
    except Exception as e:
        logger.error(f"SC-DIAG | P4 failed: {e}")


def _plot_p5_recon_grid(
    recon_c: Optional[Dict], expert_names: List[str], output_dir: str
) -> None:
    """
    P5: Final reconstruction grid from Stage C model (encode→decode via MU).
    2-row layout: row 0 = degraded y, row 1 = mixture x̂.
    """
    if recon_c is None:
        logger.warning("SC-DIAG | P5 skipped — recon_c collection failed")
        return
    try:
        y_imgs  = recon_c.get("y")
        xh_dict = recon_c.get("x_hat", {})

        # For Stage C grid, use first expert's reconstruction as representative
        # (mixture reconstruction via csmf.sample() is in MU.collect_reconstruction_metrics)
        # Here we show per-expert encode→decode from collect_reconstruction_batch
        if not xh_dict or y_imgs is None:
            logger.warning("SC-DIAG | P5: no image data from collect_reconstruction_batch")
            return

        xhat_named = {
            expert_names[k]: v
            for k, v in xh_dict.items()
            if k < len(expert_names) and v is not None
        }

        plot_reconstruction_grid(
            y           = y_imgs,
            output_path = os.path.join(output_dir, "P5_recon_grid_final.png"),
            title       = "Stage C — Final Reconstruction Grid (encode→decode per expert)",
            x_clean     = recon_c.get("x_clean"),
            x_hat       = xhat_named,
        )
    except Exception as e:
        logger.error(f"SC-DIAG | P5 failed: {e}")


def _plot_p6_recon_snapshots(
    epoch_logs: Dict, has_snapshots: bool, output_dir: str
) -> None:
    """P6: Reconstruction quality snapshots over Stage C epochs."""
    if not has_snapshots:
        logger.warning(
            "SC-DIAG | P6 skipped — 'recon_snapshots' key absent from epoch_logs. "
            "Requires CSMF-MAIN v1.3.22+."
        )
        return
    try:
        snapshots = epoch_logs.get("recon_snapshots", [])
        if not snapshots:
            logger.warning("SC-DIAG | P6: recon_snapshots empty")
            return
        plot_reconstruction_snapshots(
            snapshots   = snapshots,
            output_path = os.path.join(output_dir, "P6_recon_snapshots.png"),
            title       = "Stage C — Reconstruction Quality Over Epochs",
        )
    except Exception as e:
        logger.error(f"SC-DIAG | P6 failed: {e}")


def _plot_p7_b_vs_c(
    b_data:       Dict,
    nll_c:        Optional[Dict],
    gate_c:       Optional[Dict],
    resid_c:      Optional[Dict],
    expert_names: List[str],
    output_dir:   str,
) -> None:
    """
    P7: B vs C comparison — 4 sub-plots saved as separate PNGs.

    P7a: NLL per expert + mixture (requires per_expert_nll_mean in b_data)
    P7b: Gate weights per expert
    P7c: Neff (single value per stage)
    P7d: Residual boxplot (requires residuals_all tensor in b_data)
    """
    if not b_data:
        logger.warning("SC-DIAG | P7 skipped — no Stage B data available")
        return

    # --- P7a: NLL comparison ---
    b_nll = b_data.get("per_expert_nll_mean")
    c_nll = nll_c.get("per_expert_nll_mean") if nll_c else None
    if b_nll and c_nll:
        try:
            b_mix = b_data.get("mixture_nll_mean", float("nan"))
            c_mix = nll_c.get("mixture_nll_mean", float("nan")) if nll_c else float("nan")
            b_bars = dict(b_nll)
            c_bars = dict(c_nll)
            b_bars["Mixture"] = float(b_mix)
            c_bars["Mixture"] = float(c_mix)
            # Align keys
            all_keys = list(dict.fromkeys(list(b_bars.keys()) + list(c_bars.keys())))
            b_aligned = {k: b_bars.get(k, float("nan")) for k in all_keys}
            c_aligned = {k: c_bars.get(k, float("nan")) for k in all_keys}
            plot_comparison_bars(
                b_dict      = b_aligned,
                c_dict      = c_aligned,
                output_path = os.path.join(output_dir, "P7a_bvc_nll.png"),
                title       = "Stage B vs C — NLL Comparison",
                ylabel      = "NLL",
            )
        except Exception as e:
            logger.error(f"SC-DIAG | P7a failed: {e}")
    else:
        logger.warning(
            "SC-DIAG | P7a skipped — per_expert_nll_mean absent from Stage B data. "
            "This is expected when B data came from stage_b_summary.json (no NLL stored). "
            "Use checkpoint fallback or add NLL to SB-DIAG JSON in a future version."
        )

    # --- P7b: Gate weights comparison ---
    b_gw = b_data.get("gate_weights_mean")
    c_gw = gate_c.get("gate_weights_mean") if gate_c else None
    if b_gw and c_gw:
        try:
            all_keys  = list(dict.fromkeys(list(b_gw.keys()) + list(c_gw.keys())))
            b_aligned = {k: b_gw.get(k, float("nan")) for k in all_keys}
            c_aligned = {k: c_gw.get(k, float("nan")) for k in all_keys}
            plot_comparison_bars(
                b_dict      = b_aligned,
                c_dict      = c_aligned,
                output_path = os.path.join(output_dir, "P7b_bvc_gate_weights.png"),
                title       = "Stage B vs C — Gate Weights",
                ylabel      = "Mean Gate Weight",
            )
        except Exception as e:
            logger.error(f"SC-DIAG | P7b failed: {e}")
    else:
        logger.warning("SC-DIAG | P7b skipped — gate_weights_mean absent from B or C data")

    # --- P7c: Neff comparison ---
    b_neff_val = b_data.get("neff_mean")
    c_neff_val = gate_c.get("neff_mean") if gate_c else None
    if b_neff_val is not None and c_neff_val is not None:
        try:
            plot_comparison_bars(
                b_dict      = {"Neff": float(b_neff_val)},
                c_dict      = {"Neff": float(c_neff_val)},
                output_path = os.path.join(output_dir, "P7c_bvc_neff.png"),
                title       = "Stage B vs C — Effective Expert Count (Neff)",
                ylabel      = "Neff",
                hline       = (1.5, "Target Neff ≥ 1.5"),
            )
        except Exception as e:
            logger.error(f"SC-DIAG | P7c failed: {e}")
    else:
        logger.warning("SC-DIAG | P7c skipped — neff_mean absent from B or C data")

    # --- P7d: Residual boxplot ---
    b_resid_all = b_data.get("residuals_all")
    c_resid_all = resid_c.get("residuals_all") if resid_c else None
    if b_resid_all is not None and c_resid_all is not None:
        try:
            b_arr = b_resid_all.numpy() if hasattr(b_resid_all, "numpy") else np.asarray(b_resid_all)
            c_arr = c_resid_all.numpy() if hasattr(c_resid_all, "numpy") else np.asarray(c_resid_all)
            plot_residual_boxplot(
                b_residuals = b_arr,
                c_residuals = c_arr,
                output_path = os.path.join(output_dir, "P7d_bvc_residual.png"),
            )
        except Exception as e:
            logger.error(f"SC-DIAG | P7d failed: {e}")
    else:
        logger.warning(
            "SC-DIAG | P7d skipped — residuals_all absent from B or C data. "
            "residuals_all is only available when Stage B data came from "
            "checkpoint fallback (not stage_b_summary.json)."
        )


# =============================================================================
# Per-component loss helper [v2.2]
# =============================================================================

def _plot_p_loss_components(
    epoch_logs:    Dict,
    has_loss_comps: bool,
    output_dir:    str,
) -> None:
    """P_loss_components_c1: Per-component reconstruction-first Stage C loss over epochs.

    [v2.3 SC-RECFIRST] Plots cons_c1, img_loss, alive_penalty as separate lines.
    Replaces old NLL/SW2/cal plot — those keys no longer present in Stage C.
    Falls back gracefully: logs warning and returns if keys absent.
    Non-fatal.
    """
    if not has_loss_comps:
        logger.warning(
            "SC-DIAG | P_loss_components_c1 skipped — SC-RECFIRST keys absent "
            "(cons_c1 not in epoch_logs; old NLL-based Stage C?)"
        )
        return
    try:
        data = {}
        for key, label in [
            ("cons_c1",       "Consistency λ_cons·‖A(x̂_mix)−y‖²"),
            ("img_loss",      "Image Recon λ_img·‖x̂_mix−x‖²"),
            ("alive_penalty", "Alive Penalty λ_alive·max(0,1.5−Neff)"),
        ]:
            vals = epoch_logs.get(key, [])
            if vals:
                data[label] = vals
        if not data:
            logger.warning("SC-DIAG | P_loss_components_c1: all component lists empty")
            return
        plot_epoch_lines(
            data_dict   = data,
            output_path = os.path.join(output_dir, "P_loss_components_c1.png"),
            title       = "Stage C — Reconstruction-First Loss Components per Epoch",
            ylabel      = "Loss",
        )
        logger.info("SC-DIAG | P_loss_components_c1 saved")
    except Exception as e:
        logger.error(f"SC-DIAG | P_loss_components_c1 failed: {e}")


def _plot_p_neff_c1(
    epoch_logs: Dict,
    output_dir: str,
) -> None:
    """P_neff_c1: Neff over Stage C epochs from reconstruction-first loss.

    [v2.3 SC-RECFIRST] Reads neff_c1 from epoch_logs (computed inside
    forward_stage_c()). Plots as a line with a dashed collapse floor at 1.5
    (the alive_penalty threshold). Non-fatal.
    """
    vals = epoch_logs.get("neff_c1", [])
    if not vals:
        logger.warning("SC-DIAG | P_neff_c1 skipped — neff_c1 absent from epoch_logs")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(vals, color="steelblue", linewidth=2, label="Neff (C1)")
        ax.axhline(1.5, color="orange", linestyle="--", linewidth=1.5,
                   label="Alive penalty threshold (1.5)")
        ax.axhline(1.0, color="red",    linestyle=":",  linewidth=1.5,
                   label="Collapse floor (1.0)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Neff")
        ax.set_title("Stage C — Neff over Epochs (Reconstruction-First)",
                     fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0.8)
        fig.tight_layout()
        out = os.path.join(output_dir, "P_neff_c1.png")
        fig.savefig(out, dpi=120)
        plt.close(fig)
        logger.info("SC-DIAG | P_neff_c1 saved: %s", out)
    except Exception as e:
        logger.error("SC-DIAG | P_neff_c1 failed: %s", e)


# =============================================================================
# JSON summary builder
# =============================================================================

def _safe_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _safe_list(values: list) -> list:
    out = []
    for v in values:
        try:
            fv = float(v)
            out.append(fv if np.isfinite(fv) else None)
        except (TypeError, ValueError):
            out.append(None)
    return out


def _safe_list_of_lists(values: list) -> list:
    return [
        [_safe_float(w) for w in row]
        for row in values
    ]


def _build_summary(
    epoch_logs:   Dict[str, list],
    expert_names: List[str],
    nll_c:        Optional[Dict],
    gate_c:       Optional[Dict],
    resid_c:      Optional[Dict],
    b_data:       Dict,
    logs_ok:      bool,
) -> Dict[str, Any]:
    """
    Build the full stage_c_summary.json payload.

    Epoch arrays preserved in full. Tensors converted to Python scalars.
    stage_b_vs_c block uses scalar means only (no tensor arrays stored).
    """
    # Epoch logs — preserve full arrays for downstream analysis
    epoch_serial: Dict[str, Any] = {}
    for key in ("train_loss", "val_loss", "neff", "tau"):
        epoch_serial[key] = _safe_list(epoch_logs.get(key, []))
    gw = epoch_logs.get("gate_weights", [])
    epoch_serial["gate_weights"] = _safe_list_of_lists(gw) if gw else []
    resid_ep = epoch_logs.get("residual", [])
    epoch_serial["residual"] = _safe_list(resid_ep) if resid_ep else []
    # recon_snapshots: not serialised (tensors) — omitted from JSON
    epoch_serial["recon_snapshots_count"] = len(epoch_logs.get("recon_snapshots", []))

    # Stage C val metrics
    val_metrics_c: Dict[str, Any] = {}
    if nll_c:
        val_metrics_c["per_expert_nll_mean"]  = {
            k: _safe_float(v) for k, v in nll_c["per_expert_nll_mean"].items()
        }
        val_metrics_c["mixture_nll_mean"] = _safe_float(nll_c["mixture_nll_mean"])
    if gate_c:
        val_metrics_c["neff_mean"]         = _safe_float(gate_c["neff_mean"])
        val_metrics_c["gate_weights_mean"] = {
            k: _safe_float(v) for k, v in gate_c["gate_weights_mean"].items()
        }
        val_metrics_c["gate_winner_counts"] = gate_c.get("gate_winner_counts", {})
    if resid_c:
        val_metrics_c["residual_mean"] = _safe_float(resid_c["residual_mean"])
        val_metrics_c["residual_std"]  = _safe_float(resid_c["residual_std"])

    # B-vs-C delta (scalars only)
    bvc: Dict[str, Any] = {}
    b_neff = _safe_float(b_data.get("neff_mean"))
    c_neff = _safe_float(gate_c.get("neff_mean")) if gate_c else None
    if b_neff is not None and c_neff is not None:
        bvc["neff"] = {"B": b_neff, "C": c_neff, "delta": round(c_neff - b_neff, 4)}

    b_resid = _safe_float(b_data.get("residual_mean"))
    c_resid = _safe_float(resid_c.get("residual_mean")) if resid_c else None
    if b_resid is not None and c_resid is not None:
        bvc["residual_mean"] = {
            "B": b_resid, "C": c_resid,
            "delta": round(c_resid - b_resid, 6),
        }

    b_gw = b_data.get("gate_weights_mean")
    c_gw = gate_c.get("gate_weights_mean") if gate_c else None
    if b_gw and c_gw:
        bvc["gate_weights"] = {"B": b_gw, "C": c_gw}

    b_nll = b_data.get("per_expert_nll_mean")
    c_nll = nll_c.get("per_expert_nll_mean") if nll_c else None
    if b_nll and c_nll:
        bvc["per_expert_nll"] = {
            "B": {k: _safe_float(v) for k, v in b_nll.items()},
            "C": {k: _safe_float(v) for k, v in c_nll.items()},
        }

    return {
        "metadata": {
            "timestamp":    datetime.datetime.now().isoformat(timespec="seconds"),
            "expert_names": expert_names,
            "diag_version": "SC-DIAG-v2.0",
        },
        "epoch_logs":       epoch_serial,
        "val_metrics_c":    val_metrics_c,
        "stage_b_vs_c":     bvc,
    }
