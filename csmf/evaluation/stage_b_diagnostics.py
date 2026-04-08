# =============================================================================
# Version: DIAG-REORG-StageBDiag-v1.2 | Abbr: SB-DIAG
# Description: Stage B diagnostic runner — gate behaviour after gate-network
#              training. Delegates all plotting to PU v1.0. Saves
#              stage_b_summary.json for downstream analysis and B-vs-C
#              comparison by SC-DIAG v2.1. All plots are non-fatal.
#
#              Note: csmf.py _save_stage_b_diagnostics() still runs internally
#              inside train_stage_B() and saves 2 basic plots to its own
#              results_dir. SB-DIAG produces richer plots to output_dir
#              (stage_b_diagnostics/) — no file conflicts.
#
# Changelog:
#   v1.2 (2026-04-07): [DIAG-OUTPUT] Add P_recon reconstruction grid and
#                      P_loss_components per-component loss plot; run() gains
#                      optional model + val_loader params for P_recon (silently
#                      skipped if omitted); P_loss_components uses nll_loss,
#                      cons_loss, trans_loss, cal_loss from epoch_logs
#                      (CSMF-MAIN v1.3.29+); optional_present check guards both
#   v1.1 (2026-04-06): [ALIVE-PLOT] Add P6 alive experts vs epoch — counts
#                      experts with gate_weight >= _ALIVE_THRESHOLD (0.05) per
#                      epoch; step-line plot with K baseline; complements P2
#                      (Neff) and P3 (weights) to show exact disappearance epoch;
#                      _ALIVE_THRESHOLD constant added at module level
#   v1.0 (2026-04-04): Initial implementation — 5 plots (P1-P5) sourced
#                      from csmf._save_stage_b_diagnostics() (P2, P3) and
#                      new plots (P1 gate loss, P4 tau, P5 winner hist);
#                      P4 requires CSMF-MAIN v1.3.23+ for tau key;
#                      gate_weights transposed from epoch-major to expert-major
#                      before passing to PU.plot_epoch_lines; P5 winner hist
#                      counts argmax(w_k) per epoch across training;
#                      stage_b_summary.json includes all epoch arrays
# Dependencies: LS v1.0, PU v1.0, numpy
# =============================================================================

import datetime
import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

from .log_schema import validate_stage_b_logs, available_optional_keys_b
from .metric_utils import collect_reconstruction_batch
from .plot_utils import (
    plot_epoch_lines,
    plot_expert_bars,
    plot_reconstruction_grid,
)

logger = logging.getLogger(__name__)

# Collapse threshold preserved from csmf.py _save_stage_b_diagnostics
_NEFF_COLLAPSE_THRESHOLD = 1.1
# Gate weight threshold below which an expert is considered "dead"
_ALIVE_THRESHOLD = 0.05


# =============================================================================
# Main entry point
# =============================================================================

def run(
    epoch_logs:   Dict[str, list],
    expert_names: List[str],
    output_dir:   str,
    hyperparams:  Optional[Dict[str, Any]] = None,
    model:        Optional[Any] = None,
    val_loader:   Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run Stage B diagnostics after gate-network training.

    Pure epoch_logs consumer — no model or val_loader needed. Called from
    train_csmf.py immediately after train_stage_B() returns.

    Args:
        epoch_logs   : Dict returned by train_stage_B() — must satisfy
                       StageBEpochLogs contract (LS v1.0). Required keys:
                       train_loss, neff, gate_weights. Optional: val_loss, tau.
                       tau key requires CSMF-MAIN v1.3.23+.
        expert_names : List of expert class name strings (same order as K dim
                       in gate_weights). E.g. ["ConditionalRealNVP", ...].
        output_dir   : Directory for plots + JSON (created if absent).
        hyperparams  : Optional dict of training hyperparams to embed in JSON.
                       Recommended keys: lambda_neff, tau_start, tau_end,
                       lambda_cons, lambda_trans, lambda_cal, early_stopped,
                       best_val_loss. No keys are required.

    Returns:
        summary dict (also saved to stage_b_summary.json).
    """
    os.makedirs(output_dir, exist_ok=True)
    hyperparams = hyperparams or {}
    logger.info(
        f"SB-DIAG | Starting Stage B diagnostics | output_dir={output_dir}"
    )

    # ------------------------------------------------------------------
    # Step 1: Validate epoch_logs (LS)
    # ------------------------------------------------------------------
    logs_ok, missing_keys = validate_stage_b_logs(epoch_logs)
    if not logs_ok:
        logger.error(
            f"SB-DIAG | epoch_logs validation failed — missing: {missing_keys}. "
            f"Dependent plots will be skipped."
        )

    optional_present  = available_optional_keys_b(epoch_logs)
    has_val_loss      = "val_loss"   in optional_present
    has_tau           = "tau"        in optional_present
    has_loss_comps    = "nll_loss"   in optional_present   # [v1.2]

    if not has_tau:
        logger.warning(
            "SB-DIAG | 'tau' key absent from epoch_logs — P4 will be skipped. "
            "Requires CSMF-MAIN v1.3.23+."
        )
    if not has_loss_comps:
        logger.warning(
            "SB-DIAG | per-component loss keys absent from epoch_logs — "
            "P_loss_components will be skipped. Requires CSMF-MAIN v1.3.29+."
        )

    K = len(expert_names)

    # ------------------------------------------------------------------
    # Step 2: Plots (all non-fatal)
    # ------------------------------------------------------------------

    # P1: Gate loss train/val over epochs
    _plot_p1_gate_loss(epoch_logs, has_val_loss, logs_ok, output_dir)

    # P2: Neff over epochs
    _plot_p2_neff(epoch_logs, logs_ok, output_dir)

    # P3: Gate weights over epochs (one line per expert)
    _plot_p3_gate_weights(epoch_logs, expert_names, K, logs_ok, output_dir)

    # P4: Tau over epochs (requires CSMF-MAIN v1.3.23+)
    _plot_p4_tau(epoch_logs, has_tau, output_dir)

    # P5: Gate winner histogram (argmax per epoch)
    _plot_p5_winner_hist(epoch_logs, expert_names, K, logs_ok, output_dir)

    # P6: Alive experts vs epoch (gate_weight >= _ALIVE_THRESHOLD)
    _plot_p6_alive_experts(epoch_logs, expert_names, K, logs_ok, output_dir)

    # P_recon: Reconstruction grid — requires model + val_loader [v1.2]
    _plot_p_recon(model, val_loader, expert_names, output_dir)

    # P_loss_components: Per-component loss over epochs [v1.2]
    _plot_p_loss_components(epoch_logs, has_loss_comps, output_dir)

    # ------------------------------------------------------------------
    # Step 3: Build and save stage_b_summary.json
    # ------------------------------------------------------------------
    summary = _build_summary(
        epoch_logs   = epoch_logs,
        expert_names = expert_names,
        K            = K,
        hyperparams  = hyperparams,
        has_val_loss = has_val_loss,
        has_tau      = has_tau,
    )

    json_path = os.path.join(output_dir, "stage_b_summary.json")
    try:
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"SB-DIAG | Summary saved: {json_path}")
    except Exception as e:
        logger.error(f"SB-DIAG | Failed to save stage_b_summary.json: {e}")

    # Log final state
    final = summary.get("final", {})
    logger.info(
        f"SB-DIAG | Complete | "
        f"final_neff={final.get('neff', 'N/A')} | "
        f"gate_weights={final.get('gate_weights', 'N/A')} | "
        f"collapse={'YES' if _neff_collapsed(epoch_logs) else 'NO'}"
    )
    return summary


# =============================================================================
# Plot helpers (all non-fatal)
# =============================================================================

def _plot_p1_gate_loss(
    epoch_logs: Dict, has_val_loss: bool, logs_ok: bool, output_dir: str
) -> None:
    """P1: Gate hybrid loss train (+ optional val) over epochs."""
    if not logs_ok:
        logger.warning("SB-DIAG | P1 skipped — epoch_logs required keys missing")
        return
    try:
        data_dict: Dict[str, list] = {}
        train_loss = epoch_logs.get("train_loss", [])
        if train_loss:
            data_dict["Train Loss"] = train_loss
        else:
            logger.warning("SB-DIAG | P1: train_loss empty — skipping")
            return

        if has_val_loss:
            val_loss = epoch_logs.get("val_loss", [])
            if val_loss:
                data_dict["val Loss"] = val_loss   # "val" prefix → dashed in PU

        plot_epoch_lines(
            data_dict   = data_dict,
            output_path = os.path.join(output_dir, "P1_gate_loss.png"),
            title       = "Stage B — Gate Hybrid Loss Over Epochs",
            ylabel      = "Hybrid Loss",
        )
    except Exception as e:
        logger.error(f"SB-DIAG | P1 failed: {e}")


def _plot_p2_neff(
    epoch_logs: Dict, logs_ok: bool, output_dir: str
) -> None:
    """P2: Neff over epochs with collapse threshold hline."""
    if not logs_ok:
        logger.warning("SB-DIAG | P2 skipped — epoch_logs required keys missing")
        return
    try:
        neff = epoch_logs.get("neff", [])
        if not neff:
            logger.warning("SB-DIAG | P2: neff list empty — skipping")
            return

        plot_epoch_lines(
            data_dict   = {"Neff": neff},
            output_path = os.path.join(output_dir, "P2_neff_epochs.png"),
            title       = "Stage B — Effective Expert Count (Neff) Over Epochs",
            ylabel      = "Neff",
            hlines      = [
                (_NEFF_COLLAPSE_THRESHOLD,
                 f"Collapse threshold ({_NEFF_COLLAPSE_THRESHOLD})",
                 "red"),
            ],
        )
    except Exception as e:
        logger.error(f"SB-DIAG | P2 failed: {e}")


def _plot_p3_gate_weights(
    epoch_logs: Dict, expert_names: List[str], K: int, logs_ok: bool, output_dir: str
) -> None:
    """
    P3: Per-expert mean gate weight over epochs.

    gate_weights in epoch_logs is epoch-major: [[w0,w1,w2], [w0,w1,w2], ...].
    Transposes to expert-major for plot_epoch_lines: {name: [w_ep0, w_ep1, ...]}.
    """
    if not logs_ok:
        logger.warning("SB-DIAG | P3 skipped — epoch_logs required keys missing")
        return
    try:
        gw_epochs = epoch_logs.get("gate_weights", [])
        if not gw_epochs:
            logger.warning("SB-DIAG | P3: gate_weights empty — skipping")
            return

        # Validate shape: each entry must have K elements
        K_actual = len(gw_epochs[0]) if gw_epochs else 0
        if K_actual == 0:
            logger.error("SB-DIAG | P3: gate_weights entries have length 0")
            return
        if K_actual != K:
            logger.warning(
                f"SB-DIAG | P3: gate_weights K={K_actual} != expert_names K={K} "
                f"— using K_actual={K_actual} entries"
            )
            K_plot = min(K, K_actual)
        else:
            K_plot = K

        # Transpose: epoch-major → expert-major
        data_dict: Dict[str, list] = {}
        for k in range(K_plot):
            name = expert_names[k] if k < len(expert_names) else f"Expert {k}"
            data_dict[name] = [float(gw_epochs[ep][k]) for ep in range(len(gw_epochs))]

        uniform = round(1.0 / K_plot, 4)
        plot_epoch_lines(
            data_dict   = data_dict,
            output_path = os.path.join(output_dir, "P3_gate_weights.png"),
            title       = "Stage B — Gate Weights Over Epochs",
            ylabel      = "Mean Gate Weight",
            hlines      = [(uniform, f"Uniform 1/K={uniform}", "black")],
        )
    except Exception as e:
        logger.error(f"SB-DIAG | P3 failed: {e}")


def _plot_p4_tau(
    epoch_logs: Dict, has_tau: bool, output_dir: str
) -> None:
    """P4: Temperature schedule over epochs. Requires CSMF-MAIN v1.3.23+."""
    if not has_tau:
        logger.warning(
            "SB-DIAG | P4 skipped — 'tau' key absent. "
            "Retrain with CSMF-MAIN v1.3.23+ to enable."
        )
        return
    try:
        tau = epoch_logs.get("tau", [])
        if not tau:
            logger.warning("SB-DIAG | P4: tau list empty — skipping")
            return

        plot_epoch_lines(
            data_dict   = {"Tau (temperature)": tau},
            output_path = os.path.join(output_dir, "P4_tau_epochs.png"),
            title       = "Stage B — Gate Temperature (τ) Annealing Schedule",
            ylabel      = "τ",
        )
    except Exception as e:
        logger.error(f"SB-DIAG | P4 failed: {e}")


def _plot_p5_winner_hist(
    epoch_logs: Dict, expert_names: List[str], K: int, logs_ok: bool, output_dir: str
) -> None:
    """
    P5: Gate winner histogram — for each epoch, which expert has the highest
    mean gate weight? Histogram of epoch-level winners across all epochs.

    Complements P3 (absolute values over time) by showing dominance distribution.
    High concentration → gate collapse risk; balanced → healthy mixture.
    """
    if not logs_ok:
        logger.warning("SB-DIAG | P5 skipped — epoch_logs required keys missing")
        return
    try:
        gw_epochs = epoch_logs.get("gate_weights", [])
        if not gw_epochs:
            logger.warning("SB-DIAG | P5: gate_weights empty — skipping")
            return

        K_actual = len(gw_epochs[0]) if gw_epochs else 0
        if K_actual < 2:
            logger.warning(
                f"SB-DIAG | P5: need K >= 2 to compute winner, got K={K_actual}"
            )
            return

        K_plot = min(K, K_actual)

        # Argmax per epoch → winner index
        winner_counts: Dict[str, int] = {
            (expert_names[k] if k < len(expert_names) else f"Expert {k}"): 0
            for k in range(K_plot)
        }
        names_ordered = list(winner_counts.keys())

        for gw in gw_epochs:
            w_arr   = np.array(gw[:K_plot], dtype=float)
            winner  = int(np.argmax(w_arr))
            winner_counts[names_ordered[winner]] += 1

        total_epochs  = len(gw_epochs)
        winner_pct    = {
            n: round(c / total_epochs * 100, 2)
            for n, c in winner_counts.items()
        }

        # Log collapse warning if one expert wins > 80% of epochs
        dominant = max(winner_pct, key=winner_pct.get)
        if winner_pct[dominant] > 80.0:
            logger.warning(
                f"SB-DIAG | P5: gate collapse risk — '{dominant}' wins "
                f"{winner_pct[dominant]:.1f}% of epochs"
            )
        else:
            logger.info(
                f"SB-DIAG | P5: epoch winner distribution = {winner_pct}"
            )

        plot_expert_bars(
            data_dict   = winner_pct,
            output_path = os.path.join(output_dir, "P5_gate_winner_hist.png"),
            title       = "Stage B — Gate Winner Per Epoch (argmax of mean weight)",
            ylabel      = "Win Rate (% of epochs)",
        )
    except Exception as e:
        logger.error(f"SB-DIAG | P5 failed: {e}")


def _plot_p6_alive_experts(
    epoch_logs: Dict, expert_names: List[str], K: int, logs_ok: bool, output_dir: str
) -> None:
    """
    P6: Alive experts vs epoch — number of experts with mean gate weight
    >= _ALIVE_THRESHOLD per epoch, plotted as a step line.

    Complements P2 (Neff) and P3 (absolute weights) by showing the exact
    epoch at which experts disappear from the active mixture.
    """
    if not logs_ok:
        logger.warning("SB-DIAG | P6 skipped — epoch_logs required keys missing")
        return
    try:
        gw_epochs = epoch_logs.get("gate_weights", [])
        if not gw_epochs:
            logger.warning("SB-DIAG | P6: gate_weights empty — skipping")
            return

        K_actual = len(gw_epochs[0]) if gw_epochs else 0
        if K_actual == 0:
            logger.warning("SB-DIAG | P6: gate_weights entries have length 0 — skipping")
            return

        K_plot = min(K, K_actual)
        alive_counts = []
        for gw in gw_epochs:
            w_arr = np.array(gw[:K_plot], dtype=float)
            alive_counts.append(int(np.sum(w_arr >= _ALIVE_THRESHOLD)))

        # Log first epoch where count drops below K
        for ep, cnt in enumerate(alive_counts):
            if cnt < K_plot:
                logger.warning(
                    f"SB-DIAG | P6: first expert disappears at epoch {ep + 1} "
                    f"(alive={cnt}/{K_plot}, threshold={_ALIVE_THRESHOLD})"
                )
                break
        else:
            logger.info(
                f"SB-DIAG | P6: all {K_plot} experts alive throughout Stage B "
                f"(threshold={_ALIVE_THRESHOLD})"
            )

        plot_epoch_lines(
            data_dict   = {"Alive experts": alive_counts},
            output_path = os.path.join(output_dir, "P6_alive_experts.png"),
            title       = f"Stage B — Alive Experts vs Epoch (threshold={_ALIVE_THRESHOLD})",
            ylabel      = "# Alive Experts",
            hlines      = [(K_plot, f"K={K_plot} (all alive)", "green")],
            drawstyle   = "steps-post",
        )
    except Exception as e:
        logger.error(f"SB-DIAG | P6 failed: {e}")


# =============================================================================
# Helper utilities
# =============================================================================

def _plot_p_recon(
    model:        Optional[Any],
    val_loader:   Optional[Any],
    expert_names: List[str],
    output_dir:   str,
) -> None:
    """P_recon: Per-expert encode→decode reconstruction grid. [v1.2]

    Requires model + val_loader. Silently skipped with warning if either is None.
    """
    if model is None or val_loader is None:
        logger.warning(
            "SB-DIAG | P_recon skipped — model or val_loader not provided"
        )
        return
    try:
        import torch
        device = next(model.parameters()).device
        recon_batch = collect_reconstruction_batch(
            csmf_model = model,
            val_loader = val_loader,
            device     = device,
            n_samples  = 8,
        )
        if recon_batch is None:
            logger.error("SB-DIAG | P_recon: collect_reconstruction_batch returned None")
            return
        plot_reconstruction_grid(
            y           = recon_batch.get("y"),
            output_path = os.path.join(output_dir, "P_recon.png"),
            expert_names = expert_names,
            x_clean     = recon_batch.get("x_clean"),
            x_hat       = recon_batch.get("x_hat"),
            title       = "Stage B — Reconstruction Grid (encode→decode per expert)",
        )
    except Exception as e:
        logger.error(f"SB-DIAG | P_recon failed: {e}")


def _plot_p_loss_components(
    epoch_logs:    Dict,
    has_loss_comps: bool,
    output_dir:    str,
) -> None:
    """P_loss_components: Per-component hybrid loss over epochs. [v1.2]

    Plots NLL, consistency, SW2 transport, calibration as separate lines
    on shared axes. Requires CSMF-MAIN v1.3.29+ epoch_logs keys.
    """
    if not has_loss_comps:
        logger.warning("SB-DIAG | P_loss_components skipped — loss component keys absent")
        return
    try:
        data = {}
        for key, label in [
            ("nll_loss",   "NLL"),
            ("cons_loss",  "Consistency (λ_cons)"),
            ("trans_loss", "SW2 Transport (λ_trans)"),
            ("cal_loss",   "Calibration (λ_cal)"),
        ]:
            vals = epoch_logs.get(key, [])
            if vals:
                data[label] = vals
        if not data:
            logger.warning("SB-DIAG | P_loss_components: all component lists empty")
            return
        plot_epoch_lines(
            data_dict   = data,
            output_path = os.path.join(output_dir, "P_loss_components.png"),
            title       = "Stage B — Hybrid Loss Components per Epoch",
            ylabel      = "Loss",
        )
    except Exception as e:
        logger.error(f"SB-DIAG | P_loss_components failed: {e}")


def _neff_collapsed(epoch_logs: Dict) -> bool:
    """Return True if final Neff is below collapse threshold."""
    neff = epoch_logs.get("neff", [])
    if not neff:
        return False
    return float(neff[-1]) < _NEFF_COLLAPSE_THRESHOLD


def _safe_list(values: list) -> list:
    """Convert list of floats to JSON-safe Python floats (None for non-finite)."""
    out = []
    for v in values:
        try:
            fv = float(v)
            out.append(fv if np.isfinite(fv) else None)
        except (TypeError, ValueError):
            out.append(None)
    return out


def _safe_list_of_lists(values: list) -> list:
    """Convert list of K-dim float lists (gate_weights) to JSON-safe form."""
    return [
        [float(w) if np.isfinite(float(w)) else None for w in row]
        for row in values
    ]


# =============================================================================
# JSON summary builder
# =============================================================================

def _build_summary(
    epoch_logs:   Dict[str, list],
    expert_names: List[str],
    K:            int,
    hyperparams:  Dict[str, Any],
    has_val_loss: bool,
    has_tau:      bool,
) -> Dict[str, Any]:
    """
    Build the full stage_b_summary.json payload.

    Epoch arrays are preserved in full for downstream analysis (SC-DIAG B-vs-C,
    further grid search comparison). All tensors converted to Python scalars.
    """
    gw_epochs   = epoch_logs.get("gate_weights", [])
    neff_list   = epoch_logs.get("neff", [])
    tau_list    = epoch_logs.get("tau", [])

    # Final-epoch gate weights per expert
    final_gw: Dict[str, Optional[float]] = {}
    if gw_epochs:
        last_gw = gw_epochs[-1]
        for k in range(min(K, len(last_gw))):
            name = expert_names[k] if k < len(expert_names) else f"Expert {k}"
            final_gw[name] = round(float(last_gw[k]), 4)

    # Gate winner counts across all epochs
    winner_counts: Dict[str, int] = {
        (expert_names[k] if k < len(expert_names) else f"Expert {k}"): 0
        for k in range(K)
    }
    names_ordered = list(winner_counts.keys())
    K_actual      = len(gw_epochs[0]) if gw_epochs else 0
    K_count       = min(K, K_actual)

    for gw in gw_epochs:
        if len(gw) < K_count:
            continue
        winner = int(np.argmax(np.array(gw[:K_count], dtype=float)))
        if winner < len(names_ordered):
            winner_counts[names_ordered[winner]] += 1

    # Collapse flag
    collapsed = _neff_collapsed(epoch_logs)
    if collapsed:
        logger.warning(
            f"SB-DIAG | stage_b_summary: gate collapsed — "
            f"final Neff={neff_list[-1] if neff_list else 'N/A'} "
            f"< {_NEFF_COLLAPSE_THRESHOLD}"
        )

    return {
        "metadata": {
            "timestamp":    datetime.datetime.now().isoformat(timespec="seconds"),
            "expert_names": expert_names,
            "diag_version": "SB-DIAG-v1.0",
            "hyperparams":  hyperparams,
        },
        "epoch_logs": {
            "train_loss":   _safe_list(epoch_logs.get("train_loss", [])),
            "val_loss":     _safe_list(epoch_logs.get("val_loss", [])) if has_val_loss else [],
            "neff":         _safe_list(neff_list),
            "gate_weights": _safe_list_of_lists(gw_epochs),
            "tau":          _safe_list(tau_list) if has_tau else [],
        },
        "final": {
            "neff":               round(float(neff_list[-1]), 4) if neff_list else None,
            "gate_weights":       final_gw,
            "gate_winner_counts": winner_counts,
            "collapsed":          collapsed,
        },
    }
