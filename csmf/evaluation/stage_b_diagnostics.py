# =============================================================================
# Version: DIAG-REORG-StageBDiag-v1.5 | Abbr: SB-DIAG
# Description: Stage B diagnostic runner — gate behaviour after gate-network
#              training. Delegates all plotting to PU v1.5. Saves
#              stage_b_summary.json for downstream analysis and B-vs-C
#              comparison by SC-DIAG v2.5. All plots are non-fatal.
#              All skipped or failed plots logged at WARNING or ERROR level.
#
# Changelog:
#   v1.5 (2026-04-19): [RECON-4COL] Replace P_recon (multi-expert col grid) with
#                      P_recon_4col_{name}.png per expert via PU.plot_recon_panel_4col
#                      stage_label="Stage B"; data from collect_reconstruction_batch
#                      x_hat (cycle) + x_gen (generated); non-fatal if no model.
#                      [P_NEFF_REG] Add P_neff_reg — neff_reg_loss over epochs;
#                      gated on LS v1.7 neff_reg_loss key; non-fatal.
#                      [P_ANNEAL] Add P_anneal_lambdas — effective lambda_*_eff over
#                      epochs; gated on lambda_cons_eff key; non-fatal.
#                      [P_FI_GATE] Add P_fi_gate — FI ratio vs gate weight grouped
#                      bar; uses MU.collect_fi_gate_comparison(); non-fatal if
#                      fi_diag_summary.json absent; run() gains fi_summary_path param.
#                      [LOSS-COMP-EXT] P_loss_components extended with neff_reg_loss.
#   v1.4 (2026-04-14): [EVID-PLOT] Add P8 evidence signals per expert — mean nll_k, r_k, g_k
#                      over epochs (one subplot per signal, one line per expert); reads
#                      evidence_nll, evidence_r, evidence_g from epoch_logs (CSMF-MAIN v2.2+).
#                      [BNORM-PLOT] Add P9 raw NLL std over epochs — verifies batch-norm is
#                      fixing scale problem; reads evidence_nll_raw_std; flags epochs where
#                      std < 0.1 as degenerate with red shading.
#                      [SCORE-SPLIT] Add P10 score component contribution — s_base vs alpha_u*u_k
#                      per expert over epochs; reveals if MLP residual dominates evidence;
#                      reads score_base_mean, score_u_mean.
#                      [EVID-CORR] Add P11 evidence-weight correlation — Pearson r(s_base_k, w_k)
#                      per expert over epochs; positive = evidence driving routing; reads
#                      evid_gate_corr. All four plots non-fatal and guarded on optional keys.
#                      _build_summary() extended with full evidence arrays in epoch_logs JSON
#                      section and evidence_summary scalar stats in final section.
#   v1.3 (2026-04-09): [RECON-SNAP] Add P7 per-expert reconstruction snapshots —
#                      reads epoch_logs["recon_snapshots"][k] (dict keyed by expert
#                      index, CSMF-MAIN v1.3.30+); calls plot_reconstruction_snapshots
#                      once per expert; output P7_recon_snapshots_{name}.png; guarded
#                      directly on "recon_snapshots" in epoch_logs, no log_schema change.
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
from .metric_utils import collect_reconstruction_batch, collect_fi_gate_comparison
from .plot_utils import (
    plot_epoch_lines,
    plot_expert_bars,
    plot_reconstruction_grid,
    plot_reconstruction_snapshots,
    plot_recon_panel_4col,        # [RECON-4COL] v1.5
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
    epoch_logs:      Dict[str, list],
    expert_names:    List[str],
    output_dir:      str,
    hyperparams:     Optional[Dict[str, Any]] = None,
    model:           Optional[Any] = None,
    val_loader:      Optional[Any] = None,
    fi_summary_path: Optional[str] = None,   # [P_FI_GATE] v1.5 — path to fi_diag_summary.json
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
    # [EVID-PLOT/BNORM-PLOT/SCORE-SPLIT/EVID-CORR] v1.4: evidence signal keys
    has_evidence      = "evidence_nll"        in epoch_logs and len(epoch_logs.get("evidence_nll", [])) > 0
    has_bnorm_diag    = "evidence_nll_raw_std" in epoch_logs and len(epoch_logs.get("evidence_nll_raw_std", [])) > 0
    has_score_split   = "score_base_mean"     in epoch_logs and len(epoch_logs.get("score_base_mean", [])) > 0
    has_evid_corr     = "evid_gate_corr"      in epoch_logs and len(epoch_logs.get("evid_gate_corr", [])) > 0
    # [NEFF-REG] v1.5: new LS v1.7 optional keys
    has_neff_reg  = "neff_reg_loss"   in epoch_logs and len(epoch_logs.get("neff_reg_loss", [])) > 0
    has_anneal    = "lambda_cons_eff" in epoch_logs and len(epoch_logs.get("lambda_cons_eff", [])) > 0

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
    if not has_neff_reg:
        logger.warning(
            "SB-DIAG | 'neff_reg_loss' key absent from epoch_logs — "
            "P_neff_reg will be skipped. Requires HYBRID v1.10.0 + LS v1.7."
        )
    if not has_anneal:
        logger.warning(
            "SB-DIAG | 'lambda_cons_eff' key absent from epoch_logs — "
            "P_anneal_lambdas will be skipped. Requires HYBRID v1.10.0 + LS v1.7."
        )
    if fi_summary_path is None:
        logger.warning(
            "SB-DIAG | fi_summary_path not provided — P_fi_gate will be skipped. "
            "Pass path to fi_diag_summary.json to enable FI vs gate comparison."
        )

    K = len(expert_names)

    # ------------------------------------------------------------------
    # Step 2: Plots (all non-fatal; skips logged at WARNING or ERROR)
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

    # P7: Per-expert reconstruction snapshots over epochs [v1.3]
    _plot_p7_recon_snapshots(epoch_logs, expert_names, output_dir)

    # P_recon_4col: Per-expert 4-row panel (Original/Degraded/Cycle/Generated) [v1.5 RECON-4COL]
    # Replaces P_recon (multi-expert col grid) — matches Stage A structure exactly
    _plot_p_recon_4col(model, val_loader, expert_names, output_dir)

    # P_loss_components: Per-component loss over epochs [v1.2 + LOSS-COMP-EXT v1.5]
    _plot_p_loss_components(epoch_logs, has_loss_comps, has_neff_reg, output_dir)

    # P_neff_reg: Neff regularisation loss over epochs [v1.5 P_NEFF_REG]
    _plot_p_neff_reg(epoch_logs, has_neff_reg, output_dir)

    # P_anneal_lambdas: Effective lambda_cons/trans/cal over epochs [v1.5 P_ANNEAL]
    _plot_p_anneal_lambdas(epoch_logs, has_anneal, output_dir)

    # P_fi_gate: FI ratio vs gate weight grouped bar [v1.5 P_FI_GATE]
    _plot_p_fi_gate(model, val_loader, fi_summary_path, expert_names, output_dir)

    # P8: Evidence signals per expert over epochs [v1.4 EVID-PLOT]
    _plot_p8_evidence_signals(epoch_logs, expert_names, K, has_evidence, output_dir)

    # P9: Raw NLL std over epochs — batch-norm sanity check [v1.4 BNORM-PLOT]
    _plot_p9_nll_raw_std(epoch_logs, has_bnorm_diag, output_dir)

    # P10: Score component contribution s_base vs alpha_u*u_k [v1.4 SCORE-SPLIT]
    _plot_p10_score_components(epoch_logs, expert_names, K, has_score_split, output_dir)

    # P11: Evidence–weight Pearson correlation per expert [v1.4 EVID-CORR]
    _plot_p11_evid_gate_corr(epoch_logs, expert_names, K, has_evid_corr, output_dir)

    # ------------------------------------------------------------------
    # Step 3: Build and save stage_b_summary.json
    # ------------------------------------------------------------------
    summary = _build_summary(
        epoch_logs      = epoch_logs,
        expert_names    = expert_names,
        K               = K,
        hyperparams     = hyperparams,
        has_val_loss    = has_val_loss,
        has_tau         = has_tau,
        has_evidence    = has_evidence,
        has_score_split = has_score_split,
        has_evid_corr   = has_evid_corr,
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

def _plot_p7_recon_snapshots(
    epoch_logs:   Dict,
    expert_names: List[str],
    output_dir:   str,
) -> None:
    """P7: Per-expert reconstruction snapshots over Stage B epochs. [v1.3]

    Reads epoch_logs["recon_snapshots"][k] — dict keyed by expert index.
    Requires CSMF-MAIN v1.3.30+. Skipped gracefully if key absent or empty.
    """
    snap_dict = epoch_logs.get("recon_snapshots", None)
    if snap_dict is None or not isinstance(snap_dict, dict):
        logger.warning(
            "SB-DIAG | P7 skipped — 'recon_snapshots' key absent from epoch_logs. "
            "Requires CSMF-MAIN v1.3.30+."
        )
        return
    for k, name in enumerate(expert_names):
        try:
            snapshots = snap_dict.get(k, [])
            if not snapshots:
                logger.warning(f"SB-DIAG | P7: no snapshots for expert '{name}' — skipping")
                continue
            safe_name = name.replace(" ", "_")
            plot_reconstruction_snapshots(
                snapshots   = snapshots,
                output_path = os.path.join(output_dir, f"P7_recon_snapshots_{safe_name}.png"),
                title       = f"Stage B — {name} Reconstruction Over Epochs (encode→decode)",
            )
        except Exception as e:
            logger.error(f"SB-DIAG | P7 failed for expert '{name}': {e}")


def _plot_p_recon_4col(
    model:        Optional[Any],
    val_loader:   Optional[Any],
    expert_names: List[str],
    output_dir:   str,
) -> None:
    """P_recon_4col: Per-expert 4-row panel (Original/Degraded/Cycle/Generated). [v1.5 RECON-4COL]

    Replaces P_recon (multi-expert col grid). Matches Stage A structure exactly.
    Requires model + val_loader. Logged at WARNING if either is None.
    """
    if model is None or val_loader is None:
        logger.warning(
            "SB-DIAG | P_recon_4col skipped — model or val_loader not provided. "
            "Pass model and val_loader to run() to enable per-expert 4-col panels."
        )
        return
    try:
        import torch
        device      = next(model.parameters()).device
        recon_batch = collect_reconstruction_batch(
            csmf_model = model,
            val_loader = val_loader,
            device     = device,
            n_samples  = 8,
        )
        if recon_batch is None:
            logger.error(
                "SB-DIAG | P_recon_4col: collect_reconstruction_batch returned None — "
                "skipping all per-expert 4-col panels"
            )
            return

        x_clean  = recon_batch.get("x_clean")
        y        = recon_batch.get("y")
        x_hat    = recon_batch.get("x_hat", {})    # cycle: encode→decode per expert
        x_gen    = recon_batch.get("x_gen", {})    # generated: z~N(0,I) per expert

        for k, name in enumerate(expert_names):
            safe_name   = name.replace("Conditional", "")
            output_path = os.path.join(output_dir, f"P_recon_4col_{safe_name}.png")
            ok = plot_recon_panel_4col(
                x_clean     = x_clean,
                y           = y,
                x_cycle     = x_hat.get(k),
                x_gen       = x_gen.get(k),
                expert_name = name,
                output_path = output_path,
                n_samples   = 8,
                stage_label = "Stage B",    # [STAGE-LABEL] PU v1.5
            )
            if not ok:
                logger.error(
                    "SB-DIAG | P_recon_4col_%s: plot_recon_panel_4col returned False — "
                    "check PU logs for details", safe_name
                )
            else:
                logger.info("SB-DIAG | P_recon_4col_%s: saved %s", safe_name, output_path)

    except Exception as e:
        logger.error("SB-DIAG | P_recon_4col failed: %s", e)


def _plot_p_loss_components(
    epoch_logs:    Dict,
    has_loss_comps: bool,
    has_neff_reg:  bool,    # [LOSS-COMP-EXT] v1.5
    output_dir:    str,
) -> None:
    """P_loss_components: Per-component hybrid loss over epochs. [v1.2 + LOSS-COMP-EXT v1.5]

    Plots NLL, consistency, SW2, calibration, and neff_reg_loss as separate lines.
    Requires CSMF-MAIN v1.3.29+ epoch_logs keys; neff_reg_loss requires HYBRID v1.10.0.
    """
    if not has_loss_comps and not has_neff_reg:
        logger.warning(
            "SB-DIAG | P_loss_components skipped — no loss component keys present "
            "(nll_loss absent and neff_reg_loss absent)"
        )
        return
    try:
        data = {}
        for key, label in [
            ("nll_loss",      "NLL"),
            ("cons_loss",     "Consistency (λ_cons)"),
            ("trans_loss",    "SW2 Transport (λ_trans)"),
            ("cal_loss",      "Calibration (λ_cal)"),
            ("neff_reg_loss", "Neff Reg (λ_neff)"),   # [LOSS-COMP-EXT] v1.5
        ]:
            vals = epoch_logs.get(key, [])
            if vals:
                data[label] = vals
        if not data:
            logger.warning("SB-DIAG | P_loss_components: all component lists empty — skipping")
            return
        plot_epoch_lines(
            data_dict   = data,
            output_path = os.path.join(output_dir, "P_loss_components.png"),
            title       = "Stage B — Hybrid Loss Components per Epoch",
            ylabel      = "Loss",
        )
    except Exception as e:

        logger.error(f"SB-DIAG | P_loss_components failed: {e}")



# =============================================================================
# [EVID-PLOT] v1.4 — P8: Evidence signals per expert over epochs
# =============================================================================

def _plot_p8_evidence_signals(
    epoch_logs:   Dict,
    expert_names: List[str],
    K:            int,
    has_evidence: bool,
    output_dir:   str,
) -> bool:
    """
    P8: Mean nll_k, r_k, g_k per expert over epochs (3 subplots).
    Shows whether evidence signals are diverse across experts or degenerate.
    Requires CSMF-MAIN v2.2+ evidence_nll/r/g keys.
    """
    try:
        if not has_evidence:
            logger.warning(
                "SB-DIAG | P8 skipped — evidence_nll/r/g absent from epoch_logs. "
                "Requires CSMF-MAIN v2.2+ with --lambda-r/g active."
            )
            return False

        nll_data = epoch_logs.get("evidence_nll", [])   # List[List[float]] epochs x K
        r_data   = epoch_logs.get("evidence_r",   [])
        g_data   = epoch_logs.get("evidence_g",   [])

        epochs_ax = list(range(1, len(nll_data) + 1))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("P8: Per-Expert Evidence Signals", fontsize=12, fontweight="bold")

        for ax, data, title, ylabel in zip(
            axes,
            [nll_data, r_data, g_data],
            ["NLL_k (lower=better)", "Residual r_k = ‖Ax̂−y‖²", "Gaussian Mismatch g_k"],
            ["Mean NLL", "Mean Residual", "Mean |μ|+|σ−1|"],
        ):
            if not data:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
                ax.set_title(title)
                continue
            arr = np.array(data)   # (E, K)
            K_actual = min(K, arr.shape[1])
            for k in range(K_actual):
                name = expert_names[k] if k < len(expert_names) else f"Expert {k}"
                ax.plot(epochs_ax[:arr.shape[0]], arr[:, k], label=name, linewidth=1.5)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = os.path.join(output_dir, "P8_evidence_signals.png")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"SB-DIAG | P8 saved: {out}")
        return True
    except Exception as e:
        logger.error(f"SB-DIAG | P8 failed: {e}")
        return False


# =============================================================================
# [BNORM-PLOT] v1.4 — P9: Raw NLL std over epochs (batch-norm sanity check)
# =============================================================================

def _plot_p9_nll_raw_std(
    epoch_logs:    Dict,
    has_bnorm_diag: bool,
    output_dir:    str,
) -> bool:
    """
    P9: Mean std of raw NLL across K experts per epoch.
    Verifies batch-normalisation is fixing scale problem.
    Epochs where std < 0.1 are shaded red — degenerate signal.
    Requires CSMF-MAIN v2.2+ evidence_nll_raw_std key.
    """
    try:
        if not has_bnorm_diag:
            logger.warning(
                "SB-DIAG | P9 skipped — evidence_nll_raw_std absent. "
                "Requires CSMF-MAIN v2.2+."
            )
            return False

        std_data  = epoch_logs.get("evidence_nll_raw_std", [])
        epochs_ax = list(range(1, len(std_data) + 1))

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs_ax, std_data, color="steelblue", linewidth=1.8, label="NLL std across K")

        _DEGEN_THRESH = 0.1
        for i, v in enumerate(std_data):
            if v < _DEGEN_THRESH:
                ax.axvspan(i + 0.5, i + 1.5, color="red", alpha=0.15)

        ax.axhline(_DEGEN_THRESH, color="red", linestyle="--", linewidth=1.0,
                   label=f"Degenerate threshold ({_DEGEN_THRESH})")
        ax.set_title("P9: Raw NLL Std Across K Experts (Batch-Norm Sanity)", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean std(nll_k over K) per batch")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = os.path.join(output_dir, "P9_nll_raw_std.png")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"SB-DIAG | P9 saved: {out}")
        return True
    except Exception as e:
        logger.error(f"SB-DIAG | P9 failed: {e}")
        return False


# =============================================================================
# [SCORE-SPLIT] v1.4 — P10: Score component contribution per expert
# =============================================================================

def _plot_p10_score_components(
    epoch_logs:     Dict,
    expert_names:   List[str],
    K:              int,
    has_score_split: bool,
    output_dir:     str,
) -> bool:
    """
    P10: Mean s_base vs alpha_u*u_k per expert over epochs (2 subplots).
    Reveals whether learned MLP residual dominates the evidence-based score.
    Requires CSMF-MAIN v2.2+ score_base_mean, score_u_mean keys.
    """
    try:
        if not has_score_split:
            logger.warning(
                "SB-DIAG | P10 skipped — score_base_mean/score_u_mean absent. "
                "Requires CSMF-MAIN v2.2+."
            )
            return False

        sb_data = epoch_logs.get("score_base_mean", [])   # (E, K)
        su_data = epoch_logs.get("score_u_mean",    [])

        epochs_ax = list(range(1, len(sb_data) + 1))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("P10: Gate Score Components (s_base vs α_u·u_k)", fontsize=12, fontweight="bold")

        for ax, data, title in zip(
            axes,
            [sb_data, su_data],
            ["s_base = −z(nll)−λ_r·z(r)−λ_g·z(g)", "α_u · u_k(h)  [learned residual]"],
        ):
            if not data:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
                ax.set_title(title)
                continue
            arr = np.array(data)
            K_actual = min(K, arr.shape[1])
            for k in range(K_actual):
                name = expert_names[k] if k < len(expert_names) else f"Expert {k}"
                ax.plot(epochs_ax[:arr.shape[0]], arr[:, k], label=name, linewidth=1.5)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Mean score value")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color="black", linewidth=0.6, linestyle="--")

        plt.tight_layout()
        out = os.path.join(output_dir, "P10_score_components.png")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"SB-DIAG | P10 saved: {out}")
        return True
    except Exception as e:
        logger.error(f"SB-DIAG | P10 failed: {e}")
        return False


# =============================================================================
# [EVID-CORR] v1.4 — P11: Evidence–weight Pearson correlation per expert
# =============================================================================

def _plot_p11_evid_gate_corr(
    epoch_logs:   Dict,
    expert_names: List[str],
    K:            int,
    has_evid_corr: bool,
    output_dir:   str,
) -> bool:
    """
    P11: Pearson r(s_base_k, w_k) per expert over epochs.
    Positive r = evidence is driving routing correctly.
    Negative r = MLP override or scoring inversion.
    Near-zero = gate is ignoring evidence.
    Requires CSMF-MAIN v2.2+ evid_gate_corr key.
    """
    try:
        if not has_evid_corr:
            logger.warning(
                "SB-DIAG | P11 skipped — evid_gate_corr absent. "
                "Requires CSMF-MAIN v2.2+."
            )
            return False

        corr_data = epoch_logs.get("evid_gate_corr", [])   # (E, K)
        epochs_ax = list(range(1, len(corr_data) + 1))

        fig, ax = plt.subplots(figsize=(9, 4))
        arr = np.array(corr_data)
        K_actual = min(K, arr.shape[1])
        colors = plt.cm.tab10(np.linspace(0, 1, K_actual))

        for k in range(K_actual):
            name = expert_names[k] if k < len(expert_names) else f"Expert {k}"
            ax.plot(epochs_ax[:arr.shape[0]], arr[:, k], label=name,
                    linewidth=1.8, color=colors[k])

        ax.axhline(0.0,  color="black",  linewidth=1.0, linestyle="--", label="r=0 (no correlation)")
        ax.axhline(0.3,  color="green",  linewidth=0.7, linestyle=":",  alpha=0.6, label="r=0.3 (weak positive)")
        ax.axhline(-0.3, color="red",    linewidth=0.7, linestyle=":",  alpha=0.6, label="r=−0.3 (weak negative)")
        ax.fill_between(epochs_ax, 0.3, 1.0, alpha=0.04, color="green")
        ax.fill_between(epochs_ax, -1.0, -0.3, alpha=0.04, color="red")

        ax.set_title("P11: Pearson r(s_base_k, w_k) — Evidence–Weight Correlation",
                     fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Pearson r")
        ax.set_ylim(-1.05, 1.05)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = os.path.join(output_dir, "P11_evid_gate_corr.png")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"SB-DIAG | P11 saved: {out}")
        return True
    except Exception as e:
        logger.error(f"SB-DIAG | P11 failed: {e}")
        return False

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
    epoch_logs:      Dict[str, list],
    expert_names:    List[str],
    K:               int,
    hyperparams:     Dict[str, Any],
    has_val_loss:    bool,
    has_tau:         bool,
    has_evidence:    bool = False,
    has_score_split: bool = False,
    has_evid_corr:   bool = False,
) -> Dict[str, Any]:
    """
    Build the full stage_b_summary.json payload.

    Epoch arrays are preserved in full for downstream analysis (SC-DIAG B-vs-C,
    further grid search comparison). All tensors converted to Python scalars.
    Evidence signal arrays included when available (CSMF-MAIN v2.2+).
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

    # [EVID-LOG] v1.4: build evidence scalar summary for "final" section
    evidence_summary: Dict[str, Any] = {}
    if has_evidence:
        try:
            nll_data  = epoch_logs.get("evidence_nll", [])
            r_data    = epoch_logs.get("evidence_r",   [])
            g_data    = epoch_logs.get("evidence_g",   [])
            nll_std   = epoch_logs.get("evidence_nll_raw_std", [])

            def _last_k_means(data):
                """Return final-epoch per-expert means, keyed by expert name."""
                if not data:
                    return {}
                last = data[-1]
                return {
                    (expert_names[k] if k < len(expert_names) else f"Expert {k}"): round(float(v), 4)
                    for k, v in enumerate(last)
                }

            evidence_summary = {
                "final_nll_k":       _last_k_means(nll_data),
                "final_r_k":         _last_k_means(r_data),
                "final_g_k":         _last_k_means(g_data),
                "mean_nll_raw_std":  round(float(np.mean(nll_std)), 4) if nll_std else None,
                "min_nll_raw_std":   round(float(np.min(nll_std)),  4) if nll_std else None,
                "degenerate_epochs": int(sum(1 for v in nll_std if v < 0.1)),
            }
        except Exception as e:
            logger.error(f"SB-DIAG | evidence_summary build failed: {e}")

    score_summary: Dict[str, Any] = {}
    if has_score_split:
        try:
            sb_data = epoch_logs.get("score_base_mean", [])
            su_data = epoch_logs.get("score_u_mean",    [])
            if sb_data and su_data:
                sb_arr = np.array(sb_data)   # (E, K)
                su_arr = np.array(su_data)
                # Ratio |s_base| / (|s_base| + |s_u|) per expert — 1.0 = fully evidence-driven
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.abs(sb_arr) / (np.abs(sb_arr) + np.abs(su_arr) + 1e-8)
                score_summary = {
                    expert_names[k] if k < len(expert_names) else f"Expert {k}": {
                        "final_s_base": round(float(sb_arr[-1, k]), 4),
                        "final_s_u":    round(float(su_arr[-1, k]), 4),
                        "mean_evidence_ratio": round(float(ratio[:, k].mean()), 4),
                    }
                    for k in range(min(len(expert_names), sb_arr.shape[1]))
                }
        except Exception as e:
            logger.error(f"SB-DIAG | score_summary build failed: {e}")

    corr_summary: Dict[str, Any] = {}
    if has_evid_corr:
        try:
            corr_data = epoch_logs.get("evid_gate_corr", [])
            if corr_data:
                corr_arr = np.array(corr_data)   # (E, K)
                corr_summary = {
                    expert_names[k] if k < len(expert_names) else f"Expert {k}": {
                        "final_r":  round(float(corr_arr[-1, k]), 4),
                        "mean_r":   round(float(corr_arr[:, k].mean()), 4),
                        "positive_epochs": int((corr_arr[:, k] > 0).sum()),
                    }
                    for k in range(min(len(expert_names), corr_arr.shape[1]))
                }
        except Exception as e:
            logger.error(f"SB-DIAG | corr_summary build failed: {e}")

    return {
        "metadata": {
            "timestamp":    datetime.datetime.now().isoformat(timespec="seconds"),
            "expert_names": expert_names,
            "diag_version": "SB-DIAG-v1.4",
            "hyperparams":  hyperparams,
        },
        "epoch_logs": {
            "train_loss":   _safe_list(epoch_logs.get("train_loss", [])),
            "val_loss":     _safe_list(epoch_logs.get("val_loss", [])) if has_val_loss else [],
            "neff":         _safe_list(neff_list),
            "gate_weights": _safe_list_of_lists(gw_epochs),
            "tau":          _safe_list(tau_list) if has_tau else [],
            # [EVID-LOG] v1.4: full evidence arrays preserved for downstream analysis
            "evidence_nll":          _safe_list_of_lists(epoch_logs.get("evidence_nll", [])),
            "evidence_r":            _safe_list_of_lists(epoch_logs.get("evidence_r",   [])),
            "evidence_g":            _safe_list_of_lists(epoch_logs.get("evidence_g",   [])),
            "evidence_nll_raw_std":  _safe_list(epoch_logs.get("evidence_nll_raw_std", [])),
            "score_base_mean":       _safe_list_of_lists(epoch_logs.get("score_base_mean", [])),
            "score_u_mean":          _safe_list_of_lists(epoch_logs.get("score_u_mean",    [])),
            "evid_gate_corr":        _safe_list_of_lists(epoch_logs.get("evid_gate_corr",  [])),
        },
        "final": {
            "neff":               round(float(neff_list[-1]), 4) if neff_list else None,
            "gate_weights":       final_gw,
            "gate_winner_counts": winner_counts,
            "collapsed":          collapsed,
            # [EVID-LOG] v1.4: scalar evidence diagnostics for quick inspection
            "evidence_summary":   evidence_summary,
            "score_summary":      score_summary,
            "corr_summary":       corr_summary,
        },
    }


# =============================================================================
# New plot helpers — SB-DIAG v1.5
# =============================================================================

def _plot_p_neff_reg(
    epoch_logs: Dict,
    has_neff_reg: bool,
    output_dir: str,
) -> None:
    """P_neff_reg: λ_neff · max(0, Neff_target − Neff) over epochs. [v1.5 P_NEFF_REG]

    Confirms entropy regularisation is active. Logged at WARNING if key absent.
    """
    if not has_neff_reg:
        logger.warning(
            "SB-DIAG | P_neff_reg skipped — 'neff_reg_loss' absent from epoch_logs. "
            "Requires HYBRID v1.10.0 + CSMF-MAIN accumulating loss_dict keys."
        )
        return
    try:
        vals = epoch_logs.get("neff_reg_loss", [])
        if not vals:
            logger.warning("SB-DIAG | P_neff_reg: neff_reg_loss list is empty — skipping")
            return
        ok = plot_epoch_lines(
            data_dict   = {"Neff reg loss": vals},
            output_path = os.path.join(output_dir, "P_neff_reg.png"),
            title       = "Stage B — Neff Regularisation Loss (λ_neff · max(0, Neff_target − Neff))",
            ylabel      = "Neff reg loss",
            hlines      = [(0.0, "Zero (no penalty)", "gray")],
        )
        if not ok:
            logger.error("SB-DIAG | P_neff_reg: plot_epoch_lines returned False")
        else:
            logger.info("SB-DIAG | P_neff_reg: saved")
    except Exception as e:
        logger.error("SB-DIAG | P_neff_reg failed: %s", e)


def _plot_p_anneal_lambdas(
    epoch_logs: Dict,
    has_anneal: bool,
    output_dir: str,
) -> None:
    """P_anneal_lambdas: Effective λ_cons / λ_trans / λ_cal over epochs. [v1.5 P_ANNEAL]

    Verifies annealing schedule ramps correctly from zero. Logged at WARNING if absent.
    """
    if not has_anneal:
        logger.warning(
            "SB-DIAG | P_anneal_lambdas skipped — 'lambda_cons_eff' absent from "
            "epoch_logs. Requires HYBRID v1.10.0 anneal_schedule wired in TRAIN-MAIN v1.9.0."
        )
        return
    try:
        data = {}
        for key, label in [
            ("lambda_cons_eff",  "λ_cons (effective)"),
            ("lambda_trans_eff", "λ_trans (effective)"),
            ("lambda_cal_eff",   "λ_cal (effective)"),
        ]:
            vals = epoch_logs.get(key, [])
            if vals:
                data[label] = vals

        if not data:
            logger.warning("SB-DIAG | P_anneal_lambdas: all lambda_*_eff lists empty — skipping")
            return

        ok = plot_epoch_lines(
            data_dict   = data,
            output_path = os.path.join(output_dir, "P_anneal_lambdas.png"),
            title       = "Stage B — Effective Lambda Annealing Schedule",
            ylabel      = "Effective λ",
        )
        if not ok:
            logger.error("SB-DIAG | P_anneal_lambdas: plot_epoch_lines returned False")
        else:
            logger.info("SB-DIAG | P_anneal_lambdas: saved")
    except Exception as e:
        logger.error("SB-DIAG | P_anneal_lambdas failed: %s", e)


def _plot_p_fi_gate(
    model:           Optional[Any],
    val_loader:      Optional[Any],
    fi_summary_path: Optional[str],
    expert_names:    List[str],
    output_dir:      str,
) -> None:
    """P_fi_gate: FI ratio vs mean gate weight per expert — grouped bar. [v1.5 P_FI_GATE]

    Flags misalignment: high-FI experts with low gate weight indicate the gate
    is not routing to informative experts. Logged at WARNING if fi_summary_path
    is None or JSON is missing (non-fatal).
    """
    if model is None or val_loader is None:
        logger.warning(
            "SB-DIAG | P_fi_gate skipped — model or val_loader not provided"
        )
        return
    if fi_summary_path is None:
        logger.warning(
            "SB-DIAG | P_fi_gate skipped — fi_summary_path not provided. "
            "Run FI-DIAG after Stage A and pass the path to fi_diag_summary.json."
        )
        return
    try:
        import torch
        device   = next(model.parameters()).device
        fi_data  = collect_fi_gate_comparison(
            csmf_model      = model,
            val_loader      = val_loader,
            device          = device,
            fi_summary_path = fi_summary_path,
        )
        if fi_data is None:
            logger.warning(
                "SB-DIAG | P_fi_gate skipped — collect_fi_gate_comparison returned None "
                "(fi_diag_summary.json missing or unreadable — non-fatal)"
            )
            return

        names        = fi_data["expert_names"]
        fi_ratios    = fi_data["fi_ratios"]
        gate_weights = fi_data["gate_weights_mean"]

        # Build grouped bar using plot_expert_bars — two series
        ok = plot_expert_bars(
            expert_names = names,
            values_dict  = {
                "FI ratio (F_k / max F_k)": fi_ratios,
                "Mean gate weight":          gate_weights,
            },
            output_path  = os.path.join(output_dir, "P_fi_gate.png"),
            title        = "Stage B — Fisher Information vs Gate Weight per Expert",
            ylabel       = "Normalised score",
        )
        if not ok:
            logger.error(
                "SB-DIAG | P_fi_gate: plot_expert_bars returned False — "
                "check PU logs for details"
            )
        else:
            logger.info("SB-DIAG | P_fi_gate: saved")

    except Exception as e:
        logger.error("SB-DIAG | P_fi_gate failed: %s", e)
