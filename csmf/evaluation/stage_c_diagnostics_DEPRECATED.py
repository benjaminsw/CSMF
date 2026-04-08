# =============================================================================
# Version: WP3.3-StageCDiag-v1.1 | Abbr: SC-DIAG
# Description: Stage C diagnostic plots and B-vs-C comparison.
#              Called after Stage C completes, before final evaluation.
# Changelog:
#   v1.1 (2026-04-03): Added 4 new epoch-level plots from enriched epoch_logs
#                      (CSMF-v1.3.22+): P3 gate weights over epochs (line per
#                      expert), P4 residual ‖Ax̂-y‖² over epochs (line + nan-safe),
#                      P5 reconstruction grid from final Stage C model (y/x_hat
#                      pairs, 8 samples), P6 reconstruction snapshots over epochs
#                      (grid rows=epoch checkpoints, cols=samples); all 4 plots
#                      non-fatal — logged as error and skipped if data missing
#   v1.0 (2026-03-29): Initial implementation — 8 plots + JSON summary
#                      Plots 1,2,5 (C-only); Plots 6,7,8,10,11 (B-vs-C)
#                      Loads Stage B checkpoint temporarily for comparison;
#                      restores Stage C state after B metrics collection
# Dependencies: CSMF-MAIN v1.3.22+, matplotlib, torch
# =============================================================================

import os
import json
import copy
import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Internal: collect metrics from a model on val_loader
# =============================================================================

@torch.no_grad()
def _collect_metrics(
    csmf_model,
    val_loader,
    fwd_model,
    expert_names: List[str],
    max_batches: int = 20,
) -> Dict[str, Any]:
    """
    Collect NLL, gate weights, Neff, residual, per-sample NLL from a model.
    Used for both Stage B and Stage C models.
    """
    csmf_model.eval()
    device = csmf_model.device
    K = csmf_model.K

    per_expert_nll_samples = {k: [] for k in range(K)}
    all_mixture_nll = []
    all_neff = []
    all_gate_weights = []
    all_residuals = []
    recon_images_stored = None   # v1.1: store first batch for P5 grid
    y_images_stored     = None
    n_collected = 0

    for x_clean, y_deg in val_loader:
        if n_collected >= max_batches:
            break
        x_clean = x_clean.to(device)
        y_deg = y_deg.to(device)

        try:
            # Mixture NLL + per-expert NLL
            log_q, log_q_experts = csmf_model.forward(x_clean, y_deg)
            if torch.isnan(log_q).any():
                logger.warning("[SC-DIAG] NaN in log_q — skipping batch")
                continue
            all_mixture_nll.append(-log_q.cpu())

            for k in range(K):
                per_expert_nll_samples[k].append(-log_q_experts[:, k].cpu())

            # Gate weights + Neff
            h = csmf_model.conditioner(y_deg)
            logits = csmf_model.gate(h)
            w = torch.softmax(logits, dim=1)
            neff = csmf_model._compute_neff(w)
            all_neff.append(neff.cpu())
            all_gate_weights.append(w.mean(dim=0).cpu())

            # Physics residual
            x_samples, _ = csmf_model.sample(y_deg, num_samples=1)
            x_hat = x_samples[:, 0, :]
            x_hat_4d = x_hat.view(x_hat.shape[0], 1, 28, 28)
            Ax = fwd_model.forward(x_hat_4d)
            residual = ((Ax - y_deg) ** 2).mean(dim=[1, 2, 3])
            all_residuals.append(residual.cpu())

            # v1.1: store first-batch images for reconstruction grid (P5)
            if recon_images_stored is None:
                recon_images_stored = x_hat_4d[:8].clamp(0, 1).cpu()
                y_images_stored     = y_deg[:8].cpu()

            n_collected += 1
        except Exception as e:
            logger.error(f"[SC-DIAG] Metric collection error batch={n_collected}: {e}")
            continue

    if n_collected == 0:
        logger.error("[SC-DIAG] No batches collected")
        return {}

    # Aggregate
    mixture_nll = torch.cat(all_mixture_nll)
    neff_all = torch.cat(all_neff)
    gate_w = torch.stack(all_gate_weights).mean(dim=0)
    residuals = torch.cat(all_residuals)

    per_expert_nll = {}
    per_expert_nll_flat = {}
    for k in range(K):
        samples = torch.cat(per_expert_nll_samples[k])
        per_expert_nll[expert_names[k]] = samples.mean().item()
        per_expert_nll_flat[k] = samples

    return {
        "mixture_nll": mixture_nll.mean().item(),
        "per_expert_nll": per_expert_nll,
        "per_expert_nll_samples": per_expert_nll_flat,
        "neff_mean": neff_all.mean().item(),
        "gate_weights_mean": {expert_names[k]: gate_w[k].item() for k in range(K)},
        "residual_mean": residuals.mean().item(),
        "residual_std": residuals.std().item(),
        "residuals_all": residuals,
        "recon_images": recon_images_stored,   # v1.1: (≤8, 1, 28, 28) or None
        "y_images":     y_images_stored,       # v1.1: (≤8, 1, H, W) or None
    }


# =============================================================================
# Main entry point
# =============================================================================

def run_stage_c_diagnostics(
    csmf_model,
    val_loader,
    fwd_model,
    epoch_logs: Dict[str, list],
    ckpt_path_B: str,
    expert_names: List[str],
    output_dir: str = "results/stage_c_diagnostics",
    max_val_batches: int = 20,
) -> Dict[str, Any]:
    """
    Run Stage C diagnostics: C-only plots + B-vs-C comparison.

    Args:
        csmf_model:      CSMF model after Stage C training.
        val_loader:      Validation DataLoader.
        fwd_model:       Forward model A (SRForwardModel).
        epoch_logs:      From train_stage_C(): {train_loss:[], val_loss:[]}.
        ckpt_path_B:     Path to Stage B checkpoint for comparison.
        expert_names:    List of short expert names e.g. ["realnvp","nice","nsf"].
        output_dir:      Directory for plots and JSON.
        max_val_batches: Max batches for val pass.

    Returns:
        Summary dict with all metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"[SC-DIAG] Starting Stage C diagnostics | output_dir={output_dir}")

    K = csmf_model.K
    device = csmf_model.device

    # ------------------------------------------------------------------
    # Collect Stage C metrics
    # ------------------------------------------------------------------
    logger.info("[SC-DIAG] Collecting Stage C metrics...")
    metrics_c = _collect_metrics(
        csmf_model, val_loader, fwd_model, expert_names, max_val_batches
    )
    if not metrics_c:
        logger.error("[SC-DIAG] Failed to collect Stage C metrics — aborting")
        return {}

    # ------------------------------------------------------------------
    # Collect Stage B metrics (load checkpoint temporarily)
    # ------------------------------------------------------------------
    metrics_b = {}
    has_stage_b = os.path.isfile(ckpt_path_B)

    if has_stage_b:
        logger.info(f"[SC-DIAG] Loading Stage B checkpoint for comparison: {ckpt_path_B}")
        # Save Stage C state
        state_c = copy.deepcopy(csmf_model.state_dict())
        try:
            payload_b = torch.load(ckpt_path_B, map_location="cpu")
            csmf_model.load_state_dict(payload_b["state_dict"])
            csmf_model.to(device)

            metrics_b = _collect_metrics(
                csmf_model, val_loader, fwd_model, expert_names, max_val_batches
            )
            logger.info("[SC-DIAG] Stage B metrics collected")
        except Exception as e:
            logger.error(f"[SC-DIAG] Failed to load/eval Stage B checkpoint: {e}")
            has_stage_b = False
        finally:
            # Restore Stage C state
            csmf_model.load_state_dict(state_c)
            csmf_model.to(device)
            logger.info("[SC-DIAG] Stage C state restored")
    else:
        logger.warning(
            f"[SC-DIAG] Stage B checkpoint not found: {ckpt_path_B} — "
            f"skipping B-vs-C comparison plots (6,7,8,10,11)"
        )

    # ------------------------------------------------------------------
    # Plot 1: Stage C loss curves
    # ------------------------------------------------------------------
    try:
        _plot_loss_curves(epoch_logs, output_dir)
        logger.info("[SC-DIAG] Plot 1 (loss curves) saved")
    except Exception as e:
        logger.error(f"[SC-DIAG] Plot 1 failed: {e}")

    # ------------------------------------------------------------------
    # Plot 2: Per-expert NLL after Stage C (bar chart)
    # ------------------------------------------------------------------
    try:
        _plot_per_expert_nll(metrics_c, expert_names, output_dir)
        logger.info("[SC-DIAG] Plot 2 (per-expert NLL) saved")
    except Exception as e:
        logger.error(f"[SC-DIAG] Plot 2 failed: {e}")

    # ------------------------------------------------------------------
    # Plot 5: Physics residual distribution
    # ------------------------------------------------------------------
    try:
        _plot_residual_dist(metrics_c, output_dir)
        logger.info("[SC-DIAG] Plot 5 (residual dist) saved")
    except Exception as e:
        logger.error(f"[SC-DIAG] Plot 5 failed: {e}")

    # ------------------------------------------------------------------
    # Plot P3 (NEW v1.1): Gate weights over epochs
    # ------------------------------------------------------------------
    try:
        _plot_gate_weights_over_epochs(epoch_logs, expert_names, output_dir)
        logger.info("[SC-DIAG] Plot P3 (gate weights over epochs) saved")
    except Exception as e:
        logger.error(f"[SC-DIAG] Plot P3 failed: {e}")

    # ------------------------------------------------------------------
    # Plot P4 (NEW v1.1): Residual ‖Ax̂ − y‖² over epochs
    # ------------------------------------------------------------------
    try:
        _plot_residual_over_epochs(epoch_logs, output_dir)
        logger.info("[SC-DIAG] Plot P4 (residual over epochs) saved")
    except Exception as e:
        logger.error(f"[SC-DIAG] Plot P4 failed: {e}")

    # ------------------------------------------------------------------
    # Plot P5 (NEW v1.1): Reconstruction grid from final Stage C model
    # ------------------------------------------------------------------
    try:
        _plot_reconstruction_grid(metrics_c, output_dir)
        logger.info("[SC-DIAG] Plot P5 (reconstruction grid) saved")
    except Exception as e:
        logger.error(f"[SC-DIAG] Plot P5 failed: {e}")

    # ------------------------------------------------------------------
    # Plot P6 (NEW v1.1): Reconstruction quality snapshots over epochs
    # ------------------------------------------------------------------
    try:
        _plot_recon_snapshots(epoch_logs, output_dir)
        logger.info("[SC-DIAG] Plot P6 (recon snapshots over epochs) saved")
    except Exception as e:
        logger.error(f"[SC-DIAG] Plot P6 failed: {e}")

    # ------------------------------------------------------------------
    # B-vs-C comparison plots (only if Stage B checkpoint available)
    # ------------------------------------------------------------------
    if has_stage_b and metrics_b:
        try:
            _plot_nll_comparison(metrics_b, metrics_c, expert_names, output_dir)
            logger.info("[SC-DIAG] Plot 6 (NLL comparison) saved")
        except Exception as e:
            logger.error(f"[SC-DIAG] Plot 6 failed: {e}")

        try:
            _plot_gate_weights_comparison(metrics_b, metrics_c, expert_names, output_dir)
            logger.info("[SC-DIAG] Plot 7 (gate weights comparison) saved")
        except Exception as e:
            logger.error(f"[SC-DIAG] Plot 7 failed: {e}")

        try:
            _plot_neff_comparison(metrics_b, metrics_c, output_dir)
            logger.info("[SC-DIAG] Plot 8 (Neff comparison) saved")
        except Exception as e:
            logger.error(f"[SC-DIAG] Plot 8 failed: {e}")

        try:
            _plot_residual_comparison(metrics_b, metrics_c, output_dir)
            logger.info("[SC-DIAG] Plot 10 (residual comparison) saved")
        except Exception as e:
            logger.error(f"[SC-DIAG] Plot 10 failed: {e}")

        try:
            _plot_pairwise_nll_comparison(metrics_b, metrics_c, expert_names, output_dir)
            logger.info("[SC-DIAG] Plot 11 (pairwise NLL comparison) saved")
        except Exception as e:
            logger.error(f"[SC-DIAG] Plot 11 failed: {e}")

    # ------------------------------------------------------------------
    # Build and save JSON summary
    # ------------------------------------------------------------------
    summary = {
        "stage_c": {
            "mixture_nll":       round(metrics_c["mixture_nll"], 6),
            "per_expert_nll":    {k: round(v, 6) for k, v in metrics_c["per_expert_nll"].items()},
            "neff_mean":         round(metrics_c["neff_mean"], 4),
            "gate_weights_mean": {k: round(v, 4) for k, v in metrics_c["gate_weights_mean"].items()},
            "residual_mean":     round(metrics_c["residual_mean"], 6),
            "residual_std":      round(metrics_c["residual_std"], 6),
        },
    }

    if has_stage_b and metrics_b:
        summary["stage_b"] = {
            "mixture_nll":       round(metrics_b["mixture_nll"], 6),
            "per_expert_nll":    {k: round(v, 6) for k, v in metrics_b["per_expert_nll"].items()},
            "neff_mean":         round(metrics_b["neff_mean"], 4),
            "gate_weights_mean": {k: round(v, 4) for k, v in metrics_b["gate_weights_mean"].items()},
            "residual_mean":     round(metrics_b["residual_mean"], 6),
            "residual_std":      round(metrics_b["residual_std"], 6),
        }
        summary["delta_b_to_c"] = {
            "mixture_nll":   round(metrics_c["mixture_nll"] - metrics_b["mixture_nll"], 6),
            "neff":          round(metrics_c["neff_mean"] - metrics_b["neff_mean"], 4),
            "residual_mean": round(metrics_c["residual_mean"] - metrics_b["residual_mean"], 6),
        }

    summary_path = os.path.join(output_dir, "stage_c_diagnostics_summary.json")
    try:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"[SC-DIAG] Summary saved: {summary_path}")
    except Exception as e:
        logger.error(f"[SC-DIAG] Failed to save summary JSON: {e}")

    csmf_model.train()
    logger.info("[SC-DIAG] Stage C diagnostics complete")
    return summary


# =============================================================================
# Plot functions
# =============================================================================

def _plot_loss_curves(epoch_logs: Dict[str, list], output_dir: str) -> None:
    """Plot 1: Stage C train + val loss over epochs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    train_loss = epoch_logs.get("train_loss", [])
    val_loss = epoch_logs.get("val_loss", [])

    if train_loss:
        ax.plot(range(1, len(train_loss) + 1), train_loss,
                label="Train Loss", marker="o", markersize=3)
    if val_loss:
        ax.plot(range(1, len(val_loss) + 1), val_loss,
                label="Val Loss", linestyle="--", marker="s", markersize=3)
    if not train_loss and not val_loss:
        logger.warning("[SC-DIAG] Plot 1: no loss data in epoch_logs — skipping")
        plt.close(fig)
        return

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Stage C — Loss Over Epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stage_c_loss_curves.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_per_expert_nll(
    metrics: Dict, expert_names: List[str], output_dir: str
) -> None:
    """Plot 2: Per-expert NLL bar chart (Stage C only)."""
    K = len(expert_names)
    nll_vals = [metrics["per_expert_nll"][n] for n in expert_names]
    mixture_nll = metrics["mixture_nll"]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.Set2(np.linspace(0, 1, K + 1))
    bars = ax.bar(range(K), nll_vals, color=colors[:K],
                  edgecolor="black", linewidth=0.5, label="Per-expert")
    ax.axhline(y=mixture_nll, color="red", linestyle="--",
               linewidth=1.5, label=f"Mixture NLL={mixture_nll:.2f}")

    ax.set_xticks(range(K))
    ax.set_xticklabels(expert_names, rotation=15)
    ax.set_ylabel("NLL")
    ax.set_title("Stage C — Per-Expert NLL")
    for bar, v in zip(bars, nll_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stage_c_per_expert_nll.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_residual_dist(metrics: Dict, output_dir: str) -> None:
    """Plot 5: Physics residual histogram."""
    residuals = metrics["residuals_all"].numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=50, density=True, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axvline(x=residuals.mean(), color="red", linestyle="--",
               label=f"Mean={residuals.mean():.4f}")
    ax.set_xlabel("‖Ax̂ − y‖²")
    ax.set_ylabel("Density")
    ax.set_title("Stage C — Physics Residual Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stage_c_residual_dist.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_nll_comparison(
    metrics_b: Dict, metrics_c: Dict, expert_names: List[str], output_dir: str
) -> None:
    """Plot 6: NLL bar chart B vs C (per-expert + mixture)."""
    K = len(expert_names)
    labels = expert_names + ["Mixture"]
    b_vals = [metrics_b["per_expert_nll"][n] for n in expert_names] + [metrics_b["mixture_nll"]]
    c_vals = [metrics_c["per_expert_nll"][n] for n in expert_names] + [metrics_c["mixture_nll"]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_b = ax.bar(x - width / 2, b_vals, width, label="Stage B",
                    color="#5DA5DA", edgecolor="black", linewidth=0.5)
    bars_c = ax.bar(x + width / 2, c_vals, width, label="Stage C",
                    color="#FAA43A", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("NLL")
    ax.set_title("Stage B vs C — NLL Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar, v in zip(bars_b, b_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    for bar, v in zip(bars_c, c_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stage_bc_nll_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_gate_weights_comparison(
    metrics_b: Dict, metrics_c: Dict, expert_names: List[str], output_dir: str
) -> None:
    """Plot 7: Gate weights grouped bar B vs C."""
    K = len(expert_names)
    b_vals = [metrics_b["gate_weights_mean"][n] for n in expert_names]
    c_vals = [metrics_c["gate_weights_mean"][n] for n in expert_names]

    x = np.arange(K)
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, b_vals, width, label="Stage B",
           color="#5DA5DA", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, c_vals, width, label="Stage C",
           color="#FAA43A", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(expert_names, rotation=15)
    ax.set_ylabel("Mean Gate Weight")
    ax.set_title("Stage B vs C — Gate Weights")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stage_bc_gate_weights.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_neff_comparison(
    metrics_b: Dict, metrics_c: Dict, output_dir: str
) -> None:
    """Plot 8: Neff bar chart B vs C."""
    fig, ax = plt.subplots(figsize=(4, 4))
    labels = ["Stage B", "Stage C"]
    vals = [metrics_b["neff_mean"], metrics_c["neff_mean"]]
    colors = ["#5DA5DA", "#FAA43A"]

    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=1.5, color="red", linestyle=":", alpha=0.6,
               label="Target Neff ≥ 1.5")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Neff")
    ax.set_title("Stage B vs C — Effective Experts")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stage_bc_neff.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_residual_comparison(
    metrics_b: Dict, metrics_c: Dict, output_dir: str
) -> None:
    """Plot 10: Residual boxplot B vs C."""
    fig, ax = plt.subplots(figsize=(5, 4))
    data = [metrics_b["residuals_all"].numpy(), metrics_c["residuals_all"].numpy()]
    bp = ax.boxplot(data, labels=["Stage B", "Stage C"], patch_artist=True)
    colors = ["#5DA5DA", "#FAA43A"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("‖Ax̂ − y‖²")
    ax.set_title("Stage B vs C — Physics Residual")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stage_bc_residual_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pairwise_nll_comparison(
    metrics_b: Dict, metrics_c: Dict, expert_names: List[str], output_dir: str
) -> None:
    """Plot 11: Pairwise NLL scatter B vs C (side-by-side panels)."""
    K = len(expert_names)
    n_pairs = K * (K - 1) // 2
    if n_pairs == 0:
        return

    fig, axes = plt.subplots(2, n_pairs, figsize=(5 * n_pairs, 8), squeeze=False)
    row_labels = ["Stage B", "Stage C"]

    for row_idx, (metrics, label) in enumerate(
        [(metrics_b, "Stage B"), (metrics_c, "Stage C")]
    ):
        nll_data = metrics.get("per_expert_nll_samples", {})
        pair_idx = 0
        for i in range(K):
            for j in range(i + 1, K):
                ax = axes[row_idx, pair_idx]
                nll_i = nll_data.get(i, torch.tensor([]))
                nll_j = nll_data.get(j, torch.tensor([]))
                if nll_i.numel() == 0 or nll_j.numel() == 0:
                    ax.set_title(f"{label}\n(no data)")
                    pair_idx += 1
                    continue
                n_min = min(nll_i.shape[0], nll_j.shape[0])
                ni = nll_i[:n_min].numpy()
                nj = nll_j[:n_min].numpy()

                ax.scatter(ni, nj, alpha=0.15, s=8, edgecolors="none")
                ax.set_xlabel(f"{expert_names[i]} NLL")
                ax.set_ylabel(f"{expert_names[j]} NLL")

                if np.std(ni) > 1e-8 and np.std(nj) > 1e-8:
                    corr = np.corrcoef(ni, nj)[0, 1]
                    ax.set_title(f"{label}\n{expert_names[i]} vs {expert_names[j]}\nρ={corr:.3f}")
                else:
                    ax.set_title(f"{label}\n{expert_names[i]} vs {expert_names[j]}")

                p5, p95 = np.percentile(np.concatenate([ni, nj]), [5, 95])
                ax.plot([p5, p95], [p5, p95], "r--", alpha=0.5, linewidth=1)
                ax.grid(True, alpha=0.3)
                pair_idx += 1

    fig.suptitle("Stage B vs C — Pairwise Expert NLL Scatter", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stage_bc_pairwise_nll.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# NEW v1.1 Plot functions
# =============================================================================

def _plot_gate_weights_over_epochs(
    epoch_logs: Dict[str, list], expert_names: List[str], output_dir: str
) -> None:
    """Plot P3 (v1.1): Per-expert mean gate weight over Stage C epochs."""
    gate_data = epoch_logs.get("gate_weights", [])
    if not gate_data:
        logger.warning("[SC-DIAG] P3: no gate_weights in epoch_logs — skipping")
        return

    K = len(expert_names)
    epochs_ax = range(1, len(gate_data) + 1)
    # gate_data: list of K-dim lists
    weights_per_expert = [
        [gate_data[ep][k] for ep in range(len(gate_data))]
        for k in range(min(K, len(gate_data[0])))
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, K))
    for k, (w_k, color) in enumerate(zip(weights_per_expert, colors)):
        ax.plot(epochs_ax, w_k, label=expert_names[k] if k < len(expert_names) else f"Expert {k}",
                color=color, linewidth=1.5, marker="o", markersize=3)

    ax.axhline(y=1.0 / K, color="black", linestyle=":", alpha=0.5,
               label=f"Uniform 1/K={1/K:.2f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Gate Weight")
    ax.set_title("Stage C — Gate Weights Over Epochs")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stage_c_gate_weights_over_epochs.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_residual_over_epochs(
    epoch_logs: Dict[str, list], output_dir: str
) -> None:
    """Plot P4 (v1.1): Physics residual ‖Ax̂ − y‖² over Stage C epochs."""
    residuals = epoch_logs.get("residual", [])
    if not residuals:
        logger.warning("[SC-DIAG] P4: no residual in epoch_logs — skipping")
        return

    # Filter NaN for clean plotting
    epochs_all = list(range(1, len(residuals) + 1))
    valid = [(ep, r) for ep, r in zip(epochs_all, residuals) if not (r != r)]
    if not valid:
        logger.warning("[SC-DIAG] P4: all residual values are NaN — skipping")
        return
    epochs_valid, res_valid = zip(*valid)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs_valid, res_valid, color="#E15759", linewidth=2,
            marker="o", markersize=4, label="‖Ax̂ − y‖²")

    # Annotate first and last
    ax.annotate(f"{res_valid[0]:.4f}", xy=(epochs_valid[0], res_valid[0]),
                xytext=(epochs_valid[0] + 0.5, res_valid[0]),
                fontsize=8, color="#E15759")
    ax.annotate(f"{res_valid[-1]:.4f}", xy=(epochs_valid[-1], res_valid[-1]),
                xytext=(epochs_valid[-1] - 2, res_valid[-1]),
                fontsize=8, color="#E15759")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("‖Ax̂ − y‖²")
    ax.set_title("Stage C — Measurement Residual Over Epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stage_c_residual_over_epochs.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_reconstruction_grid(metrics_c: Dict, output_dir: str) -> None:
    """
    Plot P5 (v1.1): Reconstruction grid from final Stage C model.
    Layout: row 0 = degraded y, row 1 = x̂ (reconstruction).
    Uses x_hat already sampled in _collect_metrics.
    """
    # _collect_metrics does not store images — it only stores scalars.
    # We need raw images. Store them during _collect_metrics if available.
    # Fallback: if no image data, skip with warning.
    recon_images = metrics_c.get("recon_images")
    y_images     = metrics_c.get("y_images")

    if recon_images is None or y_images is None:
        logger.warning("[SC-DIAG] P5: no image data in metrics_c — skipping recon grid. "
                       "Ensure CSMF-v1.3.22+ is used.")
        return

    n = min(8, recon_images.shape[0])
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))

    for i in range(n):
        # Row 0: degraded y
        y_img = y_images[i].squeeze().numpy()
        axes[0, i].imshow(y_img, cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("y (degraded)", fontsize=8, loc="left")

        # Row 1: reconstruction x̂
        x_img = recon_images[i].squeeze().numpy()
        axes[1, i].imshow(x_img, cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("x̂ (Stage C)", fontsize=8, loc="left")

    fig.suptitle("Stage C — Reconstruction Grid", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stage_c_reconstruction_grid.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_recon_snapshots(epoch_logs: Dict[str, list], output_dir: str) -> None:
    """
    Plot P6 (v1.1): Reconstruction quality snapshots over Stage C epochs.
    Layout: rows=epoch snapshots, cols=8 samples.
    Each row: y (top-half), x̂ (bottom-half) interleaved per column.
    """
    snapshots = epoch_logs.get("recon_snapshots", [])
    if not snapshots:
        logger.warning("[SC-DIAG] P6: no recon_snapshots in epoch_logs — skipping")
        return

    n_snaps = len(snapshots)
    n_cols  = 8

    fig, axes = plt.subplots(n_snaps * 2, n_cols, figsize=(2 * n_cols, 4 * n_snaps))
    if n_snaps == 1:
        axes = axes.reshape(2, n_cols)

    for row_snap, snap in enumerate(snapshots):
        epoch_label = snap.get("epoch", "?")
        y_imgs      = snap.get("y",     None)   # (B, 1, H, W)
        x_hat_imgs  = snap.get("x_hat", None)   # (B, 1, H, W)

        if y_imgs is None or x_hat_imgs is None:
            logger.error(f"[SC-DIAG] P6: snapshot epoch={epoch_label} missing images")
            continue

        n_show = min(n_cols, y_imgs.shape[0])
        for col in range(n_cols):
            row_y    = row_snap * 2
            row_xhat = row_snap * 2 + 1

            if col < n_show:
                axes[row_y,    col].imshow(y_imgs[col].squeeze().numpy(),
                                           cmap="gray", vmin=0, vmax=1)
                axes[row_xhat, col].imshow(x_hat_imgs[col].squeeze().numpy(),
                                           cmap="gray", vmin=0, vmax=1)
            else:
                axes[row_y,    col].axis("off")
                axes[row_xhat, col].axis("off")

            axes[row_y,    col].axis("off")
            axes[row_xhat, col].axis("off")

        axes[row_snap * 2, 0].set_ylabel(f"Epoch {epoch_label}\ny", fontsize=8)
        axes[row_snap * 2 + 1, 0].set_ylabel("x̂", fontsize=8)

    fig.suptitle("Stage C — Reconstruction Quality Over Epochs", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stage_c_recon_snapshots.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
