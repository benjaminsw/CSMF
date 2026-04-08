# =============================================================================
# Version: WP3.3-ExpertSanity-v1.1 | Abbr: EXP-SANITY
# Description: Expert sanity checks and diagnostic visualizations.
#              Called between Stage A → Stage B to verify experts are
#              reasonable before gate training.
# Changelog:
#   v1.0 (2026-03-01): Initial implementation — 3 core checks + 6 plots
#                      (Core 1-3: NLL/epochs, inv_err/epochs, z-hist;
#                       Additional A: recon grid, D: pairwise NLL, F: NLL rank)
#                      Returns summary dict, saves JSON + PNGs to output_dir
#   v1.1 (2026-04-03): Fix 1: Plot A switched from z~N(0,1) decode to encode→decode
#                      (z=f(x,h) then f⁻¹(z,h)) — separates invertibility bugs from
#                      generation quality. Fix 2: Added identity conditioning test
#                      (_check_identity_conditioning) — feeds clean x as y to verify
#                      conditioning path. Fix 3: Added λ_cons residual ablation
#                      (_check_lambda_cons_residual) — logs ‖Ax̂−y‖² baseline at
#                      current λ and recommended 0.5 to catch instability pre-retrain.
# Dependencies: CSMF-MAIN v1.3.6+, matplotlib, torch
# =============================================================================

import os
import json
import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Force non-interactive backend before any other matplotlib import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Main entry point
# =============================================================================

@torch.no_grad()
def run_expert_sanity(
    csmf_model,
    val_loader,
    fwd_model,
    epoch_logs: Dict[str, Dict[str, list]],
    output_dir: str = "results/expert_sanity",
    plots: Optional[List[str]] = None,
    max_val_batches: int = 20,
) -> Dict[str, Any]:
    """
    Run expert sanity checks and generate diagnostic plots after Stage A.

    Args:
        csmf_model:      CSMF model with trained (frozen) experts.
        val_loader:      Validation DataLoader yielding (x_clean, y_deg).
        fwd_model:       Forward model A for physics residual (e.g. SRForwardModel).
        epoch_logs:      Dict from train_stage_A():
                         {expert_name: {train_nll: [], val_nll: [], inv_err: []}}
        output_dir:      Directory for saved plots and JSON.
        plots:           List of plot codes to generate. Default: all.
                         Codes: "1","2","3" (core), "A","D","F" (additional).
        max_val_batches: Max val batches for diagnostics (speed control).

    Returns:
        summary dict with all check results.

    Raises:
        ValueError: if sample quality check fails (fatal).
    """
    if plots is None:
        plots = ["1", "2", "3", "A", "D", "F", "I", "L"]

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"[EXP-SANITY] Starting expert sanity | output_dir={output_dir} | plots={plots}")

    csmf_model.eval()
    device = csmf_model.device
    K = csmf_model.K

    # ------------------------------------------------------------------
    # Single val pass — collect all metrics needed for checks + plots
    # ------------------------------------------------------------------
    expert_names = [type(e).__name__ for e in csmf_model.experts]
    per_expert_data = {k: {
        "nll_per_sample": [],       # per-sample NLL (for D, F)
        "log_det_all": [],          # all log_det values (for check 1)
        "z_all": [],                # latent z samples (for check 3, plot 3)
    } for k in range(K)}

    n_collected = 0
    for x_clean, y_deg in val_loader:
        if n_collected >= max_val_batches:
            break
        x_clean = x_clean.to(device)
        y_deg = y_deg.to(device)
        h = csmf_model.conditioner(y_deg)

        for k, expert in enumerate(csmf_model.experts):
            try:
                z, log_det, log_prob, z_flist = csmf_model._expert_forward(
                    expert, x_clean, y_deg, h
                )

                if torch.isnan(log_det).any():
                    logger.warning(f"[EXP-SANITY] NaN log_det | expert={k} | batch={n_collected}")
                    continue

                # Per-sample NLL
                if log_prob is not None:
                    sample_nll = -log_prob  # (B,)
                else:
                    z_flat = z.flatten(1) if z.dim() > 2 else z
                    log_p_z = csmf_model.base_dist.log_prob(z_flat).sum(dim=1)
                    sample_nll = -(log_p_z + log_det)  # (B,)

                per_expert_data[k]["nll_per_sample"].append(sample_nll.cpu())
                per_expert_data[k]["log_det_all"].append(log_det.cpu())

                z_flat = z.flatten(1) if z.dim() > 2 else z
                per_expert_data[k]["z_all"].append(z_flat.cpu())

            except Exception as e:
                logger.error(f"[EXP-SANITY] Val pass error | expert={k} | batch={n_collected}: {e}")

        n_collected += 1

    if n_collected == 0:
        logger.error("[EXP-SANITY] No val batches collected — aborting sanity checks")
        raise ValueError("EXP-SANITY: no val batches collected")

    logger.info(f"[EXP-SANITY] Collected {n_collected} val batches for {K} experts")

    # Concatenate collected tensors
    for k in range(K):
        d = per_expert_data[k]
        d["nll_per_sample"] = torch.cat(d["nll_per_sample"]) if d["nll_per_sample"] else torch.tensor([])
        d["log_det_all"] = torch.cat(d["log_det_all"]) if d["log_det_all"] else torch.tensor([])
        d["z_all"] = torch.cat(d["z_all"]) if d["z_all"] else torch.tensor([])

    # ------------------------------------------------------------------
    # Core checks
    # ------------------------------------------------------------------
    summary = {"experts": {}, "checks": {}}

    for k in range(K):
        name = expert_names[k]
        d = per_expert_data[k]
        expert_summary = {"name": name}

        # Check 1: Log-det collapse — std(log_det)
        if d["log_det_all"].numel() > 0:
            ld_std = d["log_det_all"].std().item()
            expert_summary["log_det_std"] = round(ld_std, 6)
            if ld_std < 0.01:
                logger.warning(
                    f"[EXP-SANITY] CHECK 1 WARN | expert={k} ({name}) | "
                    f"log_det std={ld_std:.6f} < 0.01 — possible mode collapse"
                )
                expert_summary["log_det_collapse"] = True
            else:
                expert_summary["log_det_collapse"] = False
        else:
            logger.error(f"[EXP-SANITY] CHECK 1 | expert={k} ({name}) | no log_det data")
            expert_summary["log_det_collapse"] = "no_data"

        # Check 2: Sample quality — generate samples, check std
        try:
            expert = csmf_model.experts[k]
            # Get a y batch for conditioning
            sample_y = y_deg[:min(16, y_deg.shape[0])].to(device)
            sample_h = csmf_model.conditioner(sample_y)
            z_base = csmf_model.base_dist.sample((sample_y.shape[0], csmf_model.dim)).to(device)
            x_hat = csmf_model._expert_inverse(
                expert, z_base, sample_y, sample_h, z_factored_list=None
            )
            x_hat_flat = x_hat.flatten(1) if x_hat.dim() > 2 else x_hat
            sample_std = x_hat_flat.std().item()
            sample_has_nan = torch.isnan(x_hat).any().item()
            expert_summary["sample_std"] = round(sample_std, 6)
            expert_summary["sample_nan"] = sample_has_nan

            if sample_has_nan:
                logger.error(
                    f"[EXP-SANITY] CHECK 2 FATAL | expert={k} ({name}) | "
                    f"NaN in generated samples"
                )
                raise ValueError(f"EXP-SANITY: expert {k} ({name}) produces NaN samples")
            if sample_std < 1e-6:
                logger.error(
                    f"[EXP-SANITY] CHECK 2 FATAL | expert={k} ({name}) | "
                    f"sample std={sample_std:.2e} < 1e-6 — all-same output"
                )
                raise ValueError(
                    f"EXP-SANITY: expert {k} ({name}) sample std={sample_std:.2e} — degenerate"
                )
        except ValueError:
            raise  # re-raise fatals
        except Exception as e:
            logger.error(f"[EXP-SANITY] CHECK 2 | expert={k} ({name}) | sample error: {e}")
            expert_summary["sample_std"] = "error"
            expert_summary["sample_nan"] = "error"

        # Check 3: Base distribution fit — mean(z), std(z) vs N(0,1)
        if d["z_all"].numel() > 0:
            z_mean = d["z_all"].mean().item()
            z_std = d["z_all"].std().item()
            expert_summary["z_mean"] = round(z_mean, 4)
            expert_summary["z_std"] = round(z_std, 4)
            if abs(z_mean) > 1.0:
                logger.warning(
                    f"[EXP-SANITY] CHECK 3 WARN | expert={k} ({name}) | "
                    f"|z_mean|={abs(z_mean):.4f} > 1.0 — latent shift"
                )
            if abs(z_std - 1.0) > 1.0:
                logger.warning(
                    f"[EXP-SANITY] CHECK 3 WARN | expert={k} ({name}) | "
                    f"|z_std - 1|={abs(z_std - 1.0):.4f} > 1.0 — latent scale mismatch"
                )
        else:
            logger.error(f"[EXP-SANITY] CHECK 3 | expert={k} ({name}) | no z data")
            expert_summary["z_mean"] = "no_data"
            expert_summary["z_std"] = "no_data"

        summary["experts"][name] = expert_summary

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    # Core Plot 1: Per-expert NLL over epochs
    if "1" in plots and epoch_logs:
        try:
            _plot_nll_over_epochs(epoch_logs, output_dir)
            logger.info("[EXP-SANITY] Plot 1 (NLL over epochs) saved")
        except Exception as e:
            logger.error(f"[EXP-SANITY] Plot 1 failed: {e}")

    # Core Plot 2: Per-expert inv_err over epochs
    if "2" in plots and epoch_logs:
        try:
            _plot_inv_err_over_epochs(epoch_logs, output_dir)
            logger.info("[EXP-SANITY] Plot 2 (inv_err over epochs) saved")
        except Exception as e:
            logger.error(f"[EXP-SANITY] Plot 2 failed: {e}")

    # Core Plot 3: Latent z histogram per expert
    if "3" in plots:
        try:
            _plot_z_histograms(per_expert_data, expert_names, output_dir)
            logger.info("[EXP-SANITY] Plot 3 (latent z histograms) saved")
        except Exception as e:
            logger.error(f"[EXP-SANITY] Plot 3 failed: {e}")

    # Additional A: Per-expert reconstruction grid
    if "A" in plots:
        try:
            _plot_reconstruction_grid(
                csmf_model, val_loader, device, expert_names, output_dir, max_samples=8
            )
            logger.info("[EXP-SANITY] Plot A (reconstruction grid) saved")
        except Exception as e:
            logger.error(f"[EXP-SANITY] Plot A failed: {e}")

    # Additional D: Pairwise expert NLL scatter
    if "D" in plots and K >= 2:
        try:
            _plot_pairwise_nll_scatter(per_expert_data, expert_names, output_dir)
            logger.info("[EXP-SANITY] Plot D (pairwise NLL scatter) saved")
        except Exception as e:
            logger.error(f"[EXP-SANITY] Plot D failed: {e}")

    # Additional F: Expert NLL rank histogram
    if "F" in plots and K >= 2:
        try:
            _plot_nll_rank_histogram(per_expert_data, expert_names, output_dir)
            logger.info("[EXP-SANITY] Plot F (NLL rank histogram) saved")
        except Exception as e:
            logger.error(f"[EXP-SANITY] Plot F failed: {e}")

    # Fix 2 — Identity conditioning test (feed clean x as y)
    if "I" in plots:
        try:
            identity_results = _check_identity_conditioning(
                csmf_model, val_loader, device, expert_names
            )
            summary["identity_conditioning"] = identity_results
            logger.info("[EXP-SANITY] Identity conditioning check complete")
        except Exception as e:
            logger.error(f"[EXP-SANITY] Identity conditioning check failed: {e}")
            summary["identity_conditioning"] = {"error": str(e)}

    # Fix 3 — λ_cons residual ablation (baseline vs recommended 0.5)
    if "L" in plots and fwd_model is not None:
        try:
            ablation_results = _check_lambda_cons_residual(
                csmf_model, val_loader, fwd_model, device, expert_names,
                per_expert_data, lambda_cons_recommended=0.5,
            )
            summary["lambda_cons_ablation"] = ablation_results
            logger.info("[EXP-SANITY] λ_cons residual ablation complete")
        except Exception as e:
            logger.error(f"[EXP-SANITY] λ_cons ablation failed: {e}")
            summary["lambda_cons_ablation"] = {"error": str(e)}
    elif "L" in plots and fwd_model is None:
        logger.warning("[EXP-SANITY] Plot L skipped — fwd_model is None")

    # ------------------------------------------------------------------
    # Save summary JSON
    # ------------------------------------------------------------------
    summary_path = os.path.join(output_dir, "expert_sanity_summary.json")
    try:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"[EXP-SANITY] Summary saved: {summary_path}")
    except Exception as e:
        logger.error(f"[EXP-SANITY] Failed to save summary JSON: {e}")

    csmf_model.train()
    logger.info("[EXP-SANITY] Expert sanity complete")
    return summary


# =============================================================================
# Plot functions
# =============================================================================

def _plot_nll_over_epochs(
    epoch_logs: Dict[str, Dict[str, list]],
    output_dir: str,
) -> None:
    """Core Plot 1: All experts' train NLL on one overlay plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, logs in epoch_logs.items():
        train_nll = logs.get("train_nll", [])
        if train_nll:
            ax.plot(range(1, len(train_nll) + 1), train_nll, label=name, marker="o", markersize=3)
        val_nll = logs.get("val_nll", [])
        if val_nll:
            ax.plot(range(1, len(val_nll) + 1), val_nll, label=f"{name} (val)",
                    linestyle="--", marker="s", markersize=3, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NLL")
    ax.set_title("Stage A — Per-Expert NLL Over Epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "nll_over_epochs.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_inv_err_over_epochs(
    epoch_logs: Dict[str, Dict[str, list]],
    output_dir: str,
) -> None:
    """Core Plot 2: All experts' invertibility error on one overlay plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    has_data = False
    for name, logs in epoch_logs.items():
        inv_err = logs.get("inv_err", [])
        if inv_err:
            ax.plot(range(1, len(inv_err) + 1), inv_err, label=name, marker="o", markersize=3)
            has_data = True
    if not has_data:
        logger.warning("[EXP-SANITY] Plot 2: no inv_err data in epoch_logs — skipping")
        plt.close(fig)
        return
    ax.axhline(y=1e-4, color="r", linestyle=":", alpha=0.6, label="Fatal threshold (1e-4)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Invertibility Error (mean |f⁻¹(f(x)) − x|)")
    ax.set_title("Stage A — Per-Expert Invertibility Error Over Epochs")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "inv_err_over_epochs.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_z_histograms(
    per_expert_data: Dict[int, dict],
    expert_names: List[str],
    output_dir: str,
    max_dims: int = 5,
) -> None:
    """Core Plot 3: Latent z histograms per expert with N(0,1) reference."""
    K = len(expert_names)
    fig, axes = plt.subplots(1, K, figsize=(5 * K, 4), squeeze=False)
    ref_x = np.linspace(-4, 4, 200)
    ref_y = np.exp(-ref_x**2 / 2) / np.sqrt(2 * np.pi)

    for k in range(K):
        ax = axes[0, k]
        z_all = per_expert_data[k]["z_all"]
        if z_all.numel() == 0:
            ax.set_title(f"{expert_names[k]}\n(no data)")
            continue
        # Sample a few dims to avoid overcrowding
        n_dims = min(max_dims, z_all.shape[1])
        for d in range(n_dims):
            z_dim = z_all[:, d].numpy()
            ax.hist(z_dim, bins=60, density=True, alpha=0.3, label=f"dim {d}")
        ax.plot(ref_x, ref_y, "k--", linewidth=1.5, label="N(0,1)")
        ax.set_title(f"{expert_names[k]}\nμ={z_all.mean():.3f} σ={z_all.std():.3f}")
        ax.set_xlim(-4, 4)
        ax.legend(fontsize=7)
    fig.suptitle("Stage A — Latent z Distributions Per Expert", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latent_z_histograms.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def _plot_reconstruction_grid(
    csmf_model,
    val_loader,
    device: torch.device,
    expert_names: List[str],
    output_dir: str,
    max_samples: int = 8,
) -> None:
    """
    Additional Plot A: Encode→decode reconstruction per expert.
    z = f(x, h), x̂ = f⁻¹(z, h) — separates invertibility bugs from generation quality.
    Grid: rows = samples, cols = [y | x_clean | expert_0 x̂ | expert_1 x̂ | ...]
    """
    K = len(expert_names)
    # Grab first batch
    x_clean, y_deg = next(iter(val_loader))
    n = min(max_samples, x_clean.shape[0])
    x_clean = x_clean[:n].to(device)
    y_deg = y_deg[:n].to(device)
    h = csmf_model.conditioner(y_deg)

    fig, axes = plt.subplots(n, K + 2, figsize=(2.5 * (K + 2), 2.5 * n), squeeze=False)
    col_labels = ["y (input)", "x (clean)"] + [f"{name}\nz=f(x,h)→f⁻¹" for name in expert_names]

    for i in range(n):
        # Col 0: degraded y
        y_img = y_deg[i].cpu().squeeze()
        axes[i, 0].imshow(y_img, cmap="gray", vmin=0, vmax=1)
        axes[i, 0].axis("off")

        # Col 1: clean x
        x_img = x_clean[i].cpu().squeeze()
        axes[i, 1].imshow(x_img, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].axis("off")

        # Cols 2+: per-expert encode→decode reconstruction
        for k, expert in enumerate(csmf_model.experts):
            try:
                # Encode: z = f(x, h)
                z, log_det, log_prob, z_flist = csmf_model._expert_forward(
                    expert, x_clean[i:i+1], y_deg[i:i+1], h[i:i+1]
                )
                # Decode: x̂ = f⁻¹(z, h)
                x_hat = csmf_model._expert_inverse(
                    expert, z, y_deg[i:i+1], h[i:i+1], z_factored_list=z_flist
                )
                x_hat_img = x_hat.cpu().view(28, 28)
                axes[i, k + 2].imshow(x_hat_img.clamp(0, 1), cmap="gray", vmin=0, vmax=1)
            except Exception as e:
                logger.error(f"[EXP-SANITY] Plot A | sample {i} expert {k} encode→decode: {e}")
                axes[i, k + 2].text(0.5, 0.5, "ERR", ha="center", va="center", fontsize=10)
            axes[i, k + 2].axis("off")

    for c, lbl in enumerate(col_labels):
        axes[0, c].set_title(lbl, fontsize=9)
    fig.suptitle("Stage A — Per-Expert Reconstructions (encode→decode: z=f(x,h), x̂=f⁻¹(z,h))", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reconstruction_grid.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pairwise_nll_scatter(
    per_expert_data: Dict[int, dict],
    expert_names: List[str],
    output_dir: str,
) -> None:
    """
    Additional Plot D: Pairwise NLL scatter for all expert pairs.
    High correlation = redundant; low correlation = complementary.
    """
    K = len(expert_names)
    n_pairs = K * (K - 1) // 2
    if n_pairs == 0:
        return
    fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 5), squeeze=False)
    pair_idx = 0
    for i in range(K):
        for j in range(i + 1, K):
            nll_i = per_expert_data[i]["nll_per_sample"]
            nll_j = per_expert_data[j]["nll_per_sample"]
            if nll_i.numel() == 0 or nll_j.numel() == 0:
                axes[0, pair_idx].set_title("no data")
                pair_idx += 1
                continue
            n_min = min(nll_i.shape[0], nll_j.shape[0])
            ni = nll_i[:n_min].numpy()
            nj = nll_j[:n_min].numpy()
            # Clamp outliers for readability
            p5, p95 = np.percentile(np.concatenate([ni, nj]), [5, 95])
            ax = axes[0, pair_idx]
            ax.scatter(ni, nj, alpha=0.15, s=8, edgecolors="none")
            ax.set_xlabel(f"{expert_names[i]} NLL")
            ax.set_ylabel(f"{expert_names[j]} NLL")
            # Compute correlation
            if np.std(ni) > 1e-8 and np.std(nj) > 1e-8:
                corr = np.corrcoef(ni, nj)[0, 1]
                ax.set_title(f"{expert_names[i]} vs {expert_names[j]}\nρ = {corr:.3f}")
            else:
                ax.set_title(f"{expert_names[i]} vs {expert_names[j]}")
            ax.plot([p5, p95], [p5, p95], "r--", alpha=0.5, linewidth=1)
            ax.grid(True, alpha=0.3)
            pair_idx += 1

    fig.suptitle("Stage A — Pairwise Expert NLL Scatter", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pairwise_nll_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_nll_rank_histogram(
    per_expert_data: Dict[int, dict],
    expert_names: List[str],
    output_dir: str,
) -> None:
    """
    Additional Plot F: For each val sample, which expert has the lowest NLL?
    Bar chart showing win-rate per expert.
    """
    K = len(expert_names)
    # Stack per-sample NLLs: (N, K)
    nll_lists = []
    min_n = float("inf")
    for k in range(K):
        nll_k = per_expert_data[k]["nll_per_sample"]
        if nll_k.numel() == 0:
            logger.warning(f"[EXP-SANITY] Plot F: expert {k} has no NLL data — skipping")
            return
        nll_lists.append(nll_k)
        min_n = min(min_n, nll_k.shape[0])

    nll_matrix = torch.stack([nl[:min_n] for nl in nll_lists], dim=1)  # (N, K)
    winners = nll_matrix.argmin(dim=1)  # (N,)

    win_counts = torch.zeros(K)
    for k in range(K):
        win_counts[k] = (winners == k).sum().item()
    win_pct = (win_counts / win_counts.sum() * 100).numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.Set2(np.linspace(0, 1, K))
    bars = ax.bar(range(K), win_pct, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(K))
    ax.set_xticklabels(expert_names, rotation=15, ha="right")
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Stage A — Expert NLL Rank (Best Expert Per Sample)")
    for bar, pct in zip(bars, win_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(win_pct) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "nll_rank_histogram.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Fix 2 — Identity conditioning check
# =============================================================================

@torch.no_grad()
def _check_identity_conditioning(
    csmf_model,
    val_loader,
    device: torch.device,
    expert_names: List[str],
    identity_err_warn_threshold: float = 0.1,
    n_batches: int = 3,
) -> Dict[str, Any]:
    """
    Fix 2: Feed clean x as y (identity conditioning test).

    Computes h = conditioner(x_clean), then encodes and decodes x_clean
    through each expert. Measures ‖x̂ − x‖ — should be near zero if the
    conditioning path and invertibility are both healthy.

    Args:
        csmf_model:                 CSMF model with frozen experts.
        val_loader:                 Validation DataLoader yielding (x_clean, y_deg).
        device:                     Compute device.
        expert_names:               List of expert class names.
        identity_err_warn_threshold: Log warning if mean abs err exceeds this.
        n_batches:                  Number of val batches to average over.

    Returns:
        Dict with per-expert identity errors and pass/warn status.
    """
    results: Dict[str, Any] = {}
    K = len(csmf_model.experts)

    per_expert_errs: Dict[int, list] = {k: [] for k in range(K)}

    collected = 0
    for x_clean, y_deg in val_loader:
        if collected >= n_batches:
            break
        x_clean = x_clean.to(device)
        # Feed x_clean as the conditioning input (identity test)
        try:
            h_id = csmf_model.conditioner(x_clean)
        except Exception as e:
            logger.error(f"[EXP-SANITY] Identity test | conditioner(x_clean) failed batch={collected}: {e}")
            collected += 1
            continue

        for k, expert in enumerate(csmf_model.experts):
            try:
                # Encode x through expert conditioned on x (not degraded y)
                z, log_det, log_prob, z_flist = csmf_model._expert_forward(
                    expert, x_clean, x_clean, h_id
                )
                # Decode back
                x_hat = csmf_model._expert_inverse(
                    expert, z, x_clean, h_id, z_factored_list=z_flist
                )
                err = (x_hat - x_clean).abs().mean().item()
                per_expert_errs[k].append(err)
            except Exception as e:
                logger.error(
                    f"[EXP-SANITY] Identity test | expert={k} ({expert_names[k]}) "
                    f"batch={collected}: {e}"
                )
        collected += 1

    if collected == 0:
        logger.error("[EXP-SANITY] Identity test: no batches collected")
        return {"error": "no batches collected"}

    for k in range(K):
        name = expert_names[k]
        errs = per_expert_errs[k]
        if not errs:
            logger.error(f"[EXP-SANITY] Identity test | expert={k} ({name}) | no error data")
            results[name] = {"mean_abs_err": "no_data", "status": "error"}
            continue
        mean_err = float(np.mean(errs))
        status = "pass" if mean_err <= identity_err_warn_threshold else "warn"
        if status == "warn":
            logger.warning(
                f"[EXP-SANITY] Identity test WARN | expert={k} ({name}) | "
                f"mean |x̂−x|={mean_err:.6f} > threshold={identity_err_warn_threshold} "
                f"— conditioning path may be impaired"
            )
        else:
            logger.info(
                f"[EXP-SANITY] Identity test PASS | expert={k} ({name}) | "
                f"mean |x̂−x|={mean_err:.6f}"
            )
        results[name] = {"mean_abs_err": round(mean_err, 6), "status": status}

    return results


# =============================================================================
# Fix 3 — λ_cons residual ablation
# =============================================================================

@torch.no_grad()
def _check_lambda_cons_residual(
    csmf_model,
    val_loader,
    fwd_model,
    device: torch.device,
    expert_names: List[str],
    per_expert_data: Dict[int, dict],
    lambda_cons_recommended: float = 0.5,
    n_batches: int = 5,
) -> Dict[str, Any]:
    """
    Fix 3: λ_cons residual ablation — logs consistency residual ‖Ax̂−y‖²
    per expert as a baseline before committing to λ_cons=0.5.

    Computes mean residual using generated samples (z~N(0,1) decode) over
    val batches. Also computes NLL/residual ratio to expose whether NLL is
    currently dominating (ratio >> 1 → λ_cons too small).

    Args:
        csmf_model:              CSMF model with frozen experts.
        val_loader:              Validation DataLoader yielding (x_clean, y_deg).
        fwd_model:               Forward model A (must implement __call__(x) → Ax).
        device:                  Compute device.
        expert_names:            List of expert class names.
        per_expert_data:         Collected val-pass data (for mean NLL lookup).
        lambda_cons_recommended: Target λ_cons to flag in recommendation.
        n_batches:               Batches to average residual over.

    Returns:
        Dict with per-expert residuals, NLL/residual ratios, and recommendation.
    """
    results: Dict[str, Any] = {"per_expert": {}, "recommendation": {}}
    K = len(csmf_model.experts)

    per_expert_residuals: Dict[int, list] = {k: [] for k in range(K)}

    collected = 0
    for x_clean, y_deg in val_loader:
        if collected >= n_batches:
            break
        x_clean = x_clean.to(device)
        y_deg = y_deg.to(device)
        h = csmf_model.conditioner(y_deg)

        for k, expert in enumerate(csmf_model.experts):
            try:
                z_base = csmf_model.base_dist.sample(
                    (y_deg.shape[0], csmf_model.dim)
                ).to(device)
                x_hat = csmf_model._expert_inverse(
                    expert, z_base, y_deg, h, z_factored_list=None
                )
                if torch.isnan(x_hat).any():
                    logger.warning(
                        f"[EXP-SANITY] λ_cons ablation | expert={k} ({expert_names[k]}) "
                        f"| NaN in x_hat at batch={collected} — skipping batch"
                    )
                    continue
                # Compute forward model output and residual
                Ax_hat = fwd_model(x_hat)
                residual = ((Ax_hat - y_deg) ** 2).mean().item()
                per_expert_residuals[k].append(residual)
            except Exception as e:
                logger.error(
                    f"[EXP-SANITY] λ_cons ablation | expert={k} ({expert_names[k]}) "
                    f"batch={collected}: {e}"
                )
        collected += 1

    if collected == 0:
        logger.error("[EXP-SANITY] λ_cons ablation: no batches collected")
        return {"error": "no batches collected"}

    for k in range(K):
        name = expert_names[k]
        resid_list = per_expert_residuals[k]
        if not resid_list:
            logger.error(f"[EXP-SANITY] λ_cons ablation | expert={k} ({name}) | no residual data")
            results["per_expert"][name] = {"mean_residual": "no_data"}
            continue

        mean_residual = float(np.mean(resid_list))

        # NLL/residual ratio — large ratio means NLL dominates
        nll_data = per_expert_data[k]["nll_per_sample"]
        mean_nll = nll_data.mean().item() if nll_data.numel() > 0 else float("nan")
        ratio = mean_nll / (mean_residual + 1e-8)

        status = "nll_dominates" if ratio > 10.0 else "balanced"
        logger.info(
            f"[EXP-SANITY] λ_cons ablation | expert={k} ({name}) | "
            f"mean_residual={mean_residual:.6f} | mean_NLL={mean_nll:.4f} | "
            f"NLL/residual ratio={ratio:.2f} | status={status}"
        )
        results["per_expert"][name] = {
            "mean_residual": round(mean_residual, 6),
            "mean_nll": round(mean_nll, 4) if not np.isnan(mean_nll) else "no_data",
            "nll_residual_ratio": round(ratio, 2),
            "status": status,
        }

    # Overall recommendation
    any_dominated = any(
        v.get("status") == "nll_dominates"
        for v in results["per_expert"].values()
        if isinstance(v, dict)
    )
    results["recommendation"] = {
        "lambda_cons_recommended": lambda_cons_recommended,
        "increase_lambda_cons": any_dominated,
        "note": (
            f"NLL/residual ratio > 10 for at least one expert — "
            f"recommend increasing λ_cons to {lambda_cons_recommended}."
        ) if any_dominated else (
            f"λ_cons appears balanced — verify residual trend at epoch 1 after "
            f"increasing to {lambda_cons_recommended} before committing."
        ),
    }
    logger.info(
        f"[EXP-SANITY] λ_cons ablation recommendation: "
        f"increase={any_dominated}, note={results['recommendation']['note']}"
    )
    return results
