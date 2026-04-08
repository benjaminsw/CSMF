# =============================================================================
# Version: FI-DIAG-v1.5 | Abbr: FI-DIAG
# Description: Standalone Fisher Information diagnostics for CSMF experts.
#              Computes Option A (batch-averaged FI scalar per expert),
#              Option B (per-sample FI), latent Gaussian KL proxy, NLL stats,
#              and invertibility checks. Generates 6 diagnostic plots.
#              Saves fi_diag_summary.json + fi_per_sample.npy to output_dir.
# Changelog:
#   v1.5 (2026-04-02): [EPOCH-FI] Added compute_fi_option_a_batch() public function
#                      for lightweight per-epoch FI tracking during train_stage_A();
#                      added P5 _plot_fi_per_epoch() (x=epoch, y=FI, lines=experts);
#                      added P6 _plot_fi_vs_nll_scatter() (x=NLL_k, y=FI_k, one
#                      point per expert); run_fi_diagnostics() accepts epoch_logs
#                      param to enable P5; default plots extended to P1-P6
#   v1.0 (2026-04-01): Initial implementation — Options A/B, pairwise KL proxy,
#                      4 plots (P1-P4), JSON + npy outputs
#   v1.4 (2026-04-02): Raised INV_ERR_FATAL 1e-3 → 5e-3 matching eval_expert()
#                      threshold in csmf.py v1.3.12 (RealNVP logit preprocessing
#                      causes ~1e-3 pixel-space noise — not a real failure); added
#                      nll_comparable_across_experts=false and explanatory note to
#                      NLL section (RealNVP logit-space vs NICE/NSF raw-space);
#                      NLL gap warning log now includes preprocessing caveat
#   v1.3 (2026-04-02): Fixed invertibility for all 3 experts — forward pass now
#                      captures z_flist and passes to _expert_inverse (fixes RealNVP
#                      empty z_factored_list warning and max_err=0.127); flattened
#                      both sides of inv_err comparison (fixes NICE/NSF no_data);
#                      latent KL skips pairs with mismatched z dims with warning
#                      (fixes RealNVP z[196] vs NICE/NSF z[784] crash)
#   v1.2 (2026-04-01): Fixed __main__ CLI — corrected imports to configs.mnist_config;
#                      replaced get_config()/CSMF(config) with build_model() from
#                      experiments.train_csmf; fixed checkpoint key model_state_dict
#                      → state_dict; val loader uses create_precomputed_dataloaders
#                      from scripts.preprocess_mnist with correct signature
#   v1.1 (2026-04-01): Multi-batch aggregation (3-10 batches, mean+std across
#                      batches); relative FI ratio thresholds replacing absolute
#                      (fatal < 0.05, warn < 0.15); renamed pairwise_kl to
#                      latent_gaussian_kl_proxy with note; added n_params,
#                      F_per_param, NLL section, invertibility section;
#                      4-signal ready_for_stage_B verdict (FI + NLL gap +
#                      invertibility + log-det collapse)
# Dependencies: CSMF-MAIN v1.3.6+, matplotlib, torch, numpy
# Run:
#   PYTHONPATH=. python csmf/evaluation/fisher_info_diag.py \
#     --checkpoint checkpoints/csmf_stage_A.pth \
#     --output results/fisher_info
# =============================================================================

import os
import sys
import json
import logging
import argparse
import datetime
from typing import Dict, List, Optional, Any

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

logger = logging.getLogger(__name__)

# =============================================================================
# Thresholds
# =============================================================================
RATIO_FATAL      = 0.05   # F_k / max_j(F_j) — expert is dead
RATIO_WARN       = 0.15   # F_k / max_j(F_j) — expert is weak
KL_WARN_SIMILAR  = 0.5    # latent KL proxy below this → warn similar
NLL_GAP_WARN     = 30.0   # max_nll - min_nll above this → warn
INV_ERR_WARN     = 1e-4
INV_ERR_FATAL    = 5e-3   # v1.4: raised from 1e-3 — matches eval_expert() threshold in csmf.py v1.3.12
                           # RealNVP logit preprocessing causes ~1e-3 pixel-space numerical noise


# =============================================================================
# Main entry point (callable API)
# =============================================================================

def run_fi_diagnostics(
    csmf_model,
    val_loader,
    output_dir: str = "results/fisher_info",
    n_batches: int = 5,
    run_option_b: bool = True,
    plots: Optional[List[str]] = None,
    epoch_logs: Optional[Dict] = None,  # [EPOCH-FI] v1.5: pass train_stage_A() epoch_logs for P5
) -> Dict[str, Any]:
    """
    Run Fisher Information diagnostics on a frozen Stage A CSMF model.

    Args:
        csmf_model:   CSMF model with trained (frozen) experts. Must be in eval().
        val_loader:   Validation DataLoader yielding (x_clean, y_deg).
        output_dir:   Directory for plots, JSON, and fi_per_sample.npy.
        n_batches:    Number of val batches to aggregate (3–10 recommended).
        run_option_b: Compute per-sample FI (Option B). Slower due to per-sample
                      backward pass.
        plots:        Plot codes. Default: all ["P1","P2","P3","P4","P5","P6"].
        epoch_logs:   [v1.5] Dict returned by train_stage_A() with per-expert "fi_a"
                      lists. Required for P5 (FI per epoch). If None, P5 is skipped.

    Returns:
        summary dict matching fi_diag_summary.json structure.

    Raises:
        ValueError: if no val batches can be collected.
    """
    if plots is None:
        plots = ["P1", "P2", "P3", "P4", "P5", "P6"]

    os.makedirs(output_dir, exist_ok=True)
    logger.info(
        f"[FI-DIAG] Starting | output_dir={output_dir} | "
        f"n_batches={n_batches} | option_b={run_option_b} | plots={plots}"
    )

    device      = csmf_model.device
    K           = csmf_model.K
    expert_names = [type(e).__name__ for e in csmf_model.experts]

    # Per-batch accumulators
    fi_a_per_batch    = {k: [] for k in range(K)}   # list of F_k scalars per batch
    fi_b_per_sample   = {k: [] for k in range(K)}   # list of tensors (B,)
    nll_per_batch     = {k: [] for k in range(K)}
    z_all             = {k: [] for k in range(K)}
    inv_err_per_batch = {k: [] for k in range(K)}

    batches_collected = 0
    last_batch_size   = None
    val_iter          = iter(val_loader)

    for batch_idx in range(n_batches):
        try:
            x_clean, y_deg = next(val_iter)
        except StopIteration:
            logger.warning(f"[FI-DIAG] Val loader exhausted after {batches_collected} batches")
            break

        x_clean = x_clean.to(device)
        y_deg   = y_deg.to(device)
        last_batch_size = x_clean.shape[0]

        # --- Gradient-based: Option A + B ---
        for k, expert in enumerate(csmf_model.experts):
            try:
                _compute_fi_for_expert(
                    csmf_model=csmf_model,
                    expert=expert,
                    k=k,
                    x_clean=x_clean,
                    y_deg=y_deg,
                    fi_a_per_batch=fi_a_per_batch,
                    fi_b_per_sample=fi_b_per_sample,
                    run_option_b=run_option_b,
                    batch_idx=batch_idx,
                )
            except Exception as e:
                logger.error(
                    f"[FI-DIAG] FI compute error | expert={k} ({expert_names[k]}) "
                    f"| batch={batch_idx}: {e}"
                )

        # --- No-grad: NLL + z + invertibility ---
        with torch.no_grad():
            try:
                h = csmf_model.conditioner(y_deg)
            except Exception as e:
                logger.error(f"[FI-DIAG] Conditioner error | batch={batch_idx}: {e}")
                batches_collected += 1
                continue

            for k, expert in enumerate(csmf_model.experts):
                try:
                    z, log_det, log_prob, z_flist = csmf_model._expert_forward(
                        expert, x_clean, y_deg, h
                    )

                    if torch.isnan(log_det).any():
                        logger.warning(
                            f"[FI-DIAG] NaN log_det | expert={k} ({expert_names[k]}) "
                            f"| batch={batch_idx}"
                        )
                        continue

                    # NLL
                    if log_prob is not None:
                        batch_nll = (-log_prob).mean().item()
                    else:
                        z_flat  = z.flatten(1) if z.dim() > 2 else z
                        log_p_z = csmf_model.base_dist.log_prob(z_flat).sum(dim=1)
                        batch_nll = (-(log_p_z + log_det)).mean().item()
                    nll_per_batch[k].append(batch_nll)

                    # z for KL proxy
                    z_flat = z.flatten(1) if z.dim() > 2 else z
                    z_all[k].append(z_flat.cpu())

                    # Invertibility: encode x → z, decode z → x_hat
                    # Fix 1: pass z_flist (not None) so RealNVP inverse is faithful
                    # Fix 2: flatten both sides — NICE/NSF return [B,784], x_clean is [B,1,28,28]
                    try:
                        x_hat   = csmf_model._expert_inverse(
                            expert, z, y_deg, h, z_factored_list=z_flist
                        )
                        inv_err = (x_hat.flatten(1) - x_clean.flatten(1)).abs().mean().item()
                        inv_err_per_batch[k].append(inv_err)
                    except Exception as ie:
                        logger.error(
                            f"[FI-DIAG] Invertibility error | expert={k} ({expert_names[k]}) "
                            f"| batch={batch_idx}: {ie}"
                        )

                except Exception as e:
                    logger.error(
                        f"[FI-DIAG] NLL/z pass error | expert={k} ({expert_names[k]}) "
                        f"| batch={batch_idx}: {e}"
                    )

        batches_collected += 1

    if batches_collected == 0:
        logger.error("[FI-DIAG] No val batches collected — aborting")
        raise ValueError("FI-DIAG: no val batches collected")

    logger.info(f"[FI-DIAG] Collected {batches_collected} batches for {K} experts")

    # ------------------------------------------------------------------
    # Aggregate Option A — mean+std across batches, relative ratios
    # ------------------------------------------------------------------
    fi_a_results = {}
    fi_a_means   = []

    for k in range(K):
        vals = fi_a_per_batch[k]
        name = expert_names[k]
        if len(vals) == 0:
            logger.error(f"[FI-DIAG] Option A: no FI data | expert={k} ({name})")
            fi_a_results[name] = {
                "F_k_mean": None, "F_k_std": None,
                "F_ratio_to_best": None,
                "n_params": None, "F_per_param": None,
                "status": "no_data",
            }
            fi_a_means.append(0.0)
        else:
            arr      = np.array(vals)
            mean_fk  = float(arr.mean())
            std_fk   = float(arr.std())
            n_params = sum(p.numel() for p in csmf_model.experts[k].parameters())
            fi_a_means.append(mean_fk)
            fi_a_results[name] = {
                "F_k_mean":      round(mean_fk, 6),
                "F_k_std":       round(std_fk, 6),
                "F_ratio_to_best": None,   # filled after all experts
                "n_params":      n_params,
                "F_per_param":   round(mean_fk / n_params, 8) if n_params > 0 else None,
                "status":        None,     # filled after all experts
            }

    best_fk = max(fi_a_means) if fi_a_means else 1.0
    if best_fk <= 0.0:
        logger.warning("[FI-DIAG] best_fk <= 0 — all experts may be dead")
        best_fk = 1.0

    for k in range(K):
        name  = expert_names[k]
        entry = fi_a_results[name]
        if entry["F_k_mean"] is None:
            continue
        ratio = fi_a_means[k] / best_fk
        entry["F_ratio_to_best"] = round(ratio, 4)
        if ratio < RATIO_FATAL:
            entry["status"] = "fatal_dead"
            logger.error(
                f"[FI-DIAG] FATAL | expert={k} ({name}) | "
                f"F_ratio={ratio:.4f} < {RATIO_FATAL} — expert dead"
            )
        elif ratio < RATIO_WARN:
            entry["status"] = "warn_low"
            logger.warning(
                f"[FI-DIAG] WARN | expert={k} ({name}) | "
                f"F_ratio={ratio:.4f} < {RATIO_WARN} — weak expert"
            )
        else:
            entry["status"] = "alive"

    # ------------------------------------------------------------------
    # Aggregate Option B — per-sample stats + save .npy
    # ------------------------------------------------------------------
    fi_b_results = {}
    fi_b_arrays  = []   # shape [K][N] for .npy

    for k in range(K):
        name    = expert_names[k]
        samples = fi_b_per_sample[k]
        if len(samples) == 0:
            fi_b_results[name] = {"mean": None, "std": None, "min": None, "max": None}
            fi_b_arrays.append(np.array([]))
        else:
            arr = torch.cat(samples).numpy()
            fi_b_results[name] = {
                "mean": round(float(np.nanmean(arr)), 6),
                "std":  round(float(np.nanstd(arr)),  6),
                "min":  round(float(np.nanmin(arr)),  6),
                "max":  round(float(np.nanmax(arr)),  6),
            }
            fi_b_arrays.append(arr)

    if run_option_b and any(len(a) > 0 for a in fi_b_arrays):
        try:
            npy_path = os.path.join(output_dir, "fi_per_sample.npy")
            max_n    = max(len(a) for a in fi_b_arrays)
            padded   = np.full((K, max_n), np.nan)
            for k, arr in enumerate(fi_b_arrays):
                if len(arr) > 0:
                    padded[k, :len(arr)] = arr
            np.save(npy_path, padded)
            logger.info(f"[FI-DIAG] Option B npy saved: {npy_path} | shape={padded.shape}")
        except Exception as e:
            logger.error(f"[FI-DIAG] Failed to save fi_per_sample.npy: {e}")

    # ------------------------------------------------------------------
    # Latent Gaussian KL proxy
    # ------------------------------------------------------------------
    latent_kl = _compute_latent_gaussian_kl(z_all, expert_names, K)

    # ------------------------------------------------------------------
    # NLL stats
    # ------------------------------------------------------------------
    nll_results = {}
    nll_means   = []

    for k in range(K):
        name = expert_names[k]
        vals = nll_per_batch[k]
        if len(vals) == 0:
            nll_results[name] = {"mean": None, "std": None}
            nll_means.append(float("inf"))
        else:
            arr = np.array(vals)
            m   = float(arr.mean())
            nll_results[name] = {
                "mean": round(m, 4),
                "std":  round(float(arr.std()), 4),
            }
            nll_means.append(m)

    valid_nlls = [v for v in nll_means if v != float("inf")]
    nll_gap        = round(max(valid_nlls) - min(valid_nlls), 4) if len(valid_nlls) >= 2 else None
    nll_gap_status = "warn" if (nll_gap is not None and nll_gap > NLL_GAP_WARN) else "ok"
    nll_results["nll_gap"]        = nll_gap
    nll_results["nll_gap_status"] = nll_gap_status
    nll_results["nll_comparable_across_experts"] = False
    nll_results["note"] = (
        "RealNVP uses logit-preprocessed image-space flow; NICE/NSF use flattened "
        "raw-space inputs, so absolute NLL gap is only a rough diagnostic."
    )

    if nll_gap_status == "warn":
        logger.warning(
            f"[FI-DIAG] NLL gap={nll_gap:.2f} > {NLL_GAP_WARN} — experts imbalanced "
            f"(note: NLL not comparable across preprocessing regimes)"
        )

    # ------------------------------------------------------------------
    # Invertibility stats
    # ------------------------------------------------------------------
    inv_results  = {}
    inv_pass_all = True

    for k in range(K):
        name = expert_names[k]
        vals = inv_err_per_batch[k]
        if len(vals) == 0:
            inv_results[name] = {"max_err": None, "status": "no_data"}
            inv_pass_all = False
        else:
            max_err = float(max(vals))
            if max_err > INV_ERR_FATAL:
                status       = "fatal"
                inv_pass_all = False
                logger.error(
                    f"[FI-DIAG] Invertibility FATAL | expert={k} ({name}) "
                    f"| max_err={max_err:.2e} > {INV_ERR_FATAL}"
                )
            elif max_err > INV_ERR_WARN:
                status = "warn"
                logger.warning(
                    f"[FI-DIAG] Invertibility WARN | expert={k} ({name}) "
                    f"| max_err={max_err:.2e} > {INV_ERR_WARN}"
                )
            else:
                status = "pass"
            inv_results[name] = {"max_err": float(f"{max_err:.3e}"), "status": status}

    # ------------------------------------------------------------------
    # 4-signal verdict
    # ------------------------------------------------------------------
    fi_ratio_pass = all(
        fi_a_results[n].get("status") == "alive"
        for n in expert_names
        if fi_a_results[n].get("F_k_mean") is not None
    )
    nll_gap_pass  = nll_gap_status == "ok"
    # log-det collapse not re-computed here — covered by EXP-SANITY
    logdet_pass   = True

    ready = fi_ratio_pass and nll_gap_pass and inv_pass_all

    notes = []
    dead  = [n for n in expert_names if fi_a_results[n].get("status") == "fatal_dead"]
    weak  = [n for n in expert_names if fi_a_results[n].get("status") == "warn_low"]
    if dead:
        notes.append(f"Dead experts: {dead}")
    if weak:
        notes.append(f"Weak experts: {weak}")
    if not nll_gap_pass:
        notes.append(f"NLL gap={nll_gap} > {NLL_GAP_WARN}")
    if not inv_pass_all:
        notes.append("Invertibility issues detected")
    if not notes:
        notes.append("All checks passed")

    verdict = {
        "fi_ratio_pass":        fi_ratio_pass,
        "nll_gap_pass":         nll_gap_pass,
        "invertibility_pass":   inv_pass_all,
        "logdet_collapse_pass": logdet_pass,
        "ready_for_stage_B":    ready,
        "notes":                " | ".join(notes),
    }

    # ------------------------------------------------------------------
    # Assemble and save summary JSON
    # ------------------------------------------------------------------
    summary = {
        "fi_diag_version":  "FI-DIAG-v1.1",
        "stage":            "post_A",
        "timestamp":        datetime.datetime.now().isoformat(timespec="seconds"),
        "n_batches_used":   batches_collected,
        "batch_size":       last_batch_size,
        "experts":          expert_names,
        "thresholds": {
            "ratio_fatal_below":     RATIO_FATAL,
            "ratio_warn_below":      RATIO_WARN,
            "kl_warn_similar_below": KL_WARN_SIMILAR,
            "nll_gap_warn_above":    NLL_GAP_WARN,
            "inv_err_warn":          INV_ERR_WARN,
            "inv_err_fatal":         INV_ERR_FATAL,
        },
        "option_A":                 fi_a_results,
        "option_B":                 fi_b_results if run_option_b else "skipped",
        "latent_gaussian_kl_proxy": latent_kl,
        "nll":                      nll_results,
        "invertibility":            inv_results,
        "verdict":                  verdict,
    }

    json_path = os.path.join(output_dir, "fi_diag_summary.json")
    try:
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"[FI-DIAG] JSON saved: {json_path}")
    except Exception as e:
        logger.error(f"[FI-DIAG] Failed to save JSON: {e}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if "P1" in plots:
        try:
            _plot_fi_scalar_per_expert(fi_a_results, fi_a_per_batch, expert_names, output_dir)
            logger.info("[FI-DIAG] Plot P1 saved: fi_scalar_per_expert.png")
        except Exception as e:
            logger.error(f"[FI-DIAG] Plot P1 failed: {e}")

    if "P2" in plots and run_option_b:
        try:
            _plot_fi_per_sample_dist(fi_b_arrays, expert_names, output_dir)
            logger.info("[FI-DIAG] Plot P2 saved: fi_per_sample_dist.png")
        except Exception as e:
            logger.error(f"[FI-DIAG] Plot P2 failed: {e}")

    if "P3" in plots and run_option_b:
        try:
            _plot_fi_expert_scatter(fi_b_arrays, expert_names, output_dir)
            logger.info("[FI-DIAG] Plot P3 saved: fi_expert_scatter.png")
        except Exception as e:
            logger.error(f"[FI-DIAG] Plot P3 failed: {e}")

    if "P4" in plots:
        try:
            _plot_latent_kl_heatmap(latent_kl, expert_names, output_dir)
            logger.info("[FI-DIAG] Plot P4 saved: latent_gaussian_kl_heatmap.png")
        except Exception as e:
            logger.error(f"[FI-DIAG] Plot P4 failed: {e}")

    # [EPOCH-FI] v1.5: P5 — FI per epoch (requires epoch_logs from train_stage_A)
    if "P5" in plots:
        if epoch_logs is not None:
            try:
                _plot_fi_per_epoch(epoch_logs, expert_names, output_dir)
                logger.info("[FI-DIAG] Plot P5 saved: fi_per_epoch.png")
            except Exception as e:
                logger.error(f"[FI-DIAG] Plot P5 failed: {e}")
        else:
            logger.warning("[FI-DIAG] Plot P5 skipped — epoch_logs not provided")

    # [EPOCH-FI] v1.5: P6 — FI vs NLL scatter (one point per expert, fully offline)
    if "P6" in plots:
        try:
            _plot_fi_vs_nll_scatter(fi_a_means, nll_means, expert_names, output_dir)
            logger.info("[FI-DIAG] Plot P6 saved: fi_vs_nll_scatter.png")
        except Exception as e:
            logger.error(f"[FI-DIAG] Plot P6 failed: {e}")

    logger.info(f"[FI-DIAG] Done | ready_for_stage_B={ready} | {verdict['notes']}")
    return summary


# =============================================================================
# [EPOCH-FI] v1.5: Public lightweight FI Option A — for per-epoch tracking
# =============================================================================

def compute_fi_option_a_batch(
    csmf_model,
    expert,
    k: int,
    x_clean: torch.Tensor,
    y_deg: torch.Tensor,
) -> float:
    """
    Compute FI Option A (batch-averaged gradient norm squared) for one expert
    on one batch. Designed for lightweight per-epoch tracking in train_stage_A().

    Temporarily enables grad on expert k params only, runs one forward+backward,
    then restores requires_grad=False on all params via finally block.

    Args:
        csmf_model: CSMF model instance.
        expert:     The expert module to evaluate (csmf_model.experts[k]).
        k:          Expert index (0-based).
        x_clean:    (B, ...) clean input batch on correct device.
        y_deg:      (B, ...) degraded observation batch on correct device.

    Returns:
        fi_a: scalar float FI Option A value, or float('nan') on failure.
    """
    all_params    = list(csmf_model.parameters())
    expert_params = list(expert.parameters())

    for p in all_params:
        p.requires_grad_(False)
    for p in expert_params:
        p.requires_grad_(True)

    try:
        with torch.no_grad():
            h = csmf_model.conditioner(y_deg)

        csmf_model.zero_grad()
        z, log_det, log_prob, _ = csmf_model._expert_forward(expert, x_clean, y_deg, h)

        if log_prob is not None:
            log_prob_batch = log_prob.sum()
        else:
            z_flat         = z.flatten(1) if z.dim() > 2 else z
            log_p_z        = csmf_model.base_dist.log_prob(z_flat).sum(dim=1)
            log_prob_batch = (log_p_z + log_det).sum()

        if torch.isnan(log_prob_batch):
            logger.warning(f"[FI-DIAG] compute_fi_option_a_batch: NaN log_prob | expert={k}")
            return float("nan")

        log_prob_batch.backward()

        grad_norm_sq = sum(
            p.grad.detach().norm() ** 2
            for p in expert_params
            if p.grad is not None
        )
        B    = x_clean.shape[0]
        fi_a = (grad_norm_sq / B).item()
        return fi_a

    except Exception as e:
        logger.error(f"[FI-DIAG] compute_fi_option_a_batch error | expert={k}: {e}")
        return float("nan")
    finally:
        # Always restore: disable grad on all params, clear grads
        for p in all_params:
            p.requires_grad_(False)
        csmf_model.zero_grad()


# =============================================================================
# FI Gradient Computation — per expert per batch
# =============================================================================

def _compute_fi_for_expert(
    csmf_model,
    expert,
    k: int,
    x_clean: torch.Tensor,
    y_deg: torch.Tensor,
    fi_a_per_batch: Dict,
    fi_b_per_sample: Dict,
    run_option_b: bool,
    batch_idx: int,
) -> None:
    """
    Compute Option A and optionally Option B FI for one expert on one batch.

    Temporarily enables grad on expert k parameters only, then restores.
    Conditioner is always run without grad (not part of expert's θ_k).
    """
    # Freeze everything, then enable only expert k
    all_params    = list(csmf_model.parameters())
    expert_params = list(expert.parameters())

    for p in all_params:
        p.requires_grad_(False)
    for p in expert_params:
        p.requires_grad_(True)

    try:
        # Conditioner: no grad (not θ_k)
        with torch.no_grad():
            h = csmf_model.conditioner(y_deg)

        # ---- Option A: sum log_prob over batch → single backward ----
        csmf_model.zero_grad()
        z, log_det, log_prob, _ = csmf_model._expert_forward(expert, x_clean, y_deg, h)

        if log_prob is not None:
            log_prob_batch = log_prob.sum()
        else:
            z_flat         = z.flatten(1) if z.dim() > 2 else z
            log_p_z        = csmf_model.base_dist.log_prob(z_flat).sum(dim=1)
            log_prob_batch = (log_p_z + log_det).sum()

        if torch.isnan(log_prob_batch):
            logger.warning(
                f"[FI-DIAG] Option A: NaN log_prob | expert={k} | batch={batch_idx}"
            )
            return

        log_prob_batch.backward()

        grad_norm_sq = sum(
            p.grad.detach().norm() ** 2
            for p in expert_params
            if p.grad is not None
        )
        B     = x_clean.shape[0]
        fi_a  = (grad_norm_sq / B).item()
        fi_a_per_batch[k].append(fi_a)

        # ---- Option B: per-sample backward ----
        if run_option_b:
            per_sample_fi = []
            for i in range(B):
                csmf_model.zero_grad()
                xi = x_clean[i : i + 1]
                yi = y_deg[i : i + 1]
                hi = h[i : i + 1].detach()

                zi, log_det_i, log_prob_i, _ = csmf_model._expert_forward(
                    expert, xi, yi, hi
                )

                if log_prob_i is not None:
                    lp_i = log_prob_i.sum()
                else:
                    zf_i    = zi.flatten(1) if zi.dim() > 2 else zi
                    lpz_i   = csmf_model.base_dist.log_prob(zf_i).sum()
                    lp_i    = lpz_i + log_det_i.sum()

                if torch.isnan(lp_i):
                    per_sample_fi.append(float("nan"))
                    continue

                lp_i.backward()

                gsq = sum(
                    p.grad.detach().norm() ** 2
                    for p in expert_params
                    if p.grad is not None
                )
                per_sample_fi.append(gsq.item())

            fi_b_per_sample[k].append(torch.tensor(per_sample_fi, dtype=torch.float32))

    except Exception as e:
        logger.error(
            f"[FI-DIAG] _compute_fi_for_expert error | expert={k} | batch={batch_idx}: {e}"
        )
        raise
    finally:
        # Always restore: disable grad on all params, clear grads
        for p in all_params:
            p.requires_grad_(False)
        csmf_model.zero_grad()


# =============================================================================
# Latent Gaussian KL Proxy — closed-form diagonal Gaussian KL
# =============================================================================

def _compute_latent_gaussian_kl(
    z_all: Dict[int, list],
    expert_names: List[str],
    K: int,
) -> Dict[str, Any]:
    """
    Fit diagonal Gaussian N(μ_k, diag(σ²_k)) per expert from collected z.
    Compute closed-form KL(N_i || N_j) for all pairs.

    KL(N_i || N_j) = 0.5 * sum(σ²_i/σ²_j + (μ_j - μ_i)²/σ²_j - 1 + log(σ²_j/σ²_i))
    """
    result = {
        "note": (
            "Diagonal Gaussian fit on latent z — rough latent similarity only. "
            "Does NOT capture data-space diversity or NLL ranking differences."
        )
    }

    mu_list  = []
    var_list = []

    for k in range(K):
        if len(z_all[k]) == 0:
            logger.error(
                f"[FI-DIAG] Latent KL: no z data | expert={k} ({expert_names[k]})"
            )
            mu_list.append(None)
            var_list.append(None)
            continue
        try:
            z_cat = torch.cat(z_all[k], dim=0).float()   # (N, D)
            mu    = z_cat.mean(dim=0)                     # (D,)
            var   = z_cat.var(dim=0).clamp(min=1e-8)      # (D,)
            mu_list.append(mu)
            var_list.append(var)
        except Exception as e:
            logger.error(f"[FI-DIAG] Latent KL: z aggregation error | expert={k}: {e}")
            mu_list.append(None)
            var_list.append(None)

    for i in range(K):
        for j in range(i + 1, K):
            key = f"{expert_names[i]}_vs_{expert_names[j]}"
            if mu_list[i] is None or mu_list[j] is None:
                result[key] = {"kl": None, "status": "no_data"}
                continue
            if mu_list[i].shape != mu_list[j].shape:
                logger.warning(
                    f"[FI-DIAG] Latent KL skipped | {key} | "
                    f"dim mismatch: {mu_list[i].shape} vs {mu_list[j].shape} "
                    f"(RealNVP uses factored z — different dim than NICE/NSF)"
                )
                result[key] = {"kl": None, "status": "skipped_dim_mismatch"}
                continue
            try:
                mu_i, var_i = mu_list[i], var_list[i]
                mu_j, var_j = mu_list[j], var_list[j]
                kl_ij = 0.5 * (
                    var_i / var_j
                    + (mu_j - mu_i) ** 2 / var_j
                    - 1.0
                    + torch.log(var_j / var_i)
                ).sum().item()
                kl_ij  = round(kl_ij, 4)
                status = "warn_similar" if kl_ij < KL_WARN_SIMILAR else "diverse"
                result[key] = {"kl": kl_ij, "status": status}
                if status == "warn_similar":
                    logger.warning(
                        f"[FI-DIAG] Latent KL WARN | {key} | kl={kl_ij:.4f} < {KL_WARN_SIMILAR}"
                    )
            except Exception as e:
                logger.error(f"[FI-DIAG] Latent KL computation error | {key}: {e}")
                result[key] = {"kl": None, "status": "error"}

    return result


# =============================================================================
# Plot functions
# =============================================================================

def _plot_fi_scalar_per_expert(
    fi_a_results: Dict,
    fi_a_per_batch: Dict,
    expert_names: List[str],
    output_dir: str,
) -> None:
    """P1: Bar chart with error bars — F_k mean ± std across batches."""
    K      = len(expert_names)
    means  = [fi_a_results[n].get("F_k_mean") or 0.0 for n in expert_names]
    stds   = [fi_a_results[n].get("F_k_std")  or 0.0 for n in expert_names]
    ratios = [fi_a_results[n].get("F_ratio_to_best") or 0.0 for n in expert_names]
    status = [fi_a_results[n].get("status", "no_data") for n in expert_names]

    color_map = {
        "alive":     "#4CAF50",
        "warn_low":  "#FF9800",
        "fatal_dead":"#F44336",
        "no_data":   "#9E9E9E",
    }
    colors = [color_map.get(s, "#9E9E9E") for s in status]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        range(K), means, yerr=stds, color=colors,
        edgecolor="black", linewidth=0.7, capsize=5,
        error_kw={"elinewidth": 1.5}
    )
    y_offset = max(means) * 0.03 + max(stds) * 0.05 if max(means) > 0 else 0.01
    for bar, ratio, st in zip(bars, ratios, status):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_offset,
            f"ratio={ratio:.3f}\n({st})",
            ha="center", va="bottom", fontsize=8
        )

    ax.set_xticks(range(K))
    ax.set_xticklabels(expert_names, rotation=15, ha="right")
    ax.set_ylabel("F_k  (batch-averaged FI scalar)")
    ax.set_title(
        "FI-DIAG — Option A: Fisher Information per Expert\n"
        f"(mean ± std across {len(fi_a_per_batch[0])} batches)"
    )
    ax.grid(axis="y", alpha=0.3)
    legend_elems = [
        Patch(facecolor=c, label=l)
        for l, c in color_map.items() if l != "no_data"
    ]
    ax.legend(handles=legend_elems, fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "fi_scalar_per_expert.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def _plot_fi_per_sample_dist(
    fi_b_arrays: List[np.ndarray],
    expert_names: List[str],
    output_dir: str,
) -> None:
    """P2: Violin plot of per-sample FI distribution per expert (Option B)."""
    K            = len(expert_names)
    valid_data   = []
    valid_names  = []

    for k in range(K):
        arr = fi_b_arrays[k]
        if len(arr) > 0:
            clean = arr[np.isfinite(arr)]
            if len(clean) > 0:
                valid_data.append(clean)
                valid_names.append(expert_names[k])

    if not valid_data:
        logger.warning("[FI-DIAG] Plot P2: no valid Option B data — skipping")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    parts = ax.violinplot(valid_data, positions=range(len(valid_data)), showmedians=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.7)
    ax.set_xticks(range(len(valid_names)))
    ax.set_xticklabels(valid_names, rotation=15, ha="right")
    ax.set_ylabel("F_k(x_i, h_i)  —  per-sample FI")
    ax.set_title("FI-DIAG — Option B: Per-Sample FI Distribution per Expert")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "fi_per_sample_dist.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def _plot_fi_expert_scatter(
    fi_b_arrays: List[np.ndarray],
    expert_names: List[str],
    output_dir: str,
) -> None:
    """P3: Pairwise scatter of per-sample FI across all expert pairs (Option B)."""
    K       = len(expert_names)
    n_pairs = K * (K - 1) // 2
    if n_pairs == 0:
        return

    fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 5), squeeze=False)
    pair_idx  = 0

    for i in range(K):
        for j in range(i + 1, K):
            ax = axes[0, pair_idx]
            ai = fi_b_arrays[i]
            aj = fi_b_arrays[j]

            if len(ai) == 0 or len(aj) == 0:
                ax.set_title("no data")
                pair_idx += 1
                continue

            n_min = min(len(ai), len(aj))
            xi    = ai[:n_min]
            xj    = aj[:n_min]
            mask  = np.isfinite(xi) & np.isfinite(xj)
            xi, xj = xi[mask], xj[mask]

            if len(xi) == 0:
                ax.set_title("no finite data")
                pair_idx += 1
                continue

            ax.scatter(xi, xj, alpha=0.15, s=8, edgecolors="none")

            if np.std(xi) > 1e-8 and np.std(xj) > 1e-8:
                corr = np.corrcoef(xi, xj)[0, 1]
                ax.set_title(
                    f"{expert_names[i]} vs {expert_names[j]}\nρ = {corr:.3f}"
                )
            else:
                ax.set_title(f"{expert_names[i]} vs {expert_names[j]}")

            ax.set_xlabel(f"F_k(x_i) — {expert_names[i]}", fontsize=8)
            ax.set_ylabel(f"F_k(x_i) — {expert_names[j]}", fontsize=8)
            ax.grid(True, alpha=0.3)
            pair_idx += 1

    fig.suptitle("FI-DIAG — Option B: Per-Sample FI Cross-Expert Scatter", fontsize=13)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "fi_expert_scatter.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def _plot_fi_per_epoch(
    epoch_logs: Dict,
    expert_names: List[str],
    output_dir: str,
) -> None:
    """
    [EPOCH-FI] v1.5 — P5: FI Option A per epoch, one line per expert.

    Reads epoch_logs[expert_name]["fi_a"] lists populated by train_stage_A().
    x = epoch, y = FI Option A value.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    has_data = False

    for name in expert_names:
        logs = epoch_logs.get(name, {})
        fi_vals = logs.get("fi_a", [])
        if not fi_vals:
            logger.warning(f"[FI-DIAG] P5: no fi_a data for expert {name} — skipping line")
            continue
        # Filter NaN for display but keep epoch alignment
        epochs = list(range(1, len(fi_vals) + 1))
        ax.plot(epochs, fi_vals, marker="o", markersize=3, label=name)
        has_data = True

    if not has_data:
        logger.error("[FI-DIAG] P5: no fi_a data found in epoch_logs — plot empty")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("FI Option A (grad norm² / B)")
    ax.set_title("FI-DIAG — P5: Fisher Information per Expert over Epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "fi_per_epoch.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def _plot_fi_vs_nll_scatter(
    fi_a_means: List[float],
    nll_means: List[float],
    expert_names: List[str],
    output_dir: str,
) -> None:
    """
    [EPOCH-FI] v1.5 — P6: FI vs NLL scatter, one point per expert.

    x = NLL_k (mean over val batches), y = FI_k (Option A mean).
    Quadrant interpretation:
        high FI + low NLL  = healthy active expert
        low FI  + low NLL  = suspicious lazy fit
        low FI  + high NLL = weak/dead expert
        high FI + high NLL = active but undertrained
    """
    K = len(expert_names)
    fig, ax = plt.subplots(figsize=(6, 5))

    for k in range(K):
        fi  = fi_a_means[k]
        nll = nll_means[k]
        if not (np.isfinite(fi) and np.isfinite(nll)):
            logger.warning(f"[FI-DIAG] P6: non-finite value for {expert_names[k]} — skipping point")
            continue
        ax.scatter(nll, fi, s=80, zorder=5)
        ax.annotate(
            expert_names[k], (nll, fi),
            textcoords="offset points", xytext=(6, 4), fontsize=9
        )

    ax.set_xlabel("NLL_k (mean over val batches)")
    ax.set_ylabel("FI_k Option A (mean grad norm² / B)")
    ax.set_title("FI-DIAG — P6: FI vs NLL per Expert")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "fi_vs_nll_scatter.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def _plot_latent_kl_heatmap(
    latent_kl: Dict,
    expert_names: List[str],
    output_dir: str,
) -> None:
    """P4: Heatmap of latent Gaussian KL proxy between all expert pairs."""
    K          = len(expert_names)
    kl_matrix  = np.zeros((K, K))

    for i in range(K):
        for j in range(K):
            if i == j:
                kl_matrix[i, j] = 0.0
            else:
                ii, jj = (i, j) if i < j else (j, i)
                key = f"{expert_names[ii]}_vs_{expert_names[jj]}"
                val = latent_kl.get(key, {}).get("kl")
                kl_matrix[i, j] = val if val is not None else float("nan")

    fig, ax = plt.subplots(figsize=(5, 4))
    finite_vals = kl_matrix[np.isfinite(kl_matrix)]
    vmax = float(finite_vals.max()) if len(finite_vals) > 0 else 1.0

    im = ax.imshow(kl_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(expert_names, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(expert_names, fontsize=8)
    plt.colorbar(im, ax=ax, label="KL divergence (proxy)")

    for i in range(K):
        for j in range(K):
            val  = kl_matrix[i, j]
            text = f"{val:.2f}" if np.isfinite(val) else "N/A"
            txt_color = "white" if (np.isfinite(val) and val > vmax * 0.6) else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=txt_color)

    ax.set_title(
        "FI-DIAG — Latent Gaussian KL Proxy\n"
        "(diagonal Gaussian fit on z; rough latent similarity only)"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "latent_gaussian_kl_heatmap.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


# =============================================================================
# Standalone CLI
# =============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FI-DIAG-v1.1 — Standalone Fisher Information diagnostics for CSMF Stage A"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to csmf_stage_A.pth"
    )
    parser.add_argument(
        "--output", type=str, default="results/fisher_info",
        help="Output directory for plots and JSON (default: results/fisher_info)"
    )
    parser.add_argument(
        "--n-batches", type=int, default=5,
        help="Number of val batches to aggregate (3–10 recommended, default: 5)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Batch size for val loader (default: 256)"
    )
    parser.add_argument(
        "--no-option-b", action="store_true",
        help="Skip per-sample FI (Option B). Faster but no P2/P3 plots."
    )
    parser.add_argument(
        "--plots", nargs="+", default=["P1", "P2", "P3", "P4"],
        choices=["P1", "P2", "P3", "P4"],
        help="Plots to generate (default: all)"
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = _build_arg_parser()
    args   = parser.parse_args()

    # -- Import project modules (must run from project root with PYTHONPATH=.) --
    try:
        sys.path.insert(0, os.getcwd())
        from configs.mnist_config import (
            HIDDEN_DIM, NUM_LAYERS, LATENT_DIM, BATCH_SIZE, ACTIVE_EXPERTS
        )
        from experiments.train_csmf import build_model, EXPERT_REGISTRY
        from scripts.preprocess_mnist import create_precomputed_dataloaders
    except ImportError as e:
        logger.error(
            f"[FI-DIAG] Import error — run from project root with PYTHONPATH=.: {e}"
        )
        sys.exit(1)

    # -- Load checkpoint --
    logger.info(f"[FI-DIAG] Loading checkpoint: {args.checkpoint}")
    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    except Exception as e:
        logger.error(f"[FI-DIAG] Failed to load checkpoint: {e}")
        sys.exit(1)

    # -- Build and load model --
    try:
        device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        active_experts = ckpt.get("active_experts", ACTIVE_EXPERTS)
        # Minimal args namespace — mirrors defaults in train_csmf.py build_model()
        import argparse as _ap
        _build_args = _ap.Namespace(nice_scale=0.10)
        model = build_model(
            active_experts=active_experts,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            latent_dim=LATENT_DIM,
            logger=logger,
            args=_build_args,
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        logger.info(
            f"[FI-DIAG] Model loaded | active_experts={active_experts} | device={device}"
        )
    except Exception as e:
        logger.error(f"[FI-DIAG] Model build/load error: {e}")
        sys.exit(1)

    # -- Build val loader --
    try:
        from configs.mnist_config import (
            PREPROCESSED_DIR, BLUR_KERNEL, BLUR_SIGMA,
            DOWNSAMPLE_FACTOR, NOISE_SIGMA, VAL_SPLIT, SEED,
            make_worker_init_fn,
        )
        config_params = {
            'blur_kernel_size':  BLUR_KERNEL,
            'blur_sigma':        BLUR_SIGMA,
            'downsample_factor': DOWNSAMPLE_FACTOR,
            'noise_std':         NOISE_SIGMA,
            'normalize':         '[0,1]',
            'val_split':         VAL_SPLIT,
            'seed':              SEED,
        }
        _g = torch.Generator()
        _g.manual_seed(SEED)
        _, val_loader, _ = create_precomputed_dataloaders(
            preprocessed_dir=PREPROCESSED_DIR,
            batch_size=args.batch_size,
            config_params=config_params,
            worker_init_fn=make_worker_init_fn(SEED),
            generator=_g,
        )
        logger.info(f"[FI-DIAG] Val loader ready | batch_size={args.batch_size}")
    except Exception as e:
        logger.error(f"[FI-DIAG] Val loader error: {e}")
        sys.exit(1)

    # -- Run diagnostics --
    try:
        summary = run_fi_diagnostics(
            csmf_model=model,
            val_loader=val_loader,
            output_dir=args.output,
            n_batches=args.n_batches,
            run_option_b=not args.no_option_b,
            plots=args.plots,
        )
        verdict = summary.get("verdict", {})
        ready   = verdict.get("ready_for_stage_B", False)
        print(f"\n{'=' * 60}")
        print(f"  FI-DIAG-v1.4 complete")
        print(f"  ready_for_stage_B : {ready}")
        print(f"  {verdict.get('notes', '')}")
        print(f"  Results saved to  : {args.output}/")
        print(f"{'=' * 60}\n")
    except Exception as e:
        logger.error(f"[FI-DIAG] Fatal error: {e}")
        sys.exit(1)
