# =============================================================================
# Version: DIAG-REORG-MetricUtils-v1.1 | Abbr: MU
# Description: Shared metric collectors for CSMF diagnostic scripts.
#              Extracted from EXP-SANITY v1.1, FI-DIAG v1.5, SC-DIAG v1.1.
#              Called by SA-DIAG, SB-DIAG, SC-DIAG. Each function is
#              independent — no shared state, no side effects. All functions
#              return None on failure (non-fatal); callers decide whether to
#              skip or abort. All errors logged via module logger.
# Changelog:
#   v1.1 (2026-04-04): BUG FIX — removed fisher_info_diag import attempt and
#                      fallback from compute_fi_option_a_batch; inline
#                      implementation is now the sole implementation; fisher_info_diag
#                      will be deleted — no imports from it anywhere in MU
#   v1.0 (2026-04-04): Initial implementation — 6 metric collectors extracted
#                      from existing diagnostic scripts; compute_fi_option_a_batch
#                      re-exported from FI-DIAG; all functions decorated with
#                      @torch.no_grad() except where gradients are required
#                      (collect_fi_option_a); docstrings document source file
# Dependencies: CSMF-MAIN v1.3.24+, torch, numpy
# =============================================================================

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# =============================================================================
# FI Option A — lightweight per-epoch tracker
# Logic ported from FI-DIAG v1.5 compute_fi_option_a_batch().
# No dependency on fisher_info_diag — that file will be deleted.
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
    on one batch.

    Temporarily enables grad on expert k params only, runs one forward+backward,
    then restores requires_grad=False on all params via finally block.

    Args:
        csmf_model : CSMF model instance.
        expert     : The expert module (csmf_model.experts[k]).
        k          : Expert index (0-based), used for logging only.
        x_clean    : (B, ...) clean input batch on correct device.
        y_deg      : (B, ...) degraded observation batch on correct device.

    Returns:
        fi_a: scalar float, or float('nan') on failure.
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
            logger.warning(
                f"MU | compute_fi_option_a_batch: NaN log_prob | expert={k}"
            )
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
        logger.error(f"MU | compute_fi_option_a_batch error | expert={k}: {e}")
        return float("nan")
    finally:
        for p in all_params:
            p.requires_grad_(False)
        csmf_model.zero_grad()


# =============================================================================
# collect_per_expert_nll
# Source: EXP-SANITY v1.1 val pass (per_expert_data) + SC-DIAG v1.1 _collect_metrics()
# =============================================================================

@torch.no_grad()
def collect_per_expert_nll(
    csmf_model,
    val_loader,
    device: torch.device,
    max_batches: int = 20,
) -> Optional[Dict[str, Any]]:
    """
    Collect per-expert NLL and mixture NLL from the validation set.

    For each val batch, runs csmf_model.forward() to get mixture log_q and
    per-expert log_q_experts. Aggregates per-sample NLL tensors and computes
    mean NLL per expert.

    Args:
        csmf_model  : CSMF model (eval mode set internally).
        val_loader  : DataLoader yielding (x_clean, y_deg).
        device      : Compute device.
        max_batches : Max val batches to collect (speed control).

    Returns:
        Dict with keys:
            "per_expert_nll_mean"    : {expert_name: float}
            "per_expert_nll_samples" : {k (int): Tensor(N,)}
            "mixture_nll_mean"       : float
        or None if no batches collected.

    Source: EXP-SANITY v1.1 (nll_per_sample) + SC-DIAG v1.1 _collect_metrics()
    """
    csmf_model.eval()
    K = csmf_model.K
    expert_names = [type(e).__name__ for e in csmf_model.experts]

    per_expert_nll_samples: Dict[int, list] = {k: [] for k in range(K)}
    all_mixture_nll: list = []
    n_collected = 0

    for x_clean, y_deg in val_loader:
        if n_collected >= max_batches:
            break
        x_clean = x_clean.to(device)
        y_deg   = y_deg.to(device)

        try:
            log_q, log_q_experts = csmf_model.forward(x_clean, y_deg)

            if torch.isnan(log_q).any():
                logger.warning(
                    f"MU | collect_per_expert_nll: NaN in log_q — "
                    f"skipping batch={n_collected}"
                )
                continue

            all_mixture_nll.append(-log_q.cpu())

            for k in range(K):
                per_expert_nll_samples[k].append(-log_q_experts[:, k].cpu())

            n_collected += 1

        except Exception as e:
            logger.error(
                f"MU | collect_per_expert_nll: error batch={n_collected}: {e}"
            )
            continue

    if n_collected == 0:
        logger.error("MU | collect_per_expert_nll: no batches collected — returning None")
        return None

    # Aggregate
    mixture_nll_mean = torch.cat(all_mixture_nll).mean().item()

    per_expert_nll_mean: Dict[str, float] = {}
    per_expert_nll_flat: Dict[int, torch.Tensor] = {}

    for k in range(K):
        samples = per_expert_nll_samples[k]
        if not samples:
            logger.error(
                f"MU | collect_per_expert_nll: no samples for expert {k} "
                f"({expert_names[k]})"
            )
            per_expert_nll_mean[expert_names[k]] = float("nan")
            per_expert_nll_flat[k] = torch.tensor([])
            continue
        flat = torch.cat(samples)
        per_expert_nll_mean[expert_names[k]] = flat.mean().item()
        per_expert_nll_flat[k] = flat

    logger.info(
        f"MU | collect_per_expert_nll: collected {n_collected} batches | "
        f"mixture_nll={mixture_nll_mean:.4f} | "
        f"per_expert={{{', '.join(f'{n}:{v:.4f}' for n, v in per_expert_nll_mean.items())}}}"
    )

    return {
        "per_expert_nll_mean":    per_expert_nll_mean,
        "per_expert_nll_samples": per_expert_nll_flat,
        "mixture_nll_mean":       mixture_nll_mean,
    }


# =============================================================================
# collect_latent_stats
# Source: EXP-SANITY v1.1 val pass (z_all) + Check 3 (z_mean, z_std)
# =============================================================================

@torch.no_grad()
def collect_latent_stats(
    csmf_model,
    val_loader,
    device: torch.device,
    max_batches: int = 20,
) -> Optional[Dict[int, Dict[str, Any]]]:
    """
    Collect latent z statistics per expert from the validation set.

    Runs _expert_forward() per expert per batch to collect z. Computes
    mean/std of the flattened latent z for each expert.

    Args:
        csmf_model  : CSMF model (eval mode set internally).
        val_loader  : DataLoader yielding (x_clean, y_deg).
        device      : Compute device.
        max_batches : Max val batches.

    Returns:
        Dict keyed by expert index k:
            {k: {"z_all": Tensor(N, D), "z_mean": float, "z_std": float}}
        or None if no batches collected.

    Source: EXP-SANITY v1.1 val pass (z_all collection) + Check 3 stats.
    """
    csmf_model.eval()
    K = csmf_model.K
    expert_names = [type(e).__name__ for e in csmf_model.experts]

    z_accum: Dict[int, list] = {k: [] for k in range(K)}
    n_collected = 0

    for x_clean, y_deg in val_loader:
        if n_collected >= max_batches:
            break
        x_clean = x_clean.to(device)
        y_deg   = y_deg.to(device)

        try:
            h = csmf_model.conditioner(y_deg)
        except Exception as e:
            logger.error(
                f"MU | collect_latent_stats: conditioner error batch={n_collected}: {e}"
            )
            n_collected += 1
            continue

        for k, expert in enumerate(csmf_model.experts):
            try:
                z, log_det, log_prob, _ = csmf_model._expert_forward(
                    expert, x_clean, y_deg, h
                )
                if torch.isnan(log_det).any():
                    logger.warning(
                        f"MU | collect_latent_stats: NaN log_det | "
                        f"expert={k} ({expert_names[k]}) batch={n_collected}"
                    )
                    continue
                z_flat = z.flatten(1) if z.dim() > 2 else z
                z_accum[k].append(z_flat.cpu())
            except Exception as e:
                logger.error(
                    f"MU | collect_latent_stats: expert={k} ({expert_names[k]}) "
                    f"batch={n_collected}: {e}"
                )

        n_collected += 1

    if n_collected == 0:
        logger.error("MU | collect_latent_stats: no batches collected — returning None")
        return None

    result: Dict[int, Dict[str, Any]] = {}
    for k in range(K):
        name = expert_names[k]
        if not z_accum[k]:
            logger.error(
                f"MU | collect_latent_stats: no z data for expert={k} ({name})"
            )
            result[k] = {"z_all": torch.tensor([]), "z_mean": float("nan"),
                         "z_std": float("nan")}
            continue
        z_cat  = torch.cat(z_accum[k], dim=0)
        z_mean = z_cat.mean().item()
        z_std  = z_cat.std().item()

        if abs(z_mean) > 1.0:
            logger.warning(
                f"MU | collect_latent_stats: |z_mean|={abs(z_mean):.4f} > 1.0 "
                f"for expert={k} ({name}) — latent shift"
            )
        if abs(z_std - 1.0) > 1.0:
            logger.warning(
                f"MU | collect_latent_stats: |z_std - 1|={abs(z_std - 1.0):.4f} > 1.0 "
                f"for expert={k} ({name}) — latent scale mismatch"
            )

        logger.info(
            f"MU | collect_latent_stats: expert={k} ({name}) | "
            f"z_mean={z_mean:.4f} z_std={z_std:.4f} shape={tuple(z_cat.shape)}"
        )
        result[k] = {"z_all": z_cat, "z_mean": z_mean, "z_std": z_std}

    return result


# =============================================================================
# collect_invertibility
# Source: EXP-SANITY v1.1 val pass (log_det_all) + Check 1 (log_det_std)
# =============================================================================

@torch.no_grad()
def collect_invertibility(
    csmf_model,
    val_loader,
    device: torch.device,
    max_batches: int = 20,
) -> Optional[Dict[int, Dict[str, Any]]]:
    """
    Collect log-det statistics per expert from the validation set.

    Collects log_det from _expert_forward() across batches and computes
    std(log_det) as a proxy for log-det collapse (NICE always = 0 by design).

    Args:
        csmf_model  : CSMF model (eval mode set internally).
        val_loader  : DataLoader yielding (x_clean, y_deg).
        device      : Compute device.
        max_batches : Max val batches.

    Returns:
        Dict keyed by expert index k:
            {k: {
                "log_det_all"     : Tensor(N,),
                "log_det_std"     : float,
                "log_det_collapse": bool,   # True if std < 0.01 (and not NICE)
            }}
        or None if no batches collected.

    Source: EXP-SANITY v1.1 val pass (log_det_all) + Check 1.
    Note: NICE log_det_std == 0.0 is architectural (additive coupling),
          not collapse. Collapse flag is suppressed when std == 0.0 and
          the expert type name contains "NICE".
    """
    csmf_model.eval()
    K = csmf_model.K
    expert_names = [type(e).__name__ for e in csmf_model.experts]

    log_det_accum: Dict[int, list] = {k: [] for k in range(K)}
    n_collected = 0

    for x_clean, y_deg in val_loader:
        if n_collected >= max_batches:
            break
        x_clean = x_clean.to(device)
        y_deg   = y_deg.to(device)

        try:
            h = csmf_model.conditioner(y_deg)
        except Exception as e:
            logger.error(
                f"MU | collect_invertibility: conditioner error batch={n_collected}: {e}"
            )
            n_collected += 1
            continue

        for k, expert in enumerate(csmf_model.experts):
            try:
                _, log_det, _, _ = csmf_model._expert_forward(
                    expert, x_clean, y_deg, h
                )
                if torch.isnan(log_det).any():
                    logger.warning(
                        f"MU | collect_invertibility: NaN log_det | "
                        f"expert={k} ({expert_names[k]}) batch={n_collected}"
                    )
                    continue
                log_det_accum[k].append(log_det.cpu())
            except Exception as e:
                logger.error(
                    f"MU | collect_invertibility: expert={k} ({expert_names[k]}) "
                    f"batch={n_collected}: {e}"
                )

        n_collected += 1

    if n_collected == 0:
        logger.error("MU | collect_invertibility: no batches collected — returning None")
        return None

    result: Dict[int, Dict[str, Any]] = {}
    for k in range(K):
        name = expert_names[k]
        if not log_det_accum[k]:
            logger.error(
                f"MU | collect_invertibility: no log_det data for expert={k} ({name})"
            )
            result[k] = {
                "log_det_all": torch.tensor([]),
                "log_det_std": float("nan"),
                "log_det_collapse": "no_data",
            }
            continue

        log_det_cat = torch.cat(log_det_accum[k])
        ld_std      = log_det_cat.std().item()

        # NICE is volume-preserving by design — log_det == 0 always, not collapse
        is_nice     = "NICE" in name or "nice" in name
        if ld_std < 0.01 and not is_nice:
            collapse = True
            logger.warning(
                f"MU | collect_invertibility: log_det std={ld_std:.6f} < 0.01 | "
                f"expert={k} ({name}) — possible mode collapse"
            )
        elif ld_std < 0.01 and is_nice:
            collapse = False
            logger.info(
                f"MU | collect_invertibility: log_det std=0 for NICE expert={k} ({name}) "
                f"— architectural (additive coupling is volume-preserving)"
            )
        else:
            collapse = False
            logger.info(
                f"MU | collect_invertibility: expert={k} ({name}) | "
                f"log_det_std={ld_std:.6f}"
            )

        result[k] = {
            "log_det_all":      log_det_cat,
            "log_det_std":      round(ld_std, 6),
            "log_det_collapse": collapse,
        }

    return result


# =============================================================================
# collect_reconstruction_batch
# Source: EXP-SANITY v1.1 _plot_reconstruction_grid() (encode→decode fix v1.1)
#         SC-DIAG v1.1 (first-batch x_hat storage)
# =============================================================================

@torch.no_grad()
def collect_reconstruction_batch(
    csmf_model,
    val_loader,
    device: torch.device,
    n_samples: int = 8,
) -> Optional[Dict[str, Any]]:
    """
    Collect one batch of encode→decode reconstructions per expert.

    Uses z = f(x, h), x̂ = f⁻¹(z, h) — NOT generation from noise.
    This separates invertibility bugs from generation quality (EXP-SANITY v1.1 fix).

    Args:
        csmf_model : CSMF model (eval mode set internally).
        val_loader : DataLoader yielding (x_clean, y_deg).
        device     : Compute device.
        n_samples  : Number of samples to collect (first n from first batch).

    Returns:
        Dict with keys:
            "y"       : Tensor(n, C, H, W)   — degraded input
            "x_clean" : Tensor(n, C, H, W)   — clean ground truth
            "x_hat"   : {k (int): Tensor(n, C, H, W)}  — per-expert reconstruction
                        None entry for any expert that fails
        or None if the first batch cannot be collected.

    Source: EXP-SANITY v1.1 _plot_reconstruction_grid() — encode→decode path preserved.
    """
    csmf_model.eval()
    K = csmf_model.K
    expert_names = [type(e).__name__ for e in csmf_model.experts]

    try:
        x_clean_batch, y_deg_batch = next(iter(val_loader))
    except StopIteration:
        logger.error("MU | collect_reconstruction_batch: val_loader is empty")
        return None
    except Exception as e:
        logger.error(
            f"MU | collect_reconstruction_batch: failed to get first batch: {e}"
        )
        return None

    n = min(n_samples, x_clean_batch.shape[0])
    x_clean = x_clean_batch[:n].to(device)
    y_deg   = y_deg_batch[:n].to(device)

    try:
        h = csmf_model.conditioner(y_deg)
    except Exception as e:
        logger.error(
            f"MU | collect_reconstruction_batch: conditioner failed: {e}"
        )
        return None

    x_hat_per_expert: Dict[int, Optional[torch.Tensor]] = {}

    for k, expert in enumerate(csmf_model.experts):
        reconstructions = []
        all_ok = True

        for i in range(n):
            try:
                z, log_det, log_prob, z_flist = csmf_model._expert_forward(
                    expert,
                    x_clean[i:i+1],
                    y_deg[i:i+1],
                    h[i:i+1],
                )
                x_hat_i = csmf_model._expert_inverse(
                    expert, z, y_deg[i:i+1], h[i:i+1], z_factored_list=z_flist
                )
                reconstructions.append(x_hat_i.cpu())
            except Exception as e:
                logger.error(
                    f"MU | collect_reconstruction_batch: encode→decode failed | "
                    f"expert={k} ({expert_names[k]}) sample={i}: {e}"
                )
                reconstructions.append(None)
                all_ok = False

        if all(r is not None for r in reconstructions):
            x_hat_per_expert[k] = torch.cat(reconstructions, dim=0)
        else:
            # Partial: fill failed entries with zeros so the grid can still render
            filled = []
            ref_shape = next((r.shape for r in reconstructions if r is not None), None)
            for r in reconstructions:
                if r is not None:
                    filled.append(r)
                elif ref_shape is not None:
                    filled.append(torch.zeros(ref_shape))
                    logger.warning(
                        f"MU | collect_reconstruction_batch: zero-filled missing "
                        f"sample for expert={k} ({expert_names[k]})"
                    )
            if filled:
                x_hat_per_expert[k] = torch.cat(filled, dim=0)
            else:
                logger.error(
                    f"MU | collect_reconstruction_batch: no reconstructions for "
                    f"expert={k} ({expert_names[k]}) — setting to None"
                )
                x_hat_per_expert[k] = None

        if all_ok:
            logger.info(
                f"MU | collect_reconstruction_batch: expert={k} ({expert_names[k]}) "
                f"| {n} samples OK"
            )

    return {
        "y":       y_deg.cpu(),
        "x_clean": x_clean.cpu(),
        "x_hat":   x_hat_per_expert,
    }


# =============================================================================
# collect_gate_metrics
# Source: SC-DIAG v1.1 _collect_metrics() (gate weights + Neff + winner argmax)
# =============================================================================

@torch.no_grad()
def collect_gate_metrics(
    csmf_model,
    val_loader,
    device: torch.device,
    max_batches: int = 20,
) -> Optional[Dict[str, Any]]:
    """
    Collect gate weight statistics from the validation set.

    Runs conditioner → gate → softmax to get per-sample weights w.
    Computes mean Neff, mean gate weights per expert, and gate winner counts
    (argmax per sample).

    Args:
        csmf_model  : CSMF model (eval mode set internally).
        val_loader  : DataLoader yielding (x_clean, y_deg).
        device      : Compute device.
        max_batches : Max val batches.

    Returns:
        Dict with keys:
            "neff_mean"          : float
            "gate_weights_mean"  : {expert_name: float}
            "gate_winner_counts" : {expert_name: int}
        or None if no batches collected.

    Source: SC-DIAG v1.1 _collect_metrics() gate weight section.
    Note: Uses temperature=1.0 (plain softmax on logits) — not tau-annealed.
          Appropriate for post-training evaluation.
    """
    csmf_model.eval()
    K = csmf_model.K
    expert_names = [type(e).__name__ for e in csmf_model.experts]

    all_neff:          list = []
    all_gate_weights:  list = []
    all_winner_idxs:   list = []
    n_collected = 0

    for x_clean, y_deg in val_loader:
        if n_collected >= max_batches:
            break
        y_deg = y_deg.to(device)

        try:
            h      = csmf_model.conditioner(y_deg)
            logits = csmf_model.gate(h)
            w      = torch.softmax(logits, dim=1)  # (B, K) — temperature=1.0

            neff   = csmf_model._compute_neff(w)   # (B,)
            all_neff.append(neff.cpu())
            all_gate_weights.append(w.mean(dim=0).cpu())
            all_winner_idxs.append(w.argmax(dim=1).cpu())

            n_collected += 1

        except Exception as e:
            logger.error(
                f"MU | collect_gate_metrics: error batch={n_collected}: {e}"
            )
            continue

    if n_collected == 0:
        logger.error("MU | collect_gate_metrics: no batches collected — returning None")
        return None

    neff_all   = torch.cat(all_neff)
    gate_w     = torch.stack(all_gate_weights).mean(dim=0)
    all_winners = torch.cat(all_winner_idxs)

    neff_mean         = neff_all.mean().item()
    gate_weights_mean = {expert_names[k]: gate_w[k].item() for k in range(K)}
    gate_winner_counts = {
        expert_names[k]: (all_winners == k).sum().item() for k in range(K)
    }

    logger.info(
        f"MU | collect_gate_metrics: {n_collected} batches | "
        f"neff_mean={neff_mean:.4f} | "
        f"weights={{{', '.join(f'{n}:{v:.4f}' for n, v in gate_weights_mean.items())}}}"
    )

    return {
        "neff_mean":          neff_mean,
        "gate_weights_mean":  gate_weights_mean,
        "gate_winner_counts": gate_winner_counts,
    }


# =============================================================================
# collect_reconstruction_metrics
# Source: SC-DIAG v1.1 _collect_metrics() (physics residual section)
# =============================================================================

@torch.no_grad()
def collect_reconstruction_metrics(
    csmf_model,
    val_loader,
    fwd_model,
    device: torch.device,
    max_batches: int = 20,
) -> Optional[Dict[str, Any]]:
    """
    Collect physics consistency residual ‖Ax̂ - y‖² from the validation set.

    Draws one sample per val batch via csmf_model.sample(), applies the
    forward model fwd_model, and computes the per-sample MSE residual.

    Args:
        csmf_model  : CSMF model (eval mode set internally).
        val_loader  : DataLoader yielding (x_clean, y_deg).
        fwd_model   : Forward model A with .forward(x) → Ax (e.g. SRForwardModel).
        device      : Compute device.
        max_batches : Max val batches.

    Returns:
        Dict with keys:
            "residual_mean" : float
            "residual_std"  : float
            "residuals_all" : Tensor(N,)   — per-sample residuals
        or None if no batches collected.

    Source: SC-DIAG v1.1 _collect_metrics() physics residual section.
    """
    csmf_model.eval()

    all_residuals: list = []
    n_collected = 0

    for x_clean, y_deg in val_loader:
        if n_collected >= max_batches:
            break
        y_deg = y_deg.to(device)

        try:
            x_samples, _ = csmf_model.sample(y_deg, num_samples=1)
            x_hat        = x_samples[:, 0, :]
            x_hat_4d     = x_hat.view(x_hat.shape[0], 1, 28, 28)

            Ax       = fwd_model.forward(x_hat_4d)
            residual = ((Ax - y_deg) ** 2).mean(dim=[1, 2, 3])  # (B,)

            if torch.isnan(residual).any():
                logger.warning(
                    f"MU | collect_reconstruction_metrics: NaN residual "
                    f"batch={n_collected} — skipping"
                )
                continue

            all_residuals.append(residual.cpu())
            n_collected += 1

        except Exception as e:
            logger.error(
                f"MU | collect_reconstruction_metrics: error batch={n_collected}: {e}"
            )
            continue

    if n_collected == 0:
        logger.error(
            "MU | collect_reconstruction_metrics: no batches collected — returning None"
        )
        return None

    residuals_all = torch.cat(all_residuals)
    residual_mean = residuals_all.mean().item()
    residual_std  = residuals_all.std().item()

    logger.info(
        f"MU | collect_reconstruction_metrics: {n_collected} batches | "
        f"residual_mean={residual_mean:.6f} ± {residual_std:.6f}"
    )

    return {
        "residual_mean": residual_mean,
        "residual_std":  residual_std,
        "residuals_all": residuals_all,
    }
