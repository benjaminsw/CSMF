# =============================================================================
# Version: DIAG-REORG-MetricUtils-v1.6 | Abbr: MU
# Description: Shared metric collectors for CSMF diagnostic scripts.
#              Extracted from EXP-SANITY v1.1, FI-DIAG v1.5, SC-DIAG v1.1.
#              Called by SA-DIAG, SB-DIAG, SC-DIAG. Each function is
#              independent — no shared state, no side effects. All functions
#              return None on failure (non-fatal); callers decide whether to
#              skip or abort. All errors logged via module logger.
# Changelog:
#   v1.6 (2026-04-19): [MIX-RECON] Add collect_mixture_recon_batch() — Stage C
#                      mixture 4-col data: cycle=argmax-expert x_hat from
#                      sample_all_experts(); generated=csmf.sample(y,1); reshapes
#                      flat (B,784)→(B,1,28,28); NaN guard on each step; returns
#                      None on failure with error log.
#                      [FI-GATE] Add collect_fi_gate_comparison() — loads
#                      fi_diag_summary.json from known output path; extracts fi_mean
#                      per expert; collects current gate weights from val_loader;
#                      returns None with warning log if JSON missing (non-fatal).
#   v1.5 (2026-04-18): [P4-4COL] Extend collect_reconstruction_batch to also
#                      collect x_gen per expert: z~N(0,I) → _expert_inverse,
#                      unconditional per-expert generation (gate-independent).
#                      Reshape applied to x_gen same as x_hat. Return dict now
#                      includes "x_gen": {k: Tensor(n,1,28,28)|None}.
#   v1.4 (2026-04-18): [RECON-RESHAPE] Reshape (B,784)→(B,1,28,28) in
#                      collect_reconstruction_batch. [LOGDET-FIX] Remove stale
#                      x_in= kwarg from collect_logdet_decomposition.
#   v1.3 (2026-04-17): [PROX-T] Add collect_prox_diagnostics().
#                      via csmf_model.sample(num_prox_steps=0), then manually
#                      applies gradient prox steps for T in T_values; computes
#                      per-step residual ||Ax^(t)-y||² independently (no PROX
#                      API dependency); returns residuals_by_T, residual_steps,
#                      sample_std_pre/post, nll_baseline; returns None on failure
#   v1.2 (2026-04-11): [LOGDET-DIAG] Add collect_logdet_decomposition() —
#                      iterates val_loader, calls _expert_forward per expert,
#                      returns per-expert {log_det: Tensor(N,), log_p_z: Tensor(N,),
#                      D: int}; used by SA-DIAG v1.5 P12–P14; NaN batches skipped
#                      with error log; returns None if no batches collected
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

import json
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

    # [RECON-RESHAPE] Reshape flat (B,784) to (B,1,28,28) for plot_reconstruction_grid.
    # Non-image experts (NSF, NICE, CSF) return flat tensors from _expert_inverse.
    for k in list(x_hat_per_expert.keys()):
        t = x_hat_per_expert[k]
        if t is not None and t.dim() == 2 and t.shape[-1] == 784:
            x_hat_per_expert[k] = t.view(t.shape[0], 1, 28, 28)
            logger.info(f"MU | collect_reconstruction_batch: expert={k} reshaped (B,784)→(B,1,28,28)")

    # [P4-4COL] Collect generated samples per expert: z~N(0,I) → _expert_inverse.
    # This is unconditional per-expert generation (gate-independent), not gate-weighted
    # sampling. Gives a direct view of what each expert's prior looks like conditioned on y.
    x_gen_per_expert: Dict[int, Optional[torch.Tensor]] = {}
    dim = csmf_model.dim
    for k, expert in enumerate(csmf_model.experts):
        generated = []
        try:
            for i in range(n):
                z = torch.randn(1, dim, device=x_clean.device)
                x_g = csmf_model._expert_inverse(
                    expert, z, y_deg[i:i+1], h[i:i+1], z_factored_list=None
                )
                generated.append(x_g.cpu())
            x_gen_k = torch.cat(generated, dim=0)
            # Reshape flat if needed
            if x_gen_k.dim() == 2 and x_gen_k.shape[-1] == 784:
                x_gen_k = x_gen_k.view(x_gen_k.shape[0], 1, 28, 28)
            x_gen_per_expert[k] = x_gen_k
            logger.info(f"MU | collect_reconstruction_batch: expert={k} generated {n} samples OK")
        except Exception as e:
            logger.error(
                f"MU | collect_reconstruction_batch: generation failed | "
                f"expert={k} ({expert_names[k]}): {e}"
            )
            x_gen_per_expert[k] = None

    return {
        "y":       y_deg.cpu(),
        "x_clean": x_clean.cpu(),
        "x_hat":   x_hat_per_expert,
        "x_gen":   x_gen_per_expert,
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


# =============================================================================
# collect_logdet_decomposition
# Source: [LOGDET-DIAG] v1.2 — new function
# =============================================================================

@torch.no_grad()
def collect_logdet_decomposition(
    csmf_model,
    val_loader,
    device: torch.device,
    max_batches: int = 20,
) -> Optional[Dict[str, Any]]:
    """
    Collect per-expert log_det and log_p(z) decomposition from the val set.

    For each val batch, runs _expert_forward per expert to get log_det [B] and
    computes log_p_z [B] from base_dist. Aggregates across batches.

    Args:
        csmf_model  : CSMF model (eval mode set internally).
        val_loader  : DataLoader yielding (x_clean, y_deg).
        device      : Compute device.
        max_batches : Max val batches to collect (speed control).

    Returns:
        Dict mapping expert_name (str) ->
            {
                "log_det"  : Tensor(N,),   # per-sample log|det J|
                "log_p_z"  : Tensor(N,),   # per-sample log p(z)
                "D"        : int,           # input dimension (for per-dim normalisation)
            }
        or None if no batches collected for any expert.
    """
    csmf_model.eval()
    K            = csmf_model.K
    expert_names = [type(e).__name__ for e in csmf_model.experts]
    D            = csmf_model.dim  # input dimension

    log_det_accum : Dict[int, list] = {k: [] for k in range(K)}
    log_p_z_accum : Dict[int, list] = {k: [] for k in range(K)}
    n_collected = 0

    for x_clean, y_deg in val_loader:
        if n_collected >= max_batches:
            break
        x_clean = x_clean.to(device)
        y_deg   = y_deg.to(device)

        try:
            h = csmf_model.conditioner(y_deg)
        except Exception as e:
            logger.error(f"MU | collect_logdet_decomposition: conditioner failed batch={n_collected}: {e}")
            continue

        batch_ok = True
        for k, expert in enumerate(csmf_model.experts):
            try:
                z, log_det, log_prob, _ = csmf_model._expert_forward(
                    expert, x_clean, y_deg, h
                )

                if torch.isnan(log_det).any():
                    logger.error(
                        f"MU | collect_logdet_decomposition: NaN log_det | "
                        f"expert={k} ({expert_names[k]}) batch={n_collected} — skipping batch"
                    )
                    batch_ok = False
                    break

                z_flat  = z.flatten(1) if z.dim() > 2 else z
                log_p_z = csmf_model.base_dist.log_prob(z_flat).sum(dim=1)  # [B]

                if torch.isnan(log_p_z).any():
                    logger.error(
                        f"MU | collect_logdet_decomposition: NaN log_p_z | "
                        f"expert={k} ({expert_names[k]}) batch={n_collected} — skipping batch"
                    )
                    batch_ok = False
                    break

                log_det_accum[k].append(log_det.cpu())
                log_p_z_accum[k].append(log_p_z.cpu())

            except Exception as e:
                logger.error(
                    f"MU | collect_logdet_decomposition: expert={k} ({expert_names[k]}) "
                    f"batch={n_collected} exception: {e} — skipping batch"
                )
                batch_ok = False
                break

        if batch_ok:
            n_collected += 1

    if n_collected == 0:
        logger.error("MU | collect_logdet_decomposition: no batches collected — returning None")
        return None

    result: Dict[str, Any] = {}
    for k in range(K):
        name = expert_names[k]
        if not log_det_accum[k]:
            logger.error(
                f"MU | collect_logdet_decomposition: no data for expert={k} ({name})"
            )
            continue
        ld = torch.cat(log_det_accum[k])  # (N,)
        lp = torch.cat(log_p_z_accum[k])  # (N,)
        result[name] = {
            "log_det": ld,
            "log_p_z": lp,
            "D":       D,
        }
        logger.info(
            f"MU | collect_logdet_decomposition: expert={k} ({name}) | "
            f"N={ld.shape[0]} | log_det_mean={ld.mean():.4f} | "
            f"log_p_z_mean={lp.mean():.4f} | D={D}"
        )

    if not result:
        logger.error("MU | collect_logdet_decomposition: all experts failed — returning None")
        return None

    return result


# =============================================================================
# collect_prox_diagnostics
# [PROX-T] v1.3 — new function
# =============================================================================

@torch.no_grad()
def collect_prox_diagnostics(
    csmf_model,
    val_loader,
    A_fn,
    At_fn,
    device: torch.device,
    T_values: Optional[List[int]] = None,
    lam: float = 0.1,
    max_batches: int = 10,
    num_samples: int = 4,
) -> Optional[Dict[str, Any]]:
    """
    Collect proximal correction diagnostics for P_PROX1, P_PROX2, P_PROX3.

    For each T in T_values, draws x^(0) from the flow (num_prox_steps=0),
    then manually applies T gradient prox steps:
        x^(t+1) = clamp(x^(t) - lam * At(Ax^(t) - y), 0, 1)

    Residuals computed directly here — no dependency on PROX module.
    NLL baseline is from T=0 only (flow NLL unchanged by post-hoc prox).

    Args:
        csmf_model  : CSMF model (eval mode set internally)
        val_loader  : DataLoader yielding (x_clean, y_deg)
        A_fn        : Forward operator callable: x (B,d) -> Ax (B,d')
        At_fn       : Adjoint operator callable: r (B,d') -> Atr (B,d)
        device      : Compute device
        T_values    : List of T steps to evaluate. Default [0, 1, 2, 3].
        lam         : Gradient step size (default 0.1)
        max_batches : Max val batches to collect
        num_samples : Samples per observation for std estimation

    Returns:
        Dict with keys:
            "T_values"         : List[int]
            "residuals_by_T"   : Dict[str, float]   — mean residual at each T
            "residual_steps"   : List[float]         — per-step for max(T_values)
            "sample_std_pre"   : float               — mean sample std at T=0
            "sample_std_post"  : float               — mean sample std at T=max
            "nll_baseline"     : float               — mean NLL at T=0
        or None if no batches collected.
    """
    if T_values is None:
        T_values = [0, 1, 2, 3]

    T_max = max(T_values)
    csmf_model.eval()

    residuals_by_step: List[List[float]] = [[] for _ in range(T_max + 1)]
    std_pre_all:  List[float] = []
    std_post_all: List[float] = []
    nll_all:      List[float] = []
    n_collected = 0

    for x_clean, y_deg in val_loader:
        if n_collected >= max_batches:
            break

        y_deg   = y_deg.to(device)
        x_clean = x_clean.to(device)

        try:
            x_samples, _ = csmf_model.sample(y_deg, num_samples=num_samples)
            x_cur = x_samples.mean(dim=1)   # (B, d)

            log_q, _ = csmf_model.forward(x_clean, y_deg)
            if torch.isnan(log_q).any():
                logger.warning(
                    "MU | collect_prox_diagnostics: NaN NLL batch=%d — skipping",
                    n_collected
                )
                continue
            nll_all.append(-log_q.mean().item())

            std_pre_all.append(x_samples.std(dim=1).mean().item())

            y_flat = y_deg.flatten(1) if y_deg.dim() > 2 else y_deg

            res_0 = (A_fn(x_cur) - y_flat).pow(2).mean().item()
            if not np.isfinite(res_0):
                logger.error(
                    "MU | collect_prox_diagnostics: non-finite residual at t=0 "
                    "batch=%d — skipping", n_collected
                )
                continue
            residuals_by_step[0].append(res_0)

            for t in range(T_max):
                Ax    = A_fn(x_cur)
                grad  = At_fn(Ax - y_flat)
                x_cur = (x_cur - lam * grad).clamp(0.0, 1.0)
                res_t = (A_fn(x_cur) - y_flat).pow(2).mean().item()
                if not np.isfinite(res_t):
                    logger.error(
                        "MU | collect_prox_diagnostics: non-finite residual at "
                        "t=%d batch=%d — stopping prox loop", t + 1, n_collected
                    )
                    break
                residuals_by_step[t + 1].append(res_t)

            x_post = x_samples.clone()
            for s in range(num_samples):
                xs = x_post[:, s, :]
                for _ in range(T_max):
                    xs = (xs - lam * At_fn(A_fn(xs) - y_flat)).clamp(0.0, 1.0)
                x_post[:, s, :] = xs
            std_post_all.append(x_post.std(dim=1).mean().item())

            n_collected += 1

        except Exception as e:
            logger.error(
                "MU | collect_prox_diagnostics: error batch=%d: %s",
                n_collected, e
            )
            continue

    if n_collected == 0:
        logger.error("MU | collect_prox_diagnostics: no batches collected — returning None")
        return None

    mean_by_step = [
        float(np.mean(residuals_by_step[t])) if residuals_by_step[t] else float("nan")
        for t in range(T_max + 1)
    ]
    residuals_by_T = {
        str(T): mean_by_step[T] if T <= T_max and np.isfinite(mean_by_step[T])
                else float("nan")
        for T in T_values
    }

    logger.info(
        "MU | collect_prox_diagnostics: %d batches | T_max=%d | "
        "residual T=0=%.6f -> T=%d=%.6f | NLL=%.4f",
        n_collected, T_max,
        mean_by_step[0], T_max, mean_by_step[T_max],
        float(np.mean(nll_all)),
    )

    return {
        "T_values":        T_values,
        "residuals_by_T":  residuals_by_T,
        "residual_steps":  mean_by_step,
        "sample_std_pre":  float(np.mean(std_pre_all)),
        "sample_std_post": float(np.mean(std_post_all)),
        "nll_baseline":    float(np.mean(nll_all)),
    }


# =============================================================================
# collect_mixture_recon_batch  [MIX-RECON] MU v1.6
# =============================================================================

def collect_mixture_recon_batch(
    csmf_model,
    val_loader,
    device: torch.device,
    n_samples: int = 8,
) -> Optional[Dict[str, Any]]:
    """
    Collect Stage C mixture 4-col reconstruction data.

    Cycle:     argmax-expert encode→decode from sample_all_experts().
    Generated: gate-sampled generation from csmf_model.sample(y, num_samples=1).

    Args:
        csmf_model : CSMF model (eval mode set internally).
        val_loader : DataLoader yielding (x_clean, y_deg).
        device     : Compute device.
        n_samples  : Number of samples to collect.

    Returns:
        Dict with keys:
            "y"           : Tensor(n, 1, H, W) — degraded input
            "x_clean"     : Tensor(n, 1, H, W) — clean ground truth
            "x_cycle_mix" : Tensor(n, 1, H, W) — argmax-expert cycle recon
            "x_gen_mix"   : Tensor(n, 1, H, W) or None — gate-sampled generation
        or None on failure.
    """
    csmf_model.eval()

    try:
        x_clean_batch, y_deg_batch = next(iter(val_loader))
    except StopIteration:
        logger.error("MU | collect_mixture_recon_batch: val_loader is empty")
        return None
    except Exception as e:
        logger.error("MU | collect_mixture_recon_batch: failed to get batch: %s", e)
        return None

    n = min(n_samples, x_clean_batch.shape[0])
    x_clean = x_clean_batch[:n].to(device)
    y_deg   = y_deg_batch[:n].to(device)

    def _reshape(t: torch.Tensor) -> torch.Tensor:
        """Reshape flat (B,784) → (B,1,28,28) if needed."""
        if t.dim() == 2 and t.shape[-1] == 784:
            return t.view(t.shape[0], 1, 28, 28)
        return t

    # ------------------------------------------------------------------
    # Cycle: argmax-expert encode→decode via sample_all_experts()
    # ------------------------------------------------------------------
    x_cycle_mix = None
    try:
        with torch.no_grad():
            w, x_hats = csmf_model.sample_all_experts(y_deg)  # (B,K), (B,K,d)

        if torch.any(torch.isnan(w)):
            logger.error("MU | collect_mixture_recon_batch: NaN in gate weights w")
            return None

        argmax_k = w.argmax(dim=1)  # (B,) expert index per sample

        cycle_list = []
        for i in range(n):
            k_i  = argmax_k[i].item()
            xhat = x_hats[i, k_i, :]   # (d,)
            cycle_list.append(xhat.unsqueeze(0).cpu())

        x_cycle_mix = _reshape(torch.cat(cycle_list, dim=0))  # (n, 1, 28, 28)

        if torch.any(torch.isnan(x_cycle_mix)):
            logger.error("MU | collect_mixture_recon_batch: NaN in x_cycle_mix")
            return None

        logger.info(
            "MU | collect_mixture_recon_batch: cycle OK | "
            "argmax experts=%s", argmax_k.cpu().tolist()
        )

    except Exception as e:
        logger.error("MU | collect_mixture_recon_batch: cycle collection failed: %s", e)
        return None

    # ------------------------------------------------------------------
    # Generated: gate-sampled generation via csmf_model.sample()
    # ------------------------------------------------------------------
    x_gen_mix = None
    try:
        with torch.no_grad():
            x_gen_raw, _ = csmf_model.sample(y_deg, num_samples=1)
            # sample() returns (B, num_samples, d) or (B, d) — normalise to (B, d)
            if x_gen_raw.dim() == 3:
                x_gen_raw = x_gen_raw[:, 0, :]   # take first sample

        x_gen_mix = _reshape(x_gen_raw.cpu())  # (n, 1, 28, 28)

        if torch.any(torch.isnan(x_gen_mix)):
            logger.error("MU | collect_mixture_recon_batch: NaN in x_gen_mix")
            x_gen_mix = None
        else:
            logger.info("MU | collect_mixture_recon_batch: generation OK")

    except Exception as e:
        logger.error(
            "MU | collect_mixture_recon_batch: generation failed (non-fatal): %s", e
        )
        x_gen_mix = None   # non-fatal — cycle still usable

    # Reshape x_clean and y_deg for consistent output format
    x_clean_out = _reshape(x_clean.cpu())
    y_out       = _reshape(y_deg.cpu())

    return {
        "y":           y_out,
        "x_clean":     x_clean_out,
        "x_cycle_mix": x_cycle_mix,
        "x_gen_mix":   x_gen_mix,
    }


# =============================================================================
# collect_fi_gate_comparison  [FI-GATE] MU v1.6
# =============================================================================

def collect_fi_gate_comparison(
    csmf_model,
    val_loader,
    device: torch.device,
    fi_summary_path: str,
) -> Optional[Dict[str, Any]]:
    """
    Load Stage A FI scores from fi_diag_summary.json and collect current
    mean gate weights from val_loader for P_fi_gate comparison plot.

    FI scores are computed on frozen Stage A weights and do not change during
    Stage B — loading from JSON avoids recomputation.

    Args:
        csmf_model      : CSMF model in eval mode.
        val_loader      : DataLoader yielding (x_clean, y_deg).
        device          : Compute device.
        fi_summary_path : Full path to fi_diag_summary.json from FI-DIAG run.

    Returns:
        Dict with keys:
            "expert_names"     : List[str]
            "fi_scores"        : List[float] — F_k mean per expert (Option A)
            "fi_ratios"        : List[float] — F_k / max(F_k) per expert
            "gate_weights_mean": List[float] — mean gate weight per expert on val
        or None on failure (non-fatal — P_fi_gate must skip with warning).
    """
    # ------------------------------------------------------------------
    # Step 1: Load FI summary JSON
    # ------------------------------------------------------------------
    if not fi_summary_path:
        logger.warning(
            "MU | collect_fi_gate_comparison: fi_summary_path not provided — "
            "P_fi_gate will be skipped"
        )
        return None

    try:
        with open(fi_summary_path, "r") as f:
            fi_data = json.load(f)
    except FileNotFoundError:
        logger.warning(
            "MU | collect_fi_gate_comparison: fi_diag_summary.json not found "
            "at '%s' — FI-DIAG may not have been run after Stage A. "
            "P_fi_gate will be skipped (non-fatal).",
            fi_summary_path,
        )
        return None
    except Exception as e:
        logger.error(
            "MU | collect_fi_gate_comparison: failed to load '%s': %s — "
            "P_fi_gate will be skipped",
            fi_summary_path, e,
        )
        return None

    # ------------------------------------------------------------------
    # Step 2: Extract fi_mean and expert names from JSON
    # Expected JSON structure: fi_data["option_a"][expert_name]["mean"]
    # ------------------------------------------------------------------
    try:
        option_a = fi_data.get("option_a", {})
        if not option_a:
            logger.warning(
                "MU | collect_fi_gate_comparison: 'option_a' key missing or empty "
                "in fi_diag_summary.json — P_fi_gate will be skipped"
            )
            return None

        expert_names_fi = list(option_a.keys())
        fi_scores = [float(option_a[name].get("mean", 0.0)) for name in expert_names_fi]
        fi_max    = max(fi_scores) if fi_scores else 1.0
        fi_ratios = [s / max(fi_max, 1e-8) for s in fi_scores]

        logger.info(
            "MU | collect_fi_gate_comparison: loaded FI for %d experts: %s",
            len(expert_names_fi),
            {n: f"{s:.4f}" for n, s in zip(expert_names_fi, fi_scores)},
        )

    except Exception as e:
        logger.error(
            "MU | collect_fi_gate_comparison: failed to parse option_a from JSON: %s — "
            "P_fi_gate will be skipped", e
        )
        return None

    # ------------------------------------------------------------------
    # Step 3: Collect mean gate weights from val_loader
    # ------------------------------------------------------------------
    csmf_model.eval()
    gate_weight_acc = None
    n_batches = 0

    try:
        with torch.no_grad():
            for x_clean, y_deg in val_loader:
                y_deg = y_deg.to(device)
                try:
                    w = csmf_model._gate_weights(y_deg)   # (B, K)
                    if torch.any(torch.isnan(w)):
                        logger.warning(
                            "MU | collect_fi_gate_comparison: NaN gate weights "
                            "batch=%d — skipping", n_batches
                        )
                        continue
                    acc = w.mean(dim=0).cpu()              # (K,)
                    gate_weight_acc = acc if gate_weight_acc is None \
                                      else gate_weight_acc + acc
                    n_batches += 1
                except Exception as e:
                    logger.error(
                        "MU | collect_fi_gate_comparison: gate forward failed "
                        "batch=%d: %s", n_batches, e
                    )
                    continue

    except Exception as e:
        logger.error(
            "MU | collect_fi_gate_comparison: val_loader iteration failed: %s", e
        )
        return None

    if n_batches == 0 or gate_weight_acc is None:
        logger.error(
            "MU | collect_fi_gate_comparison: no valid batches for gate weights — "
            "P_fi_gate will be skipped"
        )
        return None

    gate_weights_mean = (gate_weight_acc / n_batches).tolist()   # (K,)

    # ------------------------------------------------------------------
    # Step 4: Align expert ordering (FI JSON vs model expert list)
    # JSON expert names may differ from model class names — match by
    # stripping "Conditional" prefix for robust comparison.
    # ------------------------------------------------------------------
    model_expert_names = [type(e).__name__ for e in csmf_model.experts]

    def _short(name: str) -> str:
        return name.replace("Conditional", "").lower()

    aligned_fi     = []
    aligned_ratios = []
    aligned_gates  = []
    aligned_names  = []

    for k, model_name in enumerate(model_expert_names):
        short_model = _short(model_name)
        matched_fi  = None
        matched_ratio = None

        for fi_name, fi_val, fi_ratio in zip(expert_names_fi, fi_scores, fi_ratios):
            if _short(fi_name) == short_model:
                matched_fi    = fi_val
                matched_ratio = fi_ratio
                break

        if matched_fi is None:
            logger.warning(
                "MU | collect_fi_gate_comparison: expert '%s' not found in "
                "fi_diag_summary.json — using fi=0.0 for this expert",
                model_name,
            )
            matched_fi    = 0.0
            matched_ratio = 0.0

        gate_w = gate_weights_mean[k] if k < len(gate_weights_mean) else 0.0
        aligned_fi.append(matched_fi)
        aligned_ratios.append(matched_ratio)
        aligned_gates.append(gate_w)
        aligned_names.append(model_name)

    # Flag misalignment: high FI but low gate weight
    for name, fi_r, gw in zip(aligned_names, aligned_ratios, aligned_gates):
        if fi_r > 0.5 and gw < 0.1:
            logger.warning(
                "MU | collect_fi_gate_comparison: MISALIGNMENT — expert '%s' has "
                "high FI ratio=%.3f but low gate weight=%.3f. "
                "Gate may not be routing to this expert.",
                name, fi_r, gw,
            )

    logger.info(
        "MU | collect_fi_gate_comparison: complete | %d experts | "
        "gate_weights=%s | fi_ratios=%s",
        len(aligned_names),
        [f"{g:.3f}" for g in aligned_gates],
        [f"{r:.3f}" for r in aligned_ratios],
    )

    return {
        "expert_names":      aligned_names,
        "fi_scores":         aligned_fi,
        "fi_ratios":         aligned_ratios,
        "gate_weights_mean": aligned_gates,
    }
