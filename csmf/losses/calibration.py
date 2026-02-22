# =============================================================================
# Version: WP2.4-Calib-v1.1
# Abbr: CALIB
# File: csmf/losses/calibration.py
# Description: Calibration metrics for posterior evaluation
#              Energy Score (ES), CRPS, Temperature Scaling,
#              Multi-dim CRPS extension, PIT histogram, Coverage diagnostics
# Dependencies: torch, numpy, matplotlib (optional, for PIT plot)
# Changelog:
#   v1.1 - Added multi-dimensional CRPS via energy-score extension
#   v1.1 - Added PIT histogram for visual calibration diagnostics
#   v1.1 - Added coverage diagnostics at multiple alpha levels
#   v1.0 - Core ES: pairwise diversity + distance to reference
#   v1.0 - Core CRPS (1D), temperature_scaling
# =============================================================================

import logging
import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core: Energy Score
# ---------------------------------------------------------------------------

def energy_score(
    samples: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Energy Score (proper scoring rule) for posterior samples.

    ES = E‖x − x'‖ − 2·E‖x − x*‖
    where x, x' ~ q(·|y)  and  x* is the reference (clean signal).

    Args:
        samples:   (S, d) samples from q(x|y), S >= 2
        reference: (d,) or (1, d) reference / clean sample x*

    Returns:
        es: scalar Energy Score (lower = better calibrated)
    """
    S, d = samples.shape

    # Degenerate guard
    if S < 2:
        msg = f"energy_score requires S>=2 samples, got S={S}"
        logger.error(msg)
        raise ValueError(msg)

    # Ensure reference is (1, d)
    ref = reference.view(1, d) if reference.dim() == 1 else reference
    if ref.shape != (1, d):
        msg = f"energy_score: reference shape mismatch, expected (d={d},) got {reference.shape}"
        logger.error(msg)
        raise ValueError(msg)

    # Term 1: E‖x − x'‖  (pairwise sample diversity)
    pairwise = torch.cdist(samples, samples, p=2)          # (S, S)
    if torch.any(torch.isnan(pairwise)):
        logger.error("energy_score: NaN in pairwise distances (samples may be degenerate)")
        raise RuntimeError("NaN in energy_score pairwise distances")

    # Exclude diagonal (self-distances = 0)
    mask = ~torch.eye(S, dtype=torch.bool, device=samples.device)
    diversity = pairwise[mask].mean()                      # scalar

    # Term 2: E‖x − x*‖  (distance to reference)
    dist_to_ref = torch.cdist(samples, ref, p=2).mean()   # scalar

    if torch.isnan(dist_to_ref):
        logger.error("energy_score: NaN in distance to reference")
        raise RuntimeError("NaN in energy_score dist_to_ref")

    es = diversity - 2.0 * dist_to_ref

    return es


# ---------------------------------------------------------------------------
# Core: CRPS (1D)
# ---------------------------------------------------------------------------

def crps(
    samples: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """
    Continuous Ranked Probability Score (1D only).

    CRPS = ∫ [F_S(x) − 1{x ≥ x*}]² dx
         ≈ (1/S) Σ |x^(i) − x*| − (1/2S²) Σ_{i,j} |x^(i) − x^(j)|

    Args:
        samples:   (S,) or (S, 1) 1D samples from q(x|y)
        reference: scalar or (1,) reference value x*

    Returns:
        crps_val: scalar CRPS (lower = better)
    """
    samples_flat = samples.flatten()
    S = samples_flat.shape[0]

    if samples.dim() > 1 and samples.shape[1] > 1:
        logger.warning(
            "crps: called with d=%d > 1 — 1D CRPS is only valid for scalar outputs. "
            "Use crps_multidim() for d>1.", samples.shape[1]
        )

    ref_val = reference.flatten()[0] if torch.is_tensor(reference) else torch.tensor(
        float(reference), device=samples_flat.device, dtype=samples_flat.dtype
    )

    # Term 1: (1/S) Σ |x^(i) − x*|
    term1 = torch.abs(samples_flat - ref_val).mean()

    # Term 2: (1/2S²) Σ_{i,j} |x^(i) − x^(j)|  (pairwise)
    pairwise = torch.abs(samples_flat.unsqueeze(0) - samples_flat.unsqueeze(1))  # (S, S)
    term2 = pairwise.mean() / 2.0

    crps_val = term1 - term2

    if torch.isnan(crps_val):
        logger.error("crps: NaN result. term1=%.6f, term2=%.6f", term1.item(), term2.item())
        raise RuntimeError("NaN in CRPS computation")

    return crps_val


# ---------------------------------------------------------------------------
# Core: Temperature Scaling
# ---------------------------------------------------------------------------

def temperature_scaling(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Apply temperature scaling to logits to adjust posterior spread.

    tau > 1 → softer/more spread distribution
    tau < 1 → sharper/more peaked distribution
    tau = 1 → unchanged

    Args:
        logits:      (..., K) raw logits
        temperature: tau > 0

    Returns:
        scaled_logits: (..., K) temperature-scaled logits
    """
    if temperature <= 0:
        msg = f"temperature_scaling: temperature must be > 0, got {temperature}"
        logger.error(msg)
        raise ValueError(msg)

    return logits / temperature


# ---------------------------------------------------------------------------
# [Additional] Multi-dimensional CRPS via Energy Score Extension
# ---------------------------------------------------------------------------

def crps_multidim(samples: torch.Tensor, reference: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    Multivariate CRPS-style calibration term (aligned with energy_score() convention).

    For beta in (0, 2):
        score = E||x - x'||^beta - 2 E||x - x*||^beta

    Note: With beta=1 this matches energy_score() exactly.
    """
    if not (0.0 < beta < 2.0):
        raise ValueError(f"beta must be in (0, 2), got {beta}")

    if samples.dim() != 2:
        raise ValueError(f"samples must be (S, d), got {samples.shape}")

    S, d = samples.shape
    if S < 2:
        raise ValueError(f"crps_multidim requires S>=2 samples, got S={S}")

    # Reference shape to (1, d)
    ref = reference.view(1, d) if reference.dim() == 1 else reference
    if ref.shape != (1, d):
        raise ValueError(f"reference shape mismatch, expected (d,) or (1,d), got {reference.shape}")

    # Pairwise distances between samples: (S, S)
    pairwise = torch.cdist(samples, samples, p=2)

    # Exclude diagonal (self-distances)
    mask = ~torch.eye(S, dtype=torch.bool, device=samples.device)

    # diversity = E||x - x'||^beta
    diversity = (pairwise[mask] ** beta).mean()

    # dist_to_ref = E||x - x*||^beta
    dist_to_ref = (torch.cdist(samples, ref, p=2) ** beta).mean()

    return diversity - 2.0 * dist_to_ref



# ---------------------------------------------------------------------------
# [Additional] PIT Histogram
# ---------------------------------------------------------------------------

def pit_histogram(
    samples: torch.Tensor,
    references: torch.Tensor,
    num_bins: int = 20,
) -> dict:
    """
    Probability Integral Transform (PIT) histogram for calibration diagnostics.

    For each reference x*, compute the fraction of samples below x*:
        PIT = F_S(x*) = (1/S) Σ 1{x^(i) < x*}

    A perfectly calibrated model yields a uniform PIT histogram.
    Works per-dimension for d > 1 (returns per-dim histograms).

    Args:
        samples:    (S, d) samples from q(x|y)  [single condition]
                    or (B, S, d) for a batch of conditions
        references: (d,) or (B, d) reference values
        num_bins:   number of histogram bins (default 20)

    Returns:
        dict with keys:
            'pit_values':  (B, d) or (d,) PIT values in [0, 1]
            'histograms':  (d, num_bins) bin counts per dimension
            'bin_edges':   (num_bins+1,) bin edges in [0, 1]
            'uniformity_pvalue': (d,) approximate chi-squared p-value per dim
    """
    # Normalise to (B, S, d)
    if samples.dim() == 2:
        samples = samples.unsqueeze(0)       # (1, S, d)
        references = references.unsqueeze(0) if references.dim() == 1 else references.unsqueeze(0)

    B, S, d = samples.shape

    if references.shape != (B, d):
        msg = (f"pit_histogram: references shape {references.shape} "
               f"does not match (B={B}, d={d})")
        logger.error(msg)
        raise ValueError(msg)

    # PIT values: fraction of samples below reference  (B, d)
    ref_expanded = references.unsqueeze(1)                   # (B, 1, d)
    pit_values = (samples < ref_expanded).float().mean(dim=1)  # (B, d)

    pit_np = pit_values.detach().cpu().numpy()               # (B, d)
    bin_edges = np.linspace(0, 1, num_bins + 1)

    histograms = np.zeros((d, num_bins), dtype=np.int64)
    p_values = np.zeros(d)

    for dim_i in range(d):
        counts, _ = np.histogram(pit_np[:, dim_i], bins=bin_edges)
        histograms[dim_i] = counts

        # Chi-squared uniformity test (approximate p-value)
        expected = B / num_bins
        chi2 = np.sum((counts - expected) ** 2 / (expected + 1e-8))
        from scipy.stats import chi2 as chi2_dist
        p_values[dim_i] = 1.0 - chi2_dist.cdf(chi2, df=num_bins - 1)

    logger.info(
        "PIT histogram: B=%d, S=%d, d=%d, bins=%d | mean p-value=%.4f",
        B, S, d, num_bins, p_values.mean()
    )

    return {
        "pit_values": pit_values,                          # (B, d)
        "histograms": histograms,                          # (d, num_bins)
        "bin_edges": bin_edges,                            # (num_bins+1,)
        "uniformity_pvalue": p_values,                     # (d,)
    }


# ---------------------------------------------------------------------------
# [Additional] Coverage Diagnostics
# ---------------------------------------------------------------------------

def coverage_diagnostics(
    samples: torch.Tensor,
    references: torch.Tensor,
    alpha_levels: list[float] | None = None,
) -> dict:
    """
    Check empirical vs nominal coverage at multiple alpha levels.

    For each alpha, checks whether the reference falls within the
    central (1-alpha) credible interval of the samples.
    A well-calibrated model has empirical ≈ nominal coverage.

    Args:
        samples:      (B, S, d) samples from q(x|y) for B conditions
        references:   (B, d) reference values
        alpha_levels: list of alpha values in (0,1), default [0.05, 0.10, ..., 0.50]

    Returns:
        dict with keys:
            'nominal':   list of (1 - alpha) coverage levels
            'empirical': (num_alpha, d) empirical coverage per dim
            'gap':       (num_alpha, d) empirical - nominal (positive = over-covered)
    """
    if alpha_levels is None:
        alpha_levels = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

    for a in alpha_levels:
        if not (0 < a < 1):
            msg = f"coverage_diagnostics: alpha must be in (0,1), got {a}"
            logger.error(msg)
            raise ValueError(msg)

    B, S, d = samples.shape

    if references.shape != (B, d):
        msg = (f"coverage_diagnostics: references shape {references.shape} "
               f"does not match (B={B}, d={d})")
        logger.error(msg)
        raise ValueError(msg)

    nominal_levels = [1.0 - a for a in alpha_levels]
    empirical = np.zeros((len(alpha_levels), d))

    for ai, alpha in enumerate(alpha_levels):
        lo_q = alpha / 2.0
        hi_q = 1.0 - alpha / 2.0

        # Compute quantiles per (B, d)  using torch.quantile
        lo = torch.quantile(samples, lo_q, dim=1)   # (B, d)
        hi = torch.quantile(samples, hi_q, dim=1)   # (B, d)

        inside = ((references >= lo) & (references <= hi)).float()  # (B, d)
        empirical[ai] = inside.mean(dim=0).detach().cpu().numpy()   # (d,)

    nominal_arr = np.array(nominal_levels)                          # (num_alpha,)
    gap = empirical - nominal_arr[:, None]                          # (num_alpha, d)

    logger.info(
        "Coverage diagnostics: B=%d, S=%d, d=%d | mean gap=%.4f",
        B, S, d, np.abs(gap).mean()
    )

    return {
        "nominal": nominal_levels,           # list of floats
        "empirical": empirical,              # (num_alpha, d)
        "gap": gap,                          # (num_alpha, d), + = over-covered
    }
