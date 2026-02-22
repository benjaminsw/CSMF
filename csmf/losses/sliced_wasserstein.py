# =============================================================================
# Version: WP2.2-SW2-v1.1
# Abbr: SW2
# File: csmf/losses/sliced_wasserstein.py
# Description: Differentiable Sliced-Wasserstein Distance (SW2)
# Dependencies: torch
# Changelog:
#   v1.1 - Added linear interpolation for unequal sample sizes (replaces subsampling)
#   v1.1 - Added max-SW option (worst-case projection) for robustness
#   v1.1 - Added gradient checkpointing for large L (memory efficiency)
#   v1.0 - Core SW2: random projections, sort, L2 distance
# =============================================================================

import logging
import torch
import torch.utils.checkpoint as checkpoint

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core: Sliced-Wasserstein Distance
# ---------------------------------------------------------------------------

def sliced_wasserstein_distance(
    X: torch.Tensor,
    Y: torch.Tensor,
    num_projections: int = 256,
    mode: str = "mean",          # "mean" | "max"  [Additional]
    use_checkpointing: bool = False,  # [Additional] gradient checkpointing
) -> torch.Tensor:
    """
    Compute Sliced-Wasserstein-2 distance between sample sets X and Y.

    SW2² ≈ (1/L) Σ_ℓ (1/M) Σ_i [α^(i)_ℓ - α̃^(i)_ℓ]²

    Args:
        X: (N, d) samples from distribution p
        Y: (M, d) samples from distribution q
        num_projections: L = number of random directions (recommended 256-512)
        mode: "mean" for standard SW2; "max" for max-SW (worst-case projection)
        use_checkpointing: if True, use gradient checkpointing to reduce memory

    Returns:
        sw2: scalar Sliced-Wasserstein-2 distance (not squared)

    Raises:
        ValueError: on invalid inputs or mode
    """
    # ---- Input validation ------------------------------------------------
    if X.dim() != 2 or Y.dim() != 2:
        msg = f"SW2 expects 2D tensors, got X={X.shape}, Y={Y.shape}"
        logger.error(msg)
        raise ValueError(msg)

    if X.shape[1] != Y.shape[1]:
        msg = f"SW2 dimension mismatch: X.d={X.shape[1]}, Y.d={Y.shape[1]}"
        logger.error(msg)
        raise ValueError(msg)

    if mode not in ("mean", "max"):
        msg = f"SW2 mode must be 'mean' or 'max', got '{mode}'"
        logger.error(msg)
        raise ValueError(msg)

    d = X.shape[1]
    N, M = X.shape[0], Y.shape[0]

    # ---- Sample random directions on unit sphere -------------------------
    directions = torch.randn(num_projections, d, device=X.device, dtype=X.dtype)
    norms = torch.norm(directions, dim=1, keepdim=True).clamp(min=1e-8)
    directions = directions / norms                          # (L, d)

    # ---- Compute SW2 (optionally checkpointed) ---------------------------
    def _compute_sw2(X_, Y_, dirs):
        X_proj = X_ @ dirs.T   # (N, L)
        Y_proj = Y_ @ dirs.T   # (M, L)

        # NaN guard on projections
        if torch.any(torch.isnan(X_proj)) or torch.any(torch.isnan(Y_proj)):
            logger.error("SW2: NaN detected in projections. X_proj NaN: %s, Y_proj NaN: %s",
                         torch.isnan(X_proj).sum().item(),
                         torch.isnan(Y_proj).sum().item())
            raise RuntimeError("NaN in SW2 projections - check input samples")

        # Sort along sample dimension
        X_sorted = torch.sort(X_proj, dim=0)[0]            # (N, L)
        Y_sorted = torch.sort(Y_proj, dim=0)[0]            # (M, L)

        # [Additional] Interpolate when N != M (preserves all data)
        if N != M:
            X_sorted, Y_sorted = _interpolate_to_match(X_sorted, Y_sorted)

        # NaN guard on sorted output
        if torch.any(torch.isnan(X_sorted)) or torch.any(torch.isnan(Y_sorted)):
            logger.error("SW2: NaN in sorted projections after interpolation")
            raise RuntimeError("NaN in SW2 sorted projections")

        sq_diff = (X_sorted - Y_sorted) ** 2              # (K, L)

        if mode == "max":
            # [Additional] max-SW: worst-case projection
            sw2_sq = sq_diff.mean(dim=0).max()
        else:
            sw2_sq = sq_diff.mean()

        return sw2_sq

    # [Additional] Gradient checkpointing for large L
    if use_checkpointing and num_projections >= 512:
        sw2_sq = checkpoint.checkpoint(
            _compute_sw2, X, Y, directions, use_reentrant=False
        )
    else:
        sw2_sq = _compute_sw2(X, Y, directions)

    sw2 = torch.sqrt(sw2_sq.clamp(min=0.0))               # clamp avoids sqrt(neg)

    if torch.isnan(sw2) or torch.isinf(sw2):
        logger.error("SW2: result is NaN/Inf (sw2_sq=%.6f)", sw2_sq.item())
        raise RuntimeError("SW2 returned NaN or Inf")

    return sw2


# ---------------------------------------------------------------------------
# [Additional] Linear Interpolation for unequal sample sizes
# ---------------------------------------------------------------------------

def _interpolate_to_match(
    X_sorted: torch.Tensor,
    Y_sorted: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Align sorted projections of different lengths via linear interpolation.
    Interpolates the shorter tensor to match the longer one.

    Args:
        X_sorted: (N, L) sorted projections
        Y_sorted: (M, L) sorted projections

    Returns:
        X_interp, Y_interp: both (max(N,M), L)
    """
    N, L = X_sorted.shape
    M = Y_sorted.shape[0]
    K = max(N, M)

    def _interp(t: torch.Tensor, target_len: int) -> torch.Tensor:
        """Linear interpolate tensor (src_len, L) -> (target_len, L)."""
        src_len = t.shape[0]
        if src_len == target_len:
            return t
        # query positions in [0, src_len-1]
        idx = torch.linspace(0, src_len - 1, target_len, device=t.device, dtype=t.dtype)
        lo = idx.floor().long().clamp(0, src_len - 2)
        hi = (lo + 1).clamp(0, src_len - 1)
        frac = (idx - lo.float()).unsqueeze(1)              # (target_len, 1)
        return t[lo] * (1 - frac) + t[hi] * frac           # (target_len, L)

    X_interp = _interp(X_sorted, K)
    Y_interp = _interp(Y_sorted, K)
    return X_interp, Y_interp
