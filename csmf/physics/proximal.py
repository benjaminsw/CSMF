# =============================================================================
# Version: WP1.1-Prox-v1.0 | Abbr: PROX
# File: csmf/physics/proximal.py
# Description: Proximal gradient step for measurement-consistency refinement.
#              Implements the plug-and-play loop from WP1.2:
#                x^(0) ~ flow,  x^(t+1) = Prox_A(x^(t)),  t = 0..T-1
#              Each Prox_A step is one gradient descent step on ||Ax - y||².
# Dependencies: torch, FWD-MOD (csmf/physics/forward_models.py)
# Changelog:
#   v1.0 - prox_gradient_step: x_new = x - lam * Aᵀ(Ax - y), clamp [0,1]
#   v1.0 - apply_prox_steps: T-step loop; logs ||Ax^(t)-y||² before each step
#   v1.0 - NaN/Inf guard on residual and output at every step; raises on failure
#   v1.0 - make_prox_fn: factory returning (x, y) -> x callable for use in sample()
# =============================================================================

import logging
from typing import Callable

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core: single proximal gradient step
# ---------------------------------------------------------------------------

def prox_gradient_step(
    x: torch.Tensor,
    y: torch.Tensor,
    A_fn: Callable[[torch.Tensor], torch.Tensor],
    At_fn: Callable[[torch.Tensor], torch.Tensor],
    lam: float = 0.1,
) -> torch.Tensor:
    """
    One proximal gradient step on the data-fidelity term (1/2)||Ax - y||².

        x_new = x - lam · Aᵀ(Ax − y)

    This is steepest-descent on the quadratic cost w.r.t. x.
    Stable when lam < 2 / ||AᵀA|| (for MNIST blur+ds operator, ||AᵀA|| ≤ 1).

    Args:
        x:     (B, d) pixel-space samples in [0, 1] (post-sigmoid / flat)
        y:     (B, d') degraded observations matching A's output space
        A_fn:  forward operator: x (B, d) → Ax (B, d')
        At_fn: adjoint operator: r (B, d') → Aᵀr (B, d)
        lam:   gradient step size (default 0.1 — conservative for MNIST)

    Returns:
        x_new: (B, d) corrected samples, clamped to [0, 1]

    Raises:
        RuntimeError: if NaN detected in gradient or output
    """
    if lam <= 0.0:
        msg = f"PROX | lam must be > 0, got {lam}"
        logger.error(msg)
        raise ValueError(msg)

    Ax = A_fn(x)
    residual = Ax - y                    # (B, d')

    if torch.any(torch.isnan(residual)) or torch.any(torch.isinf(residual)):
        logger.error(
            "PROX | prox_gradient_step: NaN/Inf in residual | "
            "Ax range=[%.4f, %.4f] | y range=[%.4f, %.4f]",
            Ax.min().item(), Ax.max().item(),
            y.min().item(), y.max().item(),
        )
        raise RuntimeError("NaN/Inf in prox_gradient_step residual")

    grad = At_fn(residual)               # (B, d)

    if torch.any(torch.isnan(grad)) or torch.any(torch.isinf(grad)):
        logger.error("PROX | prox_gradient_step: NaN/Inf in adjoint gradient")
        raise RuntimeError("NaN/Inf in prox_gradient_step gradient")

    x_new = (x - lam * grad).clamp(0.0, 1.0)   # pixel space: keep in [0,1]

    if torch.any(torch.isnan(x_new)):
        logger.error(
            "PROX | prox_gradient_step: NaN in output | lam=%.4f | "
            "grad_norm=%.6f",
            lam, grad.norm().item(),
        )
        raise RuntimeError("NaN in prox_gradient_step output")

    return x_new


# ---------------------------------------------------------------------------
# T-step loop
# ---------------------------------------------------------------------------

def apply_prox_steps(
    x: torch.Tensor,
    y: torch.Tensor,
    A_fn: Callable[[torch.Tensor], torch.Tensor],
    At_fn: Callable[[torch.Tensor], torch.Tensor],
    num_steps: int,
    lam: float = 0.1,
) -> torch.Tensor:
    """
    Apply T proximal gradient steps: x^(t+1) = Prox_A(x^(t)).

    No resampling from the flow — the same initial x^(0) is refined T times.
    Logs ||Ax^(t) - y||² before each step for convergence diagnostics.

    Args:
        x:         (B, d) initial pixel-space samples x^(0)  (flat, in [0,1])
        y:         (B, d') degraded observations
        A_fn:      forward operator callable
        At_fn:     adjoint operator callable
        num_steps: T ∈ {1, 2, 3} — number of correction steps
        lam:       step size per step (default 0.1)

    Returns:
        x^(T): (B, d) refined samples after T steps
    """
    if num_steps <= 0:
        logger.warning("PROX | apply_prox_steps called with num_steps=%d — returning x unchanged", num_steps)
        return x

    if num_steps > 3:
        logger.warning(
            "PROX | apply_prox_steps: num_steps=%d > 3 — "
            "WP1.2 recommends T=1–3; proceeding anyway",
            num_steps,
        )

    for t in range(num_steps):
        with torch.no_grad():
            residual_sq = (A_fn(x) - y).pow(2).mean().item()

        logger.debug(
            "PROX | step %d/%d | ||Ax^(t)-y||²=%.6f",
            t + 1, num_steps, residual_sq,
        )

        if not torch.isfinite(torch.tensor(residual_sq)):
            logger.error(
                "PROX | apply_prox_steps: non-finite residual at step %d/%d | "
                "aborting prox loop",
                t + 1, num_steps,
            )
            raise RuntimeError(f"Non-finite residual at prox step {t+1}")

        x = prox_gradient_step(x, y, A_fn, At_fn, lam=lam)

    # Log final residual
    with torch.no_grad():
        final_residual = (A_fn(x) - y).pow(2).mean().item()
    logger.debug("PROX | final ||Ax^(T)-y||²=%.6f after T=%d steps", final_residual, num_steps)

    return x


# ---------------------------------------------------------------------------
# Factory: make_prox_fn
# ---------------------------------------------------------------------------

def make_prox_fn(
    A_fn: Callable[[torch.Tensor], torch.Tensor],
    At_fn: Callable[[torch.Tensor], torch.Tensor],
    num_steps: int,
    lam: float = 0.1,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Return a (x, y) -> x callable wrapping apply_prox_steps.
    Pass the returned fn as `prox_fn` to CSMF.sample().

    Args:
        A_fn:      forward operator callable
        At_fn:     adjoint operator callable
        num_steps: T correction steps
        lam:       step size

    Returns:
        prox_fn: Callable[(x, y) -> x_corrected]

    Example:
        op = BlurDownsampleOperator(...)
        prox_fn = make_prox_fn(op.forward, op.adjoint, num_steps=2, lam=0.1)
        x_refined, _ = csmf.sample(y, num_samples=4, prox_fn=prox_fn, num_prox_steps=2)
    """
    if num_steps <= 0:
        msg = f"PROX | make_prox_fn: num_steps must be > 0, got {num_steps}"
        logger.error(msg)
        raise ValueError(msg)

    def _prox_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return apply_prox_steps(x, y, A_fn, At_fn, num_steps=num_steps, lam=lam)

    return _prox_fn
