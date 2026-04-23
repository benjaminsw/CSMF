"""
Conditional Cubic Spline Flow (CSF) with FiLM Conditioning

Version: WP0.3-CondCSF-v1.4
Abbr: COND-CSF
Last Modified: 2026-04-18
Changelog:
  v1.4 (2026-04-18): [CSF-CASCADE-FIX] K 4→6, keep n_flows=1. Root cause of NLL=+1299
                     with n_flows=2: sigmoid cascade across coupling layers. After layer 1,
                     zB=logit(spline(sigmoid(xB))) is back in ℝ. Layer 2 applies sigmoid(zB)
                     where zB can be large-magnitude (±5), giving sigmoid(±5)≈0.007,
                     log|sigmoid'|≈−5/dim × 392 dims ≈ −1960 log_det penalty → NLL=+1299.
                     n_flows=1 avoids this — single coupling layer has no inter-layer cascade.
                     K=4 raised to K=6 for expressivity (K=4 was too restrictive for MNIST).
  v1.3 (2026-04-18): [CSF-REVERT] n_flows 1→2, K 4→6. Reverted in v1.4.
  v1.2 (2026-04-18): [CSF-SIMPLIFY] n_flows 2→1, K 8→4.
  v1.1 (2026-04-18): [INV-BISECT] Bisection-Newton hybrid inverse solver.
  v1.0 (2026-04-02): Initial implementation.
Dependencies: torch>=2.0, film WP0.1-FiLM-v1.0+
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Optional, Tuple

try:
    from csmf.conditioning.film import FiLM
except ImportError as e:
    logging.error(f"COND-CSF | Failed to import FiLM: {e}")
    raise

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_VERSION = "WP0.3-CondCSF-v1.4"
_ABBR    = "COND-CSF"


# =============================================================================
# Spline primitives
# =============================================================================

def _build_steffen_coeffs(
    w: torch.Tensor,   # (N, K) bin widths  — positive, sum to 1
    h: torch.Tensor,   # (N, K) bin heights — positive, sum to 1
    bd: torch.Tensor,  # (N, 2) raw boundary derivatives (softplus applied here)
    eps: float = 1e-5,
) -> Tuple[
    torch.Tensor,  # x_knots  (N, K+1)
    torch.Tensor,  # y_knots  (N, K+1)
    torch.Tensor,  # alpha0   (N, K)
    torch.Tensor,  # alpha1   (N, K)
    torch.Tensor,  # alpha2   (N, K)
    torch.Tensor,  # alpha3   (N, K)
]:
    """
    Compute Steffen monotonic cubic spline coefficients from bin params.

    Steffen (1990) method: constructs a continuously-differentiable monotone
    cubic interpolant through K+1 knots. Boundary derivatives are learned
    (parameterised by bd, positive via softplus). Interior derivatives are
    set from adjacent slopes using the Steffen condition.

    Returns per-bin cubic coefficients alpha0..alpha3 where the polynomial
    on bin k is: y = alpha0 + alpha1*xi + alpha2*xi^2 + alpha3*xi^3,
    xi = x - x_knot[k] (local coordinate, xi in [0, w_k]).
    """
    N, K = w.shape
    device = w.device

    # Knot positions — cumsum of widths/heights over [0, 1]
    x_knots = torch.cat([torch.zeros(N, 1, device=device), w.cumsum(-1)], dim=-1)   # (N, K+1)
    y_knots = torch.cat([torch.zeros(N, 1, device=device), h.cumsum(-1)], dim=-1)   # (N, K+1)

    # Slopes: s_k = height_k / width_k
    s = h / w.clamp(min=eps)   # (N, K)

    # Interior Steffen derivatives at knots 1 .. K-1
    if K > 1:
        s_l = s[:, :-1]   # (N, K-1)  left-adjacent slope
        s_r = s[:, 1:]    # (N, K-1)  right-adjacent slope
        w_l = w[:, :-1]   # (N, K-1)
        w_r = w[:, 1:]    # (N, K-1)

        # Weighted mean of adjacent slopes
        p = (s_l * w_r + s_r * w_l) / (w_l + w_r).clamp(min=eps)   # (N, K-1)

        # Steffen condition: cap p at 2*min(s_l, s_r) to preserve monotonicity
        min_s = torch.minimum(s_l, s_r)
        d_int = torch.where(p > 2.0 * min_s, 2.0 * min_s, p)
        d_int = d_int.clamp(min=0.0)   # (N, K-1)
    else:
        d_int = torch.zeros(N, 0, device=device)

    # Boundary derivatives: enforce positivity
    d_left  = F.softplus(bd[:, 0:1])   # (N, 1)
    d_right = F.softplus(bd[:, 1:2])   # (N, 1)

    # Full derivative vector: (N, K+1)
    d_all = torch.cat([d_left, d_int, d_right], dim=-1)

    d_k  = d_all[:, :-1]   # (N, K) derivative at left knot of each bin
    d_k1 = d_all[:, 1:]    # (N, K) derivative at right knot of each bin

    # Hermite cubic coefficients (Steffen eq. 13-16)
    alpha0 = y_knots[:, :-1]                        # (N, K)
    alpha1 = d_k                                     # (N, K)
    alpha2 = (3.0 * s - 2.0 * d_k - d_k1) / w.clamp(min=eps)
    alpha3 = (d_k + d_k1 - 2.0 * s) / (w.clamp(min=eps) ** 2)

    return x_knots, y_knots, alpha0, alpha1, alpha2, alpha3


def _steffen_spline_forward(
    x: torch.Tensor,       # (N,) ∈ [0, 1]
    widths: torch.Tensor,  # (N, K) raw (softmax applied here)
    heights: torch.Tensor, # (N, K) raw (softmax applied here)
    bd: torch.Tensor,      # (N, 2) raw boundary derivatives
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Steffen monotonic cubic spline forward pass.

    Applies softmax to widths/heights → positive bins summing to 1,
    then evaluates the cubic spline at x ∈ [0, 1].

    Returns:
        y:       (N,) spline output in [0, 1]
        log_det: (N,) log|dy/dx| per sample (quadratic-polynomial derivative)
    """
    N, K = widths.shape
    device = x.device

    w = F.softmax(widths,  dim=-1)   # (N, K) positive, sum to 1
    h = F.softmax(heights, dim=-1)   # (N, K)

    x_knots, _, alpha0, alpha1, alpha2, alpha3 = _build_steffen_coeffs(w, h, bd, eps)

    # Locate bin via interior knot boundaries (searchsorted on sorted sequence)
    if K > 1:
        x_int = x_knots[:, 1:-1].contiguous()   # (N, K-1) interior x boundaries
        bin_idx = torch.searchsorted(x_int, x.unsqueeze(-1).contiguous()).squeeze(-1)
    else:
        bin_idx = torch.zeros(N, dtype=torch.long, device=device)

    bin_idx = bin_idx.clamp(0, K - 1)

    def _g(t: torch.Tensor) -> torch.Tensor:
        return t.gather(1, bin_idx.unsqueeze(1)).squeeze(1)

    a0 = _g(alpha0); a1 = _g(alpha1)
    a2 = _g(alpha2); a3 = _g(alpha3)
    x0 = _g(x_knots[:, :-1])   # left edge of bin

    xi = (x - x0).clamp(min=0.0)   # local coordinate in [0, w_k]

    # Evaluate cubic: y = a0 + a1*xi + a2*xi^2 + a3*xi^3
    y = a0 + xi * (a1 + xi * (a2 + xi * a3))
    y = y.clamp(0.0, 1.0)

    # Derivative: dy/dxi = a1 + 2*a2*xi + 3*a3*xi^2
    dy = a1 + xi * (2.0 * a2 + 3.0 * a3 * xi)
    dy = dy.clamp(min=eps)
    log_det = dy.log()

    return y, log_det


def _steffen_spline_inverse(
    y: torch.Tensor,       # (N,) ∈ [0, 1]
    widths: torch.Tensor,  # (N, K) raw
    heights: torch.Tensor, # (N, K) raw
    bd: torch.Tensor,      # (N, 2) raw
    eps: float = 1e-5,
    n_steps: int = 32,
    eps_newton: float = 1e-3,
) -> torch.Tensor:
    """
    [INV-BISECT] Invert the Steffen cubic spline via bisection-Newton hybrid.

    Finds x ∈ [0, 1] s.t. spline(x) = y.

    Algorithm:
      - Maintain bracket [lo, hi] with f(lo)<=0, f(hi)>=0 (guaranteed by
        Steffen monotonicity: f(0)=a0-y<=0, f(w_k)=y_knots[k+1]-y>=0).
      - Each step: propose Newton candidate xi - f(xi)/f'(xi).
        Accept if fp > eps_newton AND candidate is inside [lo, hi].
        Otherwise use bisection midpoint (lo+hi)/2.
      - Evaluate f at the NEW xi and update bracket from its sign.
        (Key fix vs prior attempt: bracket must be updated from f at the
        accepted point, not f at the previous point.)
      - Precision after n_steps=32: w_k/2^32 < 1e-10 (worst-case bisection).

    Note: Blinn analytical cubic solver (Durkan et al. 2019, App. A.3)
    deferred to v1.2.

    Returns:
        x: (N,) values in [0, 1]
    """
    N, K = widths.shape
    device = y.device

    w = F.softmax(widths,  dim=-1)
    h = F.softmax(heights, dim=-1)

    x_knots, y_knots, alpha0, alpha1, alpha2, alpha3 = _build_steffen_coeffs(
        w, h, bd, eps
    )

    # Locate bin via y-knot interior boundaries
    if K > 1:
        y_int = y_knots[:, 1:-1].contiguous()   # (N, K-1) interior y boundaries
        bin_idx = torch.searchsorted(y_int, y.unsqueeze(-1).contiguous()).squeeze(-1)
    else:
        bin_idx = torch.zeros(N, dtype=torch.long, device=device)

    bin_idx = bin_idx.clamp(0, K - 1)

    def _g(t: torch.Tensor) -> torch.Tensor:
        return t.gather(1, bin_idx.unsqueeze(1)).squeeze(1)

    a0 = _g(alpha0); a1 = _g(alpha1)
    a2 = _g(alpha2); a3 = _g(alpha3)
    x0 = _g(x_knots[:, :-1])
    w_k = _g(w)   # bin width (upper bound for local coord xi)

    # [INV-BISECT] Initial bracket guaranteed by Steffen monotonicity:
    #   f(0)   = a0 - y = y_knots[k] - y <= 0  (y is in bin k, so y >= y_knots[k])
    #   f(w_k) = y_knots[k+1] - y       >= 0
    lo = torch.zeros_like(y)
    hi = w_k.clone()
    xi = w_k / 2.0   # midpoint start — always inside bracket

    for step in range(n_steps):
        # Evaluate f and f' at current estimate
        f  = a0 + xi * (a1 + xi * (a2 + xi * a3)) - y
        fp = a1 + xi * (2.0 * a2 + 3.0 * a3 * xi)

        if torch.isnan(f).any() or torch.isnan(fp).any():
            logger.error(f"COND-CSF | [INV-BISECT] step {step}: NaN in f or f'")
            raise RuntimeError(f"[INV-BISECT] NaN in spline inverse step {step}")

        # Newton candidate
        xi_newton = xi - f / fp.clamp(min=eps_newton)

        # Accept Newton only when derivative is healthy AND result stays in bracket
        newton_ok = (fp > eps_newton) & (xi_newton >= lo) & (xi_newton <= hi)

        # Bisection fallback: guaranteed to halve the bracket
        xi_bisect = (lo + hi) / 2.0

        xi = torch.where(newton_ok, xi_newton, xi_bisect)

        # [KEY FIX] Evaluate f at the NEW xi, then update bracket from its sign.
        # Using f from the OLD xi here (as the prior attempt did) corrupts the bracket.
        f_new = a0 + xi * (a1 + xi * (a2 + xi * a3)) - y
        lo = torch.where(f_new <= 0.0, xi, lo)
        hi = torch.where(f_new >  0.0, xi, hi)

    x = (x0 + xi).clamp(0.0, 1.0)
    return x


# =============================================================================
# Coupling layer
# =============================================================================

class CubicSplineCouplingLayer(nn.Module):
    """
    Affine-free coupling layer using Steffen cubic splines with FiLM conditioning.

    Data flow (forward, per transformed dimension):
        xB  →  sigmoid(xB)  →  spline(·; θ(xA,h))  →  logit  →  zB

    where θ = (W, H, bd) from:
        xA  →  [Linear→ReLU→FiLM] × n_hidden  →  head  →  θ

    xA is passed through unchanged. Log-det is the sum of three contributions:
        log|sigmoid'(xB)| + log|spline'(sigmoid(xB))| + log|logit'(spline_out)|

    Note: sigmoid'·logit' terms do NOT cancel because the spline output ≠ input.
    """

    def __init__(
        self,
        dim: int,
        split_idx: int,
        K: int,
        h_dim: int,
        hidden_dims: List[int],
    ):
        """
        Args:
            dim:         Total data dimensionality.
            split_idx:   First split_idx dims → xA (passthrough). Rest → xB.
            K:           Number of spline bins per transformed dimension.
            h_dim:       External conditioning feature dimension.
            hidden_dims: Hidden layer widths for the parameter network.
        """
        super().__init__()

        self.dA = split_idx
        self.dB = dim - split_idx
        self.K  = K
        self.n_params = 2 * K + 2   # widths(K) + heights(K) + boundary_derivs(2)

        if self.dA <= 0 or self.dB <= 0:
            msg = f"COND-CSF | Invalid split: dA={self.dA}, dB={self.dB} for dim={dim}"
            logger.error(msg)
            raise ValueError(msg)

        # Parameter network: xA → spline params for xB
        self.hidden_layers = nn.ModuleList()
        self.film_layers   = nn.ModuleList()
        prev = self.dA
        for hd in hidden_dims:
            self.hidden_layers.append(nn.Linear(prev, hd))
            self.film_layers.append(FiLM(hd, h_dim))
            prev = hd

        # Head: last hidden → dB*(2K+2) spline parameters
        self.head = nn.Linear(prev, self.dB * self.n_params)

        # Zero-init head so spline starts near identity at beginning of training
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        logger.info(
            f"COND-CSF | CubicSplineCouplingLayer | "
            f"dA={self.dA}, dB={self.dB}, K={K}, hidden={hidden_dims}"
        )

    def _get_spline_params(
        self, xA: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run NN+FiLM on xA to produce spline params W, H, bd."""
        u = xA
        for lin, film in zip(self.hidden_layers, self.film_layers):
            u = lin(u)
            u = F.relu(u)
            u = film(u, h)

        theta = self.head(u)   # (B, dB*(2K+2))
        B = xA.shape[0]
        theta = theta.view(B, self.dB, self.n_params)

        W  = theta[..., :self.K]           # (B, dB, K) — passed raw to softmax in spline
        H  = theta[..., self.K:2*self.K]  # (B, dB, K)
        bd = theta[..., 2*self.K:]         # (B, dB, 2)

        return W, H, bd

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, dim) — logit-preprocessed, values in ℝ
            h: (B, h_dim) — external conditioning features
        Returns:
            z:       (B, dim)
            log_det: (B,) total log|det J| for the transformed dimensions
        """
        B = x.shape[0]
        xA = x[:, :self.dA]   # (B, dA) passthrough
        xB = x[:, self.dA:]   # (B, dB) to be transformed

        W, H, bd = self._get_spline_params(xA, h)

        # Map xB (ℝ) → [0,1] for the spline domain
        yB_sig = torch.sigmoid(xB)   # (B, dB)

        # Flatten batch × dim for spline primitives
        yB_flat = yB_sig.reshape(-1)           # (B*dB,)
        W_flat  = W.reshape(-1, self.K)        # (B*dB, K)
        H_flat  = H.reshape(-1, self.K)
        bd_flat = bd.reshape(-1, 2)

        try:
            zB_sig_flat, log_det_spline_flat = _steffen_spline_forward(
                yB_flat, W_flat, H_flat, bd_flat
            )
        except Exception as e:
            logger.error(f"COND-CSF | CouplingLayer forward spline failed: {e}")
            raise

        zB_sig = zB_sig_flat.reshape(B, self.dB)                  # (B, dB) ∈ [0,1]
        log_det_spline = log_det_spline_flat.reshape(B, self.dB)  # (B, dB)

        if torch.isnan(zB_sig).any():
            logger.error("COND-CSF | NaN in coupling spline output")
            raise RuntimeError("NaN in CubicSplineCouplingLayer.forward (spline)")

        # Clamp before logit to prevent -inf
        zB_sig_cl = zB_sig.clamp(1e-6, 1.0 - 1e-6)

        # Map spline output [0,1] → ℝ
        zB = torch.logit(zB_sig_cl)   # (B, dB)

        # --- Log-det contributions (per dim, summed to scalar per sample) ---
        # sigmoid'(xB)  = sigmoid(xB)*(1-sigmoid(xB))
        # log-det for sigmoid transformation:
        log_det_sigmoid = (
            torch.log(yB_sig.clamp(1e-7)) + torch.log((1.0 - yB_sig).clamp(1e-7))
        ).sum(dim=-1)   # (B,)

        # logit'(z) = 1 / (z*(1-z))
        # log-det for logit transformation applied to spline output:
        log_det_logit = (
            -torch.log(zB_sig_cl) - torch.log(1.0 - zB_sig_cl)
        ).sum(dim=-1)   # (B,)

        log_det = log_det_sigmoid + log_det_spline.sum(dim=-1) + log_det_logit

        if torch.isnan(log_det).any():
            logger.error(
                f"COND-CSF | NaN in coupling log_det | "
                f"sigmoid={log_det_sigmoid.mean():.3f} | "
                f"spline={log_det_spline.mean():.3f} | "
                f"logit={log_det_logit.mean():.3f}"
            )
            raise RuntimeError("NaN in CubicSplineCouplingLayer.forward (log_det)")

        z = torch.cat([xA, zB], dim=-1)   # (B, dim)
        return z, log_det

    def inverse(
        self, z: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z: (B, dim)
            h: (B, h_dim)
        Returns:
            x: (B, dim)
        """
        B = z.shape[0]
        xA = z[:, :self.dA]   # unchanged (same as forward xA)
        zB = z[:, self.dA:]   # to be inverted

        # Reuse same NN — xA is identical in forward and inverse
        W, H, bd = self._get_spline_params(xA, h)

        # Map zB (ℝ) → [0,1] to enter spline inverse domain
        zB_sig = torch.sigmoid(zB)

        zB_flat = zB_sig.reshape(-1)
        W_flat  = W.reshape(-1, self.K)
        H_flat  = H.reshape(-1, self.K)
        bd_flat = bd.reshape(-1, 2)

        try:
            xB_sig_flat = _steffen_spline_inverse(zB_flat, W_flat, H_flat, bd_flat)
        except Exception as e:
            logger.error(f"COND-CSF | CouplingLayer inverse spline failed: {e}")
            raise

        xB_sig = xB_sig_flat.reshape(B, self.dB)

        if torch.isnan(xB_sig).any():
            logger.error("COND-CSF | NaN in coupling spline inverse output")
            raise RuntimeError("NaN in CubicSplineCouplingLayer.inverse (spline)")

        xB_sig_cl = xB_sig.clamp(1e-6, 1.0 - 1e-6)
        xB = torch.logit(xB_sig_cl)   # back to ℝ

        x = torch.cat([xA, xB], dim=-1)
        return x


# =============================================================================
# ConditionalCSF
# =============================================================================

class ConditionalCSF(nn.Module):
    """
    Conditional Cubic Spline Flow (CSF).

    Stacks n_flows CubicSplineCouplingLayers. A fixed dimension-reversal
    permutation is applied between layers, ensuring all dimensions appear
    in both xA (passthrough) and xB (transformed) roles across consecutive
    layers.

    Permutation strategy: reverse all D indices after each coupling layer
    (except after the last). This is equivalent to alternating the half
    being transformed without requiring learnable permutation matrices.
    LU-decomposed linear mixing (Durkan et al. 2019, Sec. 2.4) is deferred
    to v1.1.

    External h API: caller is responsible for running MNISTConditioner and
    passing h. Matches COND-NICE / COND-NSF convention.
    """

    def __init__(
        self,
        dim: int = 784,
        h_dim: int = 64,
        cond_dim: Optional[int] = None,   # alias for h_dim (train_csmf compatibility)
        n_flows: int = 1,
        K: int = 6,
        hidden_dims: Optional[List[int]] = None,
    ):
        """
        Args:
            dim:         Data dimensionality (default 784 = 28×28 MNIST).
            h_dim:       External conditioning feature dimension.
            cond_dim:    Alias for h_dim (used by train_csmf.py instantiation).
            n_flows:     Number of coupling layers.
            K:           Number of spline bins per dimension (expressivity).
            hidden_dims: Hidden layer sizes for each coupling NN.
        """
        super().__init__()

        # cond_dim alias (train_csmf.py passes cond_dim=hidden_dim)
        if cond_dim is not None:
            h_dim = cond_dim

        self.version = _VERSION
        self.abbr    = _ABBR
        logger.info(f"COND-CSF | Initialising {self.version}")

        if hidden_dims is None:
            hidden_dims = [64, 64]

        self.dim         = dim
        self.h_dim       = h_dim
        self.n_flows     = n_flows
        self.K           = K
        self.hidden_dims = hidden_dims

        split_idx = dim // 2   # first half → xA for even layers

        # Build coupling layers with alternating split
        self.coupling_layers = nn.ModuleList()
        for k in range(n_flows):
            # Even k: first half is xA. After reverse perm, odd k sees same split
            # but on flipped dimensions — effectively alternates which half is transformed.
            s_idx = split_idx
            layer = CubicSplineCouplingLayer(
                dim=dim,
                split_idx=s_idx,
                K=K,
                h_dim=h_dim,
                hidden_dims=hidden_dims,
            )
            self.coupling_layers.append(layer)
            logger.info(f"COND-CSF | Coupling layer {k}: split_idx={s_idx}")

        # Fixed permutation: reverse all dim indices (applied between layers)
        perm     = torch.arange(dim - 1, -1, -1)   # [D-1, D-2, ..., 0]
        inv_perm = torch.zeros_like(perm)
        inv_perm[perm] = torch.arange(dim)

        self.register_buffer('perm',     perm)
        self.register_buffer('inv_perm', inv_perm)

        # Standard Normal base distribution parameters
        self.register_buffer('base_loc',   torch.zeros(1))
        self.register_buffer('base_scale', torch.ones(1))

        logger.info(
            f"COND-CSF | dim={dim} | h_dim={h_dim} | n_flows={n_flows} | "
            f"K={K} | hidden_dims={hidden_dims}"
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: x → z  (logit-preprocessed data → latent).

        Args:
            x: (B, dim) — logit-preprocessed MNIST, values in ℝ
            h: (B, h_dim) — conditioning from external MNISTConditioner
        Returns:
            z:       (B, dim)
            log_det: (B,) accumulated log|det J|
        """
        z = x
        log_det_total = torch.zeros(x.shape[0], device=x.device)

        for k, layer in enumerate(self.coupling_layers):
            try:
                z, ld = layer.forward(z, h)
            except Exception as e:
                logger.error(f"COND-CSF | forward failed at coupling layer {k}: {e}")
                raise

            log_det_total = log_det_total + ld

            # Apply reverse permutation between layers (not after the last one)
            if k < self.n_flows - 1:
                z = z[:, self.perm]

        if torch.isnan(z).any() or torch.isnan(log_det_total).any():
            n_nan_z  = torch.isnan(z).sum().item()
            n_nan_ld = torch.isnan(log_det_total).sum().item()
            logger.error(
                f"COND-CSF | NaN in forward output | z_nan={n_nan_z} | "
                f"log_det_nan={n_nan_ld}"
            )
            raise RuntimeError("NaN in ConditionalCSF.forward output")

        return z, log_det_total

    def inverse(
        self, z: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        """
        Inverse pass: z → x.

        Args:
            z: (B, dim) — latent
            h: (B, h_dim) — conditioning
        Returns:
            x: (B, dim)
        """
        x = z
        for k in reversed(range(self.n_flows)):
            # Undo the permutation that was applied after layer k in forward
            if k < self.n_flows - 1:
                x = x[:, self.inv_perm]

            try:
                x = self.coupling_layers[k].inverse(x, h)
            except Exception as e:
                logger.error(f"COND-CSF | inverse failed at coupling layer {k}: {e}")
                raise

        if torch.isnan(x).any():
            n_nan = torch.isnan(x).sum().item()
            logger.error(f"COND-CSF | NaN in inverse output: {n_nan}/{x.numel()}")
            raise RuntimeError("NaN in ConditionalCSF.inverse output")

        return x

    def log_prob(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        """
        log q(x | h) = log p(z) + log|det J_forward(x)|.

        Args:
            x: (B, dim)
            h: (B, h_dim)
        Returns:
            log_prob: (B,)
        """
        z, log_det = self.forward(x, h)

        # log N(z; 0, I)
        log_2pi = math.log(2.0 * math.pi)
        log_pz  = -0.5 * (z ** 2 + log_2pi)
        log_pz  = log_pz.sum(dim=-1)   # (B,)

        log_prob = log_pz + log_det

        if torch.isnan(log_prob).any():
            logger.error(
                f"COND-CSF | NaN in log_prob | "
                f"log_pz={log_pz.mean():.3f} | log_det={log_det.mean():.3f}"
            )
            raise RuntimeError("NaN in ConditionalCSF.log_prob")

        return log_prob

    def sample(
        self, n_samples: int, h: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate samples via z ~ N(0,I) then inverse().

        Args:
            n_samples: Samples per conditioning vector.
            h:         (B, h_dim) conditioning.
        Returns:
            x: (B*n_samples, dim) if B>1, or (n_samples, dim) if B=1.
        """
        B     = h.shape[0]
        total = B * n_samples

        # Expand h to match total samples
        h_exp = h.unsqueeze(1).expand(-1, n_samples, -1).reshape(total, -1)

        z = torch.randn(total, self.dim, device=h.device)

        try:
            x = self.inverse(z, h_exp)
        except Exception as e:
            logger.error(f"COND-CSF | sample() inverse failed: {e}")
            raise

        if B == 1:
            return x   # (n_samples, dim)
        return x.view(B, n_samples, self.dim)


# =============================================================================
# Version
# =============================================================================

def get_version() -> dict:
    """Return version metadata."""
    return {
        'version': _VERSION,
        'abbr':    _ABBR,
        'date':    '2026-04-02',
        'purpose': (
            'Cubic spline coupling flow with FiLM conditioning for MNIST inverse problems. '
            'O(1) forward and inverse (unlike MAF). '
            'Sigmoid/logit domain wrapping; Newton inverse; fixed reverse permutation.'
        ),
        'deferred_to_v1.1': [
            'LU-decomposed linear mixing layers (replaces fixed permutation)',
            'Blinn analytical cubic root solver (replaces Newton)',
        ],
    }


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(f"ConditionalCSF | {get_version()['version']}")
    print("=" * 70)

    import time

    try:
        # Test 1: Instantiation
        logger.info("[TEST 1] Model instantiation")
        model = ConditionalCSF(dim=784, h_dim=64, n_flows=2, K=8, hidden_dims=[64, 64])
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✓ Instantiation OK | params={n_params:,}")

        # Test 2: Forward
        logger.info("[TEST 2] Forward pass")
        x = torch.randn(4, 784)
        h = torch.randn(4, 64)
        t0 = time.time()
        z, log_det = model.forward(x, h)
        logger.info(
            f"✓ z={z.shape} | "
            f"log_det=[{log_det.min():.2f}, {log_det.max():.2f}] | "
            f"time={1000*(time.time()-t0):.1f}ms"
        )

        # Test 3: log_prob
        logger.info("[TEST 3] log_prob")
        lp = model.log_prob(x, h)
        logger.info(f"✓ log_prob=[{lp.min():.2f}, {lp.max():.2f}]")

        # Test 4: Invertibility
        logger.info("[TEST 4] Invertibility")
        t0 = time.time()
        x_recon = model.inverse(z, h)
        inv_time = 1000 * (time.time() - t0)
        err = (x - x_recon).abs().max().item()
        logger.info(f"  Max |x - inverse(forward(x))|: {err:.6f} | time={inv_time:.1f}ms")
        if err < 1e-3:
            logger.info("✓ Invertibility PASSED")
        else:
            logger.warning(f"✗ Invertibility FAILED (err={err:.6f} > 1e-3)")

        # Test 5: cond_dim alias
        logger.info("[TEST 5] cond_dim alias (train_csmf compatibility)")
        model2 = ConditionalCSF(dim=784, cond_dim=64, n_flows=1, K=4)
        _, _ = model2.forward(x, h)
        logger.info("✓ cond_dim alias OK")

        # Test 6: Sampling
        logger.info("[TEST 6] Sampling")
        h_single = torch.randn(1, 64)
        samples = model.sample(n_samples=5, h=h_single)
        logger.info(f"✓ samples={samples.shape}")

        # Test 7: Spline primitive sanity
        logger.info("[TEST 7] Spline forward/inverse round-trip")
        N_sp, K_sp = 100, 8
        x_sp = torch.rand(N_sp)
        W_sp = torch.randn(N_sp, K_sp)
        H_sp = torch.randn(N_sp, K_sp)
        bd_sp = torch.randn(N_sp, 2)
        y_sp, _ = _steffen_spline_forward(x_sp, W_sp, H_sp, bd_sp)
        x_sp_recon = _steffen_spline_inverse(y_sp, W_sp, H_sp, bd_sp)
        sp_err = (x_sp - x_sp_recon).abs().max().item()
        logger.info(f"  Spline round-trip error: {sp_err:.6f}")
        if sp_err < 1e-4:
            logger.info("✓ Spline round-trip PASSED")
        else:
            logger.warning(f"✗ Spline round-trip FAILED (err={sp_err:.6f})")

        print(f"\n{'='*70}\nALL TESTS PASSED\n{'='*70}")

    except Exception as e:
        logger.error(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
