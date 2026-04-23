# ==============================================================================
# File    : test_csf_film_mnist.py
# Abbr    : TEST-CSF-FILM
# Version : v1.2
# Created : 2026-04-16
# Changelog:
#   v1.2 (2026-04-16): [TAIL] Removed sigmoid/logit wrapping from coupling
#                      layer — per-layer logit log_det compounds across 10
#                      steps exploiting boundary values (NLL=1.5×10⁷). Replaced
#                      with linear tails outside [-TAIL_BOUND, TAIL_BOUND]=15.0
#                      matching NSF v1.1 approach. Spline now maps [-B,B]→[-B,B]
#                      with identity outside. Removed _logit_log_det(), sigmoid/
#                      logit steps from forward/inverse; updated spline params
#                      to use 2B-scaled widths/heights for [-B,B] domain.
#   v1.1 (2026-04-16): [BUG] Fixed polynomial coefficients for normalized xi:
#                      rescaled b1=dk*wk, b2=(3sk-2dk-dk1)*wk, b3=dk+dk1-2sk;
#                      dy/dx = (b1+2b2*xi+3b3*xi²)/wk. Old unnormalized
#                      coefficients caused invertibility error=113 and NaN.
#   v1.0 (2026-04-16): Initial standalone CSF+FiLM MNIST reconstruction test.
# ==============================================================================

import csv
import logging
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ==============================================================================
# CONFIG  (CSF paper spec adapted for raw logit-space MNIST pixels)
# ==============================================================================
N_STEPS    = 10      # flow steps (paper: 10 for tabular)
N_BINS     = 8       # cubic spline bins K
TAIL_BOUND = 15.0    # [TAIL] v1.2: linear tails outside [-B,B]; covers logit MNIST [-13.8,13.8]
HIDDEN     = 128     # coupling ResNet hidden
N_BLOCKS   = 2       # residual blocks per coupling NN (paper: 2)
COND_DIM   = 128     # h dimension
DIM        = 784     # MNIST 28×28 flattened
BATCH_SIZE = 256
EPOCHS     = 60
LR         = 5e-4    # paper: Adam + cosine decay
LOGIT_EPS  = 1e-6    # clamp for logit preprocessing (input pipeline only)
BLUR_K     = 5
BLUR_S     = 1.5
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR   = "./data"
LOG_DIR    = "./tests/logs/csf_film_mnist"
SAVE_PATH  = os.path.join(LOG_DIR, "best_checkpoint.pth")
METRICS_CSV = os.path.join(LOG_DIR, "metrics.csv")
PLOT_EVERY = 5

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "run.log"), mode="a"),
    ],
)
logger = logging.getLogger("TEST-CSF-FILM")


# ==============================================================================
# HELPERS
# ==============================================================================
def logit_preprocess(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize + logit. x: (B,1,28,28) or (B,D) in [0,1]."""
    x = x + torch.zeros_like(x).uniform_(0, 1.0 / 256)
    x = x.clamp(LOGIT_EPS, 1 - LOGIT_EPS)
    log_det = (-torch.log(x) - torch.log(1 - x)).sum(dim=1)
    return torch.log(x) - torch.log(1 - x), log_det


def sigmoid_postprocess(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def gaussian_log_prob(z: torch.Tensor) -> torch.Tensor:
    """Standard Gaussian log-prob summed over D. Returns (B,)."""
    return -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(dim=1)


def gaussian_blur_batch(x: torch.Tensor,
                         kernel_size: int = BLUR_K,
                         sigma: float = BLUR_S) -> torch.Tensor:
    """Gaussian blur (B,1,28,28) → (B,1,28,28)."""
    pad    = kernel_size // 2
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - pad
    g      = torch.exp(-0.5 * (coords / sigma) ** 2)
    g      = g / g.sum()
    k2d    = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
    return F.conv2d(x, k2d, padding=pad)


# ==============================================================================
# CUBIC SPLINE (Steffen's method, Appendix A of CSF paper)
# Maps [0,1] → [0,1]. Parameterised by K+1 knots + boundary derivatives.
# ==============================================================================
def _steffen_derivatives(widths: torch.Tensor,
                          heights: torch.Tensor) -> torch.Tensor:
    """
    Compute Steffen derivatives at internal knots (Eq. 17-18 in CSF paper).
    Args:
        widths:  (B, D, K)   bin widths  (positive, sum to 1)
        heights: (B, D, K)   bin heights (positive, sum to 1)
    Returns:
        d: (B, D, K+1)  derivatives at all K+1 knots
           boundary derivatives d[0]=d[K]=1 (match slope of linear extensions)
    """
    # slopes s_k = h_k / w_k
    s = heights / widths.clamp(min=1e-8)   # (B, D, K)

    # weighted harmonic mean p_k for internal knots k=1..K-1
    w_l = widths[:, :, :-1]   # w_{k-1}
    w_r = widths[:, :, 1:]    # w_{k}
    s_l = s[:, :, :-1]        # s_{k-1}
    s_r = s[:, :, 1:]         # s_{k}

    p = (s_l * w_r + s_r * w_l) / (w_l + w_r).clamp(min=1e-8)   # (B, D, K-1)

    # Steffen monotonicity constraint: d_k = min(p_k, 2*min(s_{k-1}, s_k))
    d_internal = torch.where(
        p > 2 * torch.minimum(s_l, s_r),
        2 * torch.minimum(s_l, s_r),
        p
    )   # (B, D, K-1)

    # Boundary derivatives fixed to 1 (match linear tails outside [0,1])
    ones = torch.ones(*widths.shape[:2], 1, device=widths.device, dtype=widths.dtype)
    d    = torch.cat([ones, d_internal, ones], dim=-1)   # (B, D, K+1)
    return d


def cubic_spline_forward(x: torch.Tensor,
                          widths: torch.Tensor,
                          heights: torch.Tensor,
                          tail_bound: float = TAIL_BOUND) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forward cubic spline with linear tails ([-B,B] domain).
    [TAIL] v1.2: Replaces [0,1] sigmoid/logit wrapper — directly handles
    unconstrained input. Identity outside [-B,B] (linear tails, log_det=0).

    widths/heights: (B, D, K) softmax-scaled × 2B (span [-B,B]).
    """
    B, D = x.shape
    K    = widths.shape[-1]

    # Knot positions spanning [-B, B]
    x_knots = torch.cat([
        -tail_bound * torch.ones(B, D, 1, device=x.device),
        -tail_bound + torch.cumsum(widths, dim=-1)
    ], dim=-1)   # (B, D, K+1)

    y_knots = torch.cat([
        -tail_bound * torch.ones(B, D, 1, device=x.device),
        -tail_bound + torch.cumsum(heights, dim=-1)
    ], dim=-1)   # (B, D, K+1)

    d = _steffen_derivatives(widths, heights)   # (B, D, K+1)

    # Linear tails: identity outside [-B, B]
    in_tail = (x <= -tail_bound) | (x >= tail_bound)
    x_safe  = x.clamp(-tail_bound + 1e-6, tail_bound - 1e-6)

    bin_idx = (x_safe.unsqueeze(-1) >= x_knots[:, :, :-1]).sum(dim=-1) - 1
    bin_idx = bin_idx.clamp(0, K - 1)

    def gather_bin(t):
        return t.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)

    xk  = gather_bin(x_knots[:, :, :-1])
    xk1 = gather_bin(x_knots[:, :, 1:])
    yk  = gather_bin(y_knots[:, :, :-1])
    yk1 = gather_bin(y_knots[:, :, 1:])
    dk  = gather_bin(d[:, :, :-1])
    dk1 = gather_bin(d[:, :, 1:])
    wk  = (xk1 - xk).clamp(min=1e-8)
    sk  = (yk1 - yk) / wk

    xi = ((x_safe - xk) / wk).clamp(0.0, 1.0)

    # Rescaled polynomial coefficients for normalized xi ∈ [0,1]
    b1 = dk * wk
    b2 = (3 * sk - 2 * dk - dk1) * wk
    b3 = dk + dk1 - 2 * sk

    y_in    = yk + b1 * xi + b2 * xi ** 2 + b3 * xi ** 3
    dy_dxi  = b1 + 2 * b2 * xi + 3 * b3 * xi ** 2
    dy_dx   = dy_dxi / wk
    log_d   = torch.log(dy_dx.clamp(min=1e-8))

    # Apply linear tails (identity, log_det=0)
    y_out   = torch.where(in_tail, x, y_in)
    log_det = torch.where(in_tail, torch.zeros_like(log_d), log_d)

    if torch.isnan(y_out).any() or torch.isinf(y_out).any():
        logger.error("[CubicSpline] NaN/Inf in forward output")
        raise RuntimeError("NaN/Inf in cubic_spline_forward")

    return y_out, log_det


def _solve_cubic_blinn(a3: torch.Tensor, a2: torch.Tensor,
                        a1: torch.Tensor, a0_y: torch.Tensor) -> torch.Tensor:
    """
    Solve a3*xi³ + a2*xi² + a1*xi + a0_y = 0 for xi ∈ [0,1].
    Stabilised Blinn/Peters method (CSF paper §A.3).
    Returns xi: (B, D).
    """
    # Depress: substitute xi = t - a2/(3*a3)
    eps  = 1e-8
    a3c  = a3.clamp(min=eps)   # avoid division by zero

    p = (3 * a3c * a1 - a2 ** 2) / (3 * a3c ** 2 + eps)
    q = (2 * a2 ** 3 - 9 * a3c * a2 * a1 + 27 * a3c ** 2 * a0_y) / \
        (27 * a3c ** 3 + eps)

    disc = -(4 * p ** 3 + 27 * q ** 2)

    # Three real roots when disc >= 0 (monotonic spline always has one in [0,1])
    disc_pos = (disc >= 0)

    # Trigonometric method for three real roots
    m   = 2 * torch.sqrt((-p / 3).clamp(min=eps))
    arg = (3 * q / (p * m + eps)).clamp(-1 + eps, 1 - eps)
    theta = torch.acos(arg) / 3

    # Three candidates
    t0 = m * torch.cos(theta)
    t1 = m * torch.cos(theta - 2 * math.pi / 3)
    t2 = m * torch.cos(theta - 4 * math.pi / 3)
    shift = a2 / (3 * a3c + eps)

    xi0 = (t0 - shift).clamp(0, 1)
    xi1 = (t1 - shift).clamp(0, 1)
    xi2 = (t2 - shift).clamp(0, 1)

    # Cardano formula for one real root (disc < 0 — shouldn't happen for monotonic)
    A   = -q / 2 + torch.sqrt((q ** 2 / 4 + p ** 3 / 27).clamp(min=0))
    B   = -q / 2 - torch.sqrt((q ** 2 / 4 + p ** 3 / 27).clamp(min=0))
    xi_card = (torch.sign(A) * torch.abs(A).clamp(min=eps) ** (1/3) +
               torch.sign(B) * torch.abs(B).clamp(min=eps) ** (1/3) - shift).clamp(0, 1)

    # Select root in [0,1] — pick the one closest to clamp midpoint
    # For a monotonic spline there is exactly one real root in each bin
    xi_cand = torch.where(disc_pos, xi0, xi_card)

    # Refine: pick among three candidates the one where f(xi) is closest to 0
    def residual(xi):
        return (a3 * xi ** 3 + a2 * xi ** 2 + a1 * xi + a0_y).abs()

    best = xi_cand
    for xi_c in [xi1, xi2]:
        better = residual(xi_c) < residual(best)
        best   = torch.where(better, xi_c, best)

    return best.clamp(0, 1)


def cubic_spline_inverse(y: torch.Tensor,
                          widths: torch.Tensor,
                          heights: torch.Tensor,
                          tail_bound: float = TAIL_BOUND) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Inverse cubic spline with linear tails ([-B,B] domain).
    [TAIL] v1.2: Identity outside [-B,B].
    """
    B, D = y.shape
    K    = widths.shape[-1]

    x_knots = torch.cat([
        -tail_bound * torch.ones(B, D, 1, device=y.device),
        -tail_bound + torch.cumsum(widths, dim=-1)
    ], dim=-1)

    y_knots = torch.cat([
        -tail_bound * torch.ones(B, D, 1, device=y.device),
        -tail_bound + torch.cumsum(heights, dim=-1)
    ], dim=-1)

    d = _steffen_derivatives(widths, heights)

    in_tail = (y <= -tail_bound) | (y >= tail_bound)
    y_safe  = y.clamp(-tail_bound + 1e-6, tail_bound - 1e-6)

    bin_idx = (y_safe.unsqueeze(-1) >= y_knots[:, :, :-1]).sum(dim=-1) - 1
    bin_idx = bin_idx.clamp(0, K - 1)

    def gather_bin(t):
        return t.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)

    xk  = gather_bin(x_knots[:, :, :-1])
    xk1 = gather_bin(x_knots[:, :, 1:])
    yk  = gather_bin(y_knots[:, :, :-1])
    yk1 = gather_bin(y_knots[:, :, 1:])
    dk  = gather_bin(d[:, :, :-1])
    dk1 = gather_bin(d[:, :, 1:])
    wk  = (xk1 - xk).clamp(min=1e-8)
    sk  = (yk1 - yk) / wk

    # Rescaled coefficients for normalized xi; constant = yk - y_safe
    b1   = dk * wk
    b2   = (3 * sk - 2 * dk - dk1) * wk
    b3   = dk + dk1 - 2 * sk
    b0_y = yk - y_safe

    xi = _solve_cubic_blinn(b3, b2, b1, b0_y)

    x_in    = xk + xi * wk
    dy_dxi  = b1 + 2 * b2 * xi + 3 * b3 * xi ** 2
    dy_dx   = dy_dxi / wk
    log_d   = torch.log(dy_dx.clamp(min=1e-8))

    x_out   = torch.where(in_tail, y, x_in)
    log_det = torch.where(in_tail, torch.zeros_like(log_d), log_d)

    if torch.isnan(x_out).any() or torch.isinf(x_out).any():
        logger.error("[CubicSpline] NaN/Inf in inverse output")
        raise RuntimeError("NaN/Inf in cubic_spline_inverse")

    return x_out, log_det


# ==============================================================================
# FiLM
# ==============================================================================
class FiLM(nn.Module):
    """(1 + γ(h)) ⊙ f + β(h). Identity init."""
    def __init__(self, f_dim: int, h_dim: int):
        super().__init__()
        self.gamma = nn.Linear(h_dim, f_dim)
        self.beta  = nn.Linear(h_dim, f_dim)
        nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight);  nn.init.zeros_(self.beta.bias)

    def forward(self, f: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        out = (1.0 + self.gamma(h)) * f + self.beta(h)
        if torch.isnan(out).any() or torch.isinf(out).any():
            logger.error("[FiLM] NaN/Inf in output")
            raise RuntimeError("NaN/Inf in FiLM output")
        return out


# ==============================================================================
# ResidualBlock + CouplingNN  (paper: 2 pre-activation blocks, FC)
# ==============================================================================
class ResidualBlock(nn.Module):
    """Pre-activation residual block with FiLM after each hidden activation."""
    def __init__(self, hidden: int, h_dim: int):
        super().__init__()
        self.fc1   = nn.Linear(hidden, hidden)
        self.fc2   = nn.Linear(hidden, hidden)
        self.film1 = FiLM(hidden, h_dim)
        self.film2 = FiLM(hidden, h_dim)
        self.act   = nn.ReLU()

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.film1(self.act(self.fc1(x)), h)
        out = self.film2(self.act(self.fc2(out)), h)
        return out + residual


class CouplingNN(nn.Module):
    """
    Cubic-spline parameter network.
    Outputs (2K+2) * half_dim unconstrained params per sample:
      K widths + K heights + 2 boundary derivatives per output dim.
    2-block ResNet, FiLM per block (extension).
    """
    def __init__(self, in_dim: int, out_dim: int, h_dim: int,
                 hidden: int = HIDDEN, n_blocks: int = N_BLOCKS):
        super().__init__()
        if n_blocks < 1:
            logger.error(f"[CouplingNN] n_blocks must be >= 1, got {n_blocks}")
            raise ValueError("CouplingNN requires n_blocks >= 1")

        self.fc_in  = nn.Linear(in_dim + h_dim, hidden)
        self.blocks = nn.ModuleList([ResidualBlock(hidden, h_dim)
                                     for _ in range(n_blocks)])
        self.fc_out = nn.Linear(hidden, out_dim)
        # Zero-init output: spline starts near-uniform at init
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

        logger.debug(
            f"[CouplingNN] in={in_dim}, out={out_dim}, "
            f"hidden={hidden}, n_blocks={n_blocks}, h_dim={h_dim}"
        )

    def forward(self, xA: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if xA.shape[0] != h.shape[0]:
            logger.error(
                f"[CouplingNN] Batch mismatch: xA={xA.shape[0]}, h={h.shape[0]}"
            )
            raise ValueError("Batch mismatch in CouplingNN.forward")

        out = F.relu(self.fc_in(torch.cat([xA, h], dim=1)))
        for block in self.blocks:
            out = block(out, h)
        params = self.fc_out(out)

        if torch.isnan(params).any() or torch.isinf(params).any():
            logger.error("[CouplingNN] NaN/Inf in output params")
            raise RuntimeError("NaN/Inf in CouplingNN output")
        return params


# ==============================================================================
# CubicSplineCouplingLayer
# Key CSF design: sigmoid BEFORE spline, logit AFTER — handles unconstrained
# input natively (no TAIL_BOUND / identity tails needed).
# ==============================================================================
class CubicSplineCouplingLayer(nn.Module):
    """
    Cubic-spline coupling layer (CSF paper §2.3–2.5).

    Per flow step:
      Forward:
        1. x_A_01 = sigmoid(x_A)          map to [0,1]
        2. y_A_01 = g_θ(x_A_01; x_B, h)  cubic spline in [0,1]
        3. y_A    = logit(y_A_01)         map back to unconstrained
        log_det = logit_log_det(y_A_01) + spline_log_det - logit_log_det(x_A)
      Inverse:
        1. x_A_01 = sigmoid(y_A)
        2. y_A_01 = g_θ⁻¹(x_A_01; x_B, h)
        3. x_A    = logit(y_A_01)

    swap=False: x_A=first half, x_B=second half
    swap=True:  x_A=second half, x_B=first half
    """
    def __init__(self, dim: int, cond_dim: int, n_bins: int = N_BINS,
                 tail_bound: float = TAIL_BOUND, swap: bool = False,
                 hidden: int = HIDDEN, n_blocks: int = N_BLOCKS):
        super().__init__()
        if dim % 2 != 0:
            logger.error(f"[CSFCoupling] dim must be even, got {dim}")
            raise ValueError("CubicSplineCouplingLayer requires even dim")

        self.dim        = dim
        self.half       = dim // 2
        self.swap       = swap
        self.n_bins     = n_bins
        self.tail_bound = tail_bound   # [TAIL] v1.2

        self.nn = CouplingNN(
            in_dim=self.half,
            out_dim=self.half * (2 * n_bins + 2),
            h_dim=cond_dim,
            hidden=hidden,
            n_blocks=n_blocks,
        )

        logger.debug(
            f"[CSFCoupling] dim={dim}, swap={swap}, n_bins={n_bins}, "
            f"tail_bound={tail_bound}, cond_dim={cond_dim}"
        )

    def _split(self, x: torch.Tensor):
        xA, xB = x[:, :self.half], x[:, self.half:]
        if self.swap:
            return xB, xA
        return xA, xB

    def _merge(self, xA: torch.Tensor, xB: torch.Tensor) -> torch.Tensor:
        if self.swap:
            return torch.cat([xB, xA], dim=1)
        return torch.cat([xA, xB], dim=1)

    def _get_spline_params(self, xB: torch.Tensor, h: torch.Tensor):
        """
        Parse NN output into widths, heights for [-B,B] domain.
        widths/heights scaled by 2*TAIL_BOUND so they span the full interval.
        """
        B   = xB.shape[0]
        K   = self.n_bins
        raw = self.nn(xB, h)                            # (B, half*(2K+2))
        raw = raw.reshape(B, self.half, 2 * K + 2)

        # Scale softmax output by 2*tail_bound so knots span [-B, B]
        W = F.softmax(raw[:, :, :K],    dim=-1) * 2 * self.tail_bound
        H = F.softmax(raw[:, :, K:2*K], dim=-1) * 2 * self.tail_bound
        # Boundary derivatives: softplus → positive (Steffen handles monotonicity)
        bd = F.softplus(raw[:, :, 2*K:])

        return W, H, bd

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        """
        [TAIL] v1.2: Spline operates directly on unconstrained xA with linear
        tails outside [-TAIL_BOUND, TAIL_BOUND]. No sigmoid/logit wrapping.
        Returns (y, log_det). log_det: (B,).
        """
        if x.dim() != 2 or x.shape[1] != self.dim:
            logger.error(
                f"[CSFCoupling] forward shape mismatch: "
                f"expected (*, {self.dim}), got {tuple(x.shape)}"
            )
            raise ValueError("CubicSplineCouplingLayer.forward shape mismatch")
        if x.shape[0] != h.shape[0]:
            logger.error(
                f"[CSFCoupling] Batch mismatch: x={x.shape[0]}, h={h.shape[0]}"
            )
            raise ValueError("Batch mismatch in CubicSplineCouplingLayer.forward")

        xA, xB   = self._split(x)
        W, H, _  = self._get_spline_params(xB, h)

        yA, ld   = cubic_spline_forward(xA, W, H, self.tail_bound)
        log_det  = ld.sum(dim=1)   # (B,)

        return self._merge(yA, xB), log_det

    def inverse(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """[TAIL] v1.2: Direct cubic spline inverse, no sigmoid/logit."""
        if y.dim() != 2 or y.shape[1] != self.dim:
            logger.error(
                f"[CSFCoupling] inverse shape mismatch: "
                f"expected (*, {self.dim}), got {tuple(y.shape)}"
            )
            raise ValueError("CubicSplineCouplingLayer.inverse shape mismatch")
        if y.shape[0] != h.shape[0]:
            logger.error(
                f"[CSFCoupling] Batch mismatch inverse: y={y.shape[0]}, h={h.shape[0]}"
            )
            raise ValueError("Batch mismatch in CubicSplineCouplingLayer.inverse")

        yA, yB   = self._split(y)
        W, H, _  = self._get_spline_params(yB, h)
        xA, _    = cubic_spline_inverse(yA, W, H, self.tail_bound)

        return self._merge(xA, yB)


# ==============================================================================
# ActNorm  (data-driven init, log_scale clamped [-0.5, 0.5])
# ==============================================================================
class ActNorm(nn.Module):
    """Per-dim affine normalisation. Data-driven init on first forward."""
    LOG_SCALE_CLAMP = 0.5

    def __init__(self, dim: int):
        super().__init__()
        self.dim       = dim
        self.shift     = nn.Parameter(torch.zeros(dim))
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.register_buffer('initialized', torch.tensor(False))

    @torch.no_grad()
    def _initialize(self, x: torch.Tensor) -> None:
        mean = x.mean(dim=0)
        std  = x.std(dim=0).clamp(min=1e-6)
        self.shift.data     = -mean
        self.log_scale.data = -torch.log(std)
        self.initialized.fill_(True)

    def forward(self, x: torch.Tensor):
        if x.dim() != 2 or x.shape[1] != self.dim:
            logger.error(f"[ActNorm] forward shape mismatch: got {tuple(x.shape)}")
            raise ValueError("ActNorm.forward shape mismatch")
        if not self.initialized:
            self._initialize(x)
        ls      = self.log_scale.clamp(-self.LOG_SCALE_CLAMP, self.LOG_SCALE_CLAMP)
        y       = (x + self.shift) * torch.exp(ls)
        log_det = ls.sum().expand(x.shape[0])
        if torch.isnan(y).any() or torch.isinf(y).any():
            logger.error("[ActNorm] NaN/Inf in forward")
            raise RuntimeError("NaN/Inf in ActNorm forward")
        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        if y.dim() != 2 or y.shape[1] != self.dim:
            logger.error(f"[ActNorm] inverse shape mismatch: got {tuple(y.shape)}")
            raise ValueError("ActNorm.inverse shape mismatch")
        ls = self.log_scale.clamp(-self.LOG_SCALE_CLAMP, self.LOG_SCALE_CLAMP)
        x  = y * torch.exp(-ls) - self.shift
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("[ActNorm] NaN/Inf in inverse")
            raise RuntimeError("NaN/Inf in ActNorm inverse")
        return x


# ==============================================================================
# InvLinear  (LU decomp, identity init, triangular solve inverse)
# ==============================================================================
class InvLinear(nn.Module):
    """
    Invertible linear transform via LU decomposition.
    Init: LU = I (paper §2.4). P = random permutation (fixed).
    Inverse via triangular solves — no linalg.inv.
    """
    LOG_S_CLAMP = 3.0

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        perm = torch.randperm(dim)
        P    = torch.zeros(dim, dim)
        P[torch.arange(dim), perm] = 1.0
        self.register_buffer('P', P)
        self.L     = nn.Parameter(torch.zeros(dim, dim))
        self.U     = nn.Parameter(torch.zeros(dim, dim))
        self.log_s = nn.Parameter(torch.zeros(dim))
        L_mask = torch.tril(torch.ones(dim, dim), diagonal=-1)
        self.register_buffer('L_mask', L_mask)
        self.register_buffer('eye',    torch.eye(dim))

    def _get_LU(self):
        ls = self.log_s.clamp(-self.LOG_S_CLAMP, self.LOG_S_CLAMP)
        L  = self.L * self.L_mask + self.eye
        U  = torch.triu(self.U, diagonal=1) + torch.diag(torch.exp(ls))
        return L, U

    def forward(self, x: torch.Tensor):
        if x.dim() != 2 or x.shape[1] != self.dim:
            logger.error(f"[InvLinear] forward shape mismatch: got {tuple(x.shape)}")
            raise ValueError("InvLinear.forward shape mismatch")
        L, U    = self._get_LU()
        W       = self.P @ L @ U
        y       = x @ W.T
        log_det = self.log_s.clamp(-self.LOG_S_CLAMP, self.LOG_S_CLAMP).sum()
        log_det = log_det.expand(x.shape[0])
        if torch.isnan(y).any() or torch.isinf(y).any():
            logger.error("[InvLinear] NaN/Inf in forward")
            raise RuntimeError("NaN/Inf in InvLinear forward")
        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        if y.dim() != 2 or y.shape[1] != self.dim:
            logger.error(f"[InvLinear] inverse shape mismatch: got {tuple(y.shape)}")
            raise ValueError("InvLinear.inverse shape mismatch")
        L, U = self._get_LU()
        b    = y @ self.P
        c    = torch.linalg.solve_triangular(L.T, b.T, upper=True, unitriangular=True).T
        x    = torch.linalg.solve_triangular(U.T, c.T, upper=False).T
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("[InvLinear] NaN/Inf in inverse")
            raise RuntimeError("NaN/Inf in InvLinear inverse")
        return x


# ==============================================================================
# ConditionalCSF
# N_STEPS flow steps: InvLinear → ActNorm → CubicSplineCoupling
# ==============================================================================
class ConditionalCSF(nn.Module):
    """
    Conditional Cubic-Spline Flow (CSF paper, Durkan et al. 2019).

    Architecture:
      - N_STEPS=10 flow steps
      - Each step: InvLinear (LU, identity init) → ActNorm → CubicSplineCoupling
      - CubicSplineCoupling: sigmoid→spline→logit per-dim (handles unconstrained)
      - Steffen's cubic spline: K=8 bins, [0,1]→[0,1]
      - Coupling NN: 2 pre-activation residual blocks × 128 hidden, FiLM per block
      - Alternating swap for full-dimensional mixing
      - Gaussian prior N(0,I)

    API:
      forward(x, h) → (z, log_det)
      inverse(z, h) → x  (logit-space; sigmoid applied externally by caller)
    """
    def __init__(self, dim: int = DIM, cond_dim: int = COND_DIM,
                 n_steps: int = N_STEPS, n_bins: int = N_BINS,
                 hidden: int = HIDDEN, n_blocks: int = N_BLOCKS):
        super().__init__()
        if dim % 2 != 0:
            logger.error(f"[CSF] dim must be even, got {dim}")
            raise ValueError("ConditionalCSF requires even dim")

        self.dim     = dim
        self.n_steps = n_steps

        self.inv_linears = nn.ModuleList([InvLinear(dim) for _ in range(n_steps)])
        self.actnorms    = nn.ModuleList([ActNorm(dim)   for _ in range(n_steps)])
        self.couplings   = nn.ModuleList([
            CubicSplineCouplingLayer(
                dim=dim, cond_dim=cond_dim, n_bins=n_bins,
                tail_bound=TAIL_BOUND, swap=(i % 2 == 1),
                hidden=hidden, n_blocks=n_blocks,
            )
            for i in range(n_steps)
        ])

        logger.info(
            f"[CSF] v1.0 initialized: dim={dim}, cond_dim={cond_dim}, "
            f"n_steps={n_steps}, n_bins={n_bins}, hidden={hidden}, "
            f"n_blocks={n_blocks}, actnorm=True"
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        """x: (B,D) logit-space. Returns (z, log_det)."""
        if x.dim() != 2 or x.shape[1] != self.dim:
            logger.error(f"[CSF] forward shape mismatch: got {tuple(x.shape)}")
            raise ValueError("ConditionalCSF.forward shape mismatch")
        if x.shape[0] != h.shape[0]:
            logger.error(f"[CSF] Batch mismatch: x={x.shape[0]}, h={h.shape[0]}")
            raise ValueError("Batch mismatch in ConditionalCSF.forward")

        z       = x
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        for i in range(self.n_steps):
            z, ld = self.inv_linears[i](z);   log_det += ld
            z, ld = self.actnorms[i](z);       log_det += ld
            z, ld = self.couplings[i](z, h);   log_det += ld

        if torch.isnan(z).any() or torch.isinf(z).any():
            logger.error("[CSF] NaN/Inf in forward output")
            raise RuntimeError("NaN/Inf in ConditionalCSF.forward")
        return z, log_det

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Inverse: z → x (logit-space). Sigmoid applied externally."""
        if z.dim() != 2 or z.shape[1] != self.dim:
            logger.error(f"[CSF] inverse shape mismatch: got {tuple(z.shape)}")
            raise ValueError("ConditionalCSF.inverse shape mismatch")
        if z.shape[0] != h.shape[0]:
            logger.error(f"[CSF] Batch mismatch inverse: z={z.shape[0]}, h={h.shape[0]}")
            raise ValueError("Batch mismatch in ConditionalCSF.inverse")

        x = z
        for i in reversed(range(self.n_steps)):
            x = self.couplings[i].inverse(x, h)
            x = self.actnorms[i].inverse(x)
            x = self.inv_linears[i].inverse(x)

        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("[CSF] NaN/Inf in inverse output")
            raise RuntimeError("NaN/Inf in ConditionalCSF.inverse")
        return x

    @torch.no_grad()
    def check_invertibility(self, x: torch.Tensor, h: torch.Tensor,
                             tol: float = 1e-3) -> float:
        """max ‖x - f⁻¹(f(x))‖_∞. Logs warning if > tol."""
        z, _  = self.forward(x, h)
        x_hat = self.inverse(z, h)
        err   = (x - x_hat).abs().max().item()
        if err > tol:
            logger.warning(
                f"[CSF] Invertibility FAILED: max_err={err:.3e} > tol={tol:.3e}"
            )
        else:
            logger.info(f"[CSF] Invertibility PASSED: max_err={err:.3e}")
        return err


# ==============================================================================
# CNN Conditioner
# ==============================================================================
class CNNConditioner(nn.Module):
    """4-conv CNN: y (B,1,28,28) → h ∈ R^{cond_dim}."""
    def __init__(self, cond_dim: int = COND_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32,  3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(128, cond_dim)
        self.norm = nn.LayerNorm(cond_dim)
        logger.info(f"[CNNConditioner] initialized: cond_dim={cond_dim}")

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if y.dim() != 4 or y.shape[1] != 1:
            logger.error(f"[CNNConditioner] Expected (B,1,28,28), got {tuple(y.shape)}")
            raise ValueError("CNNConditioner expects (B,1,28,28)")
        h = self.norm(self.head(self.pool(self.net(y)).squeeze(-1).squeeze(-1)))
        if torch.isnan(h).any() or torch.isinf(h).any():
            logger.error("[CNNConditioner] NaN/Inf in h output")
            raise RuntimeError("NaN/Inf in CNNConditioner output")
        return h


# ==============================================================================
# TRAINING
# ==============================================================================
def train(model: ConditionalCSF, conditioner: CNNConditioner,
          loader: DataLoader, optimizer: torch.optim.Optimizer,
          epoch: int) -> float:
    model.train(); conditioner.train()
    total_loss = 0.0; n_batches = 0

    for batch_idx, (x_pixel, _) in enumerate(loader):
        x_pixel = x_pixel.to(DEVICE)
        y       = gaussian_blur_batch(x_pixel)
        x_flat, logdet_logit = logit_preprocess(x_pixel.view(x_pixel.shape[0], -1))

        h          = conditioner(y)
        z, log_det = model(x_flat, h)
        log_pz     = gaussian_log_prob(z)
        log_px     = log_pz + log_det + logdet_logit
        loss       = -log_px.mean()

        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(
                f"[TRAIN] NaN/Inf loss at epoch={epoch}, batch={batch_idx}. "
                f"log_pz={log_pz.mean().item():.3f}, log_det={log_det.mean().item():.3f}"
            )
            raise RuntimeError("NaN/Inf loss during training")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(conditioner.parameters()), max_norm=5.0
        )
        optimizer.step()
        total_loss += loss.item(); n_batches += 1

    avg = total_loss / max(n_batches, 1)
    logger.info(f"[TRAIN] Epoch {epoch:3d} | avg NLL = {avg:.4f}")
    return avg


# ==============================================================================
# EVALUATION
# ==============================================================================
@torch.no_grad()
def evaluate(model: ConditionalCSF, conditioner: CNNConditioner,
             loader: DataLoader, epoch: int) -> tuple[float, float]:
    """Returns (avg_nll, avg_rmse)."""
    model.eval(); conditioner.eval()
    total_nll = total_rmse = 0.0; n_batches = 0

    for x_pixel, _ in loader:
        x_pixel = x_pixel.to(DEVICE)
        y       = gaussian_blur_batch(x_pixel)
        x_flat, logdet_logit = logit_preprocess(x_pixel.view(x_pixel.shape[0], -1))

        h          = conditioner(y)
        z, log_det = model(x_flat, h)
        log_pz     = gaussian_log_prob(z)
        nll        = -(log_pz + log_det + logdet_logit).mean().item()

        x_hat = sigmoid_postprocess(model.inverse(z, h)).view_as(x_pixel)
        rmse  = ((x_pixel - x_hat) ** 2).mean().sqrt().item()

        total_nll  += nll; total_rmse += rmse; n_batches += 1

    avg_nll  = total_nll  / max(n_batches, 1)
    avg_rmse = total_rmse / max(n_batches, 1)
    logger.info(
        f"[EVAL]  Epoch {epoch:3d} | avg NLL = {avg_nll:.4f} | avg RMSE = {avg_rmse:.5f}"
    )
    return avg_nll, avg_rmse


# ==============================================================================
# PLOTS
# ==============================================================================
@torch.no_grad()
def save_reconstruction_plot(model: ConditionalCSF, conditioner: CNNConditioner,
                              loader: DataLoader, epoch: int) -> None:
    """3-row × 8-col grid: original | degraded | reconstruction. Non-fatal."""
    try:
        model.eval(); conditioner.eval()
        x_pixel, _ = next(iter(loader))
        x_pixel     = x_pixel[:8].to(DEVICE)
        y           = gaussian_blur_batch(x_pixel)
        x_flat, _   = logit_preprocess(x_pixel.view(8, -1))
        h           = conditioner(y)
        z, _        = model(x_flat, h)
        x_hat       = sigmoid_postprocess(model.inverse(z, h)).view(8, 1, 28, 28)

        fig, axes = plt.subplots(3, 8, figsize=(16, 6))
        for row, (imgs, label) in enumerate(zip(
            [x_pixel.cpu().squeeze(1).numpy(),
             y.cpu().squeeze(1).numpy(),
             x_hat.cpu().squeeze(1).numpy()],
            ["Original", "Degraded\n(blurred)", "Reconstruction"]
        )):
            for col in range(8):
                axes[row, col].imshow(imgs[col], cmap="gray", vmin=0, vmax=1)
                axes[row, col].axis("off")
            axes[row, 0].set_ylabel(label, fontsize=9, rotation=0,
                                    labelpad=60, va="center")

        plt.suptitle(f"CSF+FiLM MNIST Reconstruction — Epoch {epoch}", fontsize=11)
        plt.tight_layout()
        path = os.path.join(LOG_DIR, f"reconstruction_epoch{epoch:03d}.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[PLOT] Reconstruction grid saved: {path}")
    except Exception as e:
        logger.error(f"[PLOT] save_reconstruction_plot failed at epoch {epoch}: {e}")


def save_training_curves(train_nlls: list, val_nlls: list,
                          val_rmses: list) -> None:
    """NLL + RMSE vs epoch. Non-fatal."""
    try:
        epochs = list(range(1, len(train_nlls) + 1))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(epochs, train_nlls, label="Train NLL", color="steelblue")
        ax1.plot(epochs, val_nlls,   label="Val NLL",   color="darkorange")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("NLL")
        ax1.set_title("NLL vs Epoch"); ax1.legend(); ax1.grid(True, alpha=0.3)
        ax2.plot(epochs, val_rmses, label="Val RMSE", color="crimson")
        ax2.axhline(0.05, color="gray", linestyle="--", label="Pass threshold (0.05)")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("RMSE")
        ax2.set_title("Reconstruction RMSE vs Epoch")
        ax2.legend(); ax2.grid(True, alpha=0.3)
        plt.suptitle("CSF+FiLM MNIST Training Curves", fontsize=11)
        plt.tight_layout()
        path = os.path.join(LOG_DIR, "training_curves.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[PLOT] Training curves saved: {path}")
    except Exception as e:
        logger.error(f"[PLOT] save_training_curves failed: {e}")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    logger.info(f"[MAIN] Device: {DEVICE}")
    logger.info(
        f"[MAIN] Config: N_STEPS={N_STEPS}, N_BINS={N_BINS}, HIDDEN={HIDDEN}, "
        f"N_BLOCKS={N_BLOCKS}, COND_DIM={COND_DIM}, EPOCHS={EPOCHS}, LR={LR}"
    )

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(METRICS_CSV, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_nll", "val_nll", "val_rmse", "best"])

    tf_t = transforms.ToTensor()
    train_ds     = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=tf_t)
    val_ds       = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    logger.info(
        f"[MAIN] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
    )

    model       = ConditionalCSF(dim=DIM, cond_dim=COND_DIM, n_steps=N_STEPS,
                                 n_bins=N_BINS, hidden=HIDDEN,
                                 n_blocks=N_BLOCKS).to(DEVICE)
    conditioner = CNNConditioner(cond_dim=COND_DIM).to(DEVICE)

    n_params = (sum(p.numel() for p in model.parameters()) +
                sum(p.numel() for p in conditioner.parameters()))
    logger.info(f"[MAIN] Total parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(conditioner.parameters()), lr=LR
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=0
    )

    # ActNorm data-driven init + pre-training invertibility check
    logger.info("[MAIN] Initializing ActNorm and pre-training invertibility check ...")
    sample_x, _ = next(iter(train_loader))
    sample_x     = sample_x[:8].to(DEVICE)
    sample_y     = gaussian_blur_batch(sample_x)
    x_flat, _    = logit_preprocess(sample_x.view(8, -1))
    with torch.no_grad():
        h_test = conditioner(sample_y)
        _      = model(x_flat, h_test)   # triggers ActNorm init
    inv_err = model.check_invertibility(x_flat, h_test)
    if inv_err > 1e-3:
        logger.warning(f"[MAIN] Pre-training invertibility error: {inv_err:.3e}")

    # Training loop
    best_val_rmse = float('inf')
    train_nlls, val_nlls, val_rmses = [], [], []

    for epoch in range(1, EPOCHS + 1):
        train_nll         = train(model, conditioner, train_loader, optimizer, epoch)
        val_nll, val_rmse = evaluate(model, conditioner, val_loader, epoch)
        scheduler.step()

        train_nlls.append(train_nll)
        val_nlls.append(val_nll)
        val_rmses.append(val_rmse)

        if epoch % PLOT_EVERY == 0 or epoch == EPOCHS:
            save_reconstruction_plot(model, conditioner, val_loader, epoch)

        is_best = val_rmse < best_val_rmse
        if is_best:
            best_val_rmse = val_rmse
            torch.save({
                'epoch': epoch, 'model': model.state_dict(),
                'conditioner': conditioner.state_dict(),
                'val_nll': val_nll, 'val_rmse': val_rmse,
            }, SAVE_PATH)
            logger.info(f"[MAIN] New best saved at epoch {epoch}: RMSE={val_rmse:.5f}")

        try:
            with open(METRICS_CSV, "a", newline="") as f:
                csv.writer(f).writerow([epoch, f"{train_nll:.4f}",
                                        f"{val_nll:.4f}", f"{val_rmse:.5f}",
                                        int(is_best)])
        except Exception as e:
            logger.error(f"[MAIN] Failed to write metrics CSV at epoch {epoch}: {e}")

    logger.info(f"[MAIN] Training complete. Best val RMSE: {best_val_rmse:.5f}")
    save_training_curves(train_nlls, val_nlls, val_rmses)

    # Post-training invertibility check
    logger.info("[MAIN] Post-training invertibility check ...")
    with torch.no_grad():
        h_test  = conditioner(sample_y)
        x_flat, _ = logit_preprocess(sample_x.view(8, -1))
    model.check_invertibility(x_flat, h_test)

    if best_val_rmse < 0.05:
        logger.info("[MAIN] ✅ RECONSTRUCTION TEST PASSED: RMSE < 0.05")
    else:
        logger.warning(
            f"[MAIN] ⚠️  RECONSTRUCTION TEST: RMSE={best_val_rmse:.5f} >= 0.05"
        )


if __name__ == "__main__":
    main()
