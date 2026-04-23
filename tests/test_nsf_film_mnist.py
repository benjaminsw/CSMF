# ==============================================================================
# File    : test_nsf_film_mnist.py
# Abbr    : TEST-NSF-FILM
# Version : v1.1
# Created : 2026-04-16
# Changelog:
#   v1.1 (2026-04-16): [TAIL] TAIL_BOUND 3.0→15.0 — paper uses B=3 for VAE
#                      latent space (N(0,1)); logit-space MNIST spans [-13.8,13.8]
#                      so B=3 makes spline act as identity on ~90% of values →
#                      coupling learns nothing → noisy reconstruction. B=15.0
#                      covers full range. [ACTNORM] Added ActNorm after each
#                      InvLinear step in ConditionalNSF — normalises signal
#                      between steps, prevents numerical drift across 10 solves,
#                      ensures spline inputs are better conditioned. [EPOCHS]
#                      30→60 — NLL still dropping steeply at epoch 30.
#   v1.0 (2026-04-16): Initial standalone NSF+FiLM MNIST reconstruction test.
#                      Self-contained (no CSMF imports). Architecture from
#                      Neural Spline Flows paper (Durkan et al. 2019) §B.2
#                      MNIST VAE spec: K=8 bins, B=3 tail bound, 10 flow steps,
#                      2-block ResNet coupling NN, 64 hidden features, COND_DIM=64.
#                      RQ-spline coupling replaces affine; forward closed-form
#                      (Eq.4-5), inverse via quadratic solve (Eq.6-8). InvConv1x1
#                      (LU decomp) between coupling layers. FiLM after each
#                      residual block (extension). CNN conditioner → h ∈ R^64.
#                      Follows test_nice_film_mnist.py conventions: LOG_DIR,
#                      metrics.csv, run.log, reconstruction grid, training
#                      curves, check_invertibility() pre/post training.
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
# CONFIG  (NSF paper §B.2 MNIST spec)
# ==============================================================================
N_STEPS    = 10      # flow steps (paper: 10)
N_BINS     = 8       # RQ-spline bins K (paper: 8)
TAIL_BOUND = 15.0    # [TAIL] v1.1: 3.0→15.0 — logit-space MNIST spans [-13.8,13.8];
                     # paper uses B=3 for VAE latents (N(0,1)); raw logit needs B=15
HIDDEN     = 64      # residual net hidden features (paper: 64 for coupling)
N_BLOCKS   = 2       # residual blocks per coupling NN (paper: 2)
COND_DIM   = 64      # h dimension (paper: 64)
DIM        = 784     # MNIST 28×28 flattened
BATCH_SIZE = 256     # paper: 256
EPOCHS     = 60      # [EPOCHS] v1.1: 30→60 — NLL still dropping at epoch 30
LR         = 5e-4    # paper: 0.0005 with cosine decay
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
LOGIT_EPS  = 1e-6
BLUR_K     = 5
BLUR_S     = 1.5
DATA_DIR   = "./data"
LOG_DIR    = "./tests/logs/nsf_film_mnist"
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
logger = logging.getLogger("TEST-NSF-FILM")


# ==============================================================================
# HELPERS
# ==============================================================================
def logit_preprocess(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize + logit. Returns (x_logit, log_det_logit). x: (B,D) in [0,1]."""
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
# RQ-SPLINE  (Durkan et al. 2019, Appendix A)
# Monotonic rational-quadratic spline on [-B, B]; identity outside (linear tails)
# ==============================================================================
def rq_spline_forward(x: torch.Tensor,
                       widths: torch.Tensor,
                       heights: torch.Tensor,
                       derivatives: torch.Tensor,
                       tail_bound: float = TAIL_BOUND):
    """
    Forward RQ-spline transform (Eq. 4, 5 in paper).

    Args:
        x:           (B, D) input
        widths:      (B, D, K) bin widths (already softmax-scaled to sum 2B)
        heights:     (B, D, K) bin heights
        derivatives: (B, D, K+1) knot derivatives (softplus, boundary=1)
        tail_bound:  B — identity outside [-B, B]

    Returns:
        (y, log_det_elementwise): both (B, D)
    """
    B, D = x.shape
    K    = widths.shape[-1]

    # Knot positions via cumsum from -B
    cum_widths  = torch.cat(
        [-tail_bound * torch.ones(B, D, 1, device=x.device),
         torch.cumsum(widths,  dim=-1) - tail_bound], dim=-1
    )  # (B, D, K+1)
    cum_heights = torch.cat(
        [-tail_bound * torch.ones(B, D, 1, device=x.device),
         torch.cumsum(heights, dim=-1) - tail_bound], dim=-1
    )  # (B, D, K+1)

    # Find bin for each x — clamp to valid range for inside-tail values
    x_safe = x.clamp(-tail_bound + 1e-6, tail_bound - 1e-6)

    # Bin index: (B, D) — rightmost bin whose left edge <= x_safe
    bin_idx = (x_safe.unsqueeze(-1) >= cum_widths[:, :, :-1]).sum(dim=-1) - 1
    bin_idx = bin_idx.clamp(0, K - 1)   # (B, D)

    # Gather per-bin quantities
    def gather_bin(t):
        # t: (B, D, K) or (B, D, K+1)
        idx = bin_idx.unsqueeze(-1)   # (B, D, 1)
        return t.gather(-1, idx).squeeze(-1)  # (B, D)

    xk   = gather_bin(cum_widths[:, :, :-1])   # left x-knot
    xk1  = gather_bin(cum_widths[:, :, 1:])    # right x-knot
    yk   = gather_bin(cum_heights[:, :, :-1])
    yk1  = gather_bin(cum_heights[:, :, 1:])
    dk   = gather_bin(derivatives[:, :, :-1])
    dk1  = gather_bin(derivatives[:, :, 1:])
    sk   = (yk1 - yk) / (xk1 - xk + 1e-8)    # bin slope

    # Normalized position in bin: ξ ∈ [0, 1]
    xi = (x_safe - xk) / (xk1 - xk + 1e-8)
    xi = xi.clamp(0, 1)

    # Forward transform (Eq. 4)
    num  = (yk1 - yk) * (sk * xi ** 2 + dk * xi * (1 - xi))
    den  = sk + (dk1 + dk - 2 * sk) * xi * (1 - xi)
    y_in = yk + num / (den + 1e-8)

    # Derivative (Eq. 5) for log_det
    num_d = (sk ** 2) * (dk1 * xi ** 2 + 2 * sk * xi * (1 - xi) + dk * (1 - xi) ** 2)
    log_d = torch.log(num_d.clamp(min=1e-8)) - 2 * torch.log(den.abs().clamp(min=1e-8))

    # Linear tails: identity outside [-B, B]
    in_tail = (x <= -tail_bound) | (x >= tail_bound)
    y       = torch.where(in_tail, x, y_in)
    log_det = torch.where(in_tail, torch.zeros_like(log_d), log_d)

    if torch.isnan(y).any() or torch.isinf(y).any():
        logger.error("[RQSpline] NaN/Inf in forward output")
        raise RuntimeError("NaN/Inf in rq_spline_forward")

    return y, log_det   # (B,D) each


def rq_spline_inverse(y: torch.Tensor,
                       widths: torch.Tensor,
                       heights: torch.Tensor,
                       derivatives: torch.Tensor,
                       tail_bound: float = TAIL_BOUND):
    """
    Inverse RQ-spline via quadratic solve (Appendix A.3, Eq. 29-32).

    Returns (x, log_det) where log_det is same magnitude as forward.
    """
    B, D = y.shape
    K    = widths.shape[-1]

    cum_widths  = torch.cat(
        [-tail_bound * torch.ones(B, D, 1, device=y.device),
         torch.cumsum(widths,  dim=-1) - tail_bound], dim=-1
    )
    cum_heights = torch.cat(
        [-tail_bound * torch.ones(B, D, 1, device=y.device),
         torch.cumsum(heights, dim=-1) - tail_bound], dim=-1
    )

    y_safe  = y.clamp(-tail_bound + 1e-6, tail_bound - 1e-6)
    bin_idx = (y_safe.unsqueeze(-1) >= cum_heights[:, :, :-1]).sum(dim=-1) - 1
    bin_idx = bin_idx.clamp(0, K - 1)

    def gather_bin(t):
        idx = bin_idx.unsqueeze(-1)
        return t.gather(-1, idx).squeeze(-1)

    xk   = gather_bin(cum_widths[:, :, :-1])
    xk1  = gather_bin(cum_widths[:, :, 1:])
    yk   = gather_bin(cum_heights[:, :, :-1])
    yk1  = gather_bin(cum_heights[:, :, 1:])
    dk   = gather_bin(derivatives[:, :, :-1])
    dk1  = gather_bin(derivatives[:, :, 1:])
    sk   = (yk1 - yk) / (xk1 - xk + 1e-8)

    # Quadratic coefficients (Eq. 30-32)
    a = (yk1 - yk) * (sk - dk) + (y_safe - yk) * (dk1 + dk - 2 * sk)
    b = (yk1 - yk) * dk        - (y_safe - yk) * (dk1 + dk - 2 * sk)
    c = -sk * (y_safe - yk)

    # Numerically stable root selection (Eq. 29 second form)
    discriminant = (b ** 2 - 4 * a * c).clamp(min=0)
    xi = (2 * c) / (-b - torch.sqrt(discriminant) - 1e-8)
    xi = xi.clamp(0, 1)

    x_in = xk + xi * (xk1 - xk)

    # Log-det from forward formula at recovered xi
    num_d = (sk ** 2) * (dk1 * xi ** 2 + 2 * sk * xi * (1 - xi) + dk * (1 - xi) ** 2)
    den   = sk + (dk1 + dk - 2 * sk) * xi * (1 - xi)
    log_d = torch.log(num_d.clamp(min=1e-8)) - 2 * torch.log(den.abs().clamp(min=1e-8))

    in_tail = (y <= -tail_bound) | (y >= tail_bound)
    x_out   = torch.where(in_tail, y, x_in)
    log_det = torch.where(in_tail, torch.zeros_like(log_d), log_d)

    if torch.isnan(x_out).any() or torch.isinf(x_out).any():
        logger.error("[RQSpline] NaN/Inf in inverse output")
        raise RuntimeError("NaN/Inf in rq_spline_inverse")

    return x_out, log_det


# ==============================================================================
# FiLM  (same as other tests)
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
# ResidualBlock + CouplingNN  (paper: ResNet, 2 blocks, 64 hidden)
# Outputs (3K-1) * (D//2) unconstrained params for spline parameterization.
# FiLM after each residual block hidden activation (extension).
# ==============================================================================
class ResidualBlock(nn.Module):
    """
    Pre-activation residual block (He et al. 2016 identity mapping).
    Linear → ReLU → FiLM → Linear → ReLU → FiLM, with residual connection.
    """
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
    Residual coupling network for RQ-NSF.
    Architecture: paper §B.2 — 2 residual blocks, 64 hidden features.
    Input:  x_A (half_dim) + h (cond_dim)
    Output: (3K-1) * half_dim unconstrained spline parameters

    FiLM injected after each residual block (extension).
    """
    def __init__(self, in_dim: int, out_dim: int, h_dim: int,
                 hidden: int = HIDDEN, n_blocks: int = N_BLOCKS):
        super().__init__()
        if n_blocks < 1:
            logger.error(f"[CouplingNN] n_blocks must be >= 1, got {n_blocks}")
            raise ValueError("CouplingNN requires n_blocks >= 1")

        self.fc_in  = nn.Linear(in_dim + h_dim, hidden)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden, h_dim) for _ in range(n_blocks)]
        )
        self.fc_out = nn.Linear(hidden, out_dim)
        # zero-init output so spline params start near uniform (identity-ish)
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
# RQSplineCouplingLayer
# Replaces affine (s, t) with per-dim RQ-spline gθ(x_A).
# NN outputs (3K-1)*half_dim unconstrained params → widths, heights, derivatives.
# ==============================================================================
class RQSplineCouplingLayer(nn.Module):
    """
    RQ-Spline coupling layer (NSF paper §3.1).
        Forward: y_A = g_θ(x_A; x_B, h),  y_B = x_B
        Inverse: x_A = g_θ⁻¹(y_A; y_B, h), x_B = y_B

    Partition: swap=False → x_A=first half, x_B=second half
               swap=True  → x_A=second half, x_B=first half
    """
    def __init__(self, dim: int, cond_dim: int, n_bins: int = N_BINS,
                 tail_bound: float = TAIL_BOUND, swap: bool = False,
                 hidden: int = HIDDEN, n_blocks: int = N_BLOCKS):
        super().__init__()
        if dim % 2 != 0:
            logger.error(f"[RQCoupling] dim must be even, got {dim}")
            raise ValueError("RQSplineCouplingLayer requires even dim")

        self.dim        = dim
        self.half       = dim // 2
        self.swap       = swap
        self.n_bins     = n_bins
        self.tail_bound = tail_bound

        # Each of the half_dim output dims needs (3K-1) params
        self.nn = CouplingNN(
            in_dim=self.half,
            out_dim=self.half * (3 * n_bins - 1),
            h_dim=cond_dim,
            hidden=hidden,
            n_blocks=n_blocks,
        )

        logger.debug(
            f"[RQCoupling] dim={dim}, swap={swap}, n_bins={n_bins}, "
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
        """Run NN and parse (3K-1)*half_dim output into widths, heights, derivatives."""
        B    = xB.shape[0]
        K    = self.n_bins
        raw  = self.nn(xB, h)                         # (B, half*(3K-1))
        raw  = raw.reshape(B, self.half, 3 * K - 1)   # (B, half, 3K-1)

        # Softmax-scaled widths and heights → positive, sum to 2B
        W = F.softmax(raw[:, :, :K],      dim=-1) * 2 * self.tail_bound  # (B, half, K)
        H = F.softmax(raw[:, :, K:2*K],   dim=-1) * 2 * self.tail_bound
        # Softplus derivatives at internal knots; boundary derivatives fixed to 1
        D_int = F.softplus(raw[:, :, 2*K:])  # (B, half, K-1)
        ones  = torch.ones(B, self.half, 1, device=xB.device)
        D     = torch.cat([ones, D_int, ones], dim=-1)   # (B, half, K+1)

        return W, H, D

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        """Returns (y, log_det). log_det: (B,)."""
        if x.dim() != 2 or x.shape[1] != self.dim:
            logger.error(
                f"[RQCoupling] forward shape mismatch: "
                f"expected (*, {self.dim}), got {tuple(x.shape)}"
            )
            raise ValueError("RQSplineCouplingLayer.forward shape mismatch")
        if x.shape[0] != h.shape[0]:
            logger.error(
                f"[RQCoupling] Batch mismatch: x={x.shape[0]}, h={h.shape[0]}"
            )
            raise ValueError("Batch mismatch in RQSplineCouplingLayer.forward")

        xA, xB   = self._split(x)
        W, H, D  = self._get_spline_params(xB, h)
        yA, ld   = rq_spline_forward(xA, W, H, D, self.tail_bound)
        log_det  = ld.sum(dim=1)   # sum over D/2 dims → (B,)
        return self._merge(yA, xB), log_det

    def inverse(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if y.dim() != 2 or y.shape[1] != self.dim:
            logger.error(
                f"[RQCoupling] inverse shape mismatch: "
                f"expected (*, {self.dim}), got {tuple(y.shape)}"
            )
            raise ValueError("RQSplineCouplingLayer.inverse shape mismatch")
        if y.shape[0] != h.shape[0]:
            logger.error(
                f"[RQCoupling] Batch mismatch inverse: y={y.shape[0]}, h={h.shape[0]}"
            )
            raise ValueError("Batch mismatch in RQSplineCouplingLayer.inverse")

        yA, yB   = self._split(y)
        W, H, D  = self._get_spline_params(yB, h)
        xA, _    = rq_spline_inverse(yA, W, H, D, self.tail_bound)
        return self._merge(xA, yB)


# ==============================================================================
# InvConv1x1  (reused from Glow test — LU decomp, flat 1D version)
# ==============================================================================
class InvLinear(nn.Module):
    """
    Invertible linear transform via LU decomposition (flat 1D).
    W = P @ L @ (U + diag(exp(log_s)))

    Init: LU = I (paper §5: "LU is initialized to the identity").
    P = random fixed permutation.

    Forward:  y = x @ W.T,   log_det = log_s_clamped.sum()
    Inverse:  x = W^{-T} y  via two triangular solves (no linalg.inv).
      x = y @ P @ L^{-T} @ U^{-T}
      Solved as:
        b = y @ P                         (permute)
        c = solve(L.T, b.T, upper=True)   (lower-tri solve transposed)
        x = solve(U.T, c,   upper=False)  (upper-tri solve transposed)
    """
    LOG_S_CLAMP = 3.0

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # [INIT] P = random permutation (fixed); L = 0 (→ I with mask+eye);
        # U = 0 (strictly upper, diagonal from log_s=0 → exp(0)=1) → W=I at start
        perm = torch.randperm(dim)
        P    = torch.zeros(dim, dim)
        P[torch.arange(dim), perm] = 1.0
        self.register_buffer('P', P)

        self.L     = nn.Parameter(torch.zeros(dim, dim))   # lower off-diag only
        self.U     = nn.Parameter(torch.zeros(dim, dim))   # upper off-diag only
        self.log_s = nn.Parameter(torch.zeros(dim))        # log diagonal of U

        L_mask = torch.tril(torch.ones(dim, dim), diagonal=-1)
        self.register_buffer('L_mask', L_mask)
        self.register_buffer('eye',    torch.eye(dim))

        logger.debug(f"[InvLinear] Initialized identity: dim={dim}")

    def _get_LU(self):
        """Return (L, U) with structural constraints enforced."""
        ls = self.log_s.clamp(-self.LOG_S_CLAMP, self.LOG_S_CLAMP)
        L  = self.L * self.L_mask + self.eye              # lower tri, unit diag
        U  = torch.triu(self.U, diagonal=1) \
             + torch.diag(torch.exp(ls))                   # upper tri, exp diag
        return L, U

    def forward(self, x: torch.Tensor):
        """x: (B, D). Returns (y, log_det). y = x @ W.T"""
        if x.dim() != 2 or x.shape[1] != self.dim:
            logger.error(
                f"[InvLinear] forward shape mismatch: "
                f"expected (*, {self.dim}), got {tuple(x.shape)}"
            )
            raise ValueError("InvLinear.forward shape mismatch")

        L, U  = self._get_LU()
        W     = self.P @ L @ U          # (D, D)
        y     = x @ W.T                 # (B, D)
        log_det = self.log_s.clamp(-self.LOG_S_CLAMP, self.LOG_S_CLAMP).sum()
        log_det = log_det.expand(x.shape[0])

        if torch.isnan(y).any() or torch.isinf(y).any():
            logger.error("[InvLinear] NaN/Inf in forward output")
            raise RuntimeError("NaN/Inf in InvLinear forward")
        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        x = y @ P @ L^{-T} @ U^{-T}
        Uses triangular solves — never forms full inverse matrix.
        """
        if y.dim() != 2 or y.shape[1] != self.dim:
            logger.error(
                f"[InvLinear] inverse shape mismatch: "
                f"expected (*, {self.dim}), got {tuple(y.shape)}"
            )
            raise ValueError("InvLinear.inverse shape mismatch")

        L, U = self._get_LU()

        # Step 1: apply P (permutation)
        b = y @ self.P            # (B, D)

        # Step 2: solve L.T @ c.T = b.T  →  c.T = L.T^{-1} b.T
        # b.T: (D, B); solve_triangular returns (D, B); transpose → (B, D)
        c = torch.linalg.solve_triangular(
            L.T, b.T, upper=True, unitriangular=True
        ).T                       # (B, D)

        # Step 3: solve U.T @ x.T = c.T  →  x.T = U.T^{-1} c.T
        x = torch.linalg.solve_triangular(
            U.T, c.T, upper=False
        ).T                       # (B, D)

        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("[InvLinear] NaN/Inf in inverse output")
            raise RuntimeError("NaN/Inf in InvLinear inverse")
        return x


# ==============================================================================
# ActNorm  [ACTNORM] v1.1
# Per-dim affine normalisation with data-driven init.
# Normalises signal between flow steps so spline inputs stay within TAIL_BOUND.
# ==============================================================================
class ActNorm(nn.Module):
    """
    Activation Normalisation (Kingma & Dhariwal 2018).
    y = (x + shift) * exp(log_scale)
    Data-driven init: output ≈ N(0,1) on first batch.
    log_scale clamped to [-0.5, 0.5] to prevent log-det exploitation.
    """
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
        logger.debug("[ActNorm] Data-driven init complete.")

    def forward(self, x: torch.Tensor):
        """Returns (y, log_det). log_det: (B,)."""
        if x.dim() != 2 or x.shape[1] != self.dim:
            logger.error(
                f"[ActNorm] forward shape mismatch: "
                f"expected (*, {self.dim}), got {tuple(x.shape)}"
            )
            raise ValueError("ActNorm.forward shape mismatch")
        if not self.initialized:
            self._initialize(x)
        ls      = self.log_scale.clamp(-self.LOG_SCALE_CLAMP, self.LOG_SCALE_CLAMP)
        y       = (x + self.shift) * torch.exp(ls)
        log_det = ls.sum().expand(x.shape[0])
        if torch.isnan(y).any() or torch.isinf(y).any():
            logger.error("[ActNorm] NaN/Inf in forward output")
            raise RuntimeError("NaN/Inf in ActNorm forward")
        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        if y.dim() != 2 or y.shape[1] != self.dim:
            logger.error(
                f"[ActNorm] inverse shape mismatch: "
                f"expected (*, {self.dim}), got {tuple(y.shape)}"
            )
            raise ValueError("ActNorm.inverse shape mismatch")
        ls = self.log_scale.clamp(-self.LOG_SCALE_CLAMP, self.LOG_SCALE_CLAMP)
        x  = y * torch.exp(-ls) - self.shift
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("[ActNorm] NaN/Inf in inverse output")
            raise RuntimeError("NaN/Inf in ActNorm inverse")
        return x


# ==============================================================================
# ConditionalNSF
# N_STEPS flow steps: each = InvLinear → ActNorm → RQSplineCoupling
# ==============================================================================
class ConditionalNSF(nn.Module):
    """
    Conditional Neural Spline Flow (RQ-NSF coupling variant).

    Architecture (NSF paper §B.2 MNIST spec, adapted for logit-space pixels):
      - N_STEPS=10 flow steps
      - Each step: InvLinear (LU) → ActNorm → RQSplineCouplingLayer
      - ActNorm normalises signal so spline inputs stay within TAIL_BOUND
      - Alternating swap=True/False for full-dimensional mixing
      - RQ-spline: K=8 bins, B=15 tail bound (covers logit-space [-13.8,13.8])
      - Coupling NN: 2 residual blocks × 64 hidden, FiLM per block
      - Gaussian prior N(0,I)

    API:
      forward(x, h) → (z, log_det)
      inverse(z, h) → x  (logit-space; sigmoid applied externally)
    """
    def __init__(self, dim: int = DIM, cond_dim: int = COND_DIM,
                 n_steps: int = N_STEPS, n_bins: int = N_BINS,
                 tail_bound: float = TAIL_BOUND, hidden: int = HIDDEN,
                 n_blocks: int = N_BLOCKS):
        super().__init__()
        if dim % 2 != 0:
            logger.error(f"[NSF] dim must be even, got {dim}")
            raise ValueError("ConditionalNSF requires even dim")

        self.dim     = dim
        self.n_steps = n_steps

        self.inv_linears = nn.ModuleList([InvLinear(dim) for _ in range(n_steps)])
        self.actnorms    = nn.ModuleList([ActNorm(dim)   for _ in range(n_steps)])  # [ACTNORM]
        self.couplings   = nn.ModuleList([
            RQSplineCouplingLayer(
                dim=dim, cond_dim=cond_dim, n_bins=n_bins,
                tail_bound=tail_bound, swap=(i % 2 == 1),
                hidden=hidden, n_blocks=n_blocks,
            )
            for i in range(n_steps)
        ])

        logger.info(
            f"[NSF] v1.1 initialized: dim={dim}, cond_dim={cond_dim}, "
            f"n_steps={n_steps}, n_bins={n_bins}, tail_bound={tail_bound}, "
            f"hidden={hidden}, n_blocks={n_blocks}, actnorm=True"
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        """x: (B,D) logit-space. Returns (z, log_det)."""
        if x.dim() != 2 or x.shape[1] != self.dim:
            logger.error(
                f"[NSF] forward shape mismatch: "
                f"expected (*, {self.dim}), got {tuple(x.shape)}"
            )
            raise ValueError("ConditionalNSF.forward shape mismatch")
        if x.shape[0] != h.shape[0]:
            logger.error(
                f"[NSF] Batch mismatch: x={x.shape[0]}, h={h.shape[0]}"
            )
            raise ValueError("Batch mismatch in ConditionalNSF.forward")

        z       = x
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        for i in range(self.n_steps):
            z, ld = self.inv_linears[i](z);   log_det += ld  # InvLinear
            z, ld = self.actnorms[i](z);       log_det += ld  # [ACTNORM] normalise
            z, ld = self.couplings[i](z, h);   log_det += ld  # RQ-spline coupling

        if torch.isnan(z).any() or torch.isinf(z).any():
            logger.error("[NSF] NaN/Inf in forward output")
            raise RuntimeError("NaN/Inf in ConditionalNSF.forward")
        return z, log_det

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Inverse: z → x (logit-space). Sigmoid applied externally."""
        if z.dim() != 2 or z.shape[1] != self.dim:
            logger.error(
                f"[NSF] inverse shape mismatch: "
                f"expected (*, {self.dim}), got {tuple(z.shape)}"
            )
            raise ValueError("ConditionalNSF.inverse shape mismatch")
        if z.shape[0] != h.shape[0]:
            logger.error(
                f"[NSF] Batch mismatch inverse: z={z.shape[0]}, h={h.shape[0]}"
            )
            raise ValueError("Batch mismatch in ConditionalNSF.inverse")

        x = z
        for i in reversed(range(self.n_steps)):
            x = self.couplings[i].inverse(x, h)   # RQ-spline inverse
            x = self.actnorms[i].inverse(x)        # [ACTNORM] undo normalisation
            x = self.inv_linears[i].inverse(x)     # InvLinear inverse

        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("[NSF] NaN/Inf in inverse output")
            raise RuntimeError("NaN/Inf in ConditionalNSF.inverse")
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
                f"[NSF] Invertibility FAILED: max_err={err:.3e} > tol={tol:.3e}"
            )
        else:
            logger.info(f"[NSF] Invertibility PASSED: max_err={err:.3e}")
        return err


# ==============================================================================
# CNN Conditioner  (same as other tests — COND_DIM=64 per paper)
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
            logger.error(
                f"[CNNConditioner] Expected (B,1,28,28), got {tuple(y.shape)}"
            )
            raise ValueError("CNNConditioner expects (B,1,28,28)")
        h = self.norm(self.head(self.pool(self.net(y)).squeeze(-1).squeeze(-1)))
        if torch.isnan(h).any() or torch.isinf(h).any():
            logger.error("[CNNConditioner] NaN/Inf in h output")
            raise RuntimeError("NaN/Inf in CNNConditioner output")
        return h


# ==============================================================================
# TRAINING
# ==============================================================================
def train(model: ConditionalNSF, conditioner: CNNConditioner,
          loader: DataLoader, optimizer: torch.optim.Optimizer,
          epoch: int) -> float:
    model.train()
    conditioner.train()
    total_loss = 0.0
    n_batches  = 0

    for batch_idx, (x_pixel, _) in enumerate(loader):
        x_pixel = x_pixel.to(DEVICE)
        y       = gaussian_blur_batch(x_pixel)
        x_flat, logdet_logit = logit_preprocess(x_pixel.view(x_pixel.shape[0], -1))

        h             = conditioner(y)
        z, log_det    = model(x_flat, h)
        log_pz        = gaussian_log_prob(z)
        log_px        = log_pz + log_det + logdet_logit
        loss          = -log_px.mean()

        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(
                f"[TRAIN] NaN/Inf loss at epoch={epoch}, batch={batch_idx}. "
                f"log_pz={log_pz.mean().item():.3f}, "
                f"log_det={log_det.mean().item():.3f}"
            )
            raise RuntimeError("NaN/Inf loss during training")

        optimizer.zero_grad()
        loss.backward()
        # paper: clip grad norm to [-5, 5] (§B.1); apply same here
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(conditioner.parameters()),
            max_norm=5.0
        )
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1

    avg = total_loss / max(n_batches, 1)
    logger.info(f"[TRAIN] Epoch {epoch:3d} | avg NLL = {avg:.4f}")
    return avg


# ==============================================================================
# EVALUATION
# ==============================================================================
@torch.no_grad()
def evaluate(model: ConditionalNSF, conditioner: CNNConditioner,
             loader: DataLoader, epoch: int) -> tuple[float, float]:
    """Returns (avg_nll, avg_rmse)."""
    model.eval()
    conditioner.eval()
    total_nll = total_rmse = 0.0
    n_batches = 0

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

        total_nll  += nll
        total_rmse += rmse
        n_batches  += 1

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
def save_reconstruction_plot(model: ConditionalNSF, conditioner: CNNConditioner,
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

        orig  = x_pixel.cpu().squeeze(1).numpy()
        deg   = y.cpu().squeeze(1).numpy()
        recon = x_hat.cpu().squeeze(1).numpy()

        fig, axes = plt.subplots(3, 8, figsize=(16, 6))
        for row, (imgs, label) in enumerate(
            zip([orig, deg, recon],
                ["Original", "Degraded\n(blurred)", "Reconstruction"])
        ):
            for col in range(8):
                axes[row, col].imshow(imgs[col], cmap="gray", vmin=0, vmax=1)
                axes[row, col].axis("off")
            axes[row, 0].set_ylabel(label, fontsize=9, rotation=0,
                                    labelpad=60, va="center")

        plt.suptitle(f"NSF+FiLM MNIST Reconstruction — Epoch {epoch}", fontsize=11)
        plt.tight_layout()
        path = os.path.join(LOG_DIR, f"reconstruction_epoch{epoch:03d}.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[PLOT] Reconstruction grid saved: {path}")
    except Exception as e:
        logger.error(f"[PLOT] save_reconstruction_plot failed at epoch {epoch}: {e}")


def save_training_curves(train_nlls: list, val_nlls: list,
                          val_rmses: list) -> None:
    """NLL + RMSE vs epoch → LOG_DIR/training_curves.png. Non-fatal."""
    try:
        epochs = list(range(1, len(train_nlls) + 1))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(epochs, train_nlls, label="Train NLL", color="steelblue")
        ax1.plot(epochs, val_nlls,   label="Val NLL",   color="darkorange")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("NLL")
        ax1.set_title("NLL vs Epoch"); ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, val_rmses, label="Val RMSE", color="crimson")
        ax2.axhline(0.05, color="gray", linestyle="--",
                    label="Pass threshold (0.05)")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("RMSE")
        ax2.set_title("Reconstruction RMSE vs Epoch")
        ax2.legend(); ax2.grid(True, alpha=0.3)

        plt.suptitle("NSF+FiLM MNIST Training Curves", fontsize=11)
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
        f"[MAIN] Config: N_STEPS={N_STEPS}, N_BINS={N_BINS}, "
        f"TAIL_BOUND={TAIL_BOUND}, HIDDEN={HIDDEN}, N_BLOCKS={N_BLOCKS}, "
        f"COND_DIM={COND_DIM}, EPOCHS={EPOCHS}, LR={LR}"
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

    model       = ConditionalNSF(dim=DIM, cond_dim=COND_DIM, n_steps=N_STEPS,
                                 n_bins=N_BINS, tail_bound=TAIL_BOUND,
                                 hidden=HIDDEN, n_blocks=N_BLOCKS).to(DEVICE)
    conditioner = CNNConditioner(cond_dim=COND_DIM).to(DEVICE)

    n_params = (sum(p.numel() for p in model.parameters()) +
                sum(p.numel() for p in conditioner.parameters()))
    logger.info(f"[MAIN] Total parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(conditioner.parameters()), lr=LR
    )
    # Paper: cosine annealing LR to 0 over training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=0
    )

    # Pre-training: ActNorm data-driven init + invertibility check
    logger.info("[MAIN] Initializing ActNorm and pre-training invertibility check ...")
    sample_x, _ = next(iter(train_loader))
    sample_x     = sample_x[:8].to(DEVICE)
    sample_y     = gaussian_blur_batch(sample_x)
    x_flat, _    = logit_preprocess(sample_x.view(8, -1))
    with torch.no_grad():
        h_test = conditioner(sample_y)
        _      = model(x_flat, h_test)   # triggers ActNorm data-driven init
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
            logger.info(
                f"[MAIN] New best saved at epoch {epoch}: RMSE={val_rmse:.5f}"
            )

        try:
            with open(METRICS_CSV, "a", newline="") as f:
                csv.writer(f).writerow([epoch, f"{train_nll:.4f}",
                                        f"{val_nll:.4f}", f"{val_rmse:.5f}",
                                        int(is_best)])
        except Exception as e:
            logger.error(
                f"[MAIN] Failed to write metrics CSV at epoch {epoch}: {e}"
            )

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
