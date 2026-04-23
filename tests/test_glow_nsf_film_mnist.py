# ==============================================================================
# File    : test_glow_nsf_film_mnist.py
# Abbr    : TEST-GLOW-NSF-FILM
# Version : v1.0
# Created : 2026-04-16
# Changelog:
#   v1.0 (2026-04-16): Initial standalone Glow+NSF+FiLM MNIST reconstruction
#                      test. Self-contained (no CSMF imports). Architecture:
#                      Glow multi-scale (2 levels, 4 steps each) with NSF
#                      rational-quadratic spline coupling replacing affine.
#                      Spec: 1×28×28 → dequant+logit → L1(squeeze→4×14×14,
#                      4×[ActNorm→Inv1x1Conv→RQCoupling+FiLM]) → L2(squeeze
#                      →16×7×7, 4×[...]) → z~N(0,I). FiLM only inside conv
#                      coupling NN (not on ActNorm/InvConv paths). Spatial
#                      CouplingNN: Conv3×3→ReLU→FiLM→ResBlock→FiLM→Conv3×3
#                      (zero-init). K=8 bins, B=3 tail bound. Linear tails
#                      outside [-3,3] — no per-block sigmoid/logit. InvConv1x1
#                      identity init, triangular solve inverse. ActNorm clamped
#                      [-0.5,0.5]. Tiny CNN conditioner → h∈R^32. Follows
#                      test_nice_film_mnist.py conventions: LOG_DIR, metrics.csv,
#                      run.log, reconstruction grid, training curves,
#                      check_invertibility pre/post training.
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
# CONFIG  (exact spec)
# ==============================================================================
N_LEVELS   = 2       # Glow levels
K_STEPS    = 4       # flow steps per level
HIDDEN_C   = 32      # coupling NN hidden channels
N_BINS     = 8       # spline bins K
TAIL_BOUND = 3.0     # B — linear tails outside [-B, B]
H_DIM      = 32      # conditioner output dim
BATCH_SIZE = 256
EPOCHS     = 60
LR         = 5e-4    # cosine decay
LOGIT_EPS  = 1e-6
BLUR_K     = 5
BLUR_S     = 1.5
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR   = "./data"
LOG_DIR    = "./tests/logs/glow_nsf_film_mnist"
SAVE_PATH  = os.path.join(LOG_DIR, "best_checkpoint.pth")
METRICS_CSV = os.path.join(LOG_DIR, "metrics.csv")
PLOT_EVERY  = 5

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "run.log"), mode="a"),
    ],
)
logger = logging.getLogger("TEST-GLOW-NSF-FILM")


# ==============================================================================
# HELPERS
# ==============================================================================
def logit_preprocess(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    One global dequantize + logit (spec: no repeated sigmoid/logit inside blocks).
    x: (B,1,28,28) in [0,1]. Returns (x_logit, log_det_logit).
    """
    x = x + torch.zeros_like(x).uniform_(0, 1.0 / 256)
    x = x.clamp(LOGIT_EPS, 1 - LOGIT_EPS)
    log_det = (-torch.log(x) - torch.log(1 - x))
    log_det = log_det.reshape(x.shape[0], -1).sum(dim=1)
    return torch.log(x) - torch.log(1 - x), log_det


def sigmoid_postprocess(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def gaussian_log_prob(z: torch.Tensor) -> torch.Tensor:
    """Gaussian log-prob over all dims. Returns (B,)."""
    return -0.5 * (z.reshape(z.shape[0], -1) ** 2 +
                   math.log(2 * math.pi)).sum(dim=1)


def gaussian_blur_batch(x: torch.Tensor,
                         kernel_size: int = BLUR_K,
                         sigma: float = BLUR_S) -> torch.Tensor:
    pad    = kernel_size // 2
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - pad
    g      = torch.exp(-0.5 * (coords / sigma) ** 2)
    g      = g / g.sum()
    k2d    = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
    return F.conv2d(x, k2d, padding=pad)


def squeeze2d(x: torch.Tensor) -> torch.Tensor:
    """(B,C,H,W) → (B,4C,H/2,W/2)."""
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // 2, 2, W // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4)
    return x.reshape(B, C * 4, H // 2, W // 2)


def unsqueeze2d(x: torch.Tensor) -> torch.Tensor:
    """(B,4C,H,W) → (B,C,2H,2W)."""
    B, C4, H, W = x.shape
    C = C4 // 4
    x = x.reshape(B, C, 2, 2, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3)
    return x.reshape(B, C, H * 2, W * 2)


# ==============================================================================
# ACTNORM  (per-channel spatial, data-driven init, clamp [-0.5, 0.5])
# ==============================================================================
class ActNorm(nn.Module):
    """Per-channel ActNorm for spatial (B,C,H,W). Data-driven init."""
    LOG_SCALE_CLAMP = 0.5

    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.log_scale  = nn.Parameter(torch.zeros(1, n_channels, 1, 1))
        self.shift      = nn.Parameter(torch.zeros(1, n_channels, 1, 1))
        self.register_buffer('initialized', torch.tensor(False))

    @torch.no_grad()
    def _initialize(self, x: torch.Tensor) -> None:
        mean = x.mean(dim=[0, 2, 3], keepdim=True)
        std  = x.std(dim=[0, 2, 3],  keepdim=True).clamp(min=1e-6)
        self.shift.data     = -mean
        self.log_scale.data = -torch.log(std)
        self.initialized.fill_(True)

    def forward(self, x: torch.Tensor):
        if not self.initialized:
            self._initialize(x)
        ls      = self.log_scale.clamp(-self.LOG_SCALE_CLAMP, self.LOG_SCALE_CLAMP)
        y       = (x + self.shift) * torch.exp(ls)
        H, W    = x.shape[2], x.shape[3]
        log_det = (ls.sum() * H * W).expand(x.shape[0])
        if torch.isnan(y).any() or torch.isinf(y).any():
            logger.error("[ActNorm] NaN/Inf in forward")
            raise RuntimeError("NaN/Inf in ActNorm forward")
        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        ls = self.log_scale.clamp(-self.LOG_SCALE_CLAMP, self.LOG_SCALE_CLAMP)
        x  = y * torch.exp(-ls) - self.shift
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("[ActNorm] NaN/Inf in inverse")
            raise RuntimeError("NaN/Inf in ActNorm inverse")
        return x


# ==============================================================================
# INV1x1CONV  (LU decomp, identity init, triangular solve inverse)
# ==============================================================================
class InvConv1x1(nn.Module):
    """
    Invertible 1×1 convolution (Glow §3.2 / NSF §2.4).
    LU decomp, identity init (P=random permutation, L=0→I, U=0, log_s=0).
    Inverse via triangular solves — no linalg.inv.
    """
    LOG_S_CLAMP = 3.0

    def __init__(self, n_channels: int):
        super().__init__()
        self.C = n_channels
        perm    = torch.randperm(n_channels)
        P       = torch.zeros(n_channels, n_channels)
        P[torch.arange(n_channels), perm] = 1.0
        self.register_buffer('P', P)
        self.L     = nn.Parameter(torch.zeros(n_channels, n_channels))
        self.U     = nn.Parameter(torch.zeros(n_channels, n_channels))
        self.log_s = nn.Parameter(torch.zeros(n_channels))
        L_mask     = torch.tril(torch.ones(n_channels, n_channels), diagonal=-1)
        self.register_buffer('L_mask', L_mask)
        self.register_buffer('eye',    torch.eye(n_channels))

    def _get_LU(self):
        ls = self.log_s.clamp(-self.LOG_S_CLAMP, self.LOG_S_CLAMP)
        L  = self.L * self.L_mask + self.eye
        U  = torch.triu(self.U, diagonal=1) + torch.diag(torch.exp(ls))
        return L, U

    def forward(self, x: torch.Tensor):
        """x: (B,C,H,W). Returns (y, log_det)."""
        L, U    = self._get_LU()
        W       = (self.P @ L @ U).unsqueeze(2).unsqueeze(3)   # (C,C,1,1)
        y       = F.conv2d(x, W)
        H, W_hw = x.shape[2], x.shape[3]
        log_det = (self.log_s.clamp(-self.LOG_S_CLAMP, self.LOG_S_CLAMP).sum()
                   * H * W_hw).expand(x.shape[0])
        if torch.isnan(y).any() or torch.isinf(y).any():
            logger.error("[InvConv1x1] NaN/Inf in forward")
            raise RuntimeError("NaN/Inf in InvConv1x1 forward")
        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse via triangular solves on (B,C,H,W)."""
        L, U = self._get_LU()
        B, C, H, W = y.shape
        # Reshape to (B*H*W, C) for batch solve
        y_flat = y.permute(0, 2, 3, 1).reshape(-1, C)   # (B*H*W, C)
        # Solve: W^T x = y  →  (PLU)^T x = y
        b = y_flat @ self.P                              # (B*H*W, C)
        c = torch.linalg.solve_triangular(L.T, b.T, upper=True, unitriangular=True).T
        x_flat = torch.linalg.solve_triangular(U.T, c.T, upper=False).T
        x = x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("[InvConv1x1] NaN/Inf in inverse")
            raise RuntimeError("NaN/Inf in InvConv1x1 inverse")
        return x


# ==============================================================================
# FILM2D  (spec: (1+γ(h))⊙f + β(h), h broadcast over spatial, identity init)
# ==============================================================================
class FiLM2d(nn.Module):
    """FiLM for spatial feature maps. h: (B,h_dim) → (B,C,1,1) broadcast."""
    def __init__(self, h_dim: int, n_channels: int):
        super().__init__()
        self.gamma = nn.Linear(h_dim, n_channels)
        self.beta  = nn.Linear(h_dim, n_channels)
        nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight);  nn.init.zeros_(self.beta.bias)

    def forward(self, f: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        g   = (1.0 + self.gamma(h))[:, :, None, None]
        b   = self.beta(h)[:, :, None, None]
        out = g * f + b
        if torch.isnan(out).any() or torch.isinf(out).any():
            logger.error("[FiLM2d] NaN/Inf in output")
            raise RuntimeError("NaN/Inf in FiLM2d output")
        return out


# ==============================================================================
# COUPLING NN  (spec: Conv3×3 → ReLU → FiLM → ResBlock → FiLM → Conv3×3)
# ==============================================================================
class ResBlock2d(nn.Module):
    """Simple 2-conv residual block for spatial features."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + residual)


class CouplingNN(nn.Module):
    """
    Spatial coupling NN (spec):
      Conv3×3(C_b, hidden) → ReLU → FiLM → ResBlock → FiLM → Conv3×3(hidden, out_c)

    out_c = (3*K-1) * C_a  for spline params.
    Final conv zero-initialised (identity spline at start).
    """
    def __init__(self, in_c: int, out_c: int, h_dim: int,
                 hidden: int = HIDDEN_C):
        super().__init__()
        self.conv_in  = nn.Conv2d(in_c,   hidden, 3, padding=1)
        self.resblock = ResBlock2d(hidden)
        self.conv_out = nn.Conv2d(hidden, out_c,  3, padding=1)
        self.film1    = FiLM2d(h_dim, hidden)
        self.film2    = FiLM2d(h_dim, hidden)
        # Zero-init output so spline params start uniform (identity coupling)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

        logger.debug(
            f"[CouplingNN] in={in_c}, out={out_c}, hidden={hidden}, h_dim={h_dim}"
        )

    def forward(self, x_b: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv_in(x_b))
        out = self.film1(out, h)
        out = self.resblock(out)
        out = self.film2(out, h)
        out = self.conv_out(out)
        if torch.isnan(out).any() or torch.isinf(out).any():
            logger.error("[CouplingNN] NaN/Inf in output")
            raise RuntimeError("NaN/Inf in CouplingNN output")
        return out


# ==============================================================================
# RQ-SPLINE FUNCTIONS  (forward / inverse on spatial tensors)
# Same math as NSF test but operates on (B, C_a, H, W) not (B, D)
# ==============================================================================
def _rq_spline_fwd_spatial(x: torch.Tensor,
                             W: torch.Tensor,
                             H_sp: torch.Tensor,
                             D: torch.Tensor,
                             tail_bound: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    RQ-spline forward on spatial tensor.
    x:    (B, C_a, Hsp, Wsp)
    W:    (B, C_a, Hsp, Wsp, K) bin widths  × 2B
    H_sp: (B, C_a, Hsp, Wsp, K) bin heights × 2B
    D:    (B, C_a, Hsp, Wsp, K+1) derivatives (softplus + boundary=1)
    Returns (y, log_det): both (B, C_a, Hsp, Wsp)
    """
    K = W.shape[-1]
    B_sz, C_a, Hs, Ws = x.shape

    # Cumulative knots
    cum_W = torch.cat([
        -tail_bound * torch.ones(*W.shape[:-1], 1, device=x.device),
        torch.cumsum(W, dim=-1) - tail_bound
    ], dim=-1)   # (..., K+1)

    cum_H = torch.cat([
        -tail_bound * torch.ones(*H_sp.shape[:-1], 1, device=x.device),
        torch.cumsum(H_sp, dim=-1) - tail_bound
    ], dim=-1)

    in_tail = (x <= -tail_bound) | (x >= tail_bound)
    x_safe  = x.clamp(-tail_bound + 1e-6, tail_bound - 1e-6)

    # Find bin index
    bin_idx = (x_safe.unsqueeze(-1) >= cum_W[..., :-1]).sum(dim=-1) - 1
    bin_idx = bin_idx.clamp(0, K - 1)   # (B, C_a, Hs, Ws)

    def gather_last(t):
        return t.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)

    xk  = gather_last(cum_W[..., :-1])
    xk1 = gather_last(cum_W[..., 1:])
    yk  = gather_last(cum_H[..., :-1])
    yk1 = gather_last(cum_H[..., 1:])
    dk  = gather_last(D[..., :-1])
    dk1 = gather_last(D[..., 1:])
    sk  = (yk1 - yk) / (xk1 - xk).clamp(min=1e-8)

    xi = ((x_safe - xk) / (xk1 - xk).clamp(min=1e-8)).clamp(0, 1)

    num = (yk1 - yk) * (sk * xi**2 + dk * xi * (1 - xi))
    den = sk + (dk1 + dk - 2*sk) * xi * (1 - xi)
    y_in   = yk + num / den.clamp(min=1e-8)

    num_d  = sk**2 * (dk1*xi**2 + 2*sk*xi*(1-xi) + dk*(1-xi)**2)
    log_d  = torch.log(num_d.clamp(min=1e-8)) - 2*torch.log(den.abs().clamp(min=1e-8))

    y       = torch.where(in_tail, x, y_in)
    log_det = torch.where(in_tail, torch.zeros_like(log_d), log_d)

    if torch.isnan(y).any() or torch.isinf(y).any():
        logger.error("[RQSpline] NaN/Inf in forward")
        raise RuntimeError("NaN/Inf in rq_spline_fwd_spatial")
    return y, log_det


def _rq_spline_inv_spatial(y: torch.Tensor,
                             W: torch.Tensor,
                             H_sp: torch.Tensor,
                             D: torch.Tensor,
                             tail_bound: float) -> tuple[torch.Tensor, torch.Tensor]:
    """RQ-spline inverse on spatial tensor (quadratic solve)."""
    K = W.shape[-1]

    cum_W = torch.cat([
        -tail_bound * torch.ones(*W.shape[:-1], 1, device=y.device),
        torch.cumsum(W, dim=-1) - tail_bound
    ], dim=-1)
    cum_H = torch.cat([
        -tail_bound * torch.ones(*H_sp.shape[:-1], 1, device=y.device),
        torch.cumsum(H_sp, dim=-1) - tail_bound
    ], dim=-1)

    in_tail = (y <= -tail_bound) | (y >= tail_bound)
    y_safe  = y.clamp(-tail_bound + 1e-6, tail_bound - 1e-6)

    bin_idx = (y_safe.unsqueeze(-1) >= cum_H[..., :-1]).sum(dim=-1) - 1
    bin_idx = bin_idx.clamp(0, K - 1)

    def gather_last(t):
        return t.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)

    xk  = gather_last(cum_W[..., :-1])
    xk1 = gather_last(cum_W[..., 1:])
    yk  = gather_last(cum_H[..., :-1])
    yk1 = gather_last(cum_H[..., 1:])
    dk  = gather_last(D[..., :-1])
    dk1 = gather_last(D[..., 1:])
    sk  = (yk1 - yk) / (xk1 - xk).clamp(min=1e-8)

    a = (yk1 - yk)*(sk - dk) + (y_safe - yk)*(dk1 + dk - 2*sk)
    b = (yk1 - yk)*dk        - (y_safe - yk)*(dk1 + dk - 2*sk)
    c = -sk*(y_safe - yk)

    disc = (b**2 - 4*a*c).clamp(min=0)
    xi   = (2*c / (-b - torch.sqrt(disc) - 1e-8)).clamp(0, 1)

    x_in   = xk + xi*(xk1 - xk)
    num_d  = sk**2 * (dk1*xi**2 + 2*sk*xi*(1-xi) + dk*(1-xi)**2)
    den    = sk + (dk1 + dk - 2*sk)*xi*(1-xi)
    log_d  = torch.log(num_d.clamp(min=1e-8)) - 2*torch.log(den.abs().clamp(min=1e-8))

    x_out   = torch.where(in_tail, y, x_in)
    log_det = torch.where(in_tail, torch.zeros_like(log_d), log_d)

    if torch.isnan(x_out).any() or torch.isinf(x_out).any():
        logger.error("[RQSpline] NaN/Inf in inverse")
        raise RuntimeError("NaN/Inf in rq_spline_inv_spatial")
    return x_out, log_det


# ==============================================================================
# RQ-SPLINE COUPLING LAYER  (channel split, spatial conv NN, FiLM inside NN)
# ==============================================================================
class RQSplineCouplingLayer(nn.Module):
    """
    Spatial RQ-spline coupling (spec: channel split only, conv coupling NN).
        Forward: y_a = g_θ(x_a; x_b, h),  y_b = x_b
        Inverse: x_a = g_θ⁻¹(y_a; y_b, h), x_b = y_b

    NN outputs (3K-1)*C_a spline params per spatial position.
    Widths/heights: softmax × 2B.  Derivatives: softplus, boundary=1.
    Linear tails outside [-B, B].
    FiLM only inside NN (not on skip/invertible path).
    """
    def __init__(self, n_channels: int, h_dim: int,
                 n_bins: int = N_BINS, tail_bound: float = TAIL_BOUND,
                 hidden: int = HIDDEN_C):
        super().__init__()
        if n_channels % 2 != 0:
            logger.error(f"[RQCoupling] n_channels must be even, got {n_channels}")
            raise ValueError("RQSplineCouplingLayer requires even n_channels")

        self.C_a        = n_channels // 2
        self.C_b        = n_channels - self.C_a
        self.n_bins     = n_bins
        self.tail_bound = tail_bound

        # NN: x_b → spline params for x_a
        # output: (3K-1) values per channel per spatial position
        self.nn = CouplingNN(
            in_c=self.C_b,
            out_c=self.C_a * (3 * n_bins - 1),
            h_dim=h_dim,
            hidden=hidden,
        )

        logger.debug(
            f"[RQCoupling] C_a={self.C_a}, C_b={self.C_b}, "
            f"n_bins={n_bins}, tail_bound={tail_bound}"
        )

    def _get_spline_params(self, x_b: torch.Tensor, h: torch.Tensor):
        """
        Run NN and parse output into W, H, D tensors.
        Returns W, H: (B,C_a,Hs,Ws,K);  D: (B,C_a,Hs,Ws,K+1)
        """
        B, _, Hs, Ws = x_b.shape
        K   = self.n_bins
        raw = self.nn(x_b, h)   # (B, C_a*(3K-1), Hs, Ws)

        # Reshape to (B, C_a, 3K-1, Hs, Ws) then permute to (B, C_a, Hs, Ws, 3K-1)
        raw = raw.reshape(B, self.C_a, 3*K-1, Hs, Ws).permute(0, 1, 3, 4, 2)

        W   = F.softmax(raw[..., :K],   dim=-1) * 2 * self.tail_bound
        H   = F.softmax(raw[..., K:2*K], dim=-1) * 2 * self.tail_bound
        d_int = F.softplus(raw[..., 2*K:])   # (B,C_a,Hs,Ws, K-1)
        ones  = torch.ones(*d_int.shape[:-1], 1, device=x_b.device)
        D     = torch.cat([ones, d_int, ones], dim=-1)   # (B,C_a,Hs,Ws, K+1)
        return W, H, D

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        """x: (B,C,H,W). Returns (y, log_det). log_det: (B,)."""
        if x.dim() != 4:
            logger.error(f"[RQCoupling] forward expects 4D, got {x.dim()}D")
            raise ValueError("RQSplineCouplingLayer.forward expects (B,C,H,W)")

        x_a, x_b = x[:, :self.C_a], x[:, self.C_a:]
        W, H, D  = self._get_spline_params(x_b, h)
        y_a, ld  = _rq_spline_fwd_spatial(x_a, W, H, D, self.tail_bound)
        log_det  = ld.reshape(x.shape[0], -1).sum(dim=1)   # sum over C_a×H×W
        return torch.cat([y_a, x_b], dim=1), log_det

    def inverse(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if y.dim() != 4:
            logger.error(f"[RQCoupling] inverse expects 4D, got {y.dim()}D")
            raise ValueError("RQSplineCouplingLayer.inverse expects (B,C,H,W)")

        y_a, y_b = y[:, :self.C_a], y[:, self.C_a:]
        W, H, D  = self._get_spline_params(y_b, h)
        x_a, _   = _rq_spline_inv_spatial(y_a, W, H, D, self.tail_bound)
        return torch.cat([x_a, y_b], dim=1)


# ==============================================================================
# FLOW STEP  = ActNorm → InvConv1x1 → RQSplineCoupling
# ==============================================================================
class FlowStep(nn.Module):
    """Single Glow+NSF flow step: ActNorm → Inv1x1Conv → RQSplineCoupling."""
    def __init__(self, n_channels: int, h_dim: int,
                 n_bins: int = N_BINS, tail_bound: float = TAIL_BOUND,
                 hidden: int = HIDDEN_C):
        super().__init__()
        self.actnorm  = ActNorm(n_channels)
        self.invconv  = InvConv1x1(n_channels)
        self.coupling = RQSplineCouplingLayer(n_channels, h_dim,
                                              n_bins, tail_bound, hidden)

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        log_det = torch.zeros(x.shape[0], device=x.device)
        x, ld = self.actnorm(x);       log_det += ld
        x, ld = self.invconv(x);       log_det += ld
        x, ld = self.coupling(x, h);   log_det += ld
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("[FlowStep] NaN/Inf in forward")
            raise RuntimeError("NaN/Inf in FlowStep forward")
        return x, log_det

    def inverse(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x = self.coupling.inverse(y, h)
        x = self.invconv.inverse(x)
        x = self.actnorm.inverse(x)
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("[FlowStep] NaN/Inf in inverse")
            raise RuntimeError("NaN/Inf in FlowStep inverse")
        return x


# ==============================================================================
# FLOW LEVEL  = Squeeze → K_STEPS × FlowStep
# ==============================================================================
class FlowLevel(nn.Module):
    """One Glow level: squeeze then K steps."""
    def __init__(self, in_channels: int, h_dim: int,
                 k_steps: int = K_STEPS, n_bins: int = N_BINS,
                 tail_bound: float = TAIL_BOUND, hidden: int = HIDDEN_C):
        super().__init__()
        sq_c         = in_channels * 4
        self.in_c    = in_channels
        self.sq_c    = sq_c
        self.steps   = nn.ModuleList([
            FlowStep(sq_c, h_dim, n_bins, tail_bound, hidden)
            for _ in range(k_steps)
        ])

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        log_det = torch.zeros(x.shape[0], device=x.device)
        z = squeeze2d(x)
        for step in self.steps:
            z, ld = step(z, h)
            log_det += ld
        return z, log_det

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x = z
        for step in reversed(self.steps):
            x = step.inverse(x, h)
        return unsqueeze2d(x)


# ==============================================================================
# GLOW-NSF MODEL  (2 levels, Gaussian prior)
# ==============================================================================
class GlowNSF(nn.Module):
    """
    Glow + NSF coupling + FiLM (spec §minimum practical structure).

    Architecture:
      x [1,28,28]
        → dequant+logit (one global, outside model)
        → Level1: squeeze→[4,14,14], 4×[ActNorm→Inv1x1→RQCoupling+FiLM]
        → Level2: squeeze→[16,7,7],  4×[ActNorm→Inv1x1→RQCoupling+FiLM]
        → z [16,7,7] ~ N(0,I)

    FiLM only inside CouplingNN — not on ActNorm/InvConv paths.
    One global logit preprocessing only — no sigmoid/logit inside blocks.

    API:
      forward(x, h) → (z, log_det)   x: (B,1,28,28) logit-space
      inverse(z, h) → x              returns logit-space (B,1,28,28)
    """
    def __init__(self, h_dim: int = H_DIM, k_steps: int = K_STEPS,
                 n_bins: int = N_BINS, tail_bound: float = TAIL_BOUND,
                 hidden: int = HIDDEN_C):
        super().__init__()
        # Level 1: 1ch input → squeeze → 4ch
        self.level1 = FlowLevel(1,  h_dim, k_steps, n_bins, tail_bound, hidden)
        # Level 2: 4ch input → squeeze → 16ch
        self.level2 = FlowLevel(4,  h_dim, k_steps, n_bins, tail_bound, hidden)

        logger.info(
            f"[GlowNSF] v1.0 initialized: h_dim={h_dim}, k_steps={k_steps}, "
            f"n_bins={n_bins}, tail_bound={tail_bound}, hidden={hidden}, "
            f"levels=2, z_shape=[16,7,7]"
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        """x: (B,1,28,28) logit-space. Returns (z, log_det)."""
        if x.dim() != 4 or x.shape[1] != 1:
            logger.error(f"[GlowNSF] forward expects (B,1,H,W), got {tuple(x.shape)}")
            raise ValueError("GlowNSF.forward expects (B,1,28,28)")
        if x.shape[0] != h.shape[0]:
            logger.error(f"[GlowNSF] Batch mismatch: x={x.shape[0]}, h={h.shape[0]}")
            raise ValueError("Batch mismatch in GlowNSF.forward")

        log_det = torch.zeros(x.shape[0], device=x.device)
        z, ld   = self.level1(x, h);   log_det += ld
        z, ld   = self.level2(z, h);   log_det += ld

        if torch.isnan(z).any() or torch.isinf(z).any():
            logger.error("[GlowNSF] NaN/Inf in forward output z")
            raise RuntimeError("NaN/Inf in GlowNSF.forward")
        return z, log_det

    @torch.no_grad()
    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """z: (B,16,7,7). Returns x (B,1,28,28) logit-space."""
        if z.dim() != 4 or z.shape[1] != 16:
            logger.error(f"[GlowNSF] inverse expects (B,16,7,7), got {tuple(z.shape)}")
            raise ValueError("GlowNSF.inverse shape mismatch")

        x = self.level2.inverse(z, h)
        x = self.level1.inverse(x, h)

        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("[GlowNSF] NaN/Inf in inverse output")
            raise RuntimeError("NaN/Inf in GlowNSF.inverse")
        return x

    def check_invertibility(self, x: torch.Tensor, h: torch.Tensor,
                             tol: float = 1e-4) -> float:
        """max ‖x - f⁻¹(f(x))‖_∞. Returns max error."""
        z, _  = self.forward(x, h)
        x_hat = self.inverse(z, h)
        err   = (x.detach() - x_hat).abs().max().item()
        if err > tol:
            logger.warning(
                f"[GlowNSF] Invertibility FAILED: max_err={err:.3e} > tol={tol:.3e}"
            )
        else:
            logger.info(f"[GlowNSF] Invertibility PASSED: max_err={err:.3e}")
        return err


# ==============================================================================
# CONDITIONER  (spec: tiny CNN → h_dim=32)
# ==============================================================================
class TinyConditioner(nn.Module):
    """
    Spec conditioner: Conv3×3(1,16)→ReLU→Conv3×3(16,32,s=2)→ReLU→GAP→Linear(32,h_dim).
    """
    def __init__(self, h_dim: int = H_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(32, h_dim)
        self.norm = nn.LayerNorm(h_dim)
        logger.info(f"[TinyConditioner] initialized: h_dim={h_dim}")

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if y.dim() != 4 or y.shape[1] != 1:
            logger.error(f"[TinyConditioner] Expected (B,1,H,W), got {tuple(y.shape)}")
            raise ValueError("TinyConditioner expects (B,1,28,28)")
        h = self.norm(self.head(self.pool(self.net(y)).squeeze(-1).squeeze(-1)))
        if torch.isnan(h).any() or torch.isinf(h).any():
            logger.error("[TinyConditioner] NaN/Inf in h output")
            raise RuntimeError("NaN/Inf in TinyConditioner output")
        return h


# ==============================================================================
# TRAINING
# ==============================================================================
def train(model: GlowNSF, conditioner: TinyConditioner,
          loader: DataLoader, optimizer: torch.optim.Optimizer,
          epoch: int) -> float:
    model.train(); conditioner.train()
    total_loss = 0.0; n_batches = 0

    for batch_idx, (x_pixel, _) in enumerate(loader):
        x_pixel = x_pixel.to(DEVICE)
        y_deg   = gaussian_blur_batch(x_pixel)
        x_logit, logdet_logit = logit_preprocess(x_pixel)

        h          = conditioner(y_deg)
        z, log_det = model(x_logit, h)
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
            list(model.parameters()) + list(conditioner.parameters()),
            max_norm=5.0
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
def evaluate(model: GlowNSF, conditioner: TinyConditioner,
             loader: DataLoader, epoch: int) -> tuple[float, float]:
    model.eval(); conditioner.eval()
    total_nll = total_rmse = 0.0; n_batches = 0

    for x_pixel, _ in loader:
        x_pixel = x_pixel.to(DEVICE)
        y_deg   = gaussian_blur_batch(x_pixel)
        x_logit, logdet_logit = logit_preprocess(x_pixel)

        h          = conditioner(y_deg)
        z, log_det = model(x_logit, h)
        log_pz     = gaussian_log_prob(z)
        nll        = -(log_pz + log_det + logdet_logit).mean().item()

        x_hat = sigmoid_postprocess(model.inverse(z, h))
        rmse  = ((x_pixel - x_hat) ** 2).mean().sqrt().item()

        total_nll += nll; total_rmse += rmse; n_batches += 1

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
def save_reconstruction_plot(model: GlowNSF, conditioner: TinyConditioner,
                              loader: DataLoader, epoch: int) -> None:
    try:
        model.eval(); conditioner.eval()
        x_pixel, _ = next(iter(loader))
        x_pixel     = x_pixel[:8].to(DEVICE)
        y_deg       = gaussian_blur_batch(x_pixel)
        x_logit, _  = logit_preprocess(x_pixel)
        h           = conditioner(y_deg)
        z, _        = model(x_logit, h)
        x_hat       = sigmoid_postprocess(model.inverse(z, h))

        fig, axes = plt.subplots(3, 8, figsize=(16, 6))
        for row, (imgs, label) in enumerate(zip(
            [x_pixel.cpu().squeeze(1).numpy(),
             y_deg.cpu().squeeze(1).numpy(),
             x_hat.cpu().squeeze(1).numpy()],
            ["Original", "Degraded\n(blurred)", "Reconstruction"]
        )):
            for col in range(8):
                axes[row, col].imshow(imgs[col], cmap="gray", vmin=0, vmax=1)
                axes[row, col].axis("off")
            axes[row, 0].set_ylabel(label, fontsize=9, rotation=0,
                                    labelpad=60, va="center")
        plt.suptitle(f"Glow+NSF+FiLM MNIST Reconstruction — Epoch {epoch}", fontsize=11)
        plt.tight_layout()
        path = os.path.join(LOG_DIR, f"reconstruction_epoch{epoch:03d}.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[PLOT] Reconstruction grid saved: {path}")
    except Exception as e:
        logger.error(f"[PLOT] save_reconstruction_plot failed at epoch {epoch}: {e}")


def save_training_curves(train_nlls: list, val_nlls: list, val_rmses: list) -> None:
    try:
        epochs = list(range(1, len(train_nlls) + 1))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(epochs, train_nlls, label="Train NLL", color="steelblue")
        ax1.plot(epochs, val_nlls,   label="Val NLL",   color="darkorange")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("NLL"); ax1.set_title("NLL vs Epoch")
        ax1.legend(); ax1.grid(True, alpha=0.3)
        ax2.plot(epochs, val_rmses, label="Val RMSE", color="crimson")
        ax2.axhline(0.05, color="gray", linestyle="--", label="Pass threshold (0.05)")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("RMSE")
        ax2.set_title("Reconstruction RMSE vs Epoch")
        ax2.legend(); ax2.grid(True, alpha=0.3)
        plt.suptitle("Glow+NSF+FiLM MNIST Training Curves", fontsize=11)
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
        f"[MAIN] Config: N_LEVELS={N_LEVELS}, K_STEPS={K_STEPS}, "
        f"HIDDEN_C={HIDDEN_C}, N_BINS={N_BINS}, TAIL_BOUND={TAIL_BOUND}, "
        f"H_DIM={H_DIM}, EPOCHS={EPOCHS}, LR={LR}"
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
    logger.info(f"[MAIN] Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")

    model       = GlowNSF(h_dim=H_DIM, k_steps=K_STEPS, n_bins=N_BINS,
                          tail_bound=TAIL_BOUND, hidden=HIDDEN_C).to(DEVICE)
    conditioner = TinyConditioner(h_dim=H_DIM).to(DEVICE)

    n_params = (sum(p.numel() for p in model.parameters()) +
                sum(p.numel() for p in conditioner.parameters()))
    logger.info(f"[MAIN] Total parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(conditioner.parameters()), lr=LR
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=0
    )

    # ActNorm init + pre-training invertibility check
    logger.info("[MAIN] Initializing ActNorm and pre-training invertibility check ...")
    sample_x, _ = next(iter(train_loader))
    sample_x     = sample_x[:8].to(DEVICE)
    sample_y     = gaussian_blur_batch(sample_x)
    x_logit, _   = logit_preprocess(sample_x)
    with torch.no_grad():
        h_test = conditioner(sample_y)
        _      = model(x_logit, h_test)   # triggers ActNorm data-driven init

    inv_err = model.check_invertibility(x_logit, h_test)
    if inv_err > 1e-4:
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
        x_logit, _ = logit_preprocess(sample_x)
    model.check_invertibility(x_logit, h_test)

    if best_val_rmse < 0.05:
        logger.info("[MAIN] ✅ RECONSTRUCTION TEST PASSED: RMSE < 0.05")
    else:
        logger.warning(
            f"[MAIN] ⚠️  RECONSTRUCTION TEST: RMSE={best_val_rmse:.5f} >= 0.05"
        )


if __name__ == "__main__":
    main()
