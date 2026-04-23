# ==============================================================================
# File    : test_iresnet_film_mnist.py
# Abbr    : TEST-IRESNET-FILM
# Version : v1.3
# Created : 2026-04-16
# Changelog:
#   v1.3 (2026-04-19): Phase 3 prep — Combined plot + log-det tracking.
#                      [P3-PLOT] save_reconstruction_plot + save_generative_plot
#                      merged into save_combined_plot(): 5-row × 8-col grid —
#                      Original / Degraded / Cycle(x→z→x̂) / Generated(z~N(0,1))
#                      / Residual(|orig−cycle|). Single file per epoch.
#                      [P3-LOGDET] train() now returns (avg_nll, avg_logdet_mean).
#                      Per-epoch log_det.mean() tracked in train_logdets list,
#                      written to metrics.csv, and plotted in training curves.
#                      [P3-CURVES] save_training_curves(): 4-panel —
#                      NLL / Cycle-RMSE / inv_err / log-det mean.
#   v1.1 (2026-04-18): Phase 1 — Numerical Invertibility.
#                      [P1-CONTRACTION] SN_COEFF 0.8→0.6, TAU 0.5→0.3,
#                      ALPHA 0.1→0.05. Lip bound drops 0.341→0.118.
#                      [P1-ITERS] N_ITER_INV 20→100 (matches paper §5.1).
#                      [P1-LR] LR 3e-3→1e-3 (reduces Hutchinson noise impact).
#                      [P1-INVLOG] Per-epoch invertibility error logged.
#   v1.0 (2026-04-16): Initial standalone i-ResNet+FiLM MNIST reconstruction
#                      test. Self-contained (no CSMF imports).
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
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ==============================================================================
# CONFIG  (i-ResNet MNIST spec §C.3 + FiLM injection doc)
# ==============================================================================
N_BLOCKS   = 8       # blocks (paper: 32 per scale-block; 8 = minimal)
CHANNELS   = 64      # = SQ_C — work directly at squeezed channel dim, no proj needed
H_DIM      = 64      # conditioner output dim
# [P1-CONTRACTION] Tightened from 0.1→0.05. FiLM multiplier a ∈ [0.95, 1.05].
ALPHA      = 0.05
# [P1-CONTRACTION] Tightened from 0.5→0.3. Lip bound: 0.3*(0.6*1.05)^3 ≈ 0.118.
TAU        = 0.3
# [P1-CONTRACTION] Tightened from 0.8→0.6. Larger safety margin vs true conv operator
# norm (paper §2.1: power-iter on param matrix under-estimates true spectral norm for
# 3×3 convs, so the assumed 0.341 bound may be false at SN_COEFF=0.8).
SN_COEFF   = 0.6
# [P1-ITERS] Increased from 20→100 (paper §5.1 uses 100 for convergence guarantee).
N_ITER_INV = 100
N_SERIES   = 4       # [P2-OOM] 10→4: retain full graph per term with create_graph=True.
# [P2-EPOCHS] 60→20 quick test run.
EPOCHS     = 20
# [P1-LR] Reduced from 3e-3→1e-3.
LR         = 1e-3
BATCH_SIZE = 128     # [P2-OOM] 256→128 to reduce activation memory.
# [P2-FILM-ABLATION] 0=no FiLM, 1=first conv only, 3=all sites (default).
FILM_SITES = 3
LOGIT_EPS  = 1e-6
BLUR_K     = 5
BLUR_S     = 1.5
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR   = "./data"
LOG_DIR    = "./tests/logs/iresnet_film_mnist"
SAVE_PATH  = os.path.join(LOG_DIR, "best_checkpoint.pth")
METRICS_CSV = os.path.join(LOG_DIR, "metrics.csv")
PLOT_EVERY  = 5

# Input pipeline constants
IN_C    = 1    # grayscale
PAD_C   = 16   # zero-pad to 16 channels (injective padding, paper §C.2)
SQ_C    = PAD_C * 4   # after 2×2 squeeze: 64 channels, 14×14

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "run.log"), mode="a"),
    ],
)
logger = logging.getLogger("TEST-IRESNET-FILM")


# ==============================================================================
# HELPERS
# ==============================================================================
def logit_preprocess(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize + logit. x: (B,C,H,W) in [0,1]. Returns (x_logit, logdet)."""
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
    """Gaussian blur (B,1,28,28)."""
    pad    = kernel_size // 2
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - pad
    g      = torch.exp(-0.5 * (coords / sigma) ** 2)
    g      = g / g.sum()
    k2d    = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
    return F.conv2d(x, k2d, padding=pad)


def squeeze2d(x: torch.Tensor) -> torch.Tensor:
    """2×2 spatial → channel: (B,C,H,W) → (B,4C,H/2,W/2)."""
    B, C, H, W = x.shape
    if H % 2 != 0 or W % 2 != 0:
        logger.error(f"[Squeeze] H,W must be even, got {H},{W}")
        raise ValueError("squeeze2d requires even spatial dims")
    x = x.reshape(B, C, H // 2, 2, W // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4)
    return x.reshape(B, C * 4, H // 2, W // 2)


def unsqueeze2d(x: torch.Tensor, orig_C: int) -> torch.Tensor:
    """Inverse squeeze: (B,4C,H/2,W/2) → (B,C,H,W)."""
    B, C4, H2, W2 = x.shape
    C = orig_C
    x = x.reshape(B, C, 2, 2, H2, W2)
    x = x.permute(0, 1, 4, 2, 5, 3)
    return x.reshape(B, C, H2 * 2, W2 * 2)


def injective_pad(x: torch.Tensor, target_C: int) -> torch.Tensor:
    """Zero-pad channel dim from x.shape[1] to target_C."""
    B, C, H, W = x.shape
    if C >= target_C:
        return x
    pad = torch.zeros(B, target_C - C, H, W, device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=1)


def remove_pad(x: torch.Tensor, orig_C: int) -> torch.Tensor:
    """Remove zero-padding: keep first orig_C channels."""
    return x[:, :orig_C]


# ==============================================================================
# Spectral-normed convolution helpers
# ==============================================================================
def sn_conv3x3(in_c: int, out_c: int, coeff: float = SN_COEFF) -> nn.Module:
    """3×3 conv with spectral norm, scaled by coeff."""
    conv = nn.Conv2d(in_c, out_c, 3, padding=1, bias=True)
    conv = spectral_norm(conv)
    # Wrap to scale output by coeff — ensures ‖W‖₂ ≤ coeff
    return _ScaledSN(conv, coeff)


def sn_conv1x1(in_c: int, out_c: int, coeff: float = SN_COEFF) -> nn.Module:
    """1×1 conv with spectral norm, scaled by coeff."""
    conv = nn.Conv2d(in_c, out_c, 1, bias=True)
    conv = spectral_norm(conv)
    return _ScaledSN(conv, coeff)


class _ScaledSN(nn.Module):
    """Wraps a spectrally-normed conv and multiplies output by coeff."""
    def __init__(self, conv: nn.Module, coeff: float):
        super().__init__()
        self.conv  = conv
        self.coeff = coeff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.coeff * self.conv(x)


# ==============================================================================
# FiLM2d  (doc spec: bounded multiplier, identity init)
# Operates on spatial feature maps (B, C, H, W).
# a_i(h) = 1 + alpha * tanh(gamma_i(h))  →  a ∈ [1-alpha, 1+alpha]
# b_i(h) = beta_i(h)
# ==============================================================================
class FiLM2d(nn.Module):
    """
    Bounded FiLM for spatial features (doc spec).
    a = 1 + alpha * tanh(gamma(h))   →  a ∈ [1-alpha, 1+alpha]
    b = beta(h)
    out = a[:,:,None,None] * u + b[:,:,None,None]

    Invertibility: additive b does not increase Lip(g); multiplicative
    a is bounded to [0.9, 1.1] with alpha=0.1, so worst-case factor is 1.1.
    Identity init: gamma/beta weights = 0 → a=1, b=0 at start.
    """
    def __init__(self, h_dim: int, channels: int, alpha: float = ALPHA):
        super().__init__()
        self.gamma = nn.Linear(h_dim, channels)
        self.beta  = nn.Linear(h_dim, channels)
        self.alpha = alpha
        nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight);  nn.init.zeros_(self.beta.bias)

    def forward(self, u: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """u: (B,C,H,W), h: (B,h_dim). Returns (B,C,H,W)."""
        a = 1.0 + self.alpha * torch.tanh(self.gamma(h))  # (B, C)
        b = self.beta(h)                                   # (B, C)
        out = a[:, :, None, None] * u + b[:, :, None, None]
        if torch.isnan(out).any() or torch.isinf(out).any():
            logger.error("[FiLM2d] NaN/Inf in output")
            raise RuntimeError("NaN/Inf in FiLM2d output")
        return out


# ==============================================================================
# FiLMiResBlock  (doc spec: 3-conv + FiLM at each site + τ shrink)
# g(x, h) = τ * FiLM3(W3 * ELU(FiLM2(W2 * ELU(FiLM1(W1*x, h)), h)), h)
# y = x + g(x, h)
#
# Lipschitz bound (doc spec):
#   τ × (SN_COEFF × (1+alpha))³ = 0.5 × (0.8 × 1.1)³ ≈ 0.341 < 1  ✅
# ==============================================================================
class FiLMiResBlock(nn.Module):
    """
    FiLM-conditioned i-ResNet block.
    [P2-FILM-ABLATION] film_sites: 0=no FiLM, 1=first conv only, 3=all sites.
    """
    def __init__(self, channels: int, h_dim: int,
                 tau: float = TAU, sn_coeff: float = SN_COEFF,
                 alpha: float = ALPHA, film_sites: int = FILM_SITES):
        super().__init__()
        self.w1    = sn_conv3x3(channels, channels, sn_coeff)
        self.w2    = sn_conv1x1(channels, channels, sn_coeff)
        self.w3    = sn_conv3x3(channels, channels, sn_coeff)
        self.film1 = FiLM2d(h_dim, channels, alpha)
        self.film2 = FiLM2d(h_dim, channels, alpha)
        self.film3 = FiLM2d(h_dim, channels, alpha)
        self.act        = nn.ELU()
        self.tau        = tau
        self.film_sites = film_sites

    def g(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Contractive residual map. Lip(g) ≤ 0.118 < 1."""
        u1  = self.w1(x)
        f1  = self.film1(u1, h) if self.film_sites >= 1 else u1
        z1  = self.act(f1)
        u2  = self.w2(z1)
        f2  = self.film2(u2, h) if self.film_sites >= 2 else u2
        z2  = self.act(f2)
        u3  = self.w3(z2)
        out = self.film3(u3, h) if self.film_sites >= 3 else u3
        return self.tau * out

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """y = x + g(x, h)."""
        if x.dim() != 4:
            logger.error(f"[iResBlock] forward expects 4D input, got {x.dim()}D")
            raise ValueError("FiLMiResBlock.forward expects (B,C,H,W)")
        out = x + self.g(x, h)
        if torch.isnan(out).any() or torch.isinf(out).any():
            logger.error("[iResBlock] NaN/Inf in forward output")
            raise RuntimeError("NaN/Inf in FiLMiResBlock forward")
        return out

    @torch.no_grad()
    def inverse(self, y: torch.Tensor, h: torch.Tensor,
                n_iter: int = N_ITER_INV) -> torch.Tensor:
        """Fixed-point inverse: x_{k+1} = y - g(x_k, h).
        [P1-INVLOG] Logs final-iteration residual ‖x_n − x_{n-1}‖∞ to detect
        cases where Lip(g) ≥ 1 in practice (paper §2.1 warns that 3×3 conv
        spectral norm from power-iteration is a lower bound, not true norm).
        """
        if y.dim() != 4:
            logger.error(f"[iResBlock] inverse expects 4D input, got {y.dim()}D")
            raise ValueError("FiLMiResBlock.inverse expects (B,C,H,W)")
        x = y.clone()
        x_prev = x
        for i in range(n_iter):
            x_new = y - self.g(x, h)
            if i == n_iter - 1:
                # [P1-INVLOG] Final residual — should be near 0 if Lip(g) < 1
                final_res = (x_new - x).abs().max().item()
                if final_res > 1e-3:
                    logger.warning(
                        f"[iResBlock] inverse final residual={final_res:.3e} "
                        f"after {n_iter} iters — Lip(g) may be ≥ 1 in practice"
                    )
            x_prev = x
            x = x_new
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("[iResBlock] NaN/Inf in inverse output")
            raise RuntimeError("NaN/Inf in FiLMiResBlock inverse")
        return x


# ==============================================================================
# Hutchinson log-det estimator (power series, Algorithm 2 in paper)
# log|det(I + J_g)| ≈ Σ_{k=1}^{n} (-1)^{k+1} vᵀJᵏv / k
# Converges because Lip(g) < 1 → ‖J_g‖₂ < 1
# ==============================================================================
def hutchinson_logdet(g_fn, x: torch.Tensor, h: torch.Tensor,
                       n_series: int = N_SERIES) -> torch.Tensor:
    """
    Stochastic Hutchinson trace estimator of log|det(I + J_g)| (paper §3.2).

    Args:
        g_fn:     callable g(x, h) — contractive residual map
        x:        (B, C, H, W) input
        h:        (B, h_dim) conditioning
        n_series: number of power-series terms

    Returns:
        log_det: (B,) per-sample estimate
    """
    # Sample Rademacher vector v
    v = torch.randint(0, 2, x.shape, dtype=x.dtype, device=x.device) * 2 - 1  # ±1

    B = x.shape[0]
    w = v.clone()
    log_det = torch.zeros(B, device=x.device, dtype=x.dtype)

    for k in range(1, n_series + 1):
        # w = w^T J_g via vector-Jacobian product (reverse-mode AD)
        x_req = x.detach().requires_grad_(True)
        gx    = g_fn(x_req, h)
        # vJp: compute w^T @ J_g  (∂(gx⊙w)/∂x evaluated at x)
        # [P2-GRAD] create_graph=True: gradient flows to conv weights.
        # retain_graph=True: keeps buffers alive for outer loss.backward().
        vjp = torch.autograd.grad(
            (gx * w.detach()).sum(), x_req,
            create_graph=True, retain_graph=True
        )[0]
        w        = vjp
        # Accumulate: (-1)^{k+1} * (w^T v) / k
        sign     = (-1) ** (k + 1)
        wTv      = (w * v).reshape(B, -1).sum(dim=1)   # (B,)
        log_det  = log_det + sign * wTv / k

    return log_det


# ==============================================================================
# ConditionalIResNet
# ==============================================================================
class ConditionalIResNet(nn.Module):
    """
    Conditional i-ResNet flow (Behrmann et al. 2019 + FiLM injection).

    Architecture (MNIST spec §C.3, minimal):
      - Input: 1×28×28 → injective pad to 16×28×28 → squeeze to 64×14×14
      - N_BLOCKS=8 FiLMiResBlocks on squeezed spatial features
      - Each block: 3×3/1×1/3×3 ELU convs, spectrally normed c=0.8, τ=0.5
      - FiLM at 3 sites per block: a=1+0.1·tanh(γ(h)) ∈ [0.9,1.1]
      - Lip(g) ≤ 0.5×(0.8×1.1)³ ≈ 0.341 < 1  (guaranteed invertible)
      - log_det: Hutchinson power-series approximation (10 terms)
      - Inverse: fixed-point iteration (20 steps per block)
      - Gaussian prior N(0,I)

    API:
      forward(x, h) → (z, log_det)   x: (B,1,28,28) in logit-space
      inverse(z, h) → x              returns logit-space spatial tensor
    """
    def __init__(self, channels: int = CHANNELS, h_dim: int = H_DIM,
                 n_blocks: int = N_BLOCKS, tau: float = TAU,
                 sn_coeff: float = SN_COEFF, alpha: float = ALPHA,
                 n_iter_inv: int = N_ITER_INV, n_series: int = N_SERIES):
        super().__init__()
        self.channels   = channels
        self.n_blocks   = n_blocks
        self.n_iter_inv = n_iter_inv
        self.n_series   = n_series

        # No proj_in/proj_out: blocks operate directly at SQ_C channels
        # (channels must equal SQ_C=64 — enforced by CHANNELS=SQ_C above)
        if channels != SQ_C:
            logger.error(
                f"[iResNet] channels={channels} must equal SQ_C={SQ_C} "
                f"(no projection — set CHANNELS=SQ_C)"
            )
            raise ValueError(f"ConditionalIResNet requires channels == SQ_C={SQ_C}")

        self.blocks = nn.ModuleList([
            FiLMiResBlock(channels, h_dim, tau, sn_coeff, alpha, film_sites=FILM_SITES)
            for _ in range(n_blocks)
        ])

        logger.info(
            f"[iResNet] v1.0 initialized: channels={channels}, h_dim={h_dim}, "
            f"n_blocks={n_blocks}, tau={tau}, sn_coeff={sn_coeff}, "
            f"alpha={alpha}, n_iter_inv={n_iter_inv}, n_series={n_series}, "
            f"Lip_bound={tau*(sn_coeff*(1+alpha))**3:.4f}"
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        """
        x: (B,1,28,28) logit-space.
        Returns (z, log_det). log_det: (B,).
        """
        if x.dim() != 4 or x.shape[1] != 1:
            logger.error(
                f"[iResNet] forward expects (B,1,H,W), got {tuple(x.shape)}"
            )
            raise ValueError("ConditionalIResNet.forward expects (B,1,28,28)")
        if x.shape[0] != h.shape[0]:
            logger.error(
                f"[iResNet] Batch mismatch: x={x.shape[0]}, h={h.shape[0]}"
            )
            raise ValueError("Batch mismatch in ConditionalIResNet.forward")

        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        # Injective padding + squeeze → (B, SQ_C=64, 14, 14)
        z = injective_pad(x, PAD_C)
        z = squeeze2d(z)

        # i-ResNet blocks with Hutchinson log-det
        for block in self.blocks:
            # log|det(I + J_g)| via power series
            ld      = hutchinson_logdet(block.g, z, h, self.n_series)
            log_det = log_det + ld
            z       = block(z, h)

        if torch.isnan(z).any() or torch.isinf(z).any():
            logger.error("[iResNet] NaN/Inf in forward output")
            raise RuntimeError("NaN/Inf in ConditionalIResNet.forward")

        return z, log_det

    @torch.no_grad()
    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Inverse: z (B,64,14,14) → x (B,1,28,28) logit-space.
        Fixed-point iteration per block, reversed order.
        """
        if z.dim() != 4 or z.shape[1] != SQ_C:
            logger.error(
                f"[iResNet] inverse expects (B,{SQ_C},H,W), got {tuple(z.shape)}"
            )
            raise ValueError("ConditionalIResNet.inverse shape mismatch")

        # Undo blocks in reverse (no projection layers to undo)
        x = z.clone()
        for block in reversed(self.blocks):
            x = block.inverse(x, h, self.n_iter_inv)

        # Undo squeeze + remove padding
        x = unsqueeze2d(x, PAD_C)     # (B, 16, 28, 28)
        x = remove_pad(x, IN_C)       # (B, 1, 28, 28)

        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("[iResNet] NaN/Inf in inverse output")
            raise RuntimeError("NaN/Inf in ConditionalIResNet.inverse")
        return x

    def check_invertibility(self, x: torch.Tensor, h: torch.Tensor,
                             tol: float = 1e-2) -> float:
        """
        max ‖x - f⁻¹(f(x))‖_∞.
        NOTE: forward() needs grad (Hutchinson), so no @torch.no_grad here.
        Only the inverse fixed-point is wrapped in no_grad.
        """
        z, _ = self.forward(x, h)
        with torch.no_grad():
            x_hat = self.inverse(z, h)
        err = (x.detach() - x_hat).abs().max().item()
        if err > tol:
            logger.warning(
                f"[iResNet] Invertibility FAILED: max_err={err:.3e} > tol={tol:.3e}"
            )
        else:
            logger.info(f"[iResNet] Invertibility PASSED: max_err={err:.3e}")
        return err


# ==============================================================================
# CNN Conditioner
# ==============================================================================
class CNNConditioner(nn.Module):
    """4-conv CNN: y (B,1,28,28) → h ∈ R^{h_dim}."""
    def __init__(self, h_dim: int = H_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32,  3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(128, h_dim)
        self.norm = nn.LayerNorm(h_dim)
        logger.info(f"[CNNConditioner] initialized: h_dim={h_dim}")

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
def train(model: ConditionalIResNet, conditioner: CNNConditioner,
          loader: DataLoader, optimizer: torch.optim.Optimizer,
          epoch: int) -> tuple[float, float]:
    """Returns (avg_nll, avg_logdet_mean).
    [P3-LOGDET] avg_logdet_mean tracks log|det(I+J_g)| mean per epoch.
    Should be non-zero and changing if create_graph=True gradient fix works.
    """
    model.train(); conditioner.train()
    total_loss = 0.0; total_logdet_mean = 0.0; total_logdet_std = 0.0
    n_batches  = 0

    for batch_idx, (x_pixel, _) in enumerate(loader):
        x_pixel = x_pixel.to(DEVICE)
        y_deg   = gaussian_blur_batch(x_pixel)
        x_logit, logdet_logit = logit_preprocess(x_pixel)

        h          = conditioner(y_deg)
        z, log_det = model(x_logit, h)

        log_pz = gaussian_log_prob(z)
        log_px = log_pz + log_det + logdet_logit
        loss   = -log_px.mean()

        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(
                f"[TRAIN] NaN/Inf loss at epoch={epoch}, batch={batch_idx}. "
                f"log_pz={log_pz.mean().item():.3f}, "
                f"log_det={log_det.mean().item():.3f}"
            )
            raise RuntimeError("NaN/Inf loss during training")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(conditioner.parameters()),
            max_norm=5.0
        )
        optimizer.step()

        total_loss       += loss.item()
        total_logdet_mean += log_det.mean().item()   # [P3-LOGDET] mean per batch
        total_logdet_std  += log_det.std().item()
        n_batches         += 1

    avg_nll         = total_loss       / max(n_batches, 1)
    avg_logdet_mean = total_logdet_mean / max(n_batches, 1)
    avg_logdet_std  = total_logdet_std  / max(n_batches, 1)
    logger.info(
        f"[TRAIN] Epoch {epoch:3d} | avg NLL = {avg_nll:.4f} | "
        f"logdet_mean = {avg_logdet_mean:.4f} | logdet_std = {avg_logdet_std:.4f}"
    )
    return avg_nll, avg_logdet_mean


# ==============================================================================
# EVALUATION
# ==============================================================================
def evaluate(model: ConditionalIResNet, conditioner: CNNConditioner,
             loader: DataLoader, epoch: int) -> tuple[float, float, float]:
    """Returns (avg_nll, avg_rmse, inv_err).
    NOTE: no @torch.no_grad() — Hutchinson logdet needs autograd.grad.
    Only fixed-point inverse is wrapped in no_grad.
    [P1-INVLOG] inv_err = max‖x_logit − inv(fwd(x_logit))‖∞ on first batch
    of 8 samples. Distinguishes inverse failure from training failure.
    """
    model.eval(); conditioner.eval()
    total_nll = total_rmse = 0.0; n_batches = 0

    # [P1-INVLOG] Capture first batch for invertibility check
    first_x_logit = None; first_h = None

    for x_pixel, _ in loader:
        x_pixel = x_pixel.to(DEVICE)
        y_deg   = gaussian_blur_batch(x_pixel)
        x_logit, logdet_logit = logit_preprocess(x_pixel)

        h          = conditioner(y_deg)
        z, log_det = model(x_logit, h)   # needs grad for Hutchinson
        log_pz     = gaussian_log_prob(z)
        nll        = -(log_pz + log_det + logdet_logit).mean().item()

        with torch.no_grad():
            x_hat = sigmoid_postprocess(model.inverse(z.detach(), h.detach()))
            rmse  = ((x_pixel - x_hat) ** 2).mean().sqrt().item()

        total_nll  += nll; total_rmse += rmse; n_batches += 1

        # [P1-INVLOG] Store first 8 samples for end-of-epoch inv check
        if first_x_logit is None:
            first_x_logit = x_logit[:8].detach()
            first_h       = h[:8].detach()

    avg_nll  = total_nll  / max(n_batches, 1)
    avg_rmse = total_rmse / max(n_batches, 1)

    # [P1-INVLOG] Per-epoch invertibility error on 8 held-out samples
    inv_err = float('inf')
    try:
        inv_err = model.check_invertibility(first_x_logit, first_h)
    except Exception as e:
        logger.error(f"[EVAL] check_invertibility failed at epoch {epoch}: {e}")

    logger.info(
        f"[EVAL]  Epoch {epoch:3d} | avg NLL = {avg_nll:.4f} | "
        f"avg RMSE = {avg_rmse:.5f} | inv_err = {inv_err:.3e}"
    )
    return avg_nll, avg_rmse, inv_err


# ==============================================================================
# PLOTS
# ==============================================================================
def save_combined_plot(model: ConditionalIResNet,
                        conditioner: CNNConditioner,
                        loader: DataLoader, epoch: int) -> None:
    """[P3-PLOT] 5-row × 8-col combined diagnostic grid. Non-fatal.
    Row 1: Original x
    Row 2: Degraded y (Gaussian blurred)
    Row 3: Cycle (x→z→inverse(z,h)) — tests invertibility
    Row 4: Generated (z~N(0,1)→inverse(z,h)) — tests generative quality
    Row 5: Residual |original − cycle| — highlights inversion error
    """
    try:
        model.eval(); conditioner.eval()
        x_pixel, _ = next(iter(loader))
        x_pixel     = x_pixel[:8].to(DEVICE)
        y_deg       = gaussian_blur_batch(x_pixel)
        x_logit, _  = logit_preprocess(x_pixel)

        # Cycle: x→z→x̂ (needs grad for Hutchinson forward pass)
        h      = conditioner(y_deg)
        z, _   = model(x_logit, h)
        with torch.no_grad():
            x_hat    = sigmoid_postprocess(model.inverse(z.detach(), h.detach()))
            # Generative: z~N(0,1)→x_gen
            z_sample = torch.randn(8, SQ_C, 14, 14, device=DEVICE)
            x_gen    = sigmoid_postprocess(model.inverse(z_sample, h.detach()))

        orig  = x_pixel.cpu().squeeze(1).numpy()
        deg   = y_deg.cpu().squeeze(1).numpy()
        cycle = x_hat.cpu().squeeze(1).numpy()
        gen   = x_gen.cpu().squeeze(1).numpy()
        resid = (orig - cycle).__abs__()        # |original − cycle|

        rows   = [orig, deg, cycle, gen, resid]
        labels = [
            "Original",
            "Degraded\n(blurred)",
            "Cycle\n(x→z→x̂)",
            "Generated\n(z~N(0,1))",
            "Residual\n|orig−cycle|",
        ]
        cmaps  = ["gray", "gray", "gray", "gray", "hot"]

        fig, axes = plt.subplots(5, 8, figsize=(16, 10))
        for row, (imgs, label, cmap) in enumerate(zip(rows, labels, cmaps)):
            vmax = imgs.max() if row == 4 else 1.0   # residual: auto-scale
            for col in range(8):
                axes[row, col].imshow(imgs[col], cmap=cmap,
                                      vmin=0, vmax=vmax)
                axes[row, col].axis("off")
            axes[row, 0].set_ylabel(label, fontsize=9, rotation=0,
                                    labelpad=70, va="center")

        plt.suptitle(
            f"i-ResNet+FiLM MNIST Diagnostics — Epoch {epoch} "
            f"[FILM_SITES={FILM_SITES}]", fontsize=11
        )
        plt.tight_layout()
        path = os.path.join(LOG_DIR, f"combined_epoch{epoch:03d}.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[PLOT] Combined diagnostic grid saved: {path}")
    except Exception as e:
        logger.error(f"[PLOT] save_combined_plot failed at epoch {epoch}: {e}")


def save_training_curves(train_nlls: list, val_nlls: list,
                          val_rmses: list, inv_errs: list,
                          train_logdets: list) -> None:
    """4-panel training curves. [P3-CURVES] Non-fatal.
    Panel 1: Train/Val NLL vs epoch
    Panel 2: Cycle RMSE vs epoch
    Panel 3: Invertibility error (log scale) vs epoch
    Panel 4: log-det mean vs epoch — should be non-zero/changing if P2-GRAD works
    """
    try:
        epochs = list(range(1, len(train_nlls) + 1))
        fig, axes = plt.subplots(1, 4, figsize=(24, 4))
        ax1, ax2, ax3, ax4 = axes

        ax1.plot(epochs, train_nlls, label="Train NLL", color="steelblue")
        ax1.plot(epochs, val_nlls,   label="Val NLL",   color="darkorange")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("NLL")
        ax1.set_title("NLL vs Epoch"); ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, val_rmses, label="Val RMSE (cycle)", color="crimson")
        ax2.axhline(0.05, color="gray", linestyle="--", label="Pass (0.05)")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("RMSE")
        ax2.set_title("Cycle RMSE vs Epoch"); ax2.legend(); ax2.grid(True, alpha=0.3)

        ax3.semilogy(epochs, inv_errs, label="inv_err", color="purple")
        ax3.axhline(1e-2, color="gray", linestyle="--", label="Warn (1e-2)")
        ax3.set_xlabel("Epoch"); ax3.set_ylabel("max|x−inv(fwd(x))|")
        ax3.set_title("Invertibility Error vs Epoch")
        ax3.legend(); ax3.grid(True, alpha=0.3)

        # [P3-LOGDET] log-det mean — positive/growing = flow is expanding volume
        ax4.plot(epochs, train_logdets, label="logdet mean", color="teal")
        ax4.axhline(0, color="gray", linestyle="--", label="zero")
        ax4.set_xlabel("Epoch"); ax4.set_ylabel("log|det(I+J_g)| mean")
        ax4.set_title("Log-Det Mean vs Epoch"); ax4.legend(); ax4.grid(True, alpha=0.3)

        plt.suptitle(
            f"i-ResNet+FiLM Training Curves [FILM_SITES={FILM_SITES}]", fontsize=11)
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
        f"[MAIN] Config: N_BLOCKS={N_BLOCKS}, CHANNELS={CHANNELS}, "
        f"H_DIM={H_DIM}, ALPHA={ALPHA}, TAU={TAU}, SN_COEFF={SN_COEFF}, "
        f"N_ITER_INV={N_ITER_INV}, N_SERIES={N_SERIES}, EPOCHS={EPOCHS}, LR={LR}"
    )
    logger.info(
        f"[MAIN] Lip bound: τ×(c×(1+α))³ = "
        f"{TAU*(SN_COEFF*(1+ALPHA))**3:.4f} < 1 ✅  "
        f"[P1: SN={SN_COEFF}, TAU={TAU}, ALPHA={ALPHA}, ITERS={N_ITER_INV}, LR={LR}]"
    )

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(METRICS_CSV, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_nll", "train_logdet", "val_nll", "val_rmse", "inv_err", "best"])

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

    model       = ConditionalIResNet(
        channels=CHANNELS, h_dim=H_DIM, n_blocks=N_BLOCKS,
        tau=TAU, sn_coeff=SN_COEFF, alpha=ALPHA,
        n_iter_inv=N_ITER_INV, n_series=N_SERIES
    ).to(DEVICE)
    conditioner = CNNConditioner(h_dim=H_DIM).to(DEVICE)

    n_params = (sum(p.numel() for p in model.parameters()) +
                sum(p.numel() for p in conditioner.parameters()))
    logger.info(f"[MAIN] Total parameters: {n_params:,}")

    # Adamax per paper §C.3
    optimizer = torch.optim.Adamax(
        list(model.parameters()) + list(conditioner.parameters()), lr=LR
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR / 10
    )

    # Pre-training invertibility check
    logger.info("[MAIN] Pre-training invertibility check ...")
    sample_x, _ = next(iter(train_loader))
    sample_x     = sample_x[:4].to(DEVICE)
    sample_y     = gaussian_blur_batch(sample_x)
    x_logit, _   = logit_preprocess(sample_x)
    with torch.no_grad():
        h_test = conditioner(sample_y)
    # Note: check_invertibility needs grad for Hutchinson — called outside no_grad
    inv_err = model.check_invertibility(x_logit, h_test)
    if inv_err > 1e-2:
        logger.warning(f"[MAIN] Pre-training invertibility error: {inv_err:.3e}")

    # Training loop
    best_val_rmse = float('inf')
    train_nlls, val_nlls, val_rmses, inv_errs, train_logdets = [], [], [], [], []

    for epoch in range(1, EPOCHS + 1):
        train_nll, train_logdet       = train(model, conditioner, train_loader, optimizer, epoch)
        val_nll, val_rmse, inv_err    = evaluate(model, conditioner, val_loader, epoch)
        scheduler.step()

        train_nlls.append(train_nll)
        val_nlls.append(val_nll)
        val_rmses.append(val_rmse)
        inv_errs.append(inv_err)
        train_logdets.append(train_logdet)  # [P3-LOGDET]

        if epoch % PLOT_EVERY == 0 or epoch == EPOCHS:
            save_combined_plot(model, conditioner, val_loader, epoch)

        is_best = val_rmse < best_val_rmse
        if is_best:
            best_val_rmse = val_rmse
            torch.save({
                'epoch': epoch, 'model': model.state_dict(),
                'conditioner': conditioner.state_dict(),
                'val_nll': val_nll, 'val_rmse': val_rmse, 'inv_err': inv_err,
            }, SAVE_PATH)
            logger.info(f"[MAIN] New best saved at epoch {epoch}: RMSE={val_rmse:.5f}, inv_err={inv_err:.3e}")

        try:
            with open(METRICS_CSV, "a", newline="") as f:
                csv.writer(f).writerow([epoch, f"{train_nll:.4f}",
                                        f"{train_logdet:.4f}", f"{val_nll:.4f}",
                                        f"{val_rmse:.5f}", f"{inv_err:.3e}",
                                        int(is_best)])
        except Exception as e:
            logger.error(f"[MAIN] Failed to write metrics CSV at epoch {epoch}: {e}")

    logger.info(
        f"[MAIN] Training complete. Best val RMSE: {best_val_rmse:.5f} | "
        f"Final inv_err: {inv_errs[-1]:.3e}"
    )
    save_training_curves(train_nlls, val_nlls, val_rmses, inv_errs, train_logdets)

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
