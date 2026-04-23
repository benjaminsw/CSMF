# ==============================================================================
# File    : test_glow_film_mnist.py
# Abbr    : TEST-GLOW-FILM
# Version : v1.0
# Created : 2026-04-16
# Changelog:
#   v1.0 (2026-04-16): Initial standalone Glow+FiLM MNIST reconstruction test.
#                      Self-contained (no CSMF imports). Architecture from
#                      OpenAI Glow CIFAR-10 config (n_level=3, depth=32,
#                      hidden=512, n_bits_x=8) adapted to MNIST: input padded
#                      28→32 for squeeze compatibility; spatial pipeline
#                      (B,C,H,W) throughout flow levels; flattened only for
#                      prior log-prob. FlowStep = ActNorm → InvConv1x1 →
#                      AffineCoupling (flow_coupling=1, affine). FiLM after
#                      each hidden ReLU in coupling MLP. Multi-scale split at
#                      each level except last. Gaussian prior. CNN conditioner
#                      → h. Follows test_nice_film_mnist.py conventions:
#                      LOG_DIR, metrics.csv, run.log, reconstruction grid,
#                      training curves, check_invertibility() pre/post training.
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
# CONFIG  (Glow CIFAR-10 spec adapted for MNIST)
# ==============================================================================
N_LEVELS    = 3       # glow: n_level=3
DEPTH       = 32      # glow: depth=32 flow steps per level
HIDDEN      = 512     # glow: 512-unit coupling NN
N_HIDDEN    = 3       # glow: 3 hidden layers in coupling NN
N_BITS_X    = 8       # glow: n_bits_x=8 (uniform dequantization guard)
IMG_SIZE    = 32      # pad 28→32 for 2×2 squeeze compatibility
IN_CHANNELS = 1       # MNIST is grayscale
COND_DIM    = 128
S_CLAMP     = 2.0     # log_s clamp in affine coupling
BLUR_K      = 5
BLUR_S      = 1.5
BATCH_SIZE  = 128
EPOCHS      = 30
LR          = 1e-4    # glow: lr=0.001; lowered for stability on 1 GPU
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
LOGIT_EPS   = 1e-6
DATA_DIR    = "./data"
LOG_DIR     = "./tests/logs/glow_film_mnist"
SAVE_PATH   = os.path.join(LOG_DIR, "best_checkpoint.pth")
METRICS_CSV = os.path.join(LOG_DIR, "metrics.csv")
PLOT_EVERY  = 5

# LOG_DIR created before FileHandler
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "run.log"), mode="a"),
    ],
)
logger = logging.getLogger("TEST-GLOW-FILM")


# ==============================================================================
# HELPERS
# ==============================================================================
def pad_to_32(x: torch.Tensor) -> torch.Tensor:
    """Zero-pad (B,1,28,28) → (B,1,32,32) for squeeze compatibility."""
    return F.pad(x, (2, 2, 2, 2))


def unpad_from_32(x: torch.Tensor) -> torch.Tensor:
    """Remove padding: (B,1,32,32) → (B,1,28,28)."""
    return x[:, :, 2:30, 2:30]


def preprocess(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Glow-style preprocessing (n_bits_x=8):
      1. Dequantize: x_uint8 + U[0,1]
      2. Scale to [0,1]: / 256
      3. Logit transform with eps clamp
    Returns (x_logit, log_det_logit). x: (B,1,28,28) in [0,1].
    """
    x = x * 255.0                                      # [0,1] → [0,255]
    x = x + torch.zeros_like(x).uniform_(0, 1)        # dequantize
    x = x / 256.0                                      # back to (0,1)
    x = x.clamp(LOGIT_EPS, 1 - LOGIT_EPS)
    log_det = (-torch.log(x) - torch.log(1 - x))      # per-dim Jacobian
    log_det = log_det.view(x.shape[0], -1).sum(dim=1)  # (B,)
    x_logit = torch.log(x) - torch.log(1 - x)
    return x_logit, log_det


def sigmoid_postprocess(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def gaussian_log_prob_flat(z: torch.Tensor) -> torch.Tensor:
    """Standard Gaussian log-prob over flattened z. Returns (B,)."""
    z_flat = z.view(z.shape[0], -1)
    return -0.5 * (z_flat ** 2 + math.log(2 * math.pi)).sum(dim=1)


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
            logger.error("[FiLM] NaN/Inf in FiLM output")
            raise RuntimeError("NaN/Inf in FiLM output")
        return out


# ==============================================================================
# ActNorm  (Glow paper §3.1 — replaces BatchNorm)
# Per-channel learnable scale + shift, data-driven init from first batch.
# Operates on (B, C, H, W) spatial tensors.
# ==============================================================================
class ActNorm(nn.Module):
    """
    Activation Normalisation (Kingma & Dhariwal 2018, Glow §3.1).
    Learnable per-channel log_scale and shift.
    Data-driven init: shift = -mean, log_scale = -log(std) per channel.
    Forward:  y = (x + shift) * exp(log_scale),  log_det = H*W * log_scale.sum()
    Inverse:  x = y * exp(-log_scale) - shift
    """
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.log_scale  = nn.Parameter(torch.zeros(1, n_channels, 1, 1))
        self.shift      = nn.Parameter(torch.zeros(1, n_channels, 1, 1))
        self.register_buffer('initialized', torch.tensor(False))

    def _initialize(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            mean = x.mean(dim=[0, 2, 3], keepdim=True)
            std  = x.std(dim=[0, 2, 3], keepdim=True).clamp(min=1e-6)
            self.shift.data     = -mean
            self.log_scale.data = -torch.log(std)
        self.initialized.fill_(True)
        logger.debug(f"[ActNorm] Initialized from first batch (C={self.n_channels})")

    LOG_SCALE_CLAMP = 3.0  # [CLAMP] prevents exp(log_scale) explosion on logit-space input

    def forward(self, x: torch.Tensor):
        """x: (B,C,H,W). Returns (y, log_det) where log_det: (B,)."""
        if x.dim() != 4 or x.shape[1] != self.n_channels:
            logger.error(
                f"[ActNorm] forward shape mismatch: "
                f"expected (B,{self.n_channels},H,W), got {tuple(x.shape)}"
            )
            raise ValueError("ActNorm.forward shape mismatch")

        if not self.initialized:
            self._initialize(x)

        # [CLAMP] clamp log_scale to [-3, 3] — prevents log-det exploitation
        # and NaN/Inf in inverse when logit-space input has large variance
        ls      = self.log_scale.clamp(-self.LOG_SCALE_CLAMP, self.LOG_SCALE_CLAMP)
        y       = (x + self.shift) * torch.exp(ls)
        H, W    = x.shape[2], x.shape[3]
        log_det = ls.sum() * H * W               # scalar
        log_det = log_det.expand(x.shape[0])     # (B,)

        if torch.isnan(y).any() or torch.isinf(y).any():
            logger.error("[ActNorm] NaN/Inf in forward output")
            raise RuntimeError("NaN/Inf in ActNorm forward")
        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        if y.dim() != 4 or y.shape[1] != self.n_channels:
            logger.error(
                f"[ActNorm] inverse shape mismatch: "
                f"expected (B,{self.n_channels},H,W), got {tuple(y.shape)}"
            )
            raise ValueError("ActNorm.inverse shape mismatch")
        # [CLAMP] must use same clamped value as forward() for exact invertibility
        ls = self.log_scale.clamp(-self.LOG_SCALE_CLAMP, self.LOG_SCALE_CLAMP)
        x  = y * torch.exp(-ls) - self.shift
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("[ActNorm] NaN/Inf in inverse output")
            raise RuntimeError("NaN/Inf in ActNorm inverse")
        return x


# ==============================================================================
# InvConv1x1  (Glow paper §3.2 — invertible 1×1 convolution)
# Learned permutation via LU decomposition for tractable log_det.
# ==============================================================================
class InvConv1x1(nn.Module):
    """
    Invertible 1×1 convolution (Glow paper §3.2).
    Parameterised via LU decomposition: W = P @ L @ (U + diag(s))
    log_det = H * W * sum(log|s|)
    P fixed (random permutation), L lower-triangular, U upper-triangular.
    """
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels

        # Random orthogonal init via QR decomposition
        W_init = torch.linalg.qr(torch.randn(n_channels, n_channels))[0]

        # LU decomposition for tractable inverse/log_det
        P, L, U = torch.linalg.lu(W_init)
        self.register_buffer('P', P)                    # fixed permutation
        self.L  = nn.Parameter(L)                       # lower triangular
        self.U  = nn.Parameter(torch.triu(U, diagonal=1))  # strictly upper
        self.log_s = nn.Parameter(torch.log(U.diag().abs().clamp(min=1e-6)))

        # Masks to enforce triangular structure
        L_mask = torch.tril(torch.ones(n_channels, n_channels), diagonal=-1)
        self.register_buffer('L_mask', L_mask)
        self.register_buffer('eye', torch.eye(n_channels))

        logger.debug(f"[InvConv1x1] Initialized: n_channels={n_channels}")

    LOG_S_CLAMP = 3.0  # [CLAMP] prevents exp(log_s)→0 (singular W) or explosion

    def _get_weight(self):
        L = self.L * self.L_mask + self.eye
        # [CLAMP] clamp log_s — if log_s→-inf, exp(log_s)→0 → W singular → inv NaN
        log_s_clamped = self.log_s.clamp(-self.LOG_S_CLAMP, self.LOG_S_CLAMP)
        U = self.U + torch.diag(torch.exp(log_s_clamped))
        return self.P @ L @ U

    def forward(self, x: torch.Tensor):
        """x: (B,C,H,W). Returns (y, log_det)."""
        if x.dim() != 4 or x.shape[1] != self.n_channels:
            logger.error(
                f"[InvConv1x1] forward shape mismatch: "
                f"expected (B,{self.n_channels},H,W), got {tuple(x.shape)}"
            )
            raise ValueError("InvConv1x1.forward shape mismatch")

        W       = self._get_weight()
        W_conv  = W.unsqueeze(2).unsqueeze(3)          # (C,C,1,1)
        y       = F.conv2d(x, W_conv)
        H, W_hw = x.shape[2], x.shape[3]
        # log_det uses clamped log_s (same as _get_weight) for consistency
        log_det = self.log_s.clamp(-self.LOG_S_CLAMP, self.LOG_S_CLAMP).sum() * H * W_hw
        log_det = log_det.expand(x.shape[0])           # (B,)

        if torch.isnan(y).any() or torch.isinf(y).any():
            logger.error("[InvConv1x1] NaN/Inf in forward output")
            raise RuntimeError("NaN/Inf in InvConv1x1 forward")
        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        if y.dim() != 4 or y.shape[1] != self.n_channels:
            logger.error(
                f"[InvConv1x1] inverse shape mismatch: "
                f"expected (B,{self.n_channels},H,W), got {tuple(y.shape)}"
            )
            raise ValueError("InvConv1x1.inverse shape mismatch")

        W      = self._get_weight()
        W_inv  = torch.linalg.inv(W).unsqueeze(2).unsqueeze(3)
        x      = F.conv2d(y, W_inv)
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("[InvConv1x1] NaN/Inf in inverse output")
            raise RuntimeError("NaN/Inf in InvConv1x1 inverse")
        return x


# ==============================================================================
# AffineCouplingMLP  (Glow flow_coupling=1: affine)
# 3 hidden layers × 512 units, ReLU, FiLM after each hidden ReLU.
# Input: x_B flattened + h. Outputs (log_s, t) over x_A dims.
# ==============================================================================
class AffineCouplingMLP(nn.Module):
    """
    Scale-and-translate MLP for Glow affine coupling (flow_coupling=1).
    3 hidden layers × 512, ReLU, FiLM after each ReLU.
    log_s: tanh-activated. t: linear.
    Input: flattened x_B (half spatial channels) + h.
    """
    def __init__(self, in_dim: int, out_dim: int, h_dim: int,
                 hidden: int = HIDDEN, n_hidden: int = N_HIDDEN):
        super().__init__()
        if n_hidden < 1:
            logger.error(f"[CouplingMLP] n_hidden must be >= 1, got {n_hidden}")
            raise ValueError("AffineCouplingMLP requires n_hidden >= 1")

        self.fc_in = nn.Linear(in_dim + h_dim, hidden)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(n_hidden - 1)]
        )
        self.film_layers = nn.ModuleList(
            [FiLM(hidden, h_dim) for _ in range(n_hidden)]
        )
        self.log_s_layer = nn.Linear(hidden, out_dim)
        self.t_layer     = nn.Linear(hidden, out_dim)
        self.act         = nn.ReLU()

        logger.debug(
            f"[CouplingMLP] in={in_dim}, out={out_dim}, "
            f"hidden={hidden}, n_hidden={n_hidden}, h_dim={h_dim}"
        )

    def forward(self, xB_flat: torch.Tensor, h: torch.Tensor):
        if xB_flat.shape[0] != h.shape[0]:
            logger.error(
                f"[CouplingMLP] Batch mismatch: "
                f"xB={xB_flat.shape[0]}, h={h.shape[0]}"
            )
            raise ValueError("Batch mismatch in AffineCouplingMLP.forward")

        out = self.act(self.fc_in(torch.cat([xB_flat, h], dim=1)))
        out = self.film_layers[0](out, h)

        for i, fc in enumerate(self.hidden_layers):
            out = self.act(fc(out))
            out = self.film_layers[i + 1](out, h)

        log_s = torch.tanh(self.log_s_layer(out))
        t     = self.t_layer(out)

        if torch.isnan(log_s).any() or torch.isinf(log_s).any():
            logger.error("[CouplingMLP] NaN/Inf in log_s")
            raise RuntimeError("NaN/Inf in AffineCouplingMLP log_s")
        if torch.isnan(t).any() or torch.isinf(t).any():
            logger.error("[CouplingMLP] NaN/Inf in t")
            raise RuntimeError("NaN/Inf in AffineCouplingMLP t")
        return log_s, t


# ==============================================================================
# AffineCouplingLayer  (Glow flow_coupling=1)
# Operates on (B, C, H, W). Splits channels: x_A=first half, x_B=second half.
# y_A = x_A * exp(log_s(x_B, h)) + t(x_B, h)
# log_det = log_s_clamped.sum() per sample
# ==============================================================================
class AffineCouplingLayer(nn.Module):
    """
    Glow affine coupling (flow_coupling=1, Dinh et al. 2017).
        Forward:  y_A = x_A * exp(log_s) + t(x_B, h),  y_B = x_B
                  log_det = sum(log_s_clamped) per spatial position
        Inverse:  x_A = (y_A - t) * exp(-log_s),  x_B = y_B

    Splits along channel dim. x_B feeds the MLP; x_A is transformed.
    log_s clamped to [-S_CLAMP, S_CLAMP].
    """
    def __init__(self, n_channels: int, h_dim: int,
                 hidden: int = HIDDEN, n_hidden: int = N_HIDDEN,
                 s_clamp: float = S_CLAMP):
        super().__init__()
        if n_channels % 2 != 0:
            logger.error(
                f"[AffineCoupling] n_channels must be even, got {n_channels}"
            )
            raise ValueError("AffineCouplingLayer requires even n_channels")

        self.n_channels = n_channels
        self.half       = n_channels // 2
        self.s_clamp    = s_clamp

        # MLP operates on flattened spatial x_B: (B, half*H*W)
        # H,W are not fixed here — we flatten at call time
        # in_dim set dynamically; use lazy approach via first forward call flag
        self._mlp_built = False
        self._h_dim     = h_dim
        self._hidden    = hidden
        self._n_hidden  = n_hidden
        self.mlp        = None  # built on first forward call

    def _build_mlp(self, x_B: torch.Tensor) -> None:
        """Build MLP on first forward call when spatial dims are known."""
        in_dim  = x_B.shape[1] * x_B.shape[2] * x_B.shape[3]  # half * H * W
        out_dim = in_dim
        self.mlp = AffineCouplingMLP(
            in_dim, out_dim, self._h_dim, self._hidden, self._n_hidden
        ).to(x_B.device)
        self._mlp_built = True
        logger.debug(
            f"[AffineCoupling] MLP built: in_dim={in_dim}, "
            f"out_dim={out_dim}, h_dim={self._h_dim}"
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        """x: (B,C,H,W). Returns (y, log_det)."""
        if x.dim() != 4 or x.shape[1] != self.n_channels:
            logger.error(
                f"[AffineCoupling] forward shape mismatch: "
                f"expected (B,{self.n_channels},H,W), got {tuple(x.shape)}"
            )
            raise ValueError("AffineCouplingLayer.forward shape mismatch")
        if x.shape[0] != h.shape[0]:
            logger.error(
                f"[AffineCoupling] Batch mismatch: x={x.shape[0]}, h={h.shape[0]}"
            )
            raise ValueError("Batch mismatch in AffineCouplingLayer.forward")

        xA, xB = x[:, :self.half], x[:, self.half:]   # split on channel dim

        if not self._mlp_built:
            self._build_mlp(xB)

        B, C_B, H, W = xB.shape
        xB_flat      = xB.reshape(B, -1)
        log_s, t     = self.mlp(xB_flat, h)

        # Reshape log_s and t back to spatial
        log_s = log_s.reshape(B, C_B, H, W).clamp(-self.s_clamp, self.s_clamp)
        t     = t.reshape(B, C_B, H, W)

        yA      = xA * torch.exp(log_s) + t
        y       = torch.cat([yA, xB], dim=1)
        log_det = log_s.reshape(B, -1).sum(dim=1)   # (B,)
        return y, log_det

    def inverse(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if y.dim() != 4 or y.shape[1] != self.n_channels:
            logger.error(
                f"[AffineCoupling] inverse shape mismatch: "
                f"expected (B,{self.n_channels},H,W), got {tuple(y.shape)}"
            )
            raise ValueError("AffineCouplingLayer.inverse shape mismatch")
        if not self._mlp_built:
            logger.error("[AffineCoupling] MLP not built — call forward() first")
            raise RuntimeError("AffineCouplingLayer MLP not yet built")

        yA, yB = y[:, :self.half], y[:, self.half:]
        B, C_B, H, W = yB.shape
        yB_flat      = yB.reshape(B, -1)
        log_s, t     = self.mlp(yB_flat, h)

        log_s = log_s.reshape(B, C_B, H, W).clamp(-self.s_clamp, self.s_clamp)
        t     = t.reshape(B, C_B, H, W)
        xA    = (yA - t) * torch.exp(-log_s)
        return torch.cat([xA, yB], dim=1)


# ==============================================================================
# Squeeze  — spatial 2×2 → channel: (B,C,H,W) → (B,4C,H/2,W/2)
# ==============================================================================
class Squeeze(nn.Module):
    """
    Squeeze operation: 2×2 spatial blocks → 4× channels.
    (B, C, H, W) → (B, 4C, H//2, W//2)
    Inverse: (B, 4C, H//2, W//2) → (B, C, H, W)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if H % 2 != 0 or W % 2 != 0:
            logger.error(
                f"[Squeeze] H and W must be even, got H={H}, W={W}"
            )
            raise ValueError("Squeeze requires even spatial dims")
        x = x.reshape(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4)   # (B, C, 2, 2, H//2, W//2)
        x = x.reshape(B, C * 4, H // 2, W // 2)
        return x

    def inverse(self, x: torch.Tensor, orig_C: int) -> torch.Tensor:
        B, C4, H2, W2 = x.shape
        C = orig_C
        x = x.reshape(B, C, 2, 2, H2, W2)
        x = x.permute(0, 1, 4, 2, 5, 3)   # (B, C, H2, 2, W2, 2)
        x = x.reshape(B, C, H2 * 2, W2 * 2)
        return x


# ==============================================================================
# FlowStep  = ActNorm → InvConv1x1 → AffineCoupling
# (Glow paper Fig. 2)
# ==============================================================================
class FlowStep(nn.Module):
    """
    Single Glow flow step (Glow paper Fig. 2):
      ActNorm → InvConv1x1 → AffineCouplingLayer

    Forward:  y, log_det = ActNorm → InvConv1x1 → AffineCoupling
    Inverse:  x = AffineCoupling⁻¹ → InvConv1x1⁻¹ → ActNorm⁻¹
    """
    def __init__(self, n_channels: int, h_dim: int,
                 hidden: int = HIDDEN, n_hidden: int = N_HIDDEN,
                 s_clamp: float = S_CLAMP):
        super().__init__()
        self.actnorm  = ActNorm(n_channels)
        self.invconv  = InvConv1x1(n_channels)
        self.coupling = AffineCouplingLayer(n_channels, h_dim,
                                            hidden, n_hidden, s_clamp)

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        x, ld = self.actnorm(x);        log_det += ld
        x, ld = self.invconv(x);        log_det += ld
        x, ld = self.coupling(x, h);    log_det += ld

        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("[FlowStep] NaN/Inf in forward output")
            raise RuntimeError("NaN/Inf in FlowStep forward")
        return x, log_det

    def inverse(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x = self.coupling.inverse(y, h)
        x = self.invconv.inverse(x)
        x = self.actnorm.inverse(x)
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("[FlowStep] NaN/Inf in inverse output")
            raise RuntimeError("NaN/Inf in FlowStep inverse")
        return x


# ==============================================================================
# FlowLevel  = Squeeze → depth × FlowStep → Split (except last level)
# ==============================================================================
class FlowLevel(nn.Module):
    """
    One Glow multi-scale level:
      Squeeze → depth FlowSteps → Split (if not last level)

    Split: half channels pushed to Gaussian prior (log_prob accumulated),
           half passed to next level.
    Last level: no split — all channels passed to prior.
    """
    def __init__(self, in_channels: int, h_dim: int, depth: int,
                 is_last: bool = False,
                 hidden: int = HIDDEN, n_hidden: int = N_HIDDEN,
                 s_clamp: float = S_CLAMP):
        super().__init__()
        self.is_last     = is_last
        self.squeeze     = Squeeze()
        squeezed_C       = in_channels * 4   # after squeeze

        self.steps = nn.ModuleList([
            FlowStep(squeezed_C, h_dim, hidden, n_hidden, s_clamp)
            for _ in range(depth)
        ])

        self.in_channels  = in_channels
        self.squeezed_C   = squeezed_C
        self.split_C      = squeezed_C // 2   # channels split to prior

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        """
        Returns (z_out, z_split, log_det).
        z_split: channels pushed to prior (None if last level).
        z_out:   channels passed to next level (or final prior if last).
        """
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        x = self.squeeze(x)

        for step in self.steps:
            x, ld = step(x, h)
            log_det += ld

        if self.is_last:
            return x, None, log_det
        else:
            z_out   = x[:, :self.split_C]
            z_split = x[:, self.split_C:]
            return z_out, z_split, log_det

    def inverse(self, z_out: torch.Tensor, z_split,
                h: torch.Tensor) -> torch.Tensor:
        """Reconstruct x from z_out and z_split (None if last level)."""
        if self.is_last:
            x = z_out
        else:
            x = torch.cat([z_out, z_split], dim=1)

        for step in reversed(self.steps):
            x = step.inverse(x, h)

        x = self.squeeze.inverse(x, self.in_channels)
        return x


# ==============================================================================
# GlowModel  — N_LEVELS FlowLevels + Gaussian prior
# ==============================================================================
class GlowModel(nn.Module):
    """
    Glow generative flow model with FiLM conditioning.

    Architecture (CIFAR-10 config adapted for MNIST):
      - Input padded 28→32 for squeeze compatibility
      - N_LEVELS=3 FlowLevels; each level: Squeeze + DEPTH=32 FlowSteps + Split
      - FlowStep = ActNorm → InvConv1x1 → AffineCoupling(MLP+FiLM)
      - Multi-scale Gaussian prior: split channels at each level (except last)
        contribute log_prob; all accumulated for total NLL
      - Gaussian prior N(0,I)

    API:
      forward(x, h) → (z_list, log_det, log_pz)
        z_list:  list of latent tensors (one per level split + final)
        log_det: total log|det J| (B,)
        log_pz:  total log p(z) under prior (B,)
      inverse(z_list, h) → x  (logit-space spatial; sigmoid + unpad externally)
    """
    def __init__(self, in_channels: int = IN_CHANNELS,
                 n_levels: int = N_LEVELS, depth: int = DEPTH,
                 h_dim: int = COND_DIM, hidden: int = HIDDEN,
                 n_hidden: int = N_HIDDEN, s_clamp: float = S_CLAMP):
        super().__init__()
        self.n_levels    = n_levels
        self.in_channels = in_channels

        levels    = []
        cur_C     = in_channels
        self.level_in_channels = []   # track for inverse

        for i in range(n_levels):
            is_last = (i == n_levels - 1)
            levels.append(
                FlowLevel(cur_C, h_dim, depth, is_last, hidden, n_hidden, s_clamp)
            )
            self.level_in_channels.append(cur_C)
            if not is_last:
                cur_C = (cur_C * 4) // 2   # after squeeze+split: 4C/2 = 2C

        self.levels = nn.ModuleList(levels)

        logger.info(
            f"[GlowModel] v1.0 initialized: in_channels={in_channels}, "
            f"n_levels={n_levels}, depth={depth}, h_dim={h_dim}, "
            f"hidden={hidden}, n_hidden={n_hidden}, s_clamp={s_clamp}"
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        """
        x: (B, C, 32, 32) logit-space — padded to 32 before calling.
        h: (B, h_dim) conditioning.
        Returns (z_list, log_det, log_pz).
        """
        if x.dim() != 4:
            logger.error(f"[GlowModel] forward expects 4D input, got {x.dim()}D")
            raise ValueError("GlowModel.forward expects (B,C,H,W)")
        if x.shape[0] != h.shape[0]:
            logger.error(
                f"[GlowModel] Batch mismatch: x={x.shape[0]}, h={h.shape[0]}"
            )
            raise ValueError("Batch mismatch in GlowModel.forward")

        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        log_pz  = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        z_list  = []
        z       = x

        for level in self.levels:
            z_out, z_split, ld = level(z, h)
            log_det += ld

            if z_split is not None:
                # Accumulate prior log-prob for split channels
                log_pz += gaussian_log_prob_flat(z_split)
                z_list.append(z_split)

            z = z_out

        # Final level: all remaining channels go to prior
        log_pz += gaussian_log_prob_flat(z)
        z_list.append(z)

        if torch.isnan(log_det).any() or torch.isinf(log_det).any():
            logger.error("[GlowModel] NaN/Inf in log_det")
            raise RuntimeError("NaN/Inf in GlowModel log_det")

        return z_list, log_det, log_pz

    def inverse(self, z_list: list, h: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct x from z_list.
        z_list layout (set by forward):
          z_list[0..n_levels-2] = split tensors from levels 0..n_levels-2
          z_list[n_levels-1]    = final level latent (no split)
        Inverse order: last level first, working back to level 0.
        """
        if len(z_list) != self.n_levels:
            logger.error(
                f"[GlowModel] inverse expects {self.n_levels} z tensors, "
                f"got {len(z_list)}"
            )
            raise ValueError("GlowModel.inverse z_list length mismatch")

        z_out = z_list[-1]   # start from final level latent

        for i in range(self.n_levels - 1, -1, -1):
            level   = self.levels[i]
            # last level has no split; all others use their own split tensor
            z_split = None if i == self.n_levels - 1 else z_list[i]
            z_out   = level.inverse(z_out, z_split, h)

        if torch.isnan(z_out).any() or torch.isinf(z_out).any():
            logger.error("[GlowModel] NaN/Inf in inverse output")
            raise RuntimeError("NaN/Inf in GlowModel inverse")
        return z_out

    @torch.no_grad()
    def check_invertibility(self, x: torch.Tensor, h: torch.Tensor,
                             tol: float = 1e-3) -> float:
        """max ‖x - f⁻¹(f(x))‖_∞. Returns max error. Logs warning if > tol."""
        z_list, _, _ = self.forward(x, h)
        x_hat        = self.inverse(z_list, h)
        err          = (x - x_hat).abs().max().item()
        if err > tol:
            logger.warning(
                f"[GlowModel] Invertibility FAILED: "
                f"max_err={err:.3e} > tol={tol:.3e}"
            )
        else:
            logger.info(f"[GlowModel] Invertibility PASSED: max_err={err:.3e}")
        return err


# ==============================================================================
# CNN Conditioner  (same as other tests — accepts 28×28 degraded input)
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
def train(model: GlowModel, conditioner: CNNConditioner,
          loader: DataLoader, optimizer: torch.optim.Optimizer,
          epoch: int) -> float:
    model.train()
    conditioner.train()
    total_loss = 0.0
    n_batches  = 0

    for batch_idx, (x_pixel, _) in enumerate(loader):
        x_pixel = x_pixel.to(DEVICE)                # (B,1,28,28) in [0,1]
        y       = gaussian_blur_batch(x_pixel)       # degraded observation

        x_logit, logdet_logit = preprocess(x_pixel)
        x_spatial = pad_to_32(x_logit)              # (B,1,32,32)

        h                   = conditioner(y)
        z_list, log_det, log_pz = model(x_spatial, h)

        # NLL = -[log_pz + log_det + logdet_logit]
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
            max_norm=1.0
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
def evaluate(model: GlowModel, conditioner: CNNConditioner,
             loader: DataLoader, epoch: int) -> tuple[float, float]:
    """Returns (avg_nll, avg_rmse)."""
    model.eval()
    conditioner.eval()
    total_nll = total_rmse = 0.0
    n_batches = 0

    for x_pixel, _ in loader:
        x_pixel = x_pixel.to(DEVICE)
        y       = gaussian_blur_batch(x_pixel)

        x_logit, logdet_logit = preprocess(x_pixel)
        x_spatial = pad_to_32(x_logit)

        h = conditioner(y)
        z_list, log_det, log_pz = model(x_spatial, h)
        nll = -(log_pz + log_det + logdet_logit).mean().item()

        # Reconstruct: inverse → sigmoid → unpad
        x_rec_spatial = model.inverse(z_list, h)
        x_rec_pixel   = unpad_from_32(sigmoid_postprocess(x_rec_spatial))
        rmse = ((x_pixel - x_rec_pixel) ** 2).mean().sqrt().item()

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
def save_reconstruction_plot(model: GlowModel, conditioner: CNNConditioner,
                              loader: DataLoader, epoch: int) -> None:
    """3-row × 8-col grid: original | degraded | reconstruction. Non-fatal."""
    try:
        model.eval(); conditioner.eval()
        x_pixel, _ = next(iter(loader))
        x_pixel     = x_pixel[:8].to(DEVICE)
        y           = gaussian_blur_batch(x_pixel)
        x_logit, _  = preprocess(x_pixel)
        x_spatial   = pad_to_32(x_logit)
        h           = conditioner(y)
        z_list, _, _ = model(x_spatial, h)
        x_rec       = unpad_from_32(
            sigmoid_postprocess(model.inverse(z_list, h))
        )

        orig  = x_pixel.cpu().squeeze(1).numpy()
        deg   = y.cpu().squeeze(1).numpy()
        recon = x_rec.cpu().squeeze(1).numpy()

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

        plt.suptitle(f"Glow+FiLM MNIST Reconstruction — Epoch {epoch}", fontsize=11)
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

        plt.suptitle("Glow+FiLM MNIST Training Curves", fontsize=11)
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
        f"[MAIN] Config: N_LEVELS={N_LEVELS}, DEPTH={DEPTH}, HIDDEN={HIDDEN}, "
        f"N_HIDDEN={N_HIDDEN}, N_BITS_X={N_BITS_X}, S_CLAMP={S_CLAMP}, "
        f"EPOCHS={EPOCHS}, LR={LR}"
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

    model       = GlowModel(in_channels=IN_CHANNELS, n_levels=N_LEVELS,
                            depth=DEPTH, h_dim=COND_DIM, hidden=HIDDEN,
                            n_hidden=N_HIDDEN, s_clamp=S_CLAMP).to(DEVICE)
    conditioner = CNNConditioner(cond_dim=COND_DIM).to(DEVICE)

    n_params = (sum(p.numel() for p in model.parameters()) +
                sum(p.numel() for p in conditioner.parameters()))
    logger.info(f"[MAIN] Total parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(conditioner.parameters()), lr=LR
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Pre-training: ActNorm data-driven init + invertibility check
    logger.info("[MAIN] Initializing ActNorm and pre-training invertibility check ...")
    sample_x, _ = next(iter(train_loader))
    sample_x     = sample_x[:8].to(DEVICE)
    sample_y     = gaussian_blur_batch(sample_x)
    x_logit, _   = preprocess(sample_x)
    x_spatial    = pad_to_32(x_logit)
    with torch.no_grad():
        h_test = conditioner(sample_y)
        _      = model(x_spatial, h_test)   # triggers ActNorm init

    inv_err = model.check_invertibility(x_spatial, h_test)
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
        h_test   = conditioner(sample_y)
        x_logit, _ = preprocess(sample_x)
        x_spatial  = pad_to_32(x_logit)
    model.check_invertibility(x_spatial, h_test)

    if best_val_rmse < 0.05:
        logger.info("[MAIN] ✅ RECONSTRUCTION TEST PASSED: RMSE < 0.05")
    else:
        logger.warning(
            f"[MAIN] ⚠️  RECONSTRUCTION TEST: RMSE={best_val_rmse:.5f} >= 0.05"
        )


if __name__ == "__main__":
    main()
