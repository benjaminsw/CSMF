# ==============================================================================
# File    : test_realnvp_film_mnist.py
# Abbr    : TEST-RNVP-FILM
# Version : v1.0
# Created : 2026-04-16
# Changelog:
#   v1.0 (2026-04-16): Initial standalone RealNVP+FiLM MNIST reconstruction
#                      test. Self-contained (no CSMF imports). Architecture
#                      from realnvp_mnist.py converted to PyTorch: 5 affine
#                      coupling layers (log_s clamped [-2,2]), 2×512 CouplingNN,
#                      ActNorm between layers. FiLM after each hidden ReLU.
#                      CNN conditioner → h. Gaussian prior. Follows
#                      test_nice_film_mnist.py conventions: LOG_DIR, metrics.csv,
#                      run.log, reconstruction grid, training curves,
#                      check_invertibility() pre/post training.
# ==============================================================================

import csv
import logging
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
# CONFIG
# ==============================================================================
DIM        = 784      # MNIST 28×28 flattened
COND_DIM   = 128      # conditioning vector dimension
HIDDEN     = 512      # CouplingNN hidden units (realnvp_mnist: shape=[256,256] → 512)
N_HIDDEN   = 2        # CouplingNN hidden layers (realnvp_mnist: 2)
N_LAYERS   = 5        # affine coupling layers (realnvp_mnist: layers=5)
S_CLAMP    = 2.0      # log_s clamp — prevents exp(log_s) explosion
BLUR_K     = 5
BLUR_S     = 1.5
BATCH_SIZE = 128      # realnvp_mnist: batch_size=128
EPOCHS     = 30
LR         = 1e-4     # realnvp_mnist: base_lr=1e-4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
LOGIT_EPS  = 1e-6
DATA_DIR   = "./data"
LOG_DIR    = "./tests/logs/realnvp_film_mnist"
SAVE_PATH  = os.path.join(LOG_DIR, "best_checkpoint.pth")
METRICS_CSV = os.path.join(LOG_DIR, "metrics.csv")
PLOT_EVERY = 5

# LOG_DIR must exist before FileHandler is created
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "run.log"), mode="a"),
    ],
)
logger = logging.getLogger("TEST-RNVP-FILM")


# ==============================================================================
# HELPERS
# ==============================================================================
def logit_preprocess(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize + logit. Returns (x_logit, log_det_logit). (B,D) in [0,1]."""
    x = x + torch.zeros_like(x).uniform_(0, 1.0 / 256)
    x = x.clamp(LOGIT_EPS, 1 - LOGIT_EPS)
    log_det = (-torch.log(x) - torch.log(1 - x)).sum(dim=1)
    return torch.log(x) - torch.log(1 - x), log_det


def sigmoid_postprocess(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def gaussian_log_prob(z: torch.Tensor) -> torch.Tensor:
    """Standard Gaussian log-prob, summed over D. Returns (B,)."""
    return -0.5 * (z ** 2 + torch.log(torch.tensor(2 * 3.141592653589793))).sum(dim=1)


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
            logger.error("[FiLM] NaN/Inf in FiLM output")
            raise RuntimeError("NaN/Inf in FiLM output")
        return out


# ==============================================================================
# CouplingNN  (realnvp_mnist NN class → PyTorch + FiLM)
# 2 hidden layers × 512 units; outputs (log_s, t)
# log_s: tanh activation (paper); t: linear (paper)
# FiLM injected after each hidden ReLU
# Input: x_B (half_dim) concatenated with h (cond_dim)
# ==============================================================================
class CouplingNN(nn.Module):
    """
    Scale-and-translate network for RealNVP affine coupling.
    Outputs (log_s, t) where log_s uses tanh, t is linear.
    FiLM injected after each hidden ReLU (extension).
    Architecture: realnvp_mnist NN with 2×512 hidden layers.
    """
    def __init__(self, in_dim: int, out_dim: int, h_dim: int,
                 hidden: int = HIDDEN, n_hidden: int = N_HIDDEN):
        super().__init__()
        if n_hidden < 1:
            logger.error(f"[CouplingNN] n_hidden must be >= 1, got {n_hidden}")
            raise ValueError("CouplingNN requires n_hidden >= 1")

        self.fc_in = nn.Linear(in_dim + h_dim, hidden)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(n_hidden - 1)]
        )
        self.film_layers = nn.ModuleList(
            [FiLM(hidden, h_dim) for _ in range(n_hidden)]
        )
        # realnvp_mnist: log_s_layer uses tanh, t_layer is linear
        self.log_s_layer = nn.Linear(hidden, out_dim)
        self.t_layer     = nn.Linear(hidden, out_dim)
        self.act         = nn.ReLU()

        logger.debug(
            f"[CouplingNN] in_dim={in_dim}, out_dim={out_dim}, "
            f"h_dim={h_dim}, hidden={hidden}, n_hidden={n_hidden}"
        )

    def forward(self, xB: torch.Tensor, h: torch.Tensor):
        """Returns (log_s, t). log_s passes through tanh (paper spec)."""
        if xB.shape[0] != h.shape[0]:
            logger.error(
                f"[CouplingNN] Batch mismatch: xB={xB.shape[0]}, h={h.shape[0]}"
            )
            raise ValueError("Batch mismatch in CouplingNN.forward")

        out = self.act(self.fc_in(torch.cat([xB, h], dim=1)))
        out = self.film_layers[0](out, h)

        for i, fc in enumerate(self.hidden_layers):
            out = self.act(fc(out))
            out = self.film_layers[i + 1](out, h)

        log_s = torch.tanh(self.log_s_layer(out))  # tanh — paper spec
        t     = self.t_layer(out)                   # linear — paper spec

        if torch.isnan(log_s).any() or torch.isinf(log_s).any():
            logger.error("[CouplingNN] NaN/Inf in log_s output")
            raise RuntimeError("NaN/Inf in CouplingNN log_s")
        if torch.isnan(t).any() or torch.isinf(t).any():
            logger.error("[CouplingNN] NaN/Inf in t output")
            raise RuntimeError("NaN/Inf in CouplingNN t")
        return log_s, t


# ==============================================================================
# ActNorm  (replaces BatchNorm from realnvp_mnist — avoids batch/running stat
# inversion bug; data-driven init on first forward pass)
# ==============================================================================
class ActNorm(nn.Module):
    """
    Activation Normalisation (Kingma & Dhariwal, 2018).
    Learnable per-dim scale (log_scale) and shift, initialized from first batch.
    Forward:  y = (x + shift) * exp(log_scale),  log_det = log_scale.sum()
    Inverse:  x = y * exp(-log_scale) - shift
    Replaces BatchNorm — fully invertible in both train and eval mode.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim       = dim
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.shift     = nn.Parameter(torch.zeros(dim))
        self.register_buffer('initialized', torch.tensor(False))

    def _initialize(self, x: torch.Tensor) -> None:
        """Data-driven init: shift=-mean, log_scale=-log(std)."""
        with torch.no_grad():
            mean = x.mean(dim=0)
            std  = x.std(dim=0).clamp(min=1e-6)
            self.shift.data     = -mean
            self.log_scale.data = -torch.log(std)
        self.initialized.fill_(True)
        logger.debug("[ActNorm] Initialized from first batch.")

    def forward(self, x: torch.Tensor):
        """Returns (y, log_det). log_det: (B,) — same value broadcast."""
        if x.dim() != 2 or x.shape[1] != self.dim:
            logger.error(
                f"[ActNorm] forward shape mismatch: "
                f"expected (*, {self.dim}), got {tuple(x.shape)}"
            )
            raise ValueError("ActNorm.forward shape mismatch")

        if not self.initialized:
            self._initialize(x)

        y       = (x + self.shift) * torch.exp(self.log_scale)
        log_det = self.log_scale.sum().expand(x.shape[0])

        if torch.isnan(y).any() or torch.isinf(y).any():
            logger.error("[ActNorm] NaN/Inf after ActNorm forward")
            raise RuntimeError("NaN/Inf in ActNorm forward")
        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        if y.dim() != 2 or y.shape[1] != self.dim:
            logger.error(
                f"[ActNorm] inverse shape mismatch: "
                f"expected (*, {self.dim}), got {tuple(y.shape)}"
            )
            raise ValueError("ActNorm.inverse shape mismatch")

        x = y * torch.exp(-self.log_scale) - self.shift
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("[ActNorm] NaN/Inf after ActNorm inverse")
            raise RuntimeError("NaN/Inf in ActNorm inverse")
        return x


# ==============================================================================
# AffineCouplingLayer  (realnvp_mnist RealNVP bijector → PyTorch)
# y_A = x_A * exp(log_s(x_B, h)) + t(x_B, h)
# log_det = log_s.sum(dim=1)
# log_s clamped to [-S_CLAMP, S_CLAMP] for stability
# swap alternates which half is x_A vs x_B
# ==============================================================================
class AffineCouplingLayer(nn.Module):
    """
    RealNVP affine coupling (Dinh et al. 2017).
        Forward:  y_A = x_A * exp(log_s) + t(x_B, h),  y_B = x_B
                  log_det = sum(log_s_clamped)
        Inverse:  x_A = (y_A - t(y_B, h)) * exp(-log_s),  x_B = y_B

    swap=False: x_A = first half,  x_B = second half
    swap=True:  x_A = second half, x_B = first half
    Alternating swap achieves full-dimensional mixing (realnvp_mnist: Permute).
    log_s clamped to [-S_CLAMP, S_CLAMP] to prevent exp explosion.
    """
    def __init__(self, dim: int, cond_dim: int, swap: bool = False,
                 hidden: int = HIDDEN, n_hidden: int = N_HIDDEN,
                 s_clamp: float = S_CLAMP):
        super().__init__()
        if dim % 2 != 0:
            logger.error(f"[AffineCoupling] dim must be even, got {dim}")
            raise ValueError("AffineCouplingLayer requires even dim")

        self.dim     = dim
        self.half    = dim // 2
        self.swap    = swap
        self.s_clamp = s_clamp
        self.nn      = CouplingNN(self.half, self.half, cond_dim, hidden, n_hidden)

        logger.debug(
            f"[AffineCoupling] dim={dim}, swap={swap}, "
            f"cond_dim={cond_dim}, s_clamp={s_clamp}"
        )

    def _split(self, x: torch.Tensor):
        xA, xB = x.chunk(2, dim=1)
        if self.swap:
            return xB, xA   # swap: treat second half as x_A
        return xA, xB

    def _merge(self, xA: torch.Tensor, xB: torch.Tensor) -> torch.Tensor:
        if self.swap:
            return torch.cat([xB, xA], dim=1)
        return torch.cat([xA, xB], dim=1)

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        if x.dim() != 2 or x.shape[1] != self.dim:
            logger.error(
                f"[AffineCoupling] forward shape mismatch: "
                f"expected (*, {self.dim}), got {tuple(x.shape)}"
            )
            raise ValueError("AffineCouplingLayer.forward shape mismatch")
        if x.shape[0] != h.shape[0]:
            logger.error(
                f"[AffineCoupling] Batch mismatch: x={x.shape[0]}, h={h.shape[0]}"
            )
            raise ValueError("Batch mismatch in AffineCouplingLayer.forward")

        xA, xB   = self._split(x)
        log_s, t = self.nn(xB, h)
        log_s    = log_s.clamp(-self.s_clamp, self.s_clamp)  # stability
        yA       = xA * torch.exp(log_s) + t
        log_det  = log_s.sum(dim=1)
        return self._merge(yA, xB), log_det

    def inverse(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if y.dim() != 2 or y.shape[1] != self.dim:
            logger.error(
                f"[AffineCoupling] inverse shape mismatch: "
                f"expected (*, {self.dim}), got {tuple(y.shape)}"
            )
            raise ValueError("AffineCouplingLayer.inverse shape mismatch")
        if y.shape[0] != h.shape[0]:
            logger.error(
                f"[AffineCoupling] Batch mismatch inverse: y={y.shape[0]}, h={h.shape[0]}"
            )
            raise ValueError("Batch mismatch in AffineCouplingLayer.inverse")

        yA, yB   = self._split(y)
        log_s, t = self.nn(yB, h)
        log_s    = log_s.clamp(-self.s_clamp, self.s_clamp)  # same clamp as forward
        xA       = (yA - t) * torch.exp(-log_s)
        return self._merge(xA, yB)


# ==============================================================================
# ConditionalRealNVP
# realnvp_mnist: 5 layers of [BatchNorm + RealNVP + Permute]
# Here:          5 AffineCouplingLayers + ActNorm between layers, alternating swap
# ==============================================================================
class ConditionalRealNVP(nn.Module):
    """
    Conditional RealNVP with FiLM conditioning and ActNorm.

    Architecture (realnvp_mnist adapted to PyTorch + FiLM):
      - N_LAYERS=5 affine coupling layers, alternating swap (replaces Permute)
      - ActNorm between coupling layers (replaces BatchNorm — fully invertible)
      - CouplingNN: 2 hidden layers × 512 units, tanh log_s, linear t
      - FiLM injected after each hidden ReLU in CouplingNN
      - Gaussian prior N(0,I)
      - log_det = Σ(ActNorm log_det) + Σ(coupling log_det)

    API:
      forward(x, h) → (z, log_det)
      inverse(z, h) → x  (logit-space; sigmoid applied externally)
    """
    def __init__(self, dim: int = DIM, cond_dim: int = COND_DIM,
                 n_layers: int = N_LAYERS, hidden: int = HIDDEN,
                 n_hidden: int = N_HIDDEN, s_clamp: float = S_CLAMP):
        super().__init__()
        if dim % 2 != 0:
            logger.error(f"[COND-RNVP] dim must be even, got {dim}")
            raise ValueError("ConditionalRealNVP requires even dim")
        if n_layers < 1:
            logger.error(f"[COND-RNVP] n_layers must be >= 1, got {n_layers}")
            raise ValueError("ConditionalRealNVP requires n_layers >= 1")

        self.dim      = dim
        self.cond_dim = cond_dim
        self.n_layers = n_layers

        # Build: coupling → actnorm → coupling → actnorm → ... → coupling
        self.coupling_layers = nn.ModuleList([
            AffineCouplingLayer(dim, cond_dim, swap=(i % 2 == 1),
                                hidden=hidden, n_hidden=n_hidden,
                                s_clamp=s_clamp)
            for i in range(n_layers)
        ])
        # ActNorm between coupling layers: n_layers - 1 ActNorm modules
        self.actnorms = nn.ModuleList([
            ActNorm(dim) for _ in range(n_layers - 1)
        ])

        logger.info(
            f"[COND-RNVP] v1.0 initialized: dim={dim}, cond_dim={cond_dim}, "
            f"n_layers={n_layers}, hidden={hidden}, n_hidden={n_hidden}, "
            f"s_clamp={s_clamp}, actnorms={n_layers - 1}"
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        """
        x: (B, dim) logit-space input
        h: (B, cond_dim) conditioning features
        Returns (z, log_det): log_det includes coupling + actnorm contributions
        """
        if x.dim() != 2 or x.shape[1] != self.dim:
            logger.error(
                f"[COND-RNVP] forward shape mismatch: "
                f"expected (*, {self.dim}), got {tuple(x.shape)}"
            )
            raise ValueError("ConditionalRealNVP.forward shape mismatch")
        if h.dim() != 2:
            logger.error(f"[COND-RNVP] h must be rank-2, got {tuple(h.shape)}")
            raise ValueError("ConditionalRealNVP.forward requires rank-2 h")
        if x.shape[0] != h.shape[0]:
            logger.error(
                f"[COND-RNVP] Batch mismatch: x={x.shape[0]}, h={h.shape[0]}"
            )
            raise ValueError("Batch mismatch in ConditionalRealNVP.forward")

        z       = x
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        for i, coupling in enumerate(self.coupling_layers):
            z, ld = coupling(z, h)
            log_det += ld
            # ActNorm between coupling layers (not after last)
            if i < len(self.actnorms):
                z, ld = self.actnorms[i](z)
                log_det += ld

        if torch.isnan(z).any() or torch.isinf(z).any():
            logger.error("[COND-RNVP] NaN/Inf in forward() output z")
            raise RuntimeError("NaN/Inf in ConditionalRealNVP.forward")
        return z, log_det

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Inverse: z → x (logit-space). Sigmoid applied externally."""
        if z.dim() != 2 or z.shape[1] != self.dim:
            logger.error(
                f"[COND-RNVP] inverse shape mismatch: "
                f"expected (*, {self.dim}), got {tuple(z.shape)}"
            )
            raise ValueError("ConditionalRealNVP.inverse shape mismatch")
        if h.dim() != 2:
            logger.error(f"[COND-RNVP] h must be rank-2 in inverse, got {tuple(h.shape)}")
            raise ValueError("ConditionalRealNVP.inverse requires rank-2 h")
        if z.shape[0] != h.shape[0]:
            logger.error(
                f"[COND-RNVP] Batch mismatch inverse: z={z.shape[0]}, h={h.shape[0]}"
            )
            raise ValueError("Batch mismatch in ConditionalRealNVP.inverse")

        x = z
        for i in reversed(range(len(self.coupling_layers))):
            if i < len(self.actnorms):
                x = self.actnorms[i].inverse(x)
            x = self.coupling_layers[i].inverse(x, h)

        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("[COND-RNVP] NaN/Inf detected after inverse()")
            raise RuntimeError("NaN/Inf in ConditionalRealNVP.inverse")
        return x

    @torch.no_grad()
    def check_invertibility(self, x: torch.Tensor, h: torch.Tensor,
                             tol: float = 1e-4) -> float:
        """max ‖x - f⁻¹(f(x))‖_∞. Logs warning if > tol."""
        z, _  = self.forward(x, h)
        x_hat = self.inverse(z, h)
        err   = (x - x_hat).abs().max().item()
        if err > tol:
            logger.warning(
                f"[COND-RNVP] Invertibility FAILED: max_err={err:.3e} > tol={tol:.3e}"
            )
        else:
            logger.info(f"[COND-RNVP] Invertibility PASSED: max_err={err:.3e}")
        return err


# ==============================================================================
# CNN Conditioner  (same as test_nice_film_mnist.py)
# ==============================================================================
class CNNConditioner(nn.Module):
    """4-conv CNN encoder: y (1×28×28) → h ∈ R^{cond_dim}."""
    def __init__(self, cond_dim: int = COND_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32,  3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),   # 14×14
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),  # 7×7
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
            logger.error("[CNNConditioner] NaN/Inf in conditioner output h")
            raise RuntimeError("NaN/Inf in CNNConditioner output")
        return h


# ==============================================================================
# TRAINING
# ==============================================================================
def train(model: ConditionalRealNVP, conditioner: CNNConditioner,
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
                f"log_pz={log_pz.mean().item():.3f}, log_det={log_det.mean().item():.3f}"
            )
            raise RuntimeError("NaN/Inf loss during training")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(conditioner.parameters()), max_norm=1.0
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
def evaluate(model: ConditionalRealNVP, conditioner: CNNConditioner,
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
def save_reconstruction_plot(model: ConditionalRealNVP,
                              conditioner: CNNConditioner,
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
            zip([orig, deg, recon], ["Original", "Degraded\n(blurred)", "Reconstruction"])
        ):
            for col in range(8):
                axes[row, col].imshow(imgs[col], cmap="gray", vmin=0, vmax=1)
                axes[row, col].axis("off")
            axes[row, 0].set_ylabel(label, fontsize=9, rotation=0,
                                    labelpad=60, va="center")

        plt.suptitle(f"RealNVP+FiLM MNIST Reconstruction — Epoch {epoch}", fontsize=11)
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
        ax2.axhline(0.05, color="gray", linestyle="--", label="Pass threshold (0.05)")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("RMSE")
        ax2.set_title("Reconstruction RMSE vs Epoch")
        ax2.legend(); ax2.grid(True, alpha=0.3)

        plt.suptitle("RealNVP+FiLM MNIST Training Curves", fontsize=11)
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
        f"[MAIN] Config: DIM={DIM}, COND_DIM={COND_DIM}, HIDDEN={HIDDEN}, "
        f"N_HIDDEN={N_HIDDEN}, N_LAYERS={N_LAYERS}, S_CLAMP={S_CLAMP}, "
        f"EPOCHS={EPOCHS}, LR={LR}"
    )

    os.makedirs(DATA_DIR, exist_ok=True)

    # Metrics CSV header
    with open(METRICS_CSV, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_nll", "val_nll", "val_rmse", "best"])

    # Data
    tf_t = transforms.ToTensor()
    train_ds     = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=tf_t)
    val_ds       = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    logger.info(f"[MAIN] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Models
    model       = ConditionalRealNVP(dim=DIM, cond_dim=COND_DIM, n_layers=N_LAYERS,
                                     hidden=HIDDEN, n_hidden=N_HIDDEN,
                                     s_clamp=S_CLAMP).to(DEVICE)
    conditioner = CNNConditioner(cond_dim=COND_DIM).to(DEVICE)

    n_params = (sum(p.numel() for p in model.parameters()) +
                sum(p.numel() for p in conditioner.parameters()))
    logger.info(f"[MAIN] Total parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(conditioner.parameters()), lr=LR
    )
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=EPOCHS, power=0.5
    )  # realnvp_mnist: PolynomialDecay base_lr→end_lr

    # Pre-training: ActNorm init + invertibility check
    logger.info("[MAIN] Initializing ActNorm and pre-training invertibility check ...")
    sample_x, _ = next(iter(train_loader))
    sample_x     = sample_x[:8].to(DEVICE)
    sample_y     = gaussian_blur_batch(sample_x)
    x_flat, _    = logit_preprocess(sample_x.view(8, -1))
    with torch.no_grad():
        h_test = conditioner(sample_y)
        # Trigger ActNorm data-dependent init
        _ = model(x_flat, h_test)
    inv_err = model.check_invertibility(x_flat, h_test)
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
        h_test = conditioner(sample_y)
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
