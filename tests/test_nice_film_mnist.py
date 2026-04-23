# ==============================================================================
# File    : test_nice_film_mnist.py
# Abbr    : TEST-NICE-FILM
# Version : v1.2
# Created : 2026-04-16
# Changelog:
#   v1.2 (2026-04-16): [PLOT] Added save_reconstruction_plot() — saves 3-row
#                      grid (original/degraded/reconstruction) for 8 val samples
#                      to LOG_DIR/reconstruction_epoch{N}.png after each eval;
#                      added save_training_curves() — NLL+RMSE vs epoch saved
#                      to LOG_DIR/training_curves.png at end of training.
#   v1.1 (2026-04-16): [LOG] Added LOG_DIR=tests/logs/nice_film_mnist; file
#                      handler writes run.log alongside stdout; SAVE_PATH moved
#                      to LOG_DIR; metrics CSV written per epoch to metrics.csv;
#                      log dir created before logging setup to avoid race.
#   v1.0 (2026-04-16): Initial standalone NICE+FiLM MNIST reconstruction test.
#                      Self-contained (no CSMF imports). Paper-faithful pure
#                      additive coupling (log_det=0 per layer). Odd/even
#                      partition alternation. 4 coupling layers. FiLM injected
#                      after each MLP hidden ReLU. CNN conditioner for h.
#                      Trains on blurred MNIST, evaluates reconstruction RMSE.
# ==============================================================================

import csv
import logging
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display needed
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TEST-NICE-FILM")

# ==============================================================================
# CONFIG
# ==============================================================================
DIM        = 784          # MNIST 28×28 flattened
COND_DIM   = 128          # conditioning vector dimension
HIDDEN     = 1000         # coupling MLP hidden units (paper MNIST spec)
N_HIDDEN   = 5            # coupling MLP hidden layers (paper MNIST spec)
N_LAYERS   = 4            # number of coupling layers (paper MNIST spec)
BLUR_K     = 5            # Gaussian blur kernel size for degradation
BLUR_S     = 1.5          # Gaussian blur sigma
BATCH_SIZE = 256
EPOCHS     = 30
LR         = 1e-3
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
LOGIT_EPS  = 1e-6         # clamp for logit preprocessing
SCALE_CLAMP = 5.0         # scaling layer clamp (stability fix)
DATA_DIR    = "./data"
LOG_DIR     = "./tests/logs/nice_film_mnist"   # [LOG] v1.1
SAVE_PATH   = os.path.join(LOG_DIR, "best_checkpoint.pth")
METRICS_CSV = os.path.join(LOG_DIR, "metrics.csv")
PLOT_EVERY  = 5    # [PLOT] v1.2: save reconstruction grid every N epochs


# ==============================================================================
# HELPERS: logit preprocessing (unified CSMF convention)
# ==============================================================================
def logit_preprocess(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dequantize + logit transform.
    x: (B, D) in [0,1].  Returns (x_logit, log_det_logit).
    log_det_logit needed for correct NLL in logit space.
    """
    # Dequantize: add U[0, 1/256]
    x = x + torch.zeros_like(x).uniform_(0, 1.0 / 256)
    x = x.clamp(LOGIT_EPS, 1 - LOGIT_EPS)
    log_det = -torch.log(x) - torch.log(1 - x)   # per-dim; sum over D for per-sample
    log_det = log_det.sum(dim=1)                  # (B,)
    x_logit = torch.log(x) - torch.log(1 - x)
    return x_logit, log_det


def sigmoid_postprocess(x_logit: torch.Tensor) -> torch.Tensor:
    """Inverse of logit: sigmoid back to pixel space [0,1]."""
    return torch.sigmoid(x_logit)


# ==============================================================================
# FiLM layer
# ==============================================================================
class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation.
    Applies (1 + γ(h)) ⊙ f + β(h) where γ,β are linear projections.
    Init: γ_weight=0 → γ≡0 → identity start.
    """
    def __init__(self, f_dim: int, h_dim: int):
        super().__init__()
        self.gamma = nn.Linear(h_dim, f_dim)
        self.beta  = nn.Linear(h_dim, f_dim)
        # Identity init
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, f: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma(h)
        beta  = self.beta(h)
        out = (1.0 + gamma) * f + beta
        if torch.isnan(out).any() or torch.isinf(out).any():
            logger.error("[FiLM] NaN/Inf detected in FiLM output")
            raise RuntimeError("NaN/Inf in FiLM output")
        return out


# ==============================================================================
# Coupling MLP: 5 hidden layers × 1000 units + FiLM after each hidden ReLU
# (paper MNIST spec with FiLM extension)
# ==============================================================================
class CouplingMLP(nn.Module):
    """
    Translation network m(x_A, h) for additive coupling.
    Architecture: paper MNIST spec — 5 hidden layers of 1000 units, ReLU,
    linear output. FiLM injected after each hidden ReLU (extension).
    Input: x_A (D//2) concatenated with h (cond_dim).
    Output: translation t (D//2).
    """
    def __init__(self, in_dim: int, out_dim: int, h_dim: int,
                 hidden: int = HIDDEN, n_hidden: int = N_HIDDEN):
        super().__init__()
        if n_hidden < 1:
            raise ValueError(f"[CouplingMLP] n_hidden must be >= 1, got {n_hidden}")

        self.fc_in = nn.Linear(in_dim + h_dim, hidden)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(n_hidden - 1)]
        )
        self.film_layers = nn.ModuleList(
            [FiLM(hidden, h_dim) for _ in range(n_hidden)]
        )
        self.fc_out = nn.Linear(hidden, out_dim)  # linear output (paper spec)
        self.act    = nn.ReLU()

        logger.debug(
            f"[CouplingMLP] in_dim={in_dim}, out_dim={out_dim}, "
            f"hidden={hidden}, n_hidden={n_hidden}, h_dim={h_dim}"
        )

    def forward(self, xA: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if xA.shape[0] != h.shape[0]:
            logger.error(
                f"[CouplingMLP] Batch mismatch: xA={xA.shape[0]}, h={h.shape[0]}"
            )
            raise ValueError("Batch mismatch in CouplingMLP.forward")

        out = self.act(self.fc_in(torch.cat([xA, h], dim=1)))
        out = self.film_layers[0](out, h)

        for i, fc in enumerate(self.hidden_layers):
            out = self.act(fc(out))
            out = self.film_layers[i + 1](out, h)

        t = self.fc_out(out)   # linear output — no activation
        if torch.isnan(t).any() or torch.isinf(t).any():
            logger.error("[CouplingMLP] NaN/Inf in translation output")
            raise RuntimeError("NaN/Inf in CouplingMLP output")
        return t


# ==============================================================================
# Additive Coupling Layer (paper-faithful: log_det = 0)
# Partition: odd/even indices, alternated via mask_type ('even'|'odd')
# ==============================================================================
class AdditiveCouplingLayer(nn.Module):
    """
    Pure additive coupling (NICE paper §3.2).
        Forward:  y_B = x_B + m(x_A, h),  y_A = x_A,  log_det = 0
        Inverse:  x_B = y_B - m(y_A, h),  x_A = y_A

    mask_type='even': x_A = even-indexed dims, x_B = odd-indexed dims.
    mask_type='odd':  x_A = odd-indexed dims,  x_B = even-indexed dims.

    Log_det is identically 0 — volume preserving by construction.
    """
    def __init__(self, dim: int, h_dim: int, mask_type: str = 'even',
                 hidden: int = HIDDEN, n_hidden: int = N_HIDDEN):
        super().__init__()
        if dim % 2 != 0:
            logger.error(f"[AdditiveCoupling] dim must be even, got {dim}")
            raise ValueError("dim must be even in AdditiveCouplingLayer")
        if mask_type not in ('even', 'odd'):
            logger.error(f"[AdditiveCoupling] mask_type must be 'even' or 'odd', got {mask_type}")
            raise ValueError("mask_type must be 'even' or 'odd'")

        self.dim       = dim
        self.mask_type = mask_type
        self.half      = dim // 2

        # Build odd/even index masks (registered as buffers — device-agnostic)
        even_idx = torch.arange(0, dim, 2)
        odd_idx  = torch.arange(1, dim, 2)
        self.register_buffer('even_idx', even_idx)
        self.register_buffer('odd_idx',  odd_idx)

        self.m = CouplingMLP(self.half, self.half, h_dim, hidden, n_hidden)

        logger.debug(
            f"[AdditiveCoupling] dim={dim}, mask_type={mask_type}, "
            f"h_dim={h_dim}, half={self.half}"
        )

    def _split(self, x: torch.Tensor):
        if self.mask_type == 'even':
            return x[:, self.even_idx], x[:, self.odd_idx]   # xA, xB
        else:
            return x[:, self.odd_idx],  x[:, self.even_idx]  # xA, xB

    def _merge(self, xA: torch.Tensor, xB: torch.Tensor, orig_shape: int) -> torch.Tensor:
        """Reconstruct full vector, placing xA and xB back into their original indices."""
        out = torch.empty(xA.shape[0], orig_shape, device=xA.device, dtype=xA.dtype)
        if self.mask_type == 'even':
            out[:, self.even_idx] = xA
            out[:, self.odd_idx]  = xB
        else:
            out[:, self.odd_idx]  = xA
            out[:, self.even_idx] = xB
        return out

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        """Returns (y, log_det=0) — volume preserving."""
        if x.dim() != 2 or x.shape[1] != self.dim:
            logger.error(
                f"[AdditiveCoupling] forward shape mismatch: "
                f"expected (*, {self.dim}), got {tuple(x.shape)}"
            )
            raise ValueError("AdditiveCouplingLayer.forward shape mismatch")

        xA, xB  = self._split(x)
        t       = self.m(xA, h)          # translation: m(x_A, h)
        yB      = xB + t                 # additive coupling
        y       = self._merge(xA, yB, self.dim)
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)  # always 0
        return y, log_det

    def inverse(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Inverse: x_B = y_B - m(y_A, h)."""
        if y.dim() != 2 or y.shape[1] != self.dim:
            logger.error(
                f"[AdditiveCoupling] inverse shape mismatch: "
                f"expected (*, {self.dim}), got {tuple(y.shape)}"
            )
            raise ValueError("AdditiveCouplingLayer.inverse shape mismatch")

        yA, yB = self._split(y)
        t      = self.m(yA, h)
        xB     = yB - t
        return self._merge(yA, xB, self.dim)


# ==============================================================================
# ConditionalNICE — paper-faithful + FiLM
# ==============================================================================
class ConditionalNICE(nn.Module):
    """
    Paper-faithful NICE with FiLM conditioning.

    Architecture (MNIST paper spec):
      - 4 additive coupling layers, alternating odd/even partition
      - Each coupling MLP: 5 hidden layers × 1000 units, ReLU, linear output
      - FiLM injected after each hidden ReLU (extension)
      - Learnable diagonal scaling: z = exp(s) ⊙ h^(4), s clamped [-5,5]
      - log_det = Σ s_i (scaling only; coupling layers contribute 0)

    API:
      forward(x, h) → (z, log_det)   x in logit-space
      inverse(z, h) → x              returns logit-space; sigmoid applied externally
    """
    def __init__(self, dim: int = DIM, cond_dim: int = COND_DIM,
                 hidden: int = HIDDEN, n_hidden: int = N_HIDDEN,
                 n_layers: int = N_LAYERS):
        super().__init__()
        if dim % 2 != 0:
            logger.error(f"[COND-NICE] dim must be even, got {dim}")
            raise ValueError("ConditionalNICE requires even dim")
        if n_layers < 1:
            logger.error(f"[COND-NICE] n_layers must be >= 1, got {n_layers}")
            raise ValueError("ConditionalNICE requires n_layers >= 1")

        self.dim      = dim
        self.cond_dim = cond_dim
        self.n_layers = n_layers

        # Alternate odd/even partitions (paper: alternates which half is transformed)
        mask_types = ['even' if i % 2 == 0 else 'odd' for i in range(n_layers)]
        self.coupling_layers = nn.ModuleList([
            AdditiveCouplingLayer(dim, cond_dim, mask_type=mt,
                                  hidden=hidden, n_hidden=n_hidden)
            for mt in mask_types
        ])

        # Learnable diagonal scaling (paper §3.3): parametrized as s, z = exp(s) ⊙ h
        self.scaling = nn.Parameter(torch.zeros(dim))

        logger.info(
            f"[COND-NICE] v1.0 initialized: dim={dim}, cond_dim={cond_dim}, "
            f"n_layers={n_layers}, hidden={hidden}, n_hidden={n_hidden}, "
            f"mask_types={mask_types}, log_det_source=scaling_only"
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        """
        x: (B, D) logit-space input
        h: (B, cond_dim) conditioning features
        Returns (z, log_det): z latent, log_det per-sample scalar
        """
        if x.dim() != 2 or x.shape[1] != self.dim:
            logger.error(
                f"[COND-NICE] forward shape mismatch: "
                f"expected (*, {self.dim}), got {tuple(x.shape)}"
            )
            raise ValueError("ConditionalNICE.forward shape mismatch")
        if x.shape[0] != h.shape[0]:
            logger.error(
                f"[COND-NICE] Batch mismatch: x={x.shape[0]}, h={h.shape[0]}"
            )
            raise ValueError("Batch mismatch in ConditionalNICE.forward")

        z = x
        # Coupling layers — log_det contributions are all 0 (pure additive)
        for i, layer in enumerate(self.coupling_layers):
            z, ld = layer(z, h)
            # ld is zero by construction; assert for safety during debug
            if not torch.allclose(ld, torch.zeros_like(ld), atol=1e-6):
                logger.error(
                    f"[COND-NICE] Coupling layer {i} log_det ≠ 0: "
                    f"max={ld.abs().max().item():.3e} — not pure additive"
                )
                raise RuntimeError(f"Coupling layer {i} is not volume-preserving")

        # Diagonal scaling: z = exp(s) ⊙ h^(4)
        s_clamped = self.scaling.clamp(-SCALE_CLAMP, SCALE_CLAMP)
        z         = z * torch.exp(s_clamped)
        log_det   = s_clamped.sum().expand(x.shape[0])  # (B,) — same for all samples

        if torch.isnan(z).any() or torch.isinf(z).any():
            logger.error("[COND-NICE] NaN/Inf after final scaling in forward()")
            raise RuntimeError("NaN/Inf after scaling in ConditionalNICE.forward")

        return z, log_det

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        z: (B, D) latent
        h: (B, cond_dim) conditioning features
        Returns x in logit-space.
        """
        if z.dim() != 2 or z.shape[1] != self.dim:
            logger.error(
                f"[COND-NICE] inverse shape mismatch: "
                f"expected (*, {self.dim}), got {tuple(z.shape)}"
            )
            raise ValueError("ConditionalNICE.inverse shape mismatch")
        if z.shape[0] != h.shape[0]:
            logger.error(
                f"[COND-NICE] Batch mismatch in inverse: z={z.shape[0]}, h={h.shape[0]}"
            )
            raise ValueError("Batch mismatch in ConditionalNICE.inverse")

        # Undo scaling — must use same clamped value as forward()
        s_clamped = self.scaling.clamp(-SCALE_CLAMP, SCALE_CLAMP)
        x = z * torch.exp(-s_clamped)

        # Reverse through coupling layers
        for layer in reversed(self.coupling_layers):
            x = layer.inverse(x, h)

        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("[COND-NICE] NaN/Inf detected in inverse()")
            raise RuntimeError("NaN/Inf in ConditionalNICE.inverse")

        return x

    @torch.no_grad()
    def check_invertibility(self, x: torch.Tensor, h: torch.Tensor,
                             tol: float = 1e-4) -> float:
        """Sanity check: max ‖x - f⁻¹(f(x))‖_∞. Returns max error."""
        z, _ = self.forward(x, h)
        x_hat = self.inverse(z, h)
        err = (x - x_hat).abs().max().item()
        if err > tol:
            logger.warning(
                f"[COND-NICE] Invertibility check FAILED: "
                f"max_err={err:.3e} > tol={tol:.3e}"
            )
        else:
            logger.info(f"[COND-NICE] Invertibility check PASSED: max_err={err:.3e}")
        return err


# ==============================================================================
# CNN Conditioner: degraded y → h
# ==============================================================================
class CNNConditioner(nn.Module):
    """
    Lightweight CNN encoder: y (1×28×28 blurred) → h ∈ R^{cond_dim}.
    4 conv layers + global avg pool + linear.
    """
    def __init__(self, cond_dim: int = COND_DIM):
        super().__init__()
        self.cond_dim = cond_dim
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),  # 14×14
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(), # 7×7
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(128, cond_dim)
        self.norm = nn.LayerNorm(cond_dim)

        logger.info(
            f"[CNNConditioner] initialized: cond_dim={cond_dim}"
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """y: (B, 1, 28, 28) degraded image. Returns h: (B, cond_dim)."""
        if y.dim() != 4 or y.shape[1] != 1:
            logger.error(
                f"[CNNConditioner] Expected (B,1,28,28), got {tuple(y.shape)}"
            )
            raise ValueError("CNNConditioner expects (B,1,28,28) input")
        feat = self.net(y)
        feat = self.pool(feat).squeeze(-1).squeeze(-1)  # (B, 128)
        h    = self.norm(self.head(feat))
        if torch.isnan(h).any() or torch.isinf(h).any():
            logger.error("[CNNConditioner] NaN/Inf in conditioning output h")
            raise RuntimeError("NaN/Inf in CNNConditioner output")
        return h


# ==============================================================================
# Logistic prior log-prob
# ==============================================================================
def logistic_log_prob(z: torch.Tensor) -> torch.Tensor:
    """
    Log-prob under standard logistic: log p(z) = -log(1+exp(z)) - log(1+exp(-z)).
    Returns per-sample sum over D: (B,).
    """
    return (-F.softplus(z) - F.softplus(-z)).sum(dim=1)


# ==============================================================================
# Degradation: Gaussian blur
# ==============================================================================
def gaussian_blur_batch(x: torch.Tensor, kernel_size: int = BLUR_K,
                         sigma: float = BLUR_S) -> torch.Tensor:
    """
    Apply Gaussian blur to (B, 1, 28, 28) tensor.
    Simulates the degradation observation y = A(x) for CSMF conditioning.
    """
    k    = kernel_size
    pad  = k // 2
    coords = torch.arange(k, dtype=x.dtype, device=x.device) - pad
    g    = torch.exp(-0.5 * (coords / sigma) ** 2)
    g    = g / g.sum()
    kernel_2d = g[:, None] * g[None, :]
    kernel_4d = kernel_2d.unsqueeze(0).unsqueeze(0)   # (1,1,k,k)
    return F.conv2d(x, kernel_4d, padding=pad)


# ==============================================================================
# TRAINING
# ==============================================================================
def train(model: ConditionalNICE, conditioner: CNNConditioner,
          loader: DataLoader, optimizer: torch.optim.Optimizer,
          epoch: int) -> float:
    model.train()
    conditioner.train()
    total_loss = 0.0
    n_batches  = 0

    for batch_idx, (x_pixel, _) in enumerate(loader):
        x_pixel = x_pixel.to(DEVICE)       # (B, 1, 28, 28) in [0,1]
        y       = gaussian_blur_batch(x_pixel)

        # Flatten + logit preprocess
        x_flat, logdet_logit = logit_preprocess(x_pixel.view(x_pixel.shape[0], -1))

        h      = conditioner(y)
        z, log_det_scaling = model(x_flat, h)

        # NLL = -log p(x) = -[log p(z) + log_det_scaling + logdet_logit]
        log_pz  = logistic_log_prob(z)
        log_px  = log_pz + log_det_scaling + logdet_logit
        loss    = -log_px.mean()

        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(
                f"[TRAIN] NaN/Inf loss at epoch={epoch}, batch={batch_idx}. "
                f"log_pz={log_pz.mean().item():.3f}, "
                f"log_det={log_det_scaling.mean().item():.3f}"
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
# EVALUATION — reconstruction RMSE
# ==============================================================================
@torch.no_grad()
def evaluate(model: ConditionalNICE, conditioner: CNNConditioner,
             loader: DataLoader, epoch: int) -> tuple[float, float]:
    """
    Returns (avg_nll, avg_rmse).
    Reconstruction: encode x → z → inverse → sigmoid → x_hat_pixel.
    """
    model.eval()
    conditioner.eval()
    total_nll  = 0.0
    total_rmse = 0.0
    n_batches  = 0

    for x_pixel, _ in loader:
        x_pixel = x_pixel.to(DEVICE)
        y       = gaussian_blur_batch(x_pixel)
        x_flat, logdet_logit = logit_preprocess(x_pixel.view(x_pixel.shape[0], -1))

        h      = conditioner(y)
        z, log_det_scaling = model(x_flat, h)

        log_pz  = logistic_log_prob(z)
        log_px  = log_pz + log_det_scaling + logdet_logit
        nll     = -log_px.mean().item()

        # Reconstruct
        x_logit_hat  = model.inverse(z, h)
        x_pixel_hat  = sigmoid_postprocess(x_logit_hat).view_as(x_pixel)
        rmse         = ((x_pixel - x_pixel_hat) ** 2).mean().sqrt().item()

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
# PLOTS  [PLOT] v1.2
# ==============================================================================
@torch.no_grad()
def save_reconstruction_plot(model: ConditionalNICE, conditioner: CNNConditioner,
                              loader: DataLoader, epoch: int) -> None:
    """
    Save a 3-row × 8-col grid: original | degraded | reconstruction.
    Saved to LOG_DIR/reconstruction_epoch{epoch:03d}.png.
    Non-fatal — logs error and returns on any exception.
    """
    try:
        model.eval()
        conditioner.eval()

        x_pixel, _ = next(iter(loader))
        x_pixel = x_pixel[:8].to(DEVICE)          # (8, 1, 28, 28)
        y       = gaussian_blur_batch(x_pixel)     # degraded

        x_flat, _ = logit_preprocess(x_pixel.view(8, -1))
        h         = conditioner(y)
        z, _      = model(x_flat, h)
        x_logit_hat = model.inverse(z, h)
        x_hat     = sigmoid_postprocess(x_logit_hat).view(8, 1, 28, 28)

        # Convert to numpy for plotting
        orig  = x_pixel.cpu().squeeze(1).numpy()   # (8, 28, 28)
        deg   = y.cpu().squeeze(1).numpy()
        recon = x_hat.cpu().squeeze(1).numpy()

        fig, axes = plt.subplots(3, 8, figsize=(16, 6))
        row_labels = ["Original", "Degraded\n(blurred)", "Reconstruction"]
        for row, (imgs, label) in enumerate(zip([orig, deg, recon], row_labels)):
            for col in range(8):
                ax = axes[row, col]
                ax.imshow(imgs[col], cmap="gray", vmin=0, vmax=1)
                ax.axis("off")
            axes[row, 0].set_ylabel(label, fontsize=9, rotation=0,
                                    labelpad=60, va="center")

        plt.suptitle(f"NICE+FiLM MNIST Reconstruction — Epoch {epoch}", fontsize=11)
        plt.tight_layout()

        path = os.path.join(LOG_DIR, f"reconstruction_epoch{epoch:03d}.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[PLOT] Reconstruction grid saved: {path}")

    except Exception as e:
        logger.error(f"[PLOT] save_reconstruction_plot failed at epoch {epoch}: {e}")


def save_training_curves(train_nlls: list, val_nlls: list,
                          val_rmses: list) -> None:
    """
    Save NLL and RMSE vs epoch to LOG_DIR/training_curves.png.
    Non-fatal — logs error and returns on any exception.
    """
    try:
        epochs = list(range(1, len(train_nlls) + 1))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(epochs, train_nlls, label="Train NLL", color="steelblue")
        ax1.plot(epochs, val_nlls,   label="Val NLL",   color="darkorange")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("NLL")
        ax1.set_title("NLL vs Epoch")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, val_rmses, label="Val RMSE", color="crimson")
        ax2.axhline(0.05, color="gray", linestyle="--", label="Pass threshold (0.05)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("RMSE")
        ax2.set_title("Reconstruction RMSE vs Epoch")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle("NICE+FiLM MNIST Training Curves", fontsize=11)
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
    logger.info(f"[MAIN] Config: DIM={DIM}, COND_DIM={COND_DIM}, "
                f"HIDDEN={HIDDEN}, N_HIDDEN={N_HIDDEN}, N_LAYERS={N_LAYERS}, "
                f"EPOCHS={EPOCHS}, LR={LR}")

    os.makedirs(DATA_DIR, exist_ok=True)

    # Data
    tf = transforms.ToTensor()
    train_ds = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=tf)
    val_ds   = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    logger.info(f"[MAIN] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Models
    model       = ConditionalNICE(dim=DIM, cond_dim=COND_DIM, hidden=HIDDEN,
                                  n_hidden=N_HIDDEN, n_layers=N_LAYERS).to(DEVICE)
    conditioner = CNNConditioner(cond_dim=COND_DIM).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters()) + \
               sum(p.numel() for p in conditioner.parameters())
    logger.info(f"[MAIN] Total parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(conditioner.parameters()),
        lr=LR
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Invertibility check before training
    logger.info("[MAIN] Pre-training invertibility check ...")
    sample_x, _ = next(iter(train_loader))
    sample_x     = sample_x[:8].to(DEVICE)
    sample_y     = gaussian_blur_batch(sample_x)
    x_flat, _    = logit_preprocess(sample_x.view(8, -1))
    with torch.no_grad():
        h_test = conditioner(sample_y)
    inv_err = model.check_invertibility(x_flat, h_test)
    if inv_err > 1e-4:
        logger.warning(f"[MAIN] Pre-training invertibility error high: {inv_err:.3e}")

    # Training loop
    best_val_rmse = float('inf')
    train_nlls, val_nlls, val_rmses = [], [], []   # [PLOT] v1.2

    for epoch in range(1, EPOCHS + 1):
        train_nll = train(model, conditioner, train_loader, optimizer, epoch)
        val_nll, val_rmse = evaluate(model, conditioner, val_loader, epoch)
        scheduler.step()

        train_nlls.append(train_nll)   # [PLOT] v1.2
        val_nlls.append(val_nll)
        val_rmses.append(val_rmse)

        # [PLOT] v1.2: save reconstruction grid every PLOT_EVERY epochs
        if epoch % PLOT_EVERY == 0 or epoch == EPOCHS:
            save_reconstruction_plot(model, conditioner, val_loader, epoch)

        is_best = val_rmse < best_val_rmse
        if is_best:
            best_val_rmse = val_rmse
            torch.save({
                'epoch':       epoch,
                'model':       model.state_dict(),
                'conditioner': conditioner.state_dict(),
                'val_nll':     val_nll,
                'val_rmse':    val_rmse,
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
    save_training_curves(train_nlls, val_nlls, val_rmses)  # [PLOT] v1.2

    # Final invertibility check
    logger.info("[MAIN] Post-training invertibility check ...")
    with torch.no_grad():
        h_test = conditioner(sample_y)
        x_flat, _ = logit_preprocess(sample_x.view(8, -1))
    model.check_invertibility(x_flat, h_test)

    # Success criteria
    if best_val_rmse < 0.05:
        logger.info("[MAIN] ✅ RECONSTRUCTION TEST PASSED: RMSE < 0.05")
    else:
        logger.warning(
            f"[MAIN] ⚠️  RECONSTRUCTION TEST: RMSE={best_val_rmse:.5f} >= 0.05 — "
            "model may need more epochs or wider hidden layers"
        )


if __name__ == "__main__":
    main()
