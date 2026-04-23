# ==============================================================================
# File    : test_ffjord_film_mnist.py
# Abbr    : TEST-FFJORD-FILM
# Version : v1.0
# Created : 2026-04-16
# Changelog:
#   v1.0 (2026-04-16): Initial standalone FFJORD+FiLM MNIST reconstruction
#                      test. Self-contained (no CSMF imports). Architecture
#                      from Grathwohl et al. 2019 §B.1 MNIST encoder-decoder
#                      spec: 4 conv layers 64→64→128→128, softplus everywhere,
#                      strided downsampling every other layer; 4 mirrored
#                      transpose-conv decoder, sigmoid final output only.
#                      FFJORD dynamics: dz/dt = f(z,t,h), ε sampled ONCE
#                      outside odeint (Alg.1), Hutchinson trace estimator
#                      d(log_p)/dt = -εᵀ(∂f/∂z)ε. t concatenated at input
#                      to every dynamics layer (§4). FiLM inside dynamics net
#                      only: a=1+0.1·tanh(γ(h)), b=β(h), identity init.
#                      odeint_adjoint dopri5, atol=rtol=1e-5 (§C). NFE logged
#                      per epoch. Step-decay LR: 1e-3→1e-4 at epoch 250 (§B.1).
#                      Batch size 900 (paper). Follows test_nice_film_mnist.py
#                      conventions: LOG_DIR, metrics.csv, run.log, reconstruction
#                      grid, training curves, check_invertibility pre/post.
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

try:
    from torchdiffeq import odeint_adjoint
except ImportError:
    logger_tmp = logging.getLogger("TEST-FFJORD-FILM")
    logger_tmp.error(
        "[IMPORT] torchdiffeq not found. Install with: "
        "pip install torchdiffeq --break-system-packages"
    )
    raise

# ==============================================================================
# CONFIG  (paper §B.1 MNIST encoder-decoder spec)
# ==============================================================================
ENC_CHANNELS   = [64, 64, 128, 128]   # encoder filter sizes (paper §B.1)
DYN_HIDDEN     = 64                    # dynamics net hidden channels
H_DIM          = 64                    # conditioner output dim (FiLM extension)
LATENT_C       = 128                   # encoder output channels
BATCH_SIZE     = 900                   # paper image experiments
LR             = 1e-3                  # paper: Adam
LR_DECAY_EPOCH = 250                   # paper: decayed to 1e-4 after epoch 250
LR_DECAY_FACTOR= 0.1
EPOCHS         = 500                   # paper §B.1
ODE_TOL        = 1e-5                  # paper §C: atol=rtol=1e-5 for images
T0             = 0.0
T1             = 1.0
LOGIT_EPS      = 1e-6
BLUR_K         = 5
BLUR_S         = 1.5
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR       = "./data"
LOG_DIR        = "./tests/logs/ffjord_film_mnist"
SAVE_PATH      = os.path.join(LOG_DIR, "best_checkpoint.pth")
METRICS_CSV    = os.path.join(LOG_DIR, "metrics.csv")
PLOT_EVERY     = 10

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "run.log"), mode="a"),
    ],
)
logger = logging.getLogger("TEST-FFJORD-FILM")


# ==============================================================================
# HELPERS
# ==============================================================================
def logit_preprocess(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One global dequantize + logit. x: (B,1,28,28)."""
    x = x + torch.zeros_like(x).uniform_(0, 1.0 / 256)
    x = x.clamp(LOGIT_EPS, 1 - LOGIT_EPS)
    log_det = (-torch.log(x) - torch.log(1 - x))
    log_det = log_det.reshape(x.shape[0], -1).sum(dim=1)
    return torch.log(x) - torch.log(1 - x), log_det


def gaussian_log_prob(z: torch.Tensor) -> torch.Tensor:
    """Standard Gaussian log-prob. z: any shape. Returns (B,)."""
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


# ==============================================================================
# ENCODER  (paper §B.1: 4 conv layers, softplus everywhere, stride 2 every other)
# 1×28×28 → 64×28×28 → 64×14×14 → 128×14×14 → 128×7×7
# ==============================================================================
class Encoder(nn.Module):
    """
    Paper §B.1 MNIST encoder:
      Conv3×3(1,64,s=1) → softplus
      Conv3×3(64,64,s=2) → softplus     (28→14)
      Conv3×3(64,128,s=1) → softplus
      Conv3×3(128,128,s=2) → softplus   (14→7)
    Output: (B, 128, 7, 7)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,   64,  3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64,  64,  3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64,  128, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        logger.info("[Encoder] initialized: 1×28×28 → 128×7×7, softplus")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != 1:
            logger.error(f"[Encoder] Expected (B,1,28,28), got {tuple(x.shape)}")
            raise ValueError("Encoder expects (B,1,28,28)")
        x = F.softplus(self.conv1(x))
        x = F.softplus(self.conv2(x))
        x = F.softplus(self.conv3(x))
        x = F.softplus(self.conv4(x))
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("[Encoder] NaN/Inf in output")
            raise RuntimeError("NaN/Inf in Encoder output")
        return x   # (B, 128, 7, 7)


# ==============================================================================
# DECODER  (paper §B.1: 4 transposed-conv layers mirroring encoder)
# 128×7×7 → 128×14×14 → 64×14×14 → 64×28×28 → 1×28×28
# softplus in all layers; sigmoid on final output only
# ==============================================================================
class Decoder(nn.Module):
    """
    Paper §B.1 MNIST decoder (mirrored encoder):
      ConvT3×3(128,128,s=2) → softplus  (7→14)
      ConvT3×3(128,64,s=1)  → softplus
      ConvT3×3(64,64,s=2)   → softplus  (14→28)
      ConvT3×3(64,1,s=1)    → sigmoid
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(128, 64,  3, stride=1, padding=1)
        self.conv3 = nn.ConvTranspose2d(64,  64,  4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(64,  1,   3, stride=1, padding=1)
        logger.info("[Decoder] initialized: 128×7×7 → 1×28×28, softplus+sigmoid")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 4 or z.shape[1] != 128:
            logger.error(f"[Decoder] Expected (B,128,7,7), got {tuple(z.shape)}")
            raise ValueError("Decoder expects (B,128,7,7)")
        x = F.softplus(self.conv1(z))
        x = F.softplus(self.conv2(x))
        x = F.softplus(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))   # sigmoid only at final output
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("[Decoder] NaN/Inf in output")
            raise RuntimeError("NaN/Inf in Decoder output")
        return x   # (B, 1, 28, 28)


# ==============================================================================
# FILM2D  (FiLM injection in dynamics net — not encoder/decoder)
# a(h) = 1 + 0.1·tanh(γ(h)), b(h) = β(h); identity init
# ==============================================================================
class FiLM2d(nn.Module):
    """Bounded FiLM for spatial features. Identity init."""
    ALPHA = 0.1

    def __init__(self, h_dim: int, n_channels: int):
        super().__init__()
        self.gamma = nn.Linear(h_dim, n_channels)
        self.beta  = nn.Linear(h_dim, n_channels)
        nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight);  nn.init.zeros_(self.beta.bias)

    def forward(self, u: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """u: (B,C,H,W), h: (B,h_dim). Returns (B,C,H,W)."""
        a = 1.0 + self.ALPHA * torch.tanh(self.gamma(h))
        b = self.beta(h)
        out = a[:, :, None, None] * u + b[:, :, None, None]
        if torch.isnan(out).any() or torch.isinf(out).any():
            logger.error("[FiLM2d] NaN/Inf in output")
            raise RuntimeError("NaN/Inf in FiLM2d output")
        return out


# ==============================================================================
# DYNAMICS NET  (paper §4: t concatenated at input to every layer)
# f(z, t, h): (B, C, H, W) → (B, C, H, W)
# Conv(C+1, hidden) → softplus → FiLM
# Conv(hidden+1, hidden) → softplus → FiLM
# Conv(hidden+1, C)    ← zero-init final layer
# '+1' for t channel at every layer input
# ==============================================================================
class DynamicsNet(nn.Module):
    """
    FFJORD dynamics net f(z, t, h) (paper §4 + FiLM injection spec).

    t injected by concatenating a scalar broadcast channel at the INPUT to
    EVERY convolutional layer (paper §4: "concatenating t on to z(t) at
    the input to every layer").

    FiLM at 2 sites (after softplus layers 1 and 2).
    Zero-init final conv → zero initial velocity → stable ODE at t=0.
    """
    def __init__(self, in_channels: int, hidden: int = DYN_HIDDEN,
                 h_dim: int = H_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + 1, hidden,      3, padding=1)
        self.conv2 = nn.Conv2d(hidden + 1,       hidden,      3, padding=1)
        self.conv3 = nn.Conv2d(hidden + 1,       in_channels, 3, padding=1)
        self.film1 = FiLM2d(h_dim, hidden)
        self.film2 = FiLM2d(h_dim, hidden)
        # Zero-init: zero initial velocity → well-posed ODE at start
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)
        self.nfe   = 0   # number of function evaluations counter

    def forward(self, z: torch.Tensor, t: torch.Tensor,
                h: torch.Tensor) -> torch.Tensor:
        """
        z: (B, C, H, W)
        t: scalar tensor
        h: (B, h_dim)
        Returns dz/dt: (B, C, H, W)
        """
        self.nfe += 1
        B, C, H, W = z.shape
        # Broadcast t as extra channel at every layer input (paper §4)
        t_ch = t.reshape(1, 1, 1, 1).expand(B, 1, H, W)

        u = F.softplus(self.conv1(torch.cat([z, t_ch], dim=1)))
        u = self.film1(u, h)
        u = F.softplus(self.conv2(torch.cat([u, t_ch], dim=1)))
        u = self.film2(u, h)
        u = self.conv3(torch.cat([u, t_ch], dim=1))

        if torch.isnan(u).any() or torch.isinf(u).any():
            logger.error(f"[DynamicsNet] NaN/Inf at t={t.item():.4f}")
            raise RuntimeError("NaN/Inf in DynamicsNet output")
        return u


# ==============================================================================
# AUGMENTED DYNAMICS  (Algorithm 1)
# State = (z, log_det)
# dz/dt        = f(z, t, h)
# d(log_p)/dt  = -εᵀ(∂f/∂z)ε   (Hutchinson estimate, fixed ε per solve)
# ==============================================================================
class AugmentedDynamics(nn.Module):
    """
    Augmented ODE function (FFJORD Algorithm 1).

    State: tuple (z: (B,C,H,W), delta_log_p: (B,))
    ε is sampled ONCE outside odeint and held fixed throughout the solve.
    Hutchinson estimate: Tr(∂f/∂z) ≈ εᵀ(∂f/∂z)ε via VJP.
    """
    def __init__(self, dynamics: DynamicsNet):
        super().__init__()
        self.dynamics = dynamics
        self.eps      = None   # set externally before each odeint call

    def forward(self, t: torch.Tensor,
                state: tuple) -> tuple:
        z, delta_log_p = state

        # [FIX] torch.enable_grad() required because odeint_adjoint calls
        # this function internally under torch.no_grad() during step-size
        # adaptation and initial evaluation. Under no_grad, f_t has no
        # computation graph even when z.requires_grad=True, so autograd.grad
        # fails. enable_grad() forces graph construction regardless of outer
        # context — this is the standard pattern for Hutchinson inside odeint.
        with torch.enable_grad():
            z = z.detach().requires_grad_(True)
            f_t = self.dynamics(z, t, self._h)
            # Hutchinson: εᵀ(∂f_t/∂z)ε via VJP (Algorithm 1)
            e_dot_f = (f_t * self.eps.detach()).sum()
            vjp     = torch.autograd.grad(e_dot_f, z, create_graph=False)[0]

        trace_e = (vjp * self.eps.detach()).reshape(z.shape[0], -1).sum(dim=1)
        return f_t, -trace_e

    def set_conditioning(self, h: torch.Tensor) -> None:
        self._h = h


# ==============================================================================
# FFJORD BLOCK  (wraps encoder, augmented ODE, decoder)
# ==============================================================================
class FFJORDBlock(nn.Module):
    """
    Single-scale FFJORD with encoder-decoder (paper §B.1 MNIST spec).

    Forward (density estimation):
      x_logit → encoder → z0
      odeint(f_aug, (z0, 0), t=[0,1]) → (z1, delta_log_p)
      log_px = gaussian_log_prob(z1) - delta_log_p + logdet_logit

    Inverse (reconstruction):
      z1 ~ N(0,I)
      odeint(f_aug, (z1, 0), t=[1,0]) → (z0, _)
      x_hat = decoder(z0)

    ε sampled once outside odeint (Alg.1 line 1).
    """
    def __init__(self, h_dim: int = H_DIM, dyn_hidden: int = DYN_HIDDEN):
        super().__init__()
        self.encoder  = Encoder()
        self.decoder  = Decoder()
        self.dynamics = DynamicsNet(LATENT_C, dyn_hidden, h_dim)
        self.aug_dyn  = AugmentedDynamics(self.dynamics)

        logger.info(
            f"[FFJORDBlock] v1.0: latent=128×7×7, dyn_hidden={dyn_hidden}, "
            f"h_dim={h_dim}, ODE_TOL={ODE_TOL}"
        )

    def forward(self, x_logit: torch.Tensor, h: torch.Tensor):
        """
        x_logit: (B,1,28,28) logit-space. h: (B,h_dim).
        Returns (z1, delta_log_p): z1=(B,128,7,7), delta_log_p=(B,).
        """
        if x_logit.dim() != 4 or x_logit.shape[1] != 1:
            logger.error(
                f"[FFJORDBlock] forward expects (B,1,28,28), got {tuple(x_logit.shape)}"
            )
            raise ValueError("FFJORDBlock.forward shape mismatch")

        z0 = self.encoder(x_logit)   # (B, 128, 7, 7)
        B  = z0.shape[0]

        # ε sampled ONCE outside the integral (Alg.1 line 1)
        eps = (torch.randint(0, 2, z0.shape, device=z0.device,
                             dtype=z0.dtype) * 2 - 1)   # Rademacher ±1

        # Register ε and h in augmented dynamics
        self.aug_dyn.eps = eps
        self.aug_dyn.set_conditioning(h)
        self.dynamics.nfe = 0   # reset NFE counter

        delta_log_p0 = torch.zeros(B, device=z0.device, dtype=z0.dtype)
        t_span       = torch.tensor([T0, T1], device=z0.device, dtype=z0.dtype)

        # odeint_adjoint: memory-efficient backprop via adjoint method
        state1 = odeint_adjoint(
            self.aug_dyn,
            (z0, delta_log_p0),
            t_span,
            method='dopri5',
            atol=ODE_TOL,
            rtol=ODE_TOL,
            adjoint_params=list(self.dynamics.parameters()),
        )
        z1          = state1[0][-1]   # (B, 128, 7, 7) at t=1
        delta_log_p = state1[1][-1]   # (B,) at t=1

        if torch.isnan(z1).any() or torch.isinf(z1).any():
            logger.error("[FFJORDBlock] NaN/Inf in z1 after ODE")
            raise RuntimeError("NaN/Inf in FFJORDBlock.forward z1")

        return z1, delta_log_p

    @torch.no_grad()
    def inverse(self, z1: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Generation: integrate ODE backward t=1→0 then decode.
        z1: (B,128,7,7) latent sample. Returns x_hat: (B,1,28,28).
        """
        if z1.dim() != 4 or z1.shape[1] != LATENT_C:
            logger.error(
                f"[FFJORDBlock] inverse expects (B,{LATENT_C},7,7), "
                f"got {tuple(z1.shape)}"
            )
            raise ValueError("FFJORDBlock.inverse shape mismatch")

        eps = torch.zeros_like(z1)   # eps irrelevant for z trajectory inverse
        self.aug_dyn.eps = eps
        self.aug_dyn.set_conditioning(h)

        delta_log_p0 = torch.zeros(z1.shape[0], device=z1.device)
        t_span       = torch.tensor([T1, T0], device=z1.device, dtype=z1.dtype)

        state0 = odeint_adjoint(
            self.aug_dyn,
            (z1, delta_log_p0),
            t_span,
            method='dopri5',
            atol=ODE_TOL,
            rtol=ODE_TOL,
            adjoint_params=list(self.dynamics.parameters()),
        )
        z0    = state0[0][-1]       # (B, 128, 7, 7) at t=0
        x_hat = self.decoder(z0)    # (B, 1, 28, 28) in [0,1]

        if torch.isnan(x_hat).any() or torch.isinf(x_hat).any():
            logger.error("[FFJORDBlock] NaN/Inf in x_hat after decode")
            raise RuntimeError("NaN/Inf in FFJORDBlock.inverse x_hat")
        return x_hat

    def check_invertibility(self, x_logit: torch.Tensor, h: torch.Tensor,
                             tol: float = 1e-3) -> float:
        """
        max ‖enc(x) - ODE_backward(ODE_forward(enc(x)))‖_∞ in latent space.
        Uses no_grad for ODE solves (no Hutchinson needed here).
        """
        with torch.no_grad():
            z0       = self.encoder(x_logit)
            eps_zero = torch.zeros_like(z0)
            self.aug_dyn.eps = eps_zero
            self.aug_dyn.set_conditioning(h)

            t_fwd = torch.tensor([T0, T1], device=z0.device, dtype=z0.dtype)
            z1    = odeint_adjoint(
                self.aug_dyn, (z0, torch.zeros(z0.shape[0], device=z0.device)),
                t_fwd, method='dopri5', atol=ODE_TOL, rtol=ODE_TOL,
                adjoint_params=list(self.dynamics.parameters()),
            )[0][-1]

            t_bwd  = torch.tensor([T1, T0], device=z1.device, dtype=z1.dtype)
            z0_hat = odeint_adjoint(
                self.aug_dyn, (z1, torch.zeros(z1.shape[0], device=z1.device)),
                t_bwd, method='dopri5', atol=ODE_TOL, rtol=ODE_TOL,
                adjoint_params=list(self.dynamics.parameters()),
            )[0][-1]

        err = (z0 - z0_hat).abs().max().item()
        if err > tol:
            logger.warning(
                f"[FFJORDBlock] Invertibility FAILED: max_err={err:.3e} > tol={tol:.3e}"
            )
        else:
            logger.info(f"[FFJORDBlock] Invertibility PASSED: max_err={err:.3e}")
        return err


# ==============================================================================
# CONDITIONER  (spec: tiny CNN → h_dim=64)
# ==============================================================================
class TinyConditioner(nn.Module):
    """
    Spec: Conv3×3(1,16)→ReLU→Conv3×3(16,32,s=2)→ReLU→GAP→Linear(32,h_dim).
    """
    def __init__(self, h_dim: int = H_DIM):
        super().__init__()
        self.net  = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(32, h_dim)
        self.norm = nn.LayerNorm(h_dim)
        logger.info(f"[TinyConditioner] initialized: h_dim={h_dim}")

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if y.dim() != 4 or y.shape[1] != 1:
            logger.error(f"[TinyConditioner] Expected (B,1,28,28), got {tuple(y.shape)}")
            raise ValueError("TinyConditioner expects (B,1,28,28)")
        h = self.norm(self.head(self.pool(self.net(y)).squeeze(-1).squeeze(-1)))
        if torch.isnan(h).any() or torch.isinf(h).any():
            logger.error("[TinyConditioner] NaN/Inf in h output")
            raise RuntimeError("NaN/Inf in TinyConditioner output")
        return h


# ==============================================================================
# TRAINING
# ==============================================================================
def train(model: FFJORDBlock, conditioner: TinyConditioner,
          loader: DataLoader, optimizer: torch.optim.Optimizer,
          epoch: int) -> tuple[float, int]:
    """Returns (avg_nll, avg_nfe)."""
    model.train(); conditioner.train()
    total_loss = 0.0; total_nfe = 0; n_batches = 0

    for batch_idx, (x_pixel, _) in enumerate(loader):
        x_pixel = x_pixel.to(DEVICE)
        y_deg   = gaussian_blur_batch(x_pixel)
        x_logit, logdet_logit = logit_preprocess(x_pixel)

        h           = conditioner(y_deg)
        z1, delta_lp = model(x_logit, h)
        log_pz       = gaussian_log_prob(z1)
        log_px       = log_pz - delta_lp + logdet_logit
        loss         = -log_px.mean()

        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(
                f"[TRAIN] NaN/Inf loss at epoch={epoch}, batch={batch_idx}. "
                f"log_pz={log_pz.mean().item():.3f}, "
                f"delta_lp={delta_lp.mean().item():.3f}"
            )
            raise RuntimeError("NaN/Inf loss during training")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(conditioner.parameters()),
            max_norm=5.0
        )
        optimizer.step()

        total_loss += loss.item()
        total_nfe  += model.dynamics.nfe
        n_batches  += 1

    avg_nll = total_loss / max(n_batches, 1)
    avg_nfe = total_nfe  // max(n_batches, 1)
    logger.info(
        f"[TRAIN] Epoch {epoch:3d} | avg NLL = {avg_nll:.4f} | avg NFE = {avg_nfe}"
    )
    if avg_nfe > 200:
        logger.warning(
            f"[TRAIN] NFE={avg_nfe} > 200 — ODE solver using many steps. "
            f"Consider weight decay or reducing model complexity."
        )
    return avg_nll, avg_nfe


# ==============================================================================
# EVALUATION
# NOTE: no @torch.no_grad() — forward needs autograd for Hutchinson trace.
# Only inverse (decoder) wrapped in no_grad.
# ==============================================================================
def evaluate(model: FFJORDBlock, conditioner: TinyConditioner,
             loader: DataLoader, epoch: int) -> tuple[float, float]:
    """Returns (avg_nll, avg_rmse)."""
    model.eval(); conditioner.eval()
    total_nll = total_rmse = 0.0; n_batches = 0

    for x_pixel, _ in loader:
        x_pixel = x_pixel.to(DEVICE)
        y_deg   = gaussian_blur_batch(x_pixel)
        x_logit, logdet_logit = logit_preprocess(x_pixel)

        h           = conditioner(y_deg)
        z1, delta_lp = model(x_logit, h)      # needs grad for Hutchinson
        log_pz       = gaussian_log_prob(z1)
        nll          = -(log_pz - delta_lp + logdet_logit).mean().item()

        with torch.no_grad():
            x_hat = model.inverse(z1.detach(), h.detach())
            rmse  = ((x_pixel - x_hat) ** 2).mean().sqrt().item()

        total_nll  += nll; total_rmse += rmse; n_batches += 1

    avg_nll  = total_nll  / max(n_batches, 1)
    avg_rmse = total_rmse / max(n_batches, 1)
    logger.info(
        f"[EVAL]  Epoch {epoch:3d} | avg NLL = {avg_nll:.4f} | "
        f"avg RMSE = {avg_rmse:.5f}"
    )
    return avg_nll, avg_rmse


# ==============================================================================
# PLOTS
# ==============================================================================
def save_reconstruction_plot(model: FFJORDBlock, conditioner: TinyConditioner,
                              loader: DataLoader, epoch: int) -> None:
    """3-row × 8-col grid. Non-fatal. Forward needs grad for Hutchinson."""
    try:
        model.eval(); conditioner.eval()
        x_pixel, _ = next(iter(loader))
        x_pixel     = x_pixel[:8].to(DEVICE)
        y_deg       = gaussian_blur_batch(x_pixel)
        x_logit, _  = logit_preprocess(x_pixel)
        h           = conditioner(y_deg)
        z1, _       = model(x_logit, h)
        with torch.no_grad():
            x_hat = model.inverse(z1.detach(), h.detach())

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
        plt.suptitle(f"FFJORD+FiLM MNIST Reconstruction — Epoch {epoch}", fontsize=11)
        plt.tight_layout()
        path = os.path.join(LOG_DIR, f"reconstruction_epoch{epoch:03d}.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[PLOT] Reconstruction grid saved: {path}")
    except Exception as e:
        logger.error(f"[PLOT] save_reconstruction_plot failed at epoch {epoch}: {e}")


def save_training_curves(train_nlls: list, val_nlls: list,
                          val_rmses: list, nfes: list) -> None:
    """NLL + RMSE + NFE vs epoch. Non-fatal."""
    try:
        epochs = list(range(1, len(train_nlls) + 1))
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
        ax1.plot(epochs, train_nlls, label="Train NLL", color="steelblue")
        ax1.plot(epochs, val_nlls,   label="Val NLL",   color="darkorange")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("NLL")
        ax1.set_title("NLL vs Epoch"); ax1.legend(); ax1.grid(True, alpha=0.3)
        ax2.plot(epochs, val_rmses, label="Val RMSE", color="crimson")
        ax2.axhline(0.05, color="gray", linestyle="--", label="Pass (0.05)")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("RMSE")
        ax2.set_title("Reconstruction RMSE vs Epoch")
        ax2.legend(); ax2.grid(True, alpha=0.3)
        ax3.plot(epochs, nfes, label="Train NFE", color="purple")
        ax3.axhline(200, color="gray", linestyle="--", label="Warning (200)")
        ax3.set_xlabel("Epoch"); ax3.set_ylabel("NFE")
        ax3.set_title("ODE Function Evaluations vs Epoch (§5.2)")
        ax3.legend(); ax3.grid(True, alpha=0.3)
        plt.suptitle("FFJORD+FiLM MNIST Training Curves", fontsize=11)
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
        f"[MAIN] Config: ENC_CHANNELS={ENC_CHANNELS}, DYN_HIDDEN={DYN_HIDDEN}, "
        f"H_DIM={H_DIM}, BATCH_SIZE={BATCH_SIZE}, LR={LR}, "
        f"LR_DECAY_EPOCH={LR_DECAY_EPOCH}, EPOCHS={EPOCHS}, ODE_TOL={ODE_TOL}"
    )

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(METRICS_CSV, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_nll", "val_nll", "val_rmse", "avg_nfe", "best"]
        )

    tf_t = transforms.ToTensor()
    train_ds     = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=tf_t)
    val_ds       = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    logger.info(
        f"[MAIN] Train: {len(train_loader)} batches, Val: {len(val_loader)} batches"
    )

    model       = FFJORDBlock(h_dim=H_DIM, dyn_hidden=DYN_HIDDEN).to(DEVICE)
    conditioner = TinyConditioner(h_dim=H_DIM).to(DEVICE)

    n_params = (sum(p.numel() for p in model.parameters()) +
                sum(p.numel() for p in conditioner.parameters()))
    logger.info(f"[MAIN] Total parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(conditioner.parameters()), lr=LR
    )
    # Paper §B.1: lr decayed from 1e-3 to 1e-4 after epoch 250
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[LR_DECAY_EPOCH], gamma=LR_DECAY_FACTOR
    )

    # Pre-training invertibility check
    logger.info("[MAIN] Pre-training invertibility check ...")
    sample_x, _ = next(iter(train_loader))
    sample_x     = sample_x[:4].to(DEVICE)
    sample_y     = gaussian_blur_batch(sample_x)
    x_logit, _   = logit_preprocess(sample_x)
    with torch.no_grad():
        h_test = conditioner(sample_y)
    model.check_invertibility(x_logit, h_test)

    # Training loop
    best_val_rmse = float('inf')
    train_nlls, val_nlls, val_rmses, nfes = [], [], [], []

    for epoch in range(1, EPOCHS + 1):
        train_nll, avg_nfe    = train(model, conditioner, train_loader,
                                       optimizer, epoch)
        val_nll, val_rmse     = evaluate(model, conditioner, val_loader, epoch)
        scheduler.step()

        train_nlls.append(train_nll)
        val_nlls.append(val_nll)
        val_rmses.append(val_rmse)
        nfes.append(avg_nfe)

        if epoch % PLOT_EVERY == 0 or epoch == EPOCHS:
            save_reconstruction_plot(model, conditioner, val_loader, epoch)
            save_training_curves(train_nlls, val_nlls, val_rmses, nfes)

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
                csv.writer(f).writerow([
                    epoch, f"{train_nll:.4f}", f"{val_nll:.4f}",
                    f"{val_rmse:.5f}", avg_nfe, int(is_best)
                ])
        except Exception as e:
            logger.error(f"[MAIN] Failed to write metrics CSV at epoch {epoch}: {e}")

    logger.info(f"[MAIN] Training complete. Best val RMSE: {best_val_rmse:.5f}")

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
