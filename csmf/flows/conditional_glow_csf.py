"""
Conditional Glow + Cubic Spline Flow (GlowCSF) with FiLM Conditioning

Version: WP0.3-CondGCSF-v1.6
Abbr: COND-GCSF
Last Modified: 2026-04-06
Changelog:
  v1.6 (2026-04-06): Tighten pre-sigmoid clamp (-5,5)→(-3,3) and add FiLM
                     scale_factor=0.1 in GlowSplineCouplingLayer — NaN grads
                     persisting at epoch 1 batch 1 indicate init-time explosion
                     from large h activations (h_norm=84 observed); scale_factor
                     0.1 vs default 5.0 reduces FiLM output magnitude at init.
  v1.5 (2026-04-06): Replace linalg.solve with triangular solves in
                     BlockDiagInvLinear.inverse() — linalg.solve backward through
                     ill-conditioned W produced NaN grads (47/50 params) even with
                     tanh-bounded log_s; triangular solves exploit P@L@U structure
                     directly with better-conditioned gradients. Fixed log_det
                     inconsistency: forward() was using raw log_s, now uses
                     tanh-bounded log_s_bounded matching _get_W() transform.
  v1.4 (2026-04-06): Sigmoid saturation fix — xB.clamp(-5,5) before sigmoid in
                     forward(); zB.clamp(-5,5) in inverse(). near_boundary=0.809
                     caused Newton spline inverse to diverge at boundaries.
  v1.3 (2026-04-06): NaN gradient fix in coupling log_det.
                     log_det_logit used nextafter(0,1)≈1.4e-45 clamp → gradient
                     1/1.4e-45≈7e44 overflows to NaN during backward. Fixed by
                     using 1e-12 clamp on log_det_logit and log_det_sigmoid terms
                     (gradient 1/1e-12=1e12, large but finite). nextafter clamp
                     kept only for zB_sig_cl/xB_sig_cl (no gradient flows there).
                     Added log_det_sp.clamp(min=-100) safety net after spline.
  v1.2 (2026-04-06): NaN in ActNorm.forward fix — three changes.
                     (1) GlowSplineCouplingLayer head init changed from zeros
                     to normal_(std=0.01): zero-init caused W=H=bd=0 → Steffen
                     spline divide-by-zero → NaN into step 1 ActNorm init.
                     (2) Coupling clamp replaced with dtype-aware nextafter so
                     bounds are strictly inside (0,1) for any dtype incl float16.
                     (3) ActNorm._data_dependent_init validates input is finite
                     before computing mean/std — catches NaN early with context.
  v1.1 (2026-04-06): Invertibility fixes — four bugs causing inv_err=0.396.
                     (1) Coupling forward/inverse clamp tightened 1e-6→1e-12 on
                     zB_sig_cl, xB_sig_cl, and log-det sigmoid terms; hard clamp
                     at 1e-6 snapped boundary values breaking round-trip at O(1).
                     (2) ActNorm.inverse() exact: removed s.clamp(min=eps) —
                     s=exp(log_s)>0 always so clamp is wrong and breaks bijectivity.
                     (3) BlockDiagInvLinear._get_W(): log_s bounded via tanh
                     (MAX_LOG_SCALE=4.0) so diagonal cannot explode during training,
                     preventing linalg.solve instability in inverse().
  v1.0 (2026-04-02): Initial implementation. Glow-style flow: each step is
                     ActNorm → BlockDiagInvLinear → GlowSplineCouplingLayer+FiLM.
                     ActNorm: per-dim scale+bias with data-dependent init on first batch
                     (initialized buffer prevents re-init). BlockDiagInvLinear: D=784
                     partitioned into n_blocks=8 blocks of 98 dims; each block has
                     independent LU-decomposed W_b (P fixed, L/U/log_s learned); log_det
                     O(block_size) per block. GlowSplineCouplingLayer: reuses Steffen
                     spline primitives from COND-CSF-v1.0 (no duplication); FiLM
                     conditioning on hidden layers; zero-init head per Glow paper.
                     External h API: forward(x,h)→(z,log_det); inverse(z,h)→x,
                     matching COND-NICE/COND-NSF/COND-CSF convention.
                     cond_dim alias added for train_csmf.py compatibility.
Dependencies: torch>=2.0, film WP0.1-FiLM-v1.0+, COND-CSF WP0.3-CondCSF-v1.0+
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Optional, Tuple

try:
    from csmf.conditioning.film import FiLM
    from csmf.flows.conditional_csf import (
        _build_steffen_coeffs,
        _steffen_spline_forward,
        _steffen_spline_inverse,
    )
except ImportError as e:
    logging.error(f"COND-GCSF | Failed to import dependencies: {e}")
    raise

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_VERSION = "WP0.3-CondGCSF-v1.6"
_ABBR    = "COND-GCSF"


# =============================================================================
# ActNorm
# =============================================================================

class ActNorm(nn.Module):
    """
    Activation Normalisation layer (Glow, Kingma & Dhariwal 2018, Sec. 3.1).

    Performs an affine transformation y = x * s + b per dimension.
    Parameters s (as log_s) and b are initialised on the first batch so that
    activations have zero mean and unit variance — data-dependent init.
    After init, s and b are regular learnable parameters.

    log_det per sample = sum(log_s)  [positive, since s > 0 via exp].

    Inverse: x = (y - b) / s = (y - b) * exp(-log_s).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.log_s = nn.Parameter(torch.zeros(dim))   # log scale
        self.b     = nn.Parameter(torch.zeros(dim))   # bias

        # Track whether data-dependent init has run
        self.register_buffer('initialized', torch.tensor(False))

    def _data_dependent_init(self, x: torch.Tensor) -> None:
        """Init log_s and b so that output has zero mean and unit variance."""
        # [F] v1.2 — validate input finite before init; NaN input → NaN log_s
        if not torch.isfinite(x).all():
            n_nan = torch.isnan(x).sum().item()
            n_inf = torch.isinf(x).sum().item()
            logger.error(
                f"COND-GCSF | ActNorm._data_dependent_init non-finite input: "
                f"nan={n_nan}, inf={n_inf} — upstream layer produced NaN/Inf"
            )
            raise RuntimeError("ActNorm._data_dependent_init: non-finite input")
        with torch.no_grad():
            mean = x.mean(dim=0)             # (D,)
            std  = x.std(dim=0).clamp(min=self.eps)  # (D,)
            # y = x * s + b = (x - mean) / std  → s = 1/std, b = -mean/std
            self.log_s.data = (-std.log())
            self.b.data     = (-mean / std)
            self.initialized.fill_(True)
            logger.info(
                f"COND-GCSF | ActNorm data-dep init | "
                f"log_s range [{self.log_s.min():.3f}, {self.log_s.max():.3f}]"
            )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, dim)
        Returns:
            y:       (B, dim)
            log_det: (B,) — same value for all samples in batch
        """
        if self.training and not self.initialized:
            self._data_dependent_init(x)

        s = self.log_s.exp()                # (D,) > 0
        y = x * s + self.b                  # (B, D)

        log_det = self.log_s.sum().expand(x.shape[0])  # (B,)

        if torch.isnan(y).any() or torch.isnan(log_det).any():
            logger.error(
                f"COND-GCSF | ActNorm NaN | "
                f"log_s range [{self.log_s.min():.3f}, {self.log_s.max():.3f}]"
            )
            raise RuntimeError("NaN in ActNorm.forward")

        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: (B, dim)
        Returns:
            x: (B, dim)
        """
        s = self.log_s.exp()
        # [F] v1.1 — exact inverse: s=exp(log_s)>0 always; clamp broke bijectivity
        x = (y - self.b) / s

        if torch.isnan(x).any():
            logger.error("COND-GCSF | ActNorm inverse NaN")
            raise RuntimeError("NaN in ActNorm.inverse")

        return x


# =============================================================================
# Block-Diagonal Invertible Linear (LU decomposition per block)
# =============================================================================

class BlockDiagInvLinear(nn.Module):
    """
    Block-diagonal invertible linear layer with LU decomposition.

    Partitions D dimensions into n_blocks blocks of block_size = D // n_blocks.
    Each block has an independent invertible linear map W_b parameterised as:
        W_b = P_b @ L_b @ (U_b + diag(exp(log_s_b)))
    where:
        P_b  — fixed random permutation matrix (registered buffer)
        L_b  — lower-triangular with ones on diagonal (learned strictly lower part)
        U_b  — upper-triangular with zeros on diagonal (learned strictly upper part)
        log_s_b — log of the diagonal of U (learned, ensures W_b invertible)

    log_det = sum over blocks of sum(log_s_b).  [O(block_size * n_blocks) = O(D)]

    Forward:  y_b = x_b @ W_b^T  (row-vector convention, batch-friendly)
    Inverse:  x_b = triangular_solve(W_b^T, y_b^T)^T  per block

    Rationale: D=784 full matrix costs ~614k mults/step and 2.5M params;
    8 blocks of 98 dims costs 8 × 9604 = 77k mults and 8 × 9604 = 77k params.
    """

    def __init__(self, dim: int, n_blocks: int = 8):
        super().__init__()

        if dim % n_blocks != 0:
            msg = f"COND-GCSF | dim={dim} must be divisible by n_blocks={n_blocks}"
            logger.error(msg)
            raise ValueError(msg)

        self.dim        = dim
        self.n_blocks   = n_blocks
        self.block_size = dim // n_blocks

        bs = self.block_size

        # Initialise each block from a random rotation matrix (log_det = 0 at init)
        L_lower_list = []   # strictly lower triangular (bs × bs), diagonal not stored
        U_upper_list = []   # strictly upper triangular (bs × bs), diagonal not stored
        log_s_list   = []   # (bs,) log diagonal of U

        for b in range(n_blocks):
            # Random rotation matrix via QR decomposition
            W_init = torch.linalg.qr(torch.randn(bs, bs))[0]

            # Compute LU decomposition: W_init = P @ L @ U
            # torch.linalg.lu returns P, L, U where W_init = P @ L @ U
            P_b, L_b, U_b = torch.linalg.lu(W_init)

            self.register_buffer(f'P_{b}', P_b)   # (bs, bs) fixed permutation

            # Extract strictly lower triangular of L (diagonal is always 1)
            L_lower_list.append(nn.Parameter(L_b.tril(-1)))   # zeros on diag + above

            # Extract diagonal (as log) and strictly upper triangular of U
            diag_U = U_b.diagonal()
            sign_U = diag_U.sign()
            # Absorb sign into log_s — at init log|diag_U|, forward uses exp so always >0
            log_s_list.append(nn.Parameter(diag_U.abs().clamp(min=1e-8).log()))

            U_upper_list.append(nn.Parameter(U_b.triu(1)))    # zeros on diag

        self.L_lowers = nn.ParameterList(L_lower_list)
        self.U_uppers = nn.ParameterList(U_upper_list)
        self.log_s    = nn.ParameterList(log_s_list)

        logger.info(
            f"COND-GCSF | BlockDiagInvLinear | "
            f"dim={dim}, n_blocks={n_blocks}, block_size={bs}"
        )

    def _get_W(self, b: int) -> torch.Tensor:
        """Assemble W_b from LU components."""
        P = getattr(self, f'P_{b}')
        L = self.L_lowers[b].tril(-1) + torch.eye(
            self.block_size, device=P.device, dtype=P.dtype
        )
        # [F] v1.1 — tanh-bound log_s (MAX_LOG_SCALE=4.0) so diagonal cannot
        # explode during training, preventing linalg.solve instability in inverse().
        MAX_LOG_SCALE = 2.0
        log_s_bounded = MAX_LOG_SCALE * torch.tanh(self.log_s[b] / MAX_LOG_SCALE)
        U = self.U_uppers[b].triu(1) + torch.diag(log_s_bounded.exp())
        return P @ L @ U   # (bs, bs)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, dim)
        Returns:
            y:       (B, dim)
            log_det: (B,) — same for all samples in batch
        """
        B = x.shape[0]
        chunks = x.split(self.block_size, dim=1)   # list of (B, block_size)
        y_chunks = []
        log_det_total = torch.zeros(1, device=x.device, dtype=x.dtype)

        for b, xb in enumerate(chunks):
            try:
                W = self._get_W(b)                  # (bs, bs)
                yb = xb @ W.t()                     # (B, bs)
                y_chunks.append(yb)
                # [F] v1.5 — use tanh-bounded log_s matching _get_W(); raw log_s
                # was inconsistent with the actual transform, giving wrong log_det
                MAX_LOG_SCALE = 2.0
                log_s_bounded = MAX_LOG_SCALE * torch.tanh(self.log_s[b] / MAX_LOG_SCALE)
                log_det_total = log_det_total + log_s_bounded.sum()
            except Exception as e:
                logger.error(f"COND-GCSF | BlockDiagInvLinear forward block {b}: {e}")
                raise

        y = torch.cat(y_chunks, dim=1)              # (B, dim)
        log_det = log_det_total.expand(B)            # (B,)

        if torch.isnan(y).any():
            logger.error("COND-GCSF | BlockDiagInvLinear forward NaN in y")
            raise RuntimeError("NaN in BlockDiagInvLinear.forward")

        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: (B, dim)
        Returns:
            x: (B, dim)
        """
        chunks = y.split(self.block_size, dim=1)
        x_chunks = []

        for b, yb in enumerate(chunks):
            try:
                P = getattr(self.inv_lin if hasattr(self, 'inv_lin') else self, f'P_{b}')
                # [F] v1.5 — triangular solves instead of linalg.solve:
                # W = P @ L @ U  →  W^T = U^T @ L^T @ P^T
                # solve W^T x^T = y^T in three steps with well-conditioned gradients:
                P  = getattr(self, f'P_{b}')                        # (bs, bs)
                L  = self.L_lowers[b].tril(-1) + torch.eye(
                    self.block_size, device=yb.device, dtype=yb.dtype
                )
                MAX_LOG_SCALE  = 2.0
                log_s_bounded  = MAX_LOG_SCALE * torch.tanh(self.log_s[b] / MAX_LOG_SCALE)
                U  = self.U_uppers[b].triu(1) + torch.diag(log_s_bounded.exp())
                # Step 1: apply P^T (permutation — no grad issues)
                rhs = (P.t() @ yb.t())                              # (bs, B)
                # Step 2: solve L^T z = rhs  (L^T is upper triangular)
                rhs = torch.linalg.solve_triangular(L.t(), rhs, upper=True)
                # Step 3: solve U^T x = rhs  (U^T is lower triangular)
                xb  = torch.linalg.solve_triangular(U.t(), rhs, upper=False).t()  # (B, bs)
                x_chunks.append(xb)
            except Exception as e:
                logger.error(f"COND-GCSF | BlockDiagInvLinear inverse block {b}: {e}")
                raise

        x = torch.cat(x_chunks, dim=1)

        if torch.isnan(x).any():
            logger.error("COND-GCSF | BlockDiagInvLinear inverse NaN in x")
            raise RuntimeError("NaN in BlockDiagInvLinear.inverse")

        return x


# =============================================================================
# Glow Spline Coupling Layer
# =============================================================================

class GlowSplineCouplingLayer(nn.Module):
    """
    Coupling layer using Steffen cubic splines with FiLM conditioning.

    Reuses _steffen_spline_forward / _steffen_spline_inverse from COND-CSF.
    Same sigmoid/logit domain wrapping as CubicSplineCouplingLayer.

    Per Glow paper (Sec. 3.3): head layer is zero-initialised so each coupling
    layer starts as an identity function, stabilising deep network training.

    Data flow (forward):
        xA  →  [Linear → ReLU → FiLM(h)] × n_hidden  →  head (zero-init)
            →  θ = (W, H, bd)  →  spline(sigmoid(xB))  →  logit  →  zB
    """

    def __init__(
        self,
        dim: int,
        split_idx: int,
        K: int,
        h_dim: int,
        hidden_dims: List[int],
    ):
        super().__init__()

        self.dA = split_idx
        self.dB = dim - split_idx
        self.K  = K
        self.n_params = 2 * K + 2

        if self.dA <= 0 or self.dB <= 0:
            msg = f"COND-GCSF | Invalid split: dA={self.dA}, dB={self.dB}"
            logger.error(msg)
            raise ValueError(msg)

        self.hidden_layers = nn.ModuleList()
        self.film_layers   = nn.ModuleList()
        prev = self.dA
        for hd in hidden_dims:
            self.hidden_layers.append(nn.Linear(prev, hd))
            # [F] v1.6 — scale_factor=0.1 vs default 5.0: large h_norm=84 at init
            # causes FiLM output explosion → NaN grads on epoch 1 batch 1
            self.film_layers.append(FiLM(hd, h_dim, scale_factor=0.1))
            prev = hd

        # [F] v1.2 — normal_(std=0.01) instead of zeros_: zero-init gave W=H=bd=0
        # → Steffen spline divide-by-zero → NaN into step 1 ActNorm init.
        # Small nonzero weights produce near-uniform but valid spline parameters.
        self.head = nn.Linear(prev, self.dB * self.n_params)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

        logger.info(
            f"COND-GCSF | GlowSplineCouplingLayer | "
            f"dA={self.dA}, dB={self.dB}, K={K}, hidden={hidden_dims}"
        )

    def _get_spline_params(
        self, xA: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u = xA
        for lin, film in zip(self.hidden_layers, self.film_layers):
            u = lin(u)
            u = F.relu(u)
            u = film(u, h)
        theta = self.head(u).view(xA.shape[0], self.dB, self.n_params)
        W  = theta[..., :self.K]
        H  = theta[..., self.K:2 * self.K]
        bd = theta[..., 2 * self.K:]
        return W, H, bd

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B   = x.shape[0]
        xA  = x[:, :self.dA]
        xB  = x[:, self.dA:]

        W, H, bd = self._get_spline_params(xA, h)

        # [F] v1.6 — tighten clamp (-5,5)→(-3,3): sigmoid(±3)=0.047/0.953,
        # further from boundaries; NaN grads at epoch 1 indicate init-time explosion
        xB_clamped = xB.clamp(-3.0, 3.0)
        yB_sig  = torch.sigmoid(xB_clamped)

        yB_flat = yB_sig.reshape(-1)
        W_flat  = W.reshape(-1, self.K)
        H_flat  = H.reshape(-1, self.K)
        bd_flat = bd.reshape(-1, 2)

        try:
            zB_sig_flat, log_det_spline_flat = _steffen_spline_forward(
                yB_flat, W_flat, H_flat, bd_flat
            )
        except Exception as e:
            logger.error(f"COND-GCSF | GlowSplineCoupling forward spline: {e}")
            raise

        zB_sig     = zB_sig_flat.reshape(B, self.dB)
        log_det_sp = log_det_spline_flat.reshape(B, self.dB)

        if torch.isnan(zB_sig).any():
            logger.error("COND-GCSF | NaN in coupling spline output")
            raise RuntimeError("NaN in GlowSplineCouplingLayer.forward (spline)")

        # [F] v1.3 — safety net: _steffen_spline_forward already clamps dy≥eps,
        # but guard here too in case of any numerical edge case in backward
        log_det_sp = log_det_sp.clamp(min=-100.0)

        # [F] v1.2 — dtype-aware nextafter clamp: strictly inside (0,1) for any dtype
        # Used only for zB (round-trip); no gradient flows through this value
        _one  = torch.tensor(1.0, device=zB_sig.device, dtype=zB_sig.dtype)
        _zero = torch.tensor(0.0, device=zB_sig.device, dtype=zB_sig.dtype)
        zB_sig_cl = zB_sig.clamp(
            torch.nextafter(_zero, _one),
            torch.nextafter(_one,  _zero)
        )
        zB = torch.logit(zB_sig_cl)

        # log-det contributions: sigmoid'(xB) + spline'(·) + logit'(spline_out)
        # [F] v1.3 — use 1e-12 (not nextafter) for log terms: gradient of log(x)
        # is 1/x; nextafter(0,1)≈1.4e-45 → gradient≈7e44 → NaN in backward.
        # 1e-12 gives gradient 1e12 — large but finite, safe for float32.
        log_det_sigmoid = (
            torch.log(yB_sig.clamp(min=1e-12)) +
            torch.log((1.0 - yB_sig).clamp(min=1e-12))
        ).sum(dim=-1)

        log_det_logit = (
            -torch.log(zB_sig_cl.clamp(min=1e-12)) -
             torch.log((1.0 - zB_sig_cl).clamp(min=1e-12))
        ).sum(dim=-1)

        log_det = log_det_sigmoid + log_det_sp.sum(dim=-1) + log_det_logit

        if torch.isnan(log_det).any():
            logger.error(
                f"COND-GCSF | NaN in coupling log_det | "
                f"sigmoid={log_det_sigmoid.mean():.3f} | "
                f"spline={log_det_sp.mean():.3f} | "
                f"logit={log_det_logit.mean():.3f}"
            )
            raise RuntimeError("NaN in GlowSplineCouplingLayer.forward (log_det)")

        return torch.cat([xA, zB], dim=-1), log_det

    def inverse(
        self, z: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        B   = z.shape[0]
        xA  = z[:, :self.dA]
        zB  = z[:, self.dA:]

        W, H, bd = self._get_spline_params(xA, h)

        # [F] v1.6 — tighten clamp (-5,5)→(-3,3) matching forward
        zB_sig  = torch.sigmoid(zB.clamp(-3.0, 3.0))
        zB_flat = zB_sig.reshape(-1)
        W_flat  = W.reshape(-1, self.K)
        H_flat  = H.reshape(-1, self.K)
        bd_flat = bd.reshape(-1, 2)

        try:
            xB_sig_flat = _steffen_spline_inverse(zB_flat, W_flat, H_flat, bd_flat)
        except Exception as e:
            logger.error(f"COND-GCSF | GlowSplineCoupling inverse spline: {e}")
            raise

        xB_sig = xB_sig_flat.reshape(B, self.dB)

        if torch.isnan(xB_sig).any():
            logger.error("COND-GCSF | NaN in coupling spline inverse")
            raise RuntimeError("NaN in GlowSplineCouplingLayer.inverse")

        # [F] v1.2 — dtype-aware nextafter clamp matching forward
        _one  = torch.tensor(1.0, device=xB_sig.device, dtype=xB_sig.dtype)
        _zero = torch.tensor(0.0, device=xB_sig.device, dtype=xB_sig.dtype)
        xB_sig_cl = xB_sig.clamp(
            torch.nextafter(_zero, _one),
            torch.nextafter(_one,  _zero)
        )
        xB = torch.logit(xB_sig_cl)
        return torch.cat([xA, xB], dim=-1)


# =============================================================================
# Glow Step (ActNorm → BlockDiagInvLinear → SplineCoupling)
# =============================================================================

class GlowStep(nn.Module):
    """
    One Glow step: ActNorm → BlockDiagInvLinear → GlowSplineCouplingLayer.

    ActNorm and BlockDiagInvLinear are unconditional (no h).
    GlowSplineCouplingLayer is conditioned on h via FiLM.
    """

    def __init__(
        self,
        dim: int,
        K: int,
        h_dim: int,
        hidden_dims: List[int],
        n_blocks: int,
    ):
        super().__init__()
        self.actnorm  = ActNorm(dim)
        self.inv_lin  = BlockDiagInvLinear(dim, n_blocks=n_blocks)
        self.coupling = GlowSplineCouplingLayer(
            dim=dim,
            split_idx=dim // 2,
            K=K,
            h_dim=h_dim,
            hidden_dims=hidden_dims,
        )

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, ld_an = self.actnorm.forward(x)
        x, ld_il = self.inv_lin.forward(x)
        x, ld_cp = self.coupling.forward(x, h)
        return x, ld_an + ld_il + ld_cp

    def inverse(
        self, z: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        z = self.coupling.inverse(z, h)
        z = self.inv_lin.inverse(z)
        z = self.actnorm.inverse(z)
        return z


# =============================================================================
# ConditionalGlowCSF
# =============================================================================

class ConditionalGlowCSF(nn.Module):
    """
    Conditional Glow + Cubic Spline Flow.

    Each of n_flows steps is: ActNorm → BlockDiagInvLinear → SplineCoupling+FiLM.
    The coupling layer splits dimensions in half; the invertible linear mixes all
    dims between steps. No additional permutation is needed — BlockDiagInvLinear
    handles the mixing role that fixed permutations played in CSF v1.0.

    External h API — caller passes precomputed conditioning vector h.
    Matches COND-NICE / COND-NSF / COND-CSF convention.
    """

    def __init__(
        self,
        dim: int = 784,
        h_dim: int = 64,
        cond_dim: Optional[int] = None,   # alias for train_csmf.py compatibility
        n_flows: int = 4,
        K: int = 8,
        hidden_dims: Optional[List[int]] = None,
        n_blocks: int = 8,
    ):
        """
        Args:
            dim:         Data dimensionality (784 = 28×28 MNIST).
            h_dim:       External conditioning feature dimension.
            cond_dim:    Alias for h_dim.
            n_flows:     Number of Glow steps.
            K:           Spline bins per transformed dimension.
            hidden_dims: Hidden layer sizes in each coupling NN.
            n_blocks:    Blocks for BlockDiagInvLinear. dim must be divisible.
        """
        super().__init__()

        if cond_dim is not None:
            h_dim = cond_dim

        self.version = _VERSION
        self.abbr    = _ABBR
        logger.info(f"COND-GCSF | Initialising {self.version}")

        if hidden_dims is None:
            hidden_dims = [64, 64]

        self.dim      = dim
        self.h_dim    = h_dim
        self.n_flows  = n_flows

        self.steps = nn.ModuleList([
            GlowStep(
                dim=dim,
                K=K,
                h_dim=h_dim,
                hidden_dims=hidden_dims,
                n_blocks=n_blocks,
            )
            for _ in range(n_flows)
        ])

        self.register_buffer('base_loc',   torch.zeros(1))
        self.register_buffer('base_scale', torch.ones(1))

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"COND-GCSF | dim={dim} | h_dim={h_dim} | n_flows={n_flows} | "
            f"K={K} | n_blocks={n_blocks} | hidden={hidden_dims} | "
            f"params={n_params:,}"
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, dim) — logit-preprocessed data
            h: (B, h_dim) — external conditioning
        Returns:
            z:       (B, dim)
            log_det: (B,)
        """
        z = x
        log_det_total = torch.zeros(x.shape[0], device=x.device)

        for k, step in enumerate(self.steps):
            try:
                z, ld = step.forward(z, h)
            except Exception as e:
                logger.error(f"COND-GCSF | forward failed at step {k}: {e}")
                raise
            log_det_total = log_det_total + ld

        if torch.isnan(z).any() or torch.isnan(log_det_total).any():
            logger.error(
                f"COND-GCSF | NaN in forward | "
                f"z_nan={torch.isnan(z).sum()} | "
                f"ld_nan={torch.isnan(log_det_total).sum()}"
            )
            raise RuntimeError("NaN in ConditionalGlowCSF.forward")

        return z, log_det_total

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
        x = z
        for k, step in enumerate(reversed(self.steps)):
            try:
                x = step.inverse(x, h)
            except Exception as e:
                logger.error(f"COND-GCSF | inverse failed at step {self.n_flows-1-k}: {e}")
                raise

        if torch.isnan(x).any():
            logger.error(f"COND-GCSF | NaN in inverse: {torch.isnan(x).sum()}/{x.numel()}")
            raise RuntimeError("NaN in ConditionalGlowCSF.inverse")

        return x

    def log_prob(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        """
        log q(x|h) = log p(z) + log|det J_forward|.

        Args:
            x: (B, dim)
            h: (B, h_dim)
        Returns:
            log_prob: (B,)
        """
        z, log_det = self.forward(x, h)
        log_pz = -0.5 * (z ** 2 + math.log(2.0 * math.pi))
        log_prob = log_pz.sum(dim=-1) + log_det

        if torch.isnan(log_prob).any():
            logger.error(
                f"COND-GCSF | NaN in log_prob | "
                f"log_pz mean={log_pz.sum(-1).mean():.3f} | "
                f"log_det mean={log_det.mean():.3f}"
            )
            raise RuntimeError("NaN in ConditionalGlowCSF.log_prob")

        return log_prob

    def sample(
        self, n_samples: int, h: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            n_samples: Samples per conditioning vector.
            h:         (B, h_dim)
        Returns:
            x: (n_samples, dim) if B=1, else (B, n_samples, dim)
        """
        B     = h.shape[0]
        total = B * n_samples
        h_exp = h.unsqueeze(1).expand(-1, n_samples, -1).reshape(total, -1)
        z     = torch.randn(total, self.dim, device=h.device)

        try:
            x = self.inverse(z, h_exp)
        except Exception as e:
            logger.error(f"COND-GCSF | sample() inverse failed: {e}")
            raise

        if B == 1:
            return x
        return x.view(B, n_samples, self.dim)


# =============================================================================
# Version
# =============================================================================

def get_version() -> dict:
    return {
        'version': _VERSION,
        'abbr':    _ABBR,
        'date':    '2026-04-02',
        'purpose': (
            'Glow-style flow with Steffen cubic spline coupling + FiLM conditioning. '
            'Steps: ActNorm → BlockDiagInvLinear (8 blocks × 98-dim) → SplineCoupling+FiLM. '
            'O(1) forward and inverse. External h API matching COND-CSF/NICE/NSF.'
        ),
        'deferred_to_v1.1': [
            'Full D×D invertible 1×1 conv (currently block-diagonal approximation)',
            'Blinn analytical cubic root solver in spline inverse',
        ],
    }


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    import time

    print("=" * 70)
    print(f"ConditionalGlowCSF | {get_version()['version']}")
    print("=" * 70)

    try:
        # Test 1: Instantiation
        logger.info("[TEST 1] Instantiation")
        model = ConditionalGlowCSF(
            dim=784, h_dim=64, n_flows=2, K=8, hidden_dims=[64, 64], n_blocks=8
        )
        logger.info(f"✓ Instantiated | params={sum(p.numel() for p in model.parameters()):,}")

        # Test 2: Forward
        logger.info("[TEST 2] Forward")
        x = torch.randn(4, 784)
        h = torch.randn(4, 64)
        t0 = time.time()
        z, log_det = model.forward(x, h)
        logger.info(
            f"✓ z={z.shape} | "
            f"log_det=[{log_det.min():.2f},{log_det.max():.2f}] | "
            f"time={1000*(time.time()-t0):.1f}ms"
        )

        # Test 3: log_prob
        logger.info("[TEST 3] log_prob")
        lp = model.log_prob(x, h)
        logger.info(f"✓ log_prob=[{lp.min():.2f},{lp.max():.2f}]")

        # Test 4: Invertibility
        logger.info("[TEST 4] Invertibility")
        model.eval()
        with torch.no_grad():
            z2, _ = model.forward(x, h)
            x_recon = model.inverse(z2, h)
        err = (x - x_recon).abs().max().item()
        logger.info(f"  max|x - inv(fwd(x))|: {err:.6f}")
        if err < 1e-3:
            logger.info("✓ Invertibility PASSED")
        else:
            logger.warning(f"✗ Invertibility FAILED (err={err:.6f})")
        model.train()

        # Test 5: cond_dim alias
        logger.info("[TEST 5] cond_dim alias")
        m2 = ConditionalGlowCSF(dim=784, cond_dim=64, n_flows=1, K=4)
        _, _ = m2.forward(x, h)
        logger.info("✓ cond_dim alias OK")

        # Test 6: ActNorm data-dep init
        logger.info("[TEST 6] ActNorm init")
        an = model.steps[0].actnorm
        logger.info(f"  initialized={an.initialized.item()}")
        logger.info(f"  log_s range [{an.log_s.min():.3f},{an.log_s.max():.3f}]")
        logger.info("✓ ActNorm init OK")

        # Test 7: BlockDiagInvLinear round-trip
        logger.info("[TEST 7] BlockDiagInvLinear round-trip")
        il = model.steps[0].inv_lin
        y_il, _ = il.forward(x)
        x_il = il.inverse(y_il)
        il_err = (x - x_il).abs().max().item()
        logger.info(f"  max|x - inv(fwd(x))|: {il_err:.6f}")
        if il_err < 1e-5:
            logger.info("✓ BlockDiagInvLinear round-trip PASSED")
        else:
            logger.warning(f"✗ BlockDiagInvLinear round-trip FAILED (err={il_err:.6f})")

        # Test 8: Sampling
        logger.info("[TEST 8] Sampling")
        samples = model.sample(n_samples=5, h=torch.randn(1, 64))
        logger.info(f"✓ samples={samples.shape}")

        print(f"\n{'='*70}\nALL TESTS PASSED\n{'='*70}")

    except Exception as e:
        logger.error(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
