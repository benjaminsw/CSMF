# Version: WP0.3-CondNICE-v1.2
# Abbr: COND-NICE
# Last Modified: 2026-02-28
# Changelog:
#   v1.2 (2026-02-28): [BN] Removed BatchNorm1d between coupling layers — BN uses batch stats
#                      in forward but running stats in inverse causing inv_err=2.12e+02;
#                      FiLM inside coupling layers provides sufficient stabilisation;
#                      deleted _batchnorm1d_inverse(); forward/inverse loops simplified;
#                      additive coupling is volume-preserving so no scale explosion risk
#   v1.1 (2026-02-24): [F1] Added FiLM modulation to ConditionalAdditiveCoupling
#   v1.0 (original):   Initial additive coupling + BatchNorm stack
# Dependencies: torch>=2.0, film.py WP0.1-FiLM-v1.0+

import logging
import torch
import torch.nn as nn
from csmf.conditioning.film import FiLM  # [F1] v1.1 — shared FiLM module

logger = logging.getLogger(__name__)

class ConditionalAdditiveCoupling(nn.Module):
    """
    Additive coupling for NICE: x_B' = x_B + t(x_A, h)

    Simpler than affine (no scale), but still tractable + invertible.

    [F1] v1.1: t_net replaced with explicit fc1/fc2/fc3 + FiLM after each hidden ReLU.
    h is still concatenated at input AND guides hidden layers via FiLM.
    """
    def __init__(self, dim, cond_dim, hidden=128):
        super().__init__()
        self.cond_dim = cond_dim

        # [F1] v1.1 — Explicit layers (replacing nn.Sequential) to allow FiLM insertion
        self.fc1   = nn.Linear(dim // 2 + cond_dim, hidden)  # input: [xA | h]
        self.fc2   = nn.Linear(hidden, hidden)
        self.fc3   = nn.Linear(hidden, dim // 2)              # output: t (no FiLM after this)
        self.act   = nn.ReLU()
        self.film1 = FiLM(f_dim=hidden, h_dim=cond_dim)      # [F1] after hidden layer 1
        self.film2 = FiLM(f_dim=hidden, h_dim=cond_dim)      # [F1] after hidden layer 2

        logger.info(f"ConditionalAdditiveCoupling v1.1 initialized: dim={dim}, cond_dim={cond_dim}, hidden={hidden}, FiLM=True")

    def _compute_t(self, xA, h):
        """
        Compute translation t(xA, h) with FiLM modulation at each hidden layer.
        Shared by forward() and inverse() to avoid code duplication.
        """
        inp = torch.cat([xA, h], dim=1)          # [xA | h] — h at input (kept from v1.0)
        out = self.act(self.fc1(inp))
        out = self.film1(out, h)                  # [F1] FiLM after hidden layer 1
        if torch.isnan(out).any() or torch.isinf(out).any():
            logger.error("[COND-NICE] NaN/Inf after film1 in _compute_t")
            raise RuntimeError("NaN/Inf after film1 in _compute_t")
        out = self.act(self.fc2(out))
        out = self.film2(out, h)                  # [F1] FiLM after hidden layer 2
        if torch.isnan(out).any() or torch.isinf(out).any():
            logger.error("[COND-NICE] NaN/Inf after film2 in _compute_t")
            raise RuntimeError("NaN/Inf after film2 in _compute_t")
        return self.fc3(out)                      # output t — no FiLM (would corrupt translation)

    def forward(self, x, h):
        """
        Args:
            x: (B, d) input
            h: (B, cond_dim) conditioning features

        Returns:
            z: (B, d) transformed output
            log_det: (B,) log-determinant (always 0 for additive)
        """
        xA, xB = x.chunk(2, dim=1)
        t = self._compute_t(xA, h)               # [F1] FiLM-conditioned translation
        xB_new = xB + t
        log_det = torch.zeros(x.shape[0], device=x.device)
        return torch.cat([xA, xB_new], dim=1), log_det

    def inverse(self, z, h):
        """Inverse is trivial: x_B = z_B - t(z_A, h)"""
        zA, zB = z.chunk(2, dim=1)
        t = self._compute_t(zA, h)               # [F1] same _compute_t — no duplication
        return torch.cat([zA, zB - t], dim=1)
class ConditionalNICE(nn.Module):
    """
    Conditional NICE: stack additive couplings (no BatchNorm).

    BatchNorm removed in v1.2 — batch/running stat mismatch breaks exact invertibility.
    FiLM inside each coupling layer provides sufficient stabilisation.

    Original NICE: Dinh et al. (2014)
    Extension: Condition on h from degraded observation y
    """
    def __init__(self, dim, cond_dim, num_layers=4, hidden=128):
        super().__init__()
        self.dim = dim

        # [BN] v1.2: BatchNorm1d removed — coupling layers only
        layers = []
        for i in range(num_layers):
            layers.append(ConditionalAdditiveCoupling(dim, cond_dim, hidden))

        self.layers = nn.ModuleList(layers)

        # Learnable scaling (NICE paper Section 4.2)
        self.scaling = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x, h):
        """
        Forward: x → z

        Returns:
            z: (B, d) latent code
            log_det: (B,) total log-determinant
        """
        log_det = torch.zeros(x.shape[0], device=x.device)

        z = x
        for layer in self.layers:
            # [BN] v1.2: all layers are ConditionalAdditiveCoupling — no BN branch needed
            z, ld = layer(z, h)
            log_det += ld  # Always 0 for additive coupling

        # Final scaling (learned per dimension)
        z = z * torch.exp(self.scaling)
        log_det += self.scaling.sum()

        return z, log_det
    
    def inverse(self, z, h):
        # Undo final scaling
        x = z * torch.exp(-self.scaling)

        # [BN] v1.2: all layers are ConditionalAdditiveCoupling — no BN to invert
        for layer in reversed(self.layers):
            x = layer.inverse(x, h)

        return x
