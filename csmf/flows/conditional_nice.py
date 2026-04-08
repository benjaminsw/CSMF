# Version: WP0.3-CondNICE-v1.8
# Abbr: COND-NICE
# Last Modified: 2026-04-02
# Changelog:
#   v1.8 (2026-04-02): [PIPELINE] Removed use_logit, _logit_preprocess, _sigmoid_postprocess
#                      — csmf._prepare_x_for_expert now applies dequantize+logit for all
#                      experts uniformly (unified pipeline v1.3.20); NICE forward() receives
#                      logit-space input directly; inverse() returns logit-space; sigmoid
#                      applied by csmf._expert_inverse; FI/NLL now comparable across experts
#   v1.7 (2026-04-02): [LOGIT] Added use_logit flag (default=True) to ConditionalNICE;
#                      _logit_preprocess() clamps x to (1e-6,1-1e-6) and applies logit,
#                      accumulates log-det into total; _sigmoid_postprocess() applies
#                      sigmoid at end of inverse() to recover x_hat_pixel; consistent
#                      with RealNVP clamp(1e-6) → range ~[-13.8,13.8]
#   v1.6 (2026-04-01): [SS] Added scale_strength arg to ConditionalAdditiveCoupling and
#                      ConditionalNICE (default=0.05, backward compatible); replaces hardcoded
#                      0.05 in _compute_st() with self.scale_strength; ConditionalNICE passes
#                      scale_strength through to each coupling layer; enables --nice-scale CLI
#                      sweep (0.05/0.08/0.1) without code changes between runs
#   v1.5 (2026-04-01): [AFF] Upgraded coupling blocks from additive-only to safe affine-lite
#                      scaling using m(s)=1+0.05*tanh(s); added dedicated scale head fc_s and
#                      shared _compute_st() path for forward/inverse; per-layer log_det now sums
#                      log(m(s)) instead of staying identically zero, while keeping transform much
#                      weaker than RealNVP's exp(s); class/docstrings updated to reflect affine-lite
#                      behaviour and exact inverse x_B=(z_B-t)/m(s)
#   v1.4 (2026-03-31): [CAP] Increased default hidden width 128→256 and default depth
#                      4→8 to improve additive-coupling capacity; [PERM] inserted fixed
#                      reverse permutation between coupling blocks so the same split does
#                      not persist across all layers; forward()/inverse() now branch over
#                      coupling and permutation modules while preserving exact invertibility;
#                      added module-level note that NICE-specific LR boost should be applied
#                      in the training script, not inside the model definition
#   v1.3 (2026-03-25): [SC] Clamped self.scaling to [-5, 5] in forward() and inverse() —
#                      unclamped scaling allows optimizer to push scaling→+inf to minimize
#                      NLL via log_det alone without learning structure; clamp prevents
#                      collapse; both z scaling and log_det accumulation now use clamped value
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


class ReversePermutation(nn.Module):
    """
    Fixed reverse permutation used between additive coupling blocks.

    Deterministic, parameter-free, and exactly invertible.
    log|det J| = 0, so it does not contribute to the flow log-determinant.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.dim:
            logger.error(
                "[COND-NICE] ReversePermutation.forward shape mismatch: "
                f"expected (*, {self.dim}), got {tuple(x.shape)}"
            )
            raise ValueError("ReversePermutation.forward shape mismatch")
        return torch.flip(x, dims=[1])

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2 or z.shape[1] != self.dim:
            logger.error(
                "[COND-NICE] ReversePermutation.inverse shape mismatch: "
                f"expected (*, {self.dim}), got {tuple(z.shape)}"
            )
            raise ValueError("ReversePermutation.inverse shape mismatch")
        return torch.flip(z, dims=[1])


class ConditionalAdditiveCoupling(nn.Module):
    """
    Affine-lite coupling for NICE: x_B' = x_B * m(s) + t(x_A, h)

    Uses a deliberately weak multiplicative term
        m(s) = 1 + 0.05 * tanh(s)
    so the block gains non-constant log_det without becoming equivalent to
    RealNVP's full exp(s) affine coupling.

    [F1] v1.1: t_net replaced with explicit fc1/fc2/fc3 + FiLM after each hidden ReLU.
    h is still concatenated at input AND guides hidden layers via FiLM.
    [CAP] v1.4: default hidden width increased to 256.
    [AFF] v1.5: added a separate scale head and safe affine-lite scaling.
    """
    def __init__(self, dim, cond_dim, hidden=256, scale_strength=0.05):
        super().__init__()
        if dim % 2 != 0:
            logger.error(f"[COND-NICE] ConditionalAdditiveCoupling requires even dim, got dim={dim}")
            raise ValueError("ConditionalAdditiveCoupling requires even dim")

        self.dim = dim
        self.cond_dim = cond_dim
        self.scale_strength = scale_strength  # [SS] v1.6: affine-lite strength, m = 1 + scale_strength * tanh(s)

        # [F1] v1.1 — Explicit layers (replacing nn.Sequential) to allow FiLM insertion
        # [CAP] v1.4 — wider hidden representation for stronger translation network
        self.fc1 = nn.Linear(dim // 2 + cond_dim, hidden)   # input: [xA | h]
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_t = nn.Linear(hidden, dim // 2)             # translation head
        self.fc_s = nn.Linear(hidden, dim // 2)             # affine-lite scale head
        self.act = nn.ReLU()
        self.film1 = FiLM(f_dim=hidden, h_dim=cond_dim)     # [F1] after hidden layer 1
        self.film2 = FiLM(f_dim=hidden, h_dim=cond_dim)     # [F1] after hidden layer 2

        logger.info(
            "ConditionalAdditiveCoupling v1.6 initialized: "
            f"dim={dim}, cond_dim={cond_dim}, hidden={hidden}, "
            f"scale_strength={scale_strength}, FiLM=True"
        )

    def _compute_st(self, xA, h):
        """
        Compute translation t(xA, h) and affine-lite scale factor m(s).
        Shared by forward() and inverse() to avoid code duplication.
        """
        if xA.shape[0] != h.shape[0]:
            logger.error(
                "[COND-NICE] Batch mismatch in _compute_st: "
                f"xA batch={xA.shape[0]}, h batch={h.shape[0]}"
            )
            raise ValueError("Batch mismatch between xA and h in _compute_st")

        inp = torch.cat([xA, h], dim=1)          # [xA | h] — h at input (kept from v1.0)
        out = self.act(self.fc1(inp))
        out = self.film1(out, h)                 # [F1] FiLM after hidden layer 1
        if torch.isnan(out).any() or torch.isinf(out).any():
            logger.error("[COND-NICE] NaN/Inf after film1 in _compute_st")
            raise RuntimeError("NaN/Inf after film1 in _compute_st")

        out = self.act(self.fc2(out))
        out = self.film2(out, h)                 # [F1] FiLM after hidden layer 2
        if torch.isnan(out).any() or torch.isinf(out).any():
            logger.error("[COND-NICE] NaN/Inf after film2 in _compute_st")
            raise RuntimeError("NaN/Inf after film2 in _compute_st")

        t = self.fc_t(out)                      # translation output
        s = self.fc_s(out)                      # raw scale logits

        if torch.isnan(t).any() or torch.isinf(t).any():
            logger.error("[COND-NICE] NaN/Inf in translation output t")
            raise RuntimeError("NaN/Inf in translation output t")
        if torch.isnan(s).any() or torch.isinf(s).any():
            logger.error("[COND-NICE] NaN/Inf in scale output s")
            raise RuntimeError("NaN/Inf in scale output s")

        m = 1.0 + self.scale_strength * torch.tanh(s)  # [SS] v1.6: scale_strength replaces hardcoded 0.05
        if torch.any(m <= 0.0):
            logger.error("[COND-NICE] Non-positive affine-lite scale factor m(s)")
            raise RuntimeError("Non-positive affine-lite scale factor m(s)")

        return t, m

    def forward(self, x, h):
        """
        Args:
            x: (B, d) input
            h: (B, cond_dim) conditioning features

        Returns:
            z: (B, d) transformed output
            log_det: (B,) log-determinant from affine-lite scaling
        """
        if x.dim() != 2 or x.shape[1] != self.dim:
            logger.error(
                "[COND-NICE] ConditionalAdditiveCoupling.forward shape mismatch: "
                f"expected (*, {self.dim}), got {tuple(x.shape)}"
            )
            raise ValueError("ConditionalAdditiveCoupling.forward shape mismatch")

        xA, xB = x.chunk(2, dim=1)
        t, m = self._compute_st(xA, h)           # [AFF] FiLM-conditioned shift + safe scale
        xB_new = xB * m + t
        log_det = torch.log(m).sum(dim=1)
        z = torch.cat([xA, xB_new], dim=1)
        return z, log_det

    def inverse(self, z, h):
        """Inverse uses the same affine-lite factor: x_B = (z_B - t(z_A, h)) / m(s)"""
        if z.dim() != 2 or z.shape[1] != self.dim:
            logger.error(
                "[COND-NICE] ConditionalAdditiveCoupling.inverse shape mismatch: "
                f"expected (*, {self.dim}), got {tuple(z.shape)}"
            )
            raise ValueError("ConditionalAdditiveCoupling.inverse shape mismatch")

        zA, zB = z.chunk(2, dim=1)
        t, m = self._compute_st(zA, h)           # [AFF] same shared path — no duplication
        x = torch.cat([zA, (zB - t) / m], dim=1)
        return x


class ConditionalNICE(nn.Module):
    """
    Conditional NICE: stack affine-lite couplings with fixed permutations (no BatchNorm).

    BatchNorm removed in v1.2 — batch/running stat mismatch breaks exact invertibility.
    FiLM inside each coupling layer provides sufficient stabilisation.
    [PERM] v1.4 adds reverse permutations between couplings to mix dimensions.
    [CAP]  v1.4 increases default depth 4→8 and hidden width 128→256.
    [AFF]  v1.5 replaces pure additive updates with safe affine-lite scaling.

    Original NICE: Dinh et al. (2014)
    Extension: Condition on h from degraded observation y

    Note:
        NICE-specific learning-rate boosts should be applied in the training script
        or experiment config, not inside this model definition.
    """
    def __init__(self, dim, cond_dim, num_layers=8, hidden=256, scale_strength=0.05):
        super().__init__()
        if dim % 2 != 0:
            logger.error(f"[COND-NICE] ConditionalNICE requires even dim, got dim={dim}")
            raise ValueError("ConditionalNICE requires even dim")
        if num_layers < 1:
            logger.error(f"[COND-NICE] ConditionalNICE requires num_layers >= 1, got {num_layers}")
            raise ValueError("ConditionalNICE requires num_layers >= 1")

        self.dim = dim
        self.cond_dim = cond_dim
        self.num_layers = num_layers
        self.hidden = hidden
        self.scale_strength = scale_strength  # [SS] v1.6: stored for logging/inspection

        # [PERM] v1.4: alternate additive couplings with fixed reverse permutations.
        layers = []
        for i in range(num_layers):
            layers.append(ConditionalAdditiveCoupling(dim, cond_dim, hidden, scale_strength=scale_strength))  # [SS] v1.6
            if i < num_layers - 1:
                layers.append(ReversePermutation(dim))

        self.layers = nn.ModuleList(layers)

        # Learnable scaling (NICE paper Section 4.2)
        self.scaling = nn.Parameter(torch.zeros(dim))

        logger.info(
            "ConditionalNICE v1.8 initialized: "
            f"dim={dim}, cond_dim={cond_dim}, num_layers={num_layers}, hidden={hidden}, "
            f"scale_strength={scale_strength}, permutations=True"
        )

    def forward(self, x, h):
        """
        Forward: x (logit-space, from csmf._prepare_x_for_expert) → z

        Args:
            x: (B, d) logit-space input — dequantize+logit applied by csmf (unified pipeline)
            h: (B, cond_dim) conditioning features

        Returns:
            z:       (B, d) latent code
            log_det: (B,) total log-determinant
        """
        if x.dim() != 2 or x.shape[1] != self.dim:
            logger.error(
                "[COND-NICE] forward shape mismatch: "
                f"expected (*, {self.dim}), got {tuple(x.shape)}"
            )
            raise ValueError("ConditionalNICE.forward shape mismatch")
        if h.dim() != 2:
            logger.error(f"[COND-NICE] h must be rank-2 in forward(), got shape={tuple(h.shape)}")
            raise ValueError("ConditionalNICE.forward requires rank-2 conditioning features")
        if x.shape[0] != h.shape[0]:
            logger.error(
                "[COND-NICE] Batch mismatch in forward(): "
                f"x batch={x.shape[0]}, h batch={h.shape[0]}"
            )
            raise ValueError("Batch mismatch between x and h in ConditionalNICE.forward")

        z = x
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        for layer in self.layers:
            if isinstance(layer, ConditionalAdditiveCoupling):
                z, ld = layer(z, h)
                log_det += ld  # Per-layer affine-lite log_det
            elif isinstance(layer, ReversePermutation):
                z = layer.forward(z)
            else:
                logger.error(f"[COND-NICE] Unsupported layer type in forward(): {type(layer)}")
                raise TypeError("Unsupported layer type in ConditionalNICE.forward")

        # Final scaling (learned per dimension)
        # [SC] v1.3: clamp scaling to [-5, 5] — prevents optimizer from pushing
        # scaling→+inf to minimize NLL via log_det alone without learning structure
        s_clamped = self.scaling.clamp(-5.0, 5.0)
        z = z * torch.exp(s_clamped)
        log_det += s_clamped.sum().to(log_det.dtype)
        if torch.isnan(z).any() or torch.isinf(z).any():
            logger.error("[COND-NICE] NaN/Inf after final scaling in forward()")
            raise RuntimeError("[COND-NICE] NaN/Inf after final scaling in forward()")

        return z, log_det

    def inverse(self, z, h):
        """
        Inverse: z → x (logit-space). Sigmoid applied by csmf._expert_inverse.
        """
        if z.dim() != 2 or z.shape[1] != self.dim:
            logger.error(
                "[COND-NICE] inverse shape mismatch: "
                f"expected (*, {self.dim}), got {tuple(z.shape)}"
            )
            raise ValueError("ConditionalNICE.inverse shape mismatch")
        if h.dim() != 2:
            logger.error(f"[COND-NICE] h must be rank-2 in inverse(), got shape={tuple(h.shape)}")
            raise ValueError("ConditionalNICE.inverse requires rank-2 conditioning features")
        if z.shape[0] != h.shape[0]:
            logger.error(
                "[COND-NICE] Batch mismatch in inverse(): "
                f"z batch={z.shape[0]}, h batch={h.shape[0]}"
            )
            raise ValueError("Batch mismatch between z and h in ConditionalNICE.inverse")

        # Undo final scaling
        # [SC] v1.3: must use same clamped value as forward() for exact invertibility
        s_clamped = self.scaling.clamp(-5.0, 5.0)
        x = z * torch.exp(-s_clamped)

        for layer in reversed(self.layers):
            if isinstance(layer, ConditionalAdditiveCoupling):
                x = layer.inverse(x, h)
            elif isinstance(layer, ReversePermutation):
                x = layer.inverse(x)
            else:
                logger.error(f"[COND-NICE] Unsupported layer type in inverse(): {type(layer)}")
                raise TypeError("Unsupported layer type in ConditionalNICE.inverse")

        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("[COND-NICE] NaN/Inf detected after inverse()")
            raise RuntimeError("[COND-NICE] NaN/Inf detected after inverse()")

        return x
