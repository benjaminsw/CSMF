# Version: WP0.3-CondNSF-v1.6
# Abbr: COND-NSF
# Last Modified: 2026-04-01
# Changelog:
#   v1.6 (2026-04-01): [PERM] Added random inter-layer permutations to ConditionalNSF —
#                      num_layers-1 random perms registered as buffers (perm_{i},
#                      inv_perm_{i}) applied before each coupling layer except layer 0;
#                      ConditionalNSF now uses swap_halves=False for all layers since
#                      permutations are more general and avoid redundant mixing mechanisms
#   v1.5 (2026-04-01): [SW] Added swap_halves param to ConditionalRQSplineCoupling —
#                      when True, second half conditions first half (xB→xA) instead of
#                      default xA→xB; ConditionalNSF alternated swap_halves per layer;
#                      half_only path unaffected by swap_halves
#   v1.4 (2026-04-01): [NSF] Added NSF-only FiLM wrapper with gentler residual modulation
#                      (scale_factor=2.0, residual_alpha=0.3) so NSF conditioning can be
#                      tuned without changing shared FiLM behaviour in other flows; reduced
#                      default spline bins K from 8 to 6 for better stability
#   v1.3 (2026-03-25): [SB] Increased spline boundary B from 3.0 to 6.0 in both
#                      ConditionalRQSplineCoupling and ConditionalNSF defaults — B=3.0
#                      too small for logit-preprocessed MNIST (range ~[-6,+6]) causing
#                      linear tail fallback for many values; z_std stuck at ~0.71;
#                      B=6.0 ensures spline covers full data range
#   v1.2 (2026-02-28): [BN] Removed BatchNorm1d between coupling layers — same batch/running
#                      stat mismatch as NICE would cause inv_err explosion; RQ-spline outputs
#                      bounded to [-B,B] so no scale explosion risk without BN; FiLM inside
#                      coupling layers provides sufficient stabilisation; deleted _bn_inverse();
#                      forward/inverse loops simplified to coupling layers only
#   v1.1 (2026-02-24): [F1] Added FiLM modulation to ConditionalRQSplineCoupling
#   v1.0 (original):   Initial RQ-spline coupling + BatchNorm stack
# Dependencies: torch>=2.0, film.py WP0.1-FiLM-v1.0+
# Reference: Durkan et al. (2019) - Neural Spline Flows

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from csmf.conditioning.film import FiLM  # shared FiLM module

logger = logging.getLogger(__name__)


class NSFFiLM(nn.Module):
    """
    NSF-only FiLM wrapper.

    Uses the shared FiLM implementation internally, but applies a gentler
    residual modulation so NSF conditioning can be adjusted without changing
    FiLM behaviour for RealNVP/NICE or other flows.
    """

    def __init__(self, f_dim, h_dim, hidden_dims=[128, 128], scale_factor=2.0,
                 residual_alpha=0.3, debug=False):
        super().__init__()
        self.base = FiLM(
            f_dim=f_dim,
            h_dim=h_dim,
            hidden_dims=hidden_dims,
            scale_factor=scale_factor,
            debug=debug,
        )
        self.residual_alpha = residual_alpha

    def forward(self, f, h):
        modulated = self.base(f, h)
        return f + self.residual_alpha * (modulated - f)


class RationalQuadraticSpline:
    """
    Monotonic rational-quadratic spline transformation.

    Maps [-B, B] → [-B, B] using K piecewise rational-quadratic segments.
    Identity (linear) outside [-B, B].

    Reference: Gregory & Delbourgo (1982), Durkan et al. (2019)
    """

    @staticmethod
    def forward(x, widths, heights, derivatives, B=3.0):
        """
        Apply monotonic RQ-spline transform.

        Args:
            x: (B, D) input values
            widths: (B, D, K) bin widths (positive, sum to 2B)
            heights: (B, D, K) bin heights (positive, sum to 2B)
            derivatives: (B, D, K-1) internal knot derivatives (positive)
            B: tail bound

        Returns:
            y: (B, D) transformed values
            log_det: (B, D) log-Jacobian determinant
        """
        inside_mask = (x >= -B) & (x <= B)

        knots_x = torch.cumsum(widths, dim=-1) - B
        knots_y = torch.cumsum(heights, dim=-1) - B

        knots_x = torch.cat([torch.full_like(knots_x[..., :1], -B), knots_x], dim=-1)
        knots_y = torch.cat([torch.full_like(knots_y[..., :1], -B), knots_y], dim=-1)

        derivatives = torch.cat([
            torch.ones_like(derivatives[..., :1]),
            derivatives,
            torch.ones_like(derivatives[..., :1])
        ], dim=-1)

        bin_idx = torch.searchsorted(knots_x.contiguous(), x.unsqueeze(-1).contiguous())
        bin_idx = torch.clamp(bin_idx - 1, 0, widths.shape[-1] - 1).squeeze(-1)

        x_k       = torch.gather(knots_x,      -1, bin_idx.unsqueeze(-1)).squeeze(-1)
        x_kp1     = torch.gather(knots_x,      -1, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)
        y_k       = torch.gather(knots_y,      -1, bin_idx.unsqueeze(-1)).squeeze(-1)
        y_kp1     = torch.gather(knots_y,      -1, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)
        delta_k   = torch.gather(derivatives,  -1, bin_idx.unsqueeze(-1)).squeeze(-1)
        delta_kp1 = torch.gather(derivatives,  -1, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)

        s_k = (y_kp1 - y_k) / (x_kp1 - x_k + 1e-8)
        xi  = (x - x_k) / (x_kp1 - x_k + 1e-8)

        # Rational-quadratic formula (Eq. 4 in paper)
        numerator   = (y_kp1 - y_k) * (s_k * xi**2 + delta_k * xi * (1 - xi))
        denominator = s_k + (delta_kp1 + delta_k - 2 * s_k) * xi * (1 - xi) + 1e-8
        y = y_k + numerator / denominator

        # Derivative (Eq. 5 in paper)
        derivative = (s_k**2 * (delta_kp1 * xi**2 + 2 * s_k * xi * (1 - xi) + delta_k * (1 - xi)**2)) / (denominator**2 + 1e-8)
        log_det = torch.log(derivative + 1e-8)

        y       = torch.where(inside_mask, y, x)
        log_det = torch.where(inside_mask, log_det, torch.zeros_like(log_det))

        return y, log_det

    @staticmethod
    def inverse(y, widths, heights, derivatives, B=3.0):
        """
        Invert RQ-spline by solving quadratic equation.

        Returns:
            x: (B, D) inverse-transformed values
        """
        inside_mask = (y >= -B) & (y <= B)

        knots_x = torch.cumsum(widths, dim=-1) - B
        knots_y = torch.cumsum(heights, dim=-1) - B

        knots_x = torch.cat([torch.full_like(knots_x[..., :1], -B), knots_x], dim=-1)
        knots_y = torch.cat([torch.full_like(knots_y[..., :1], -B), knots_y], dim=-1)

        derivatives = torch.cat([
            torch.ones_like(derivatives[..., :1]),
            derivatives,
            torch.ones_like(derivatives[..., :1])
        ], dim=-1)

        bin_idx = torch.searchsorted(knots_y.contiguous(), y.unsqueeze(-1).contiguous())
        bin_idx = torch.clamp(bin_idx - 1, 0, widths.shape[-1] - 1).squeeze(-1)

        x_k       = torch.gather(knots_x,      -1, bin_idx.unsqueeze(-1)).squeeze(-1)
        x_kp1     = torch.gather(knots_x,      -1, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)
        y_k       = torch.gather(knots_y,      -1, bin_idx.unsqueeze(-1)).squeeze(-1)
        y_kp1     = torch.gather(knots_y,      -1, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)
        delta_k   = torch.gather(derivatives,  -1, bin_idx.unsqueeze(-1)).squeeze(-1)
        delta_kp1 = torch.gather(derivatives,  -1, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)

        s_k = (y_kp1 - y_k) / (x_kp1 - x_k + 1e-8)

        # Solve quadratic: aξ² + bξ + c = 0 (Eq. 6-8 in paper)
        a = (y_kp1 - y_k) * (s_k - delta_k) + (y - y_k) * (delta_kp1 + delta_k - 2 * s_k)
        b = (y_kp1 - y_k) * delta_k - (y - y_k) * (delta_kp1 + delta_k - 2 * s_k)
        c = -s_k * (y - y_k)

        discriminant = b**2 - 4 * a * c
        xi = 2 * c / (-b - torch.sqrt(discriminant + 1e-8) + 1e-8)

        x = x_k + xi * (x_kp1 - x_k)
        x = torch.where(inside_mask, x, y)

        return x


class ConditionalRQSplineCoupling(nn.Module):
    """
    Coupling layer with rational-quadratic spline transforms.

    x_B' = RQSpline(x_B; θ(x_A, h))
    where θ = {widths, heights, derivatives}

    [F1] v1.1: param_net replaced with explicit fc1/fc2/fc3 + FiLM after each hidden ReLU.
    h is still concatenated at input AND guides hidden layers via FiLM.

    [SW] v1.5: swap_halves=True reverses the conditioner/transformed roles:
    conditioner=xB, transformed=xA. Output dimension order is always preserved
    ([first_half, second_half]) so downstream cat/chunk is consistent.
    half_only path is unaffected by swap_halves.

    [PERM] v1.6: swap_halves kept for standalone/ablation use; ConditionalNSF sets
    swap_halves=False for all layers and relies on permutations for mixing instead.
    """
    def __init__(self, dim, cond_dim, hidden=128, K=6, B=6.0, swap_halves=False):
        super().__init__()
        self.K = K
        self.B = B
        self.cond_dim = cond_dim
        self.swap_halves = swap_halves
        out_dim = (dim // 2) * (3 * K - 1)   # widths + heights + derivatives

        self.fc1   = nn.Linear(dim // 2 + cond_dim, hidden)
        self.fc2   = nn.Linear(hidden, hidden)
        self.fc3   = nn.Linear(hidden, out_dim)
        self.act   = nn.ReLU()
        self.film1 = NSFFiLM(f_dim=hidden, h_dim=cond_dim)
        self.film2 = NSFFiLM(f_dim=hidden, h_dim=cond_dim)

        logger.info(
            f"ConditionalRQSplineCoupling v1.6 initialized: dim={dim}, cond_dim={cond_dim}, "
            f"hidden={hidden}, K={K}, B={B}, FiLM=NSF-only, swap_halves={swap_halves}"
        )

    def _compute_params(self, cond_half, h):
        """
        Compute raw spline params with FiLM modulation at each hidden layer.
        Shared by forward() and inverse() to avoid code duplication.
        cond_half: the half used as conditioner (xA or xB depending on swap_halves).
        """
        inp = torch.cat([cond_half, h], dim=1)
        out = self.act(self.fc1(inp))
        out = self.film1(out, h)
        if torch.isnan(out).any() or torch.isinf(out).any():
            logger.error("[COND-NSF] NaN/Inf after film1 in _compute_params")
            raise RuntimeError("NaN/Inf after film1 in _compute_params")
        out = self.act(self.fc2(out))
        out = self.film2(out, h)
        if torch.isnan(out).any() or torch.isinf(out).any():
            logger.error("[COND-NSF] NaN/Inf after film2 in _compute_params")
            raise RuntimeError("NaN/Inf after film2 in _compute_params")
        return self.fc3(out)

    def forward(self, x, h):
        # half_only path: swap_halves not applicable
        if x.shape[1] == self.fc1.in_features - h.shape[1]:
            cond_half  = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
            trans_half = x
            half_only  = True
        else:
            first_half, second_half = x.chunk(2, dim=1)
            if self.swap_halves:
                cond_half, trans_half = second_half, first_half
            else:
                cond_half, trans_half = first_half, second_half
            half_only = False

        params = self._compute_params(cond_half, h)
        params  = params.reshape(x.shape[0], -1, 3 * self.K - 1)

        widths      = F.softmax(params[..., :self.K],         dim=-1) * 2 * self.B
        heights     = F.softmax(params[..., self.K:2*self.K], dim=-1) * 2 * self.B
        derivatives = F.softplus(params[..., 2*self.K:]) + 1e-3

        trans_half_new, log_det_B = RationalQuadraticSpline.forward(
            trans_half, widths, heights, derivatives, B=self.B
        )
        log_det = log_det_B.sum(dim=1)

        if half_only:
            return trans_half_new, log_det

        if self.swap_halves:
            return torch.cat([trans_half_new, cond_half], dim=1), log_det
        else:
            return torch.cat([cond_half, trans_half_new], dim=1), log_det

    def inverse(self, z, h):
        # half_only path: swap_halves not applicable
        if z.shape[1] == (self.fc1.in_features - h.shape[1]):
            cond_half  = torch.zeros(z.shape[0], z.shape[1], device=z.device, dtype=z.dtype)
            trans_half = z
            half_only  = True
        else:
            first_half, second_half = z.chunk(2, dim=1)
            if self.swap_halves:
                cond_half, trans_half = second_half, first_half
            else:
                cond_half, trans_half = first_half, second_half
            half_only = False

        params = self._compute_params(cond_half, h)
        params  = params.reshape(z.shape[0], -1, 3 * self.K - 1)

        widths      = F.softmax(params[..., :self.K],         dim=-1) * 2 * self.B
        heights     = F.softmax(params[..., self.K:2*self.K], dim=-1) * 2 * self.B
        derivatives = F.softplus(params[..., 2*self.K:]) + 1e-3

        trans_half_orig = RationalQuadraticSpline.inverse(
            trans_half, widths, heights, derivatives, B=self.B
        )

        if half_only:
            return trans_half_orig

        if self.swap_halves:
            return torch.cat([trans_half_orig, cond_half], dim=1)
        else:
            return torch.cat([cond_half, trans_half_orig], dim=1)


class ConditionalNSF(nn.Module):
    """
    Conditional Neural Spline Flow (coupling variant).

    Stacks RQ-spline coupling layers (no BatchNorm).
    BatchNorm removed in v1.2 — batch/running stat mismatch breaks exact invertibility.
    RQ-spline outputs bounded to [-B,B]; FiLM provides stabilisation.

    [PERM] v1.6: Random permutations registered as buffers (perm_{i}, inv_perm_{i})
    for i in range(1, num_layers) and applied between coupling layers in forward/inverse.
    All layers use swap_halves=False — permutations are more general and avoid
    redundant mixing mechanisms.
    """
    def __init__(self, dim, cond_dim, num_layers=4, hidden=128, K=6, B=6.0):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers

        # [BN] v1.2: BatchNorm1d removed — coupling layers only
        # [PERM] v1.6: all layers swap_halves=False; permutations handle mixing
        layers = []
        for i in range(num_layers):
            #layers.append(ConditionalRQSplineCoupling(dim, cond_dim, hidden, K, B, swap_halves=False))
            layers.append(ConditionalRQSplineCoupling(dim, cond_dim, hidden, K, B, swap_halves=(i % 2 == 1)))
        self.layers = nn.ModuleList(layers)

        # [PERM] v1.6: register num_layers-1 random permutations as buffers
        # perm_{i} applied before layer i (for i in 1..num_layers-1)
        # inv_perm_{i} = argsort(perm_{i}) used in inverse pass
        for i in range(1, num_layers):
            perm     = torch.randperm(dim)
            inv_perm = torch.argsort(perm)
            self.register_buffer(f"perm_{i}",     perm)
            self.register_buffer(f"inv_perm_{i}", inv_perm)

        logger.info(
            f"ConditionalNSF v1.6 initialized: dim={dim}, cond_dim={cond_dim}, "
            f"num_layers={num_layers}, K={K}, B={B}, permutations={num_layers - 1}"
        )

    def forward(self, x, h):
        """
        Forward: x → z
        Pattern: coupling_0 → perm_1 → coupling_1 → perm_2 → coupling_2 → ...

        Returns:
            z: (B, d) latent
            log_det: (B,) total log-Jacobian
        """
        log_det = torch.zeros(x.shape[0], device=x.device)
        z = x

        for i, layer in enumerate(self.layers):
            # [PERM] v1.6: permute before each layer except the first
            if i > 0:
                perm = getattr(self, f"perm_{i}")
                z = z[:, perm]

            z, ld = layer(z, h)
            log_det += ld

        return z, log_det

    def inverse(self, z, h):
        """
        Inverse: z → x
        Pattern (reversed): inv_coupling_N → inv_perm_N → ... → inv_perm_1 → inv_coupling_0
        """
        x = z

        for i, layer in enumerate(reversed(self.layers)):
            layer_idx = self.num_layers - 1 - i  # original layer index

            x = layer.inverse(x, h)

            # [PERM] v1.6: undo permutation after inverse coupling (except for layer 0)
            if layer_idx > 0:
                inv_perm = getattr(self, f"inv_perm_{layer_idx}")
                x = x[:, inv_perm]

        return x
