# Version: WP0.3-CondNSF-v1.2
# Abbr: COND-NSF
# Last Modified: 2026-02-28
# Changelog:
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
from csmf.conditioning.film import FiLM  # [F1] v1.1 — shared FiLM module

logger = logging.getLogger(__name__)

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
        # Clamp to valid range (linear tails outside)
        inside_mask = (x >= -B) & (x <= B)
        
        # Compute knot positions
        knots_x = torch.cumsum(widths, dim=-1) - B  # (B, D, K)
        knots_y = torch.cumsum(heights, dim=-1) - B  # (B, D, K)
        
        # Prepend boundary knots
        knots_x = torch.cat([
            torch.full_like(knots_x[..., :1], -B),
            knots_x
        ], dim=-1)  # (B, D, K+1)
        
        knots_y = torch.cat([
            torch.full_like(knots_y[..., :1], -B),
            knots_y
        ], dim=-1)  # (B, D, K+1)
        
        # Derivatives at boundaries = 1 (match linear tails)
        derivatives = torch.cat([
            torch.ones_like(derivatives[..., :1]),
            derivatives,
            torch.ones_like(derivatives[..., :1])
        ], dim=-1)  # (B, D, K+1)
        
        # Find which bin each x falls into
        bin_idx = torch.searchsorted(knots_x.contiguous(), x.unsqueeze(-1).contiguous())
        bin_idx = torch.clamp(bin_idx - 1, 0, widths.shape[-1] - 1).squeeze(-1)  # (B, D)
        
        # Gather bin parameters
        x_k = torch.gather(knots_x, -1, bin_idx.unsqueeze(-1)).squeeze(-1)
        x_kp1 = torch.gather(knots_x, -1, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)
        y_k = torch.gather(knots_y, -1, bin_idx.unsqueeze(-1)).squeeze(-1)
        y_kp1 = torch.gather(knots_y, -1, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)
        delta_k = torch.gather(derivatives, -1, bin_idx.unsqueeze(-1)).squeeze(-1)
        delta_kp1 = torch.gather(derivatives, -1, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)
        
        # Compute ξ (normalized position in bin)
        s_k = (y_kp1 - y_k) / (x_kp1 - x_k + 1e-8)  # slope
        xi = (x - x_k) / (x_kp1 - x_k + 1e-8)
        
        # Rational-quadratic formula (Eq. 4 in paper)
        numerator = (y_kp1 - y_k) * (s_k * xi**2 + delta_k * xi * (1 - xi))
        denominator = s_k + (delta_kp1 + delta_k - 2 * s_k) * xi * (1 - xi) + 1e-8
        y = y_k + numerator / denominator
        
        # Derivative (Eq. 5 in paper)
        derivative = (s_k**2 * (delta_kp1 * xi**2 + 2 * s_k * xi * (1 - xi) + delta_k * (1 - xi)**2)) / (denominator**2 + 1e-8)
        log_det = torch.log(derivative + 1e-8)
        
        # Apply linear tails (identity outside [-B, B])
        y = torch.where(inside_mask, y, x)
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
        
        # Compute knots (same as forward)
        knots_x = torch.cumsum(widths, dim=-1) - B
        knots_y = torch.cumsum(heights, dim=-1) - B
        
        knots_x = torch.cat([torch.full_like(knots_x[..., :1], -B), knots_x], dim=-1)
        knots_y = torch.cat([torch.full_like(knots_y[..., :1], -B), knots_y], dim=-1)
        
        derivatives = torch.cat([
            torch.ones_like(derivatives[..., :1]),
            derivatives,
            torch.ones_like(derivatives[..., :1])
        ], dim=-1)
        
        # Find bin for y
        bin_idx = torch.searchsorted(knots_y.contiguous(), y.unsqueeze(-1).contiguous())
        bin_idx = torch.clamp(bin_idx - 1, 0, widths.shape[-1] - 1).squeeze(-1)
        
        # Gather bin parameters
        x_k = torch.gather(knots_x, -1, bin_idx.unsqueeze(-1)).squeeze(-1)
        x_kp1 = torch.gather(knots_x, -1, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)
        y_k = torch.gather(knots_y, -1, bin_idx.unsqueeze(-1)).squeeze(-1)
        y_kp1 = torch.gather(knots_y, -1, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)
        delta_k = torch.gather(derivatives, -1, bin_idx.unsqueeze(-1)).squeeze(-1)
        delta_kp1 = torch.gather(derivatives, -1, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)
        
        s_k = (y_kp1 - y_k) / (x_kp1 - x_k + 1e-8)
        
        # Solve quadratic: aξ² + bξ + c = 0 (Eq. 6-8 in paper)
        a = (y_kp1 - y_k) * (s_k - delta_k) + (y - y_k) * (delta_kp1 + delta_k - 2 * s_k)
        b = (y_kp1 - y_k) * delta_k - (y - y_k) * (delta_kp1 + delta_k - 2 * s_k)
        c = -s_k * (y - y_k)
        
        # Use stable quadratic formula: ξ = 2c / (-b - sqrt(b² - 4ac))
        discriminant = b**2 - 4 * a * c
        xi = 2 * c / (-b - torch.sqrt(discriminant + 1e-8) + 1e-8)
        
        x = x_k + xi * (x_kp1 - x_k)
        
        # Linear tails
        x = torch.where(inside_mask, x, y)
        
        return x


class ConditionalRQSplineCoupling(nn.Module):
    """
    Coupling layer with rational-quadratic spline transforms.

    x_B' = RQSpline(x_B; θ(x_A, h))
    where θ = {widths, heights, derivatives}

    [F1] v1.1: param_net replaced with explicit fc1/fc2/fc3 + FiLM after each hidden ReLU.
    h is still concatenated at input AND guides hidden layers via FiLM.
    """
    def __init__(self, dim, cond_dim, hidden=128, K=8, B=3.0):
        super().__init__()
        self.K = K
        self.B = B
        self.cond_dim = cond_dim
        out_dim = (dim // 2) * (3 * K - 1)   # widths + heights + derivatives

        # [F1] v1.1 — Explicit layers (replacing nn.Sequential) to allow FiLM insertion
        self.fc1   = nn.Linear(dim // 2 + cond_dim, hidden)  # input: [xA | h]
        self.fc2   = nn.Linear(hidden, hidden)
        self.fc3   = nn.Linear(hidden, out_dim)               # output: spline params (no FiLM)
        self.act   = nn.ReLU()
        self.film1 = FiLM(f_dim=hidden, h_dim=cond_dim)      # [F1] after hidden layer 1
        self.film2 = FiLM(f_dim=hidden, h_dim=cond_dim)      # [F1] after hidden layer 2

        logger.info(f"ConditionalRQSplineCoupling v1.1 initialized: dim={dim}, cond_dim={cond_dim}, hidden={hidden}, K={K}, B={B}, FiLM=True")

    def _compute_params(self, xA, h):
        """
        Compute raw spline params with FiLM modulation at each hidden layer.
        Shared by forward() and inverse() to avoid code duplication.
        """
        inp = torch.cat([xA, h], dim=1)          # [xA | h] — h at input (kept from v1.0)
        out = self.act(self.fc1(inp))
        out = self.film1(out, h)                  # [F1] FiLM after hidden layer 1
        if torch.isnan(out).any() or torch.isinf(out).any():
            logger.error("[COND-NSF] NaN/Inf after film1 in _compute_params")
            raise RuntimeError("NaN/Inf after film1 in _compute_params")
        out = self.act(self.fc2(out))
        out = self.film2(out, h)                  # [F1] FiLM after hidden layer 2
        if torch.isnan(out).any() or torch.isinf(out).any():
            logger.error("[COND-NSF] NaN/Inf after film2 in _compute_params")
            raise RuntimeError("NaN/Inf after film2 in _compute_params")
        return self.fc3(out)                      # raw spline params — no FiLM (would corrupt outputs)

    def forward(self, x, h):
        # Accept either full (B, dim) or half (B, dim//2)
        # [F1] v1.1: half_only check updated from param_net[0].in_features → fc1.in_features
        if x.shape[1] == self.fc1.in_features - h.shape[1]:  # dim//2
            xA = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
            xB = x
            half_only = True
        else:
            xA, xB = x.chunk(2, dim=1)
            half_only = False

        params = self._compute_params(xA, h)      # [F1] FiLM-conditioned spline params

        params = params.reshape(x.shape[0], -1, 3*self.K - 1)
        widths_raw = params[..., :self.K]
        heights_raw = params[..., self.K:2*self.K]
        derivatives_raw = params[..., 2*self.K:]

        widths = F.softmax(widths_raw, dim=-1) * 2 * self.B
        heights = F.softmax(heights_raw, dim=-1) * 2 * self.B
        derivatives = F.softplus(derivatives_raw) + 1e-3

        xB_new, log_det_B = RationalQuadraticSpline.forward(
            xB, widths, heights, derivatives, B=self.B
        )
        log_det = log_det_B.sum(dim=1)

        if half_only:
            return xB_new, log_det
        return torch.cat([xA, xB_new], dim=1), log_det

    def inverse(self, z, h):
        # [F1] v1.1: half_only check updated from param_net[0].in_features → fc1.in_features
        if z.shape[1] == (self.fc1.in_features - h.shape[1]):  # dim//2
            zA = torch.zeros(z.shape[0], z.shape[1], device=z.device, dtype=z.dtype)
            zB = z
            half_only = True
        else:
            zA, zB = z.chunk(2, dim=1)
            half_only = False

        params = self._compute_params(zA, h)      # [F1] same _compute_params — no duplication

        params = params.reshape(z.shape[0], -1, 3*self.K - 1)
        widths_raw = params[..., :self.K]
        heights_raw = params[..., self.K:2*self.K]
        derivatives_raw = params[..., 2*self.K:]

        widths = F.softmax(widths_raw, dim=-1) * 2 * self.B
        heights = F.softmax(heights_raw, dim=-1) * 2 * self.B
        derivatives = F.softplus(derivatives_raw) + 1e-3

        xB = RationalQuadraticSpline.inverse(
            zB, widths, heights, derivatives, B=self.B
        )

        if half_only:
            return xB
        return torch.cat([zA, xB], dim=1)
class ConditionalNSF(nn.Module):
    """
    Conditional Neural Spline Flow (coupling variant).

    Stacks RQ-spline coupling layers (no BatchNorm).
    BatchNorm removed in v1.2 — batch/running stat mismatch breaks exact invertibility.
    RQ-spline outputs bounded to [-B,B]; FiLM provides stabilisation.
    """
    def __init__(self, dim, cond_dim, num_layers=4, hidden=128, K=8, B=3.0):
        super().__init__()
        self.dim = dim

        # [BN] v1.2: BatchNorm1d removed — coupling layers only
        layers = []
        for i in range(num_layers):
            layers.append(ConditionalRQSplineCoupling(dim, cond_dim, hidden, K, B))

        self.layers = nn.ModuleList(layers)
    
    def forward(self, x, h):
        """
        Forward: x → z

        Returns:
            z: (B, d) latent
            log_det: (B,) total log-Jacobian
        """
        log_det = torch.zeros(x.shape[0], device=x.device)

        z = x
        for layer in self.layers:
            # [BN] v1.2: all layers are ConditionalRQSplineCoupling — no BN branch needed
            z, ld = layer(z, h)
            log_det += ld

        return z, log_det
    
    
    def inverse(self, z, h):
        # [BN] v1.2: all layers are ConditionalRQSplineCoupling — no BN to invert
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x, h)
        return x