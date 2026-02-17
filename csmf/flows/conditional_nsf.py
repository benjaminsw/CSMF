# Version: WP0.3-CondNSF-v1.0
# Abbr: COND-NSF
# Dependencies: ConditionalAffineCoupling (Level 2)
# Reference: Durkan et al. (2019) - Neural Spline Flows

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """
    def __init__(self, dim, cond_dim, hidden=128, K=8, B=3.0):
        super().__init__()
        self.K = K
        self.B = B
        
        # Network outputs spline parameters
        self.param_net = nn.Sequential(
            nn.Linear(dim//2 + cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, (dim//2) * (3*K - 1))  # widths + heights + derivatives
        )
    
    def forward(self, x, h):
        # Accept either full (B, dim) or half (B, dim//2)
        if x.shape[1] == self.param_net[0].in_features - h.shape[1]:  # dim//2
            xA = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
            xB = x
            half_only = True
        else:
            xA, xB = x.chunk(2, dim=1)
            half_only = False

        inp = torch.cat([xA, h], dim=1)
        params = self.param_net(inp)

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
        if z.shape[1] == (self.param_net[0].in_features - h.shape[1]):  # dim//2
            zA = torch.zeros(z.shape[0], z.shape[1], device=z.device, dtype=z.dtype)
            zB = z
            half_only = True
        else:
            zA, zB = z.chunk(2, dim=1)
            half_only = False

        inp = torch.cat([zA, h], dim=1)
        params = self.param_net(inp)

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
    
    Stacks RQ-spline coupling layers with batch norm.
    """
    def __init__(self, dim, cond_dim, num_layers=4, hidden=128, K=8, B=3.0):
        super().__init__()
        self.dim = dim
        
        layers = []
        for i in range(num_layers):
            # RQ-spline coupling
            layers.append(ConditionalRQSplineCoupling(dim, cond_dim, hidden, K, B))
            
            # Batch norm for stability
            layers.append(nn.BatchNorm1d(dim))
        
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
            if isinstance(layer, ConditionalRQSplineCoupling):
                z, ld = layer(z, h)
                log_det += ld
            else:  # BatchNorm
                z = layer(z)
        
        return z, log_det
    
    
    def _bn_inverse(self, bn: nn.BatchNorm1d, y: torch.Tensor) -> torch.Tensor:
        # In eval mode BN is affine using running stats
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps

        if bn.affine:
            w = bn.weight
            b = bn.bias
            w = torch.where(w == 0, torch.ones_like(w), w)  # safety
            x = (y - b) / w
        else:
            x = y

        x = x * torch.sqrt(var + eps) + mean
        return x

        
    def inverse(self, z, h):
        x = z
        for layer in reversed(self.layers):
            if isinstance(layer, ConditionalRQSplineCoupling):
                x = layer.inverse(x, h)
            elif isinstance(layer, nn.BatchNorm1d):
                x = self._bn_inverse(layer, x)
        return x

    
    
    
    #def inverse(self, z, h):
        """
        Inverse: z → x (for sampling)
        """
    #    x = z
    #    for layer in reversed(self.layers):
    #        if isinstance(layer, ConditionalRQSplineCoupling):
    #            x = layer.inverse(x, h)
            # Skip BatchNorm in inverse
        
    #    return x