# Version: WP0.3-CondNICE-v1.0
# Abbr: COND-NICE
# Dependencies: ConditionalAffineCoupling (Level 2)

import torch
import torch.nn as nn

class ConditionalAdditiveCoupling(nn.Module):
    """
    Additive coupling for NICE: x_B' = x_B + t(x_A, h)
    
    Simpler than affine (no scale), but still tractable + invertible.
    """
    def __init__(self, dim, cond_dim, hidden=128):
        super().__init__()
        self.t_net = nn.Sequential(
            nn.Linear(dim//2 + cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim//2)
        )
    
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
        
        # Translation from xA and h
        inp = torch.cat([xA, h], dim=1)
        t = self.t_net(inp)
        
        # Additive coupling (no scaling)
        xB_new = xB + t
        
        # Log-det = 0 (volume preserving)
        log_det = torch.zeros(x.shape[0], device=x.device)
        
        return torch.cat([xA, xB_new], dim=1), log_det
    
    def inverse(self, z, h):
        """Inverse is trivial: x_B = z_B - t(z_A, h)"""
        zA, zB = z.chunk(2, dim=1)
        
        inp = torch.cat([zA, h], dim=1)
        t = self.t_net(inp)
        
        xB = zB - t
        
        return torch.cat([zA, xB], dim=1)


class ConditionalNICE(nn.Module):
    """
    Conditional NICE: stack additive couplings + batch norm.
    
    Original NICE: Dinh et al. (2014)
    Extension: Condition on h from degraded observation y
    """
    def __init__(self, dim, cond_dim, num_layers=4, hidden=128):
        super().__init__()
        self.dim = dim
        
        layers = []
        for i in range(num_layers):
            # Additive coupling
            layers.append(ConditionalAdditiveCoupling(dim, cond_dim, hidden))
            
            # Batch norm for stability
            layers.append(nn.BatchNorm1d(dim))
        
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
            if isinstance(layer, ConditionalAdditiveCoupling):
                z, ld = layer(z, h)
                log_det += ld  # Always 0 for additive
            else:  # BatchNorm
                z = layer(z)
        
        # Final scaling (learned per dimension)
        z = z * torch.exp(self.scaling)
        log_det += self.scaling.sum()
        
        return z, log_det
    
    #def inverse(self, z, h):
        """
        Inverse: z → x (for sampling)
        
        Reverse order + undo scaling
        """
    #    # Undo final scaling
    #    x = z * torch.exp(-self.scaling)
        
        # Reverse layers
    #    for layer in reversed(self.layers):
    #        if isinstance(layer, ConditionalAdditiveCoupling):
    #            x = layer.inverse(x, h)
            # Skip BatchNorm in inverse (assume eval mode)
        
    #    return x
    
    # inside ConditionalNICE class

    def _batchnorm1d_inverse(self, bn: nn.BatchNorm1d, y: torch.Tensor) -> torch.Tensor:
        # Works in eval mode (uses running_mean / running_var)
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps

        if bn.affine:
            weight = bn.weight
            bias = bn.bias
            # avoid divide-by-zero if someone ever sets weight=0
            weight_safe = torch.where(weight == 0, torch.ones_like(weight), weight)
            x = (y - bias) / weight_safe
        else:
            x = y

        x = x * torch.sqrt(var + eps) + mean
        return x

    def inverse(self, z, h):
        # Undo final scaling
        x = z * torch.exp(-self.scaling)

        # Reverse layers, and invert BOTH coupling and batchnorm
        for layer in reversed(self.layers):
            if isinstance(layer, ConditionalAdditiveCoupling):
                x = layer.inverse(x, h)
            elif isinstance(layer, nn.BatchNorm1d):
                x = self._batchnorm1d_inverse(layer, x)

        return x
