import torch
import torch.nn as nn
from typing import Tuple

class RealNVPFlow(nn.Module):
    """Improved Real-NVP with better masking and stability."""
    
    def __init__(self, dim: int, n_layers: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.dim = dim
        
        # Better masking strategy for 2D data
        self.masks = []
        for i in range(n_layers):
            mask = torch.zeros(dim)
            if dim == 2:
                # For 2D: alternate [1,0] and [0,1] but with more layers
                mask[i % 2] = 1
            else:
                # For higher dimensions: use checkerboard pattern
                mask[i % 2::2] = 1
            
            self.register_buffer(f'mask_{i}', mask)
            self.masks.append(mask)
        
        # Improved networks with better initialization
        self.scale_nets = nn.ModuleList()
        self.translate_nets = nn.ModuleList()
        
        for _ in range(n_layers):
            # Scale network with tanh output for stability
            scale_net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim),
                nn.Tanh()  # Bound output to [-1, 1]
            )
            
            # Translation network
            translate_net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim)
            )
            
            # Initialize final layers to be close to identity
            nn.init.zeros_(scale_net[-2].weight)
            nn.init.zeros_(scale_net[-2].bias)
            nn.init.zeros_(translate_net[-1].weight)
            nn.init.zeros_(translate_net[-1].bias)
            
            self.scale_nets.append(scale_net)
            self.translate_nets.append(translate_net)
    
    def forward_and_log_det(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = x
        log_det_total = torch.zeros(x.size(0), device=x.device)
        
        for i, mask in enumerate(self.masks):
            mask = mask.to(x.device)
            z_masked = z * mask
            
            # Compute scale and translate
            s = self.scale_nets[i](z_masked) * (1 - mask)
            # Scale factor: use smaller range for stability
            s = s * 0.5  # Scale down the tanh output
            
            t = self.translate_nets[i](z_masked) * (1 - mask)
            
            # Apply transformation
            z = z_masked + (1 - mask) * (z * torch.exp(s) + t)
            
            # Accumulate log determinant
            log_det_total += s.sum(dim=1)
        
        return z, log_det_total
    
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x = z
        
        for i in reversed(range(len(self.masks))):
            mask = self.masks[i].to(z.device)
            x_masked = x * mask
            
            s = self.scale_nets[i](x_masked) * (1 - mask)
            s = s * 0.5  # Same scaling as forward
            
            t = self.translate_nets[i](x_masked) * (1 - mask)
            
            x = x_masked + (1 - mask) * ((x - t) * torch.exp(-s))
        
        return x
    
    def log_prob(self, x):
        z, log_det = self.forward_and_log_det(x)
        # Standard Gaussian log prob
        log_prob_z = -0.5 * (z**2).sum(dim=1) - 0.5 * z.size(1) * torch.log(2 * torch.tensor(torch.pi))
        return log_prob_z + log_det
    
    def sample(self, n_samples):
        device = next(self.parameters()).device
        # Sample from standard Gaussian
        z = torch.randn(n_samples, self.dim, device=device)
        return self.inverse(z)