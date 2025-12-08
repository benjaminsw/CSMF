import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple

class BaseFlow(nn.Module, ABC):
    """Simplified base flow class for PoC."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def _get_device(self) -> torch.device:
        """Get device for this module, handling cases with only buffers."""
        try:
            # Try parameters first (for flows with learnable parameters)
            return next(self.parameters()).device
        except StopIteration:
            try:
                # Fall back to buffers (for flows like RBIG with only buffers)
                return next(self.buffers()).device
            except StopIteration:
                # Fall back to CPU if no parameters or buffers
                return torch.device('cpu')
    
    @abstractmethod
    def forward_and_log_det(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform x to z and compute log determinant."""
        pass
    
    @abstractmethod
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Transform z back to x."""
        pass
    
    def sample(self, n_samples: int) -> torch.Tensor:
        """Sample from the flow."""
        device = self._get_device()
        z = torch.randn(n_samples, self.dim, device=device)
        return self.inverse(z)
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log probability."""
        z, log_det = self.forward_and_log_det(x)
        log_prob_base = -0.5 * (z**2).sum(dim=1) - 0.5 * self.dim * torch.log(torch.tensor(2 * torch.pi))
        return log_prob_base + log_det