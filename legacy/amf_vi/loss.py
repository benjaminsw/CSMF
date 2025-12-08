import torch
import torch.nn as nn
from typing import Callable

class SimpleIWAELoss(nn.Module):
    """Simplified IWAE loss for PoC."""
    
    def __init__(self, n_importance_samples: int = 5):
        super().__init__()
        self.n_importance_samples = n_importance_samples
    
    def forward(self, model_output: dict, target_log_prob_fn: Callable, x: torch.Tensor) -> torch.Tensor:
        """Compute IWAE loss."""
        log_prob_components = model_output['log_prob']  # [batch, n_flows]
        weights = model_output['weights']  # [batch, n_flows]
        
        # Weighted log probabilities
        weighted_log_probs = log_prob_components + torch.log(weights + 1e-8)
        
        # For simplicity, use the mixture log probability as IWAE approximation
        iwae_bound = torch.logsumexp(weighted_log_probs, dim=1).clone()
        
        # Add target log probability if available (for supervised case)
        if target_log_prob_fn is not None:
            target_log_prob = target_log_prob_fn(x)
            # Simple approximation: maximize target probability
            iwae_bound += target_log_prob
        
        return -iwae_bound.mean()  # Negative for minimization