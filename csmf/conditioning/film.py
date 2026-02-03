"""
FiLM (Feature-wise Linear Modulation)
Applies γ(h) ⊙ f + β(h) transformation with LayerNorm and identity initialization

Version: WP0.1-FiLM-v1.1
Last Modified: 2025-02-03
Changelog:
  v1.1 (2025-02-03): Added LayerNorm for h normalization, identity initialization (gamma=1+δ, beta=δ), configurable scale_factor
  v1.0 (2025-12-09): Initial implementation with spatial/vector support
Dependencies: torch>=2.0
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple MLP for generating gamma or beta parameters"""
    
    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            # ReLU for all layers except the last
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation with LayerNorm and identity initialization
    
    Args:
        f_dim: Dimension of features to modulate f
        h_dim: Dimension of conditioning features h
        hidden_dims: List of hidden layer dimensions for MLPs
        scale_factor: Scale for modulation parameters (default=0.1)
                     Controls strength of conditioning at initialization
        use_layer_norm: Whether to normalize h before MLPs (default=True)
    """
    
    def __init__(self, f_dim, h_dim, hidden_dims=[128, 128], scale_factor=0.1, use_layer_norm=True):
        super().__init__()
        
        # Set seed for reproducibility
        torch.manual_seed(2026)
        
        self.h_dim = h_dim
        self.f_dim = f_dim
        self.scale_factor = scale_factor
        self.use_layer_norm = use_layer_norm
        
        # LayerNorm for conditioning features
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(h_dim)
        
        # Separate MLPs for gamma (scale) and beta (shift)
        self.gamma_mlp = MLP(h_dim, f_dim, hidden_dims)
        self.beta_mlp = MLP(h_dim, f_dim, hidden_dims)
    
    def forward(self, f, h):
        """
        Apply FiLM transformation: (1 + scale*γ_mlp(h)) ⊙ f + scale*β_mlp(h)
        
        Args:
            f: Features to modulate [B, f_dim] or [B, f_dim, H, W]
            h: Conditioning features [B, h_dim] or [B, h_dim, H', W']
        
        Returns:
            Modulated features (same shape as f)
        """
        # Handle spatial conditioning features
        if h.dim() == 4:  # [B, h_dim, H, W]
            h = torch.mean(h, dim=[2, 3])  # Global average pooling → [B, h_dim]
        
        # Normalize conditioning features
        if self.use_layer_norm:
            h = self.layer_norm(h)
        
        # Compute modulation parameters with identity initialization
        gamma = 1.0 + self.scale_factor * self.gamma_mlp(h)  # [B, f_dim], starts at 1
        beta = self.scale_factor * self.beta_mlp(h)          # [B, f_dim], starts at 0
        
        # Handle spatial features to modulate
        if f.dim() == 4:  # [B, f_dim, H, W]
            # Reshape for broadcasting
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, f_dim, 1, 1]
            beta = beta.unsqueeze(-1).unsqueeze(-1)    # [B, f_dim, 1, 1]
        
        # Apply FiLM transformation
        return gamma * f + beta