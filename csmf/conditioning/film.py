"""
FiLM (Feature-wise Linear Modulation)
Applies γ(h) ⊙ f + β(h) transformation
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
    Feature-wise Linear Modulation
    
    Args:
        h_dim: Dimension of conditioning features h
        f_dim: Dimension of features to modulate f
        hidden_dims: List of hidden layer dimensions for MLPs
    """
    
    def __init__(self, f_dim, h_dim, hidden_dims=[128, 128]):
        super().__init__()
        
        self.h_dim = h_dim
        self.f_dim = f_dim
        
        # Separate MLPs for gamma (scale) and beta (shift)
        self.gamma_mlp = MLP(h_dim, f_dim, hidden_dims)
        self.beta_mlp = MLP(h_dim, f_dim, hidden_dims)
    
    def forward(self, f, h):
        """
        Apply FiLM transformation: γ(h) ⊙ f + β(h)
        
        Args:
            f: Features to modulate [B, f_dim] or [B, f_dim, H, W]
            h: Conditioning features [B, h_dim] or [B, h_dim, H', W']
        
        Returns:
            Modulated features (same shape as f)
        """
        # Handle spatial conditioning features
        if h.dim() == 4:  # [B, h_dim, H, W]
            h = torch.mean(h, dim=[2, 3])  # Global average pooling → [B, h_dim]
        
        # Compute modulation parameters
        gamma = self.gamma_mlp(h)  # [B, f_dim]
        beta = self.beta_mlp(h)    # [B, f_dim]
        
        # Handle spatial features to modulate
        if f.dim() == 4:  # [B, f_dim, H, W]
            # Reshape for broadcasting
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, f_dim, 1, 1]
            beta = beta.unsqueeze(-1).unsqueeze(-1)    # [B, f_dim, 1, 1]
        
        # Apply FiLM transformation
        return gamma * f + beta
