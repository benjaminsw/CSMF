import torch
import torch.nn as nn
from typing import Tuple
from .base_flow import BaseFlow

class MaskedLinear(nn.Module):
    """Masked linear layer for autoregressive flows."""
    
    def __init__(self, in_features, out_features, mask):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        # Ensure mask has correct shape: [out_features, in_features]
        if mask.dim() == 1:
            mask = mask.unsqueeze(0).expand(out_features, -1)
        self.register_buffer('mask', mask)
    
    def forward(self, x):
        # Apply mask to weights during forward pass
        masked_weight = self.linear.weight * self.mask
        return nn.functional.linear(x, masked_weight, self.linear.bias)

class MADE(nn.Module):
    """Masked Autoencoder for Distribution Estimation."""
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=None):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim * 2  # For scale and translation
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Simplified autoregressive masking
        # Each dimension can only depend on previous dimensions
        
        # Input to hidden: each hidden unit sees up to a certain input dimension
        max_degree = input_dim - 1
        degrees = torch.randint(0, max_degree, (hidden_dim,))
        
        # Input mask: hidden unit h can see inputs 0 to degrees[h]
        input_mask = torch.zeros(hidden_dim, input_dim)
        for h in range(hidden_dim):
            input_mask[h, :degrees[h]+1] = 1
        
        # Output mask: output for dimension d can only use hidden units with degree < d
        output_mask = torch.zeros(output_dim, hidden_dim)
        for d in range(input_dim):
            # Scale and translation for dimension d
            scale_idx = d
            trans_idx = d + input_dim
            
            for h in range(hidden_dim):
                if degrees[h] < d:
                    if scale_idx < output_dim:
                        output_mask[scale_idx, h] = 1
                    if trans_idx < output_dim:
                        output_mask[trans_idx, h] = 1
        
        # Create masked layers
        self.input_layer = MaskedLinear(input_dim, hidden_dim, input_mask)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = MaskedLinear(hidden_dim, output_dim, output_mask)
        
        self.activation = nn.ReLU()
        
        # Store degrees for debugging
        self.register_buffer('degrees', degrees)
    
    def forward(self, x):
        h = self.activation(self.input_layer(x))
        h = self.activation(self.hidden_layer(h))
        return self.output_layer(h)

class MAFFlow(BaseFlow):
    """Masked Autoregressive Flow."""
    
    def __init__(self, dim: int, n_layers: int = 4, hidden_dim: int = 64):
        super().__init__(dim)
        self.n_layers = n_layers
        
        # Create MADE networks for each layer
        self.made_networks = nn.ModuleList([
            MADE(dim, hidden_dim) for _ in range(n_layers)
        ])
        
        # Create random permutations for each layer to increase expressiveness
        self.permutations = []
        for i in range(n_layers):
            if i == 0:
                # Identity permutation for first layer
                perm = torch.arange(dim)
            else:
                # Random permutations for subsequent layers
                perm = torch.randperm(dim)
            self.register_buffer(f'perm_{i}', perm)
            self.permutations.append(perm)
    
    def forward_and_log_det(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = x.clone()  # FIXED: Start with a copy to avoid modifying input
        log_det_total = torch.zeros(x.size(0), device=x.device)
        
        for i in range(self.n_layers):
            # Apply permutation to increase expressiveness
            z = z[:, self.permutations[i]]
            
            # Get scale and translation from MADE
            made_output = self.made_networks[i](z)
            
            # Split output into scale and translation
            scale = made_output[:, :self.dim]
            translation = made_output[:, self.dim:]
            
            # Clamp scale to prevent numerical instability
            scale = torch.clamp(scale, min=-5, max=5)
            
            # Apply autoregressive transformation without in-place operations
            # Create a new tensor to avoid in-place modifications
            z_new = z.clone()
            for j in range(1, self.dim):
                z_new[:, j] = z[:, j] * torch.exp(scale[:, j]) + translation[:, j]
                log_det_total = log_det_total + scale[:, j]  # FIXED: Avoid += in-place operation
            z = z_new
        
        return z, log_det_total
    
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x = z.clone()  # FIXED: Start with a copy
        
        # Apply inverse transformations in reverse order
        for i in reversed(range(self.n_layers)):
            # Get scale and translation using current x values
            made_output = self.made_networks[i](x)
            scale = made_output[:, :self.dim]
            translation = made_output[:, self.dim:]
            
            # Clamp scale
            scale = torch.clamp(scale, min=-5, max=5)
            
            # Apply inverse transformation in reverse dimension order
            # Create new tensor to avoid in-place operations
            x_new = x.clone()
            for j in reversed(range(1, self.dim)):
                x_new[:, j] = (x[:, j] - translation[:, j]) * torch.exp(-scale[:, j])
            x = x_new
            
            # Apply inverse permutation
            inv_perm = torch.argsort(self.permutations[i])
            x = x[:, inv_perm]
        
        return x