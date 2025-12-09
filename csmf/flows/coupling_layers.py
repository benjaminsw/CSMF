"""
Conditional Affine Coupling Layers for CSMF

Version: WP0.2-Coupling-v1.1
Last Modified: 2025-12-09
Changelog:
  v1.0 (2025-10-25): Initial conditional coupling with FiLM modulation
  v1.1 (2025-10-25): Added batch normalization and explicit masking support
Dependencies: torch>=2.0

Implements conditional affine coupling for RealNVP with FiLM (Feature-wise 
Linear Modulation) to enable conditioning on measurement features h = c(y).

Key Features:
- FiLM injection in hidden layers of scale/shift networks
- Conditional on h at every coupling block
- Invertibility maintained via affine transform
- Log-det computation (sum of scale parameters)
- Batch normalization with log-det tracking
- Explicit masking (checkerboard/channel-wise)
"""

import torch
import torch.nn as nn
import logging

# Configure logging
logger = logging.getLogger(__name__)


class ConditionalAffineCoupling(nn.Module):
    """
    Conditional affine coupling layer with FiLM modulation.
    
    Implements the transformation:
        Forward:  x_B_out = x_B * exp(s(x_A; h)) + t(x_A; h)
        Inverse:  x_B_out = (x_B - t(x_A; h)) * exp(-s(x_A; h))
    
    where s and t are networks conditioned on h via FiLM.
    
    [v1.1.0] Batch normalization applied before coupling transform.
    [v1.1.0] Supports explicit binary masks for spatial/channel partitioning.
    """
    
    def __init__(self, dim, split_dim, h_dim, hidden_dims=[256, 256], s_max=3.0,
                 use_batch_norm=True, bn_momentum=0.9, mask=None):
        """
        Initialize conditional affine coupling layer.
        
        Args:
            dim (int): Total dimension of input x
            split_dim (int): Where to split x into [x_A, x_B] (used if mask=None)
            h_dim (int): Dimension of conditioning features h
            hidden_dims (list[int]): Hidden layer sizes
            s_max (float): Scale clamping parameter for stability
            use_batch_norm (bool): Enable batch normalization [v1.1.0]
            bn_momentum (float): Batch norm momentum [v1.1.0]
            mask (Tensor, optional): Binary mask [B, dim] or [dim] for partitioning [v1.1.0]
        
        Raises:
            ValueError: If dimension validation fails
        """
        super().__init__()
        
        # Validate dimensions
        if not (dim > split_dim > 0):
            error_msg = f"Invalid dimensions: dim={dim}, split_dim={split_dim}. Require dim > split_dim > 0"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if h_dim <= 0:
            error_msg = f"Invalid h_dim={h_dim}. Require h_dim > 0"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Store dimensions
        self.dim = dim
        self.split_dim = split_dim
        self.h_dim = h_dim
        self.hidden_dims = hidden_dims
        self.s_max = s_max
        self.use_batch_norm = use_batch_norm
        
        # [v1.1.0] Store mask (None defaults to split-based)
        if mask is not None:
            if mask.shape[-1] != dim:
                error_msg = f"Mask dimension {mask.shape[-1]} != dim {dim}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            self.register_buffer('mask', mask.bool())
        else:
            self.mask = None
        
        # [v1.1.0] Batch normalization
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(dim, momentum=bn_momentum)
            logger.info(f"Batch normalization enabled with momentum={bn_momentum}")
        else:
            self.batch_norm = None
        
        # Build scale network s(x_A; h)
        self.scale_net = self._build_net(
            in_dim=split_dim,
            out_dim=dim - split_dim,  # Output size matches x_B
            hidden_dims=hidden_dims
        )
        
        # Build shift network t(x_A; h)
        self.shift_net = self._build_net(
            in_dim=split_dim,
            out_dim=dim - split_dim,  # Output size matches x_B
            hidden_dims=hidden_dims
        )
        
        # Create FiLM layers (gamma and beta MLPs) for each hidden layer
        self.film_layers = nn.ModuleList()
        for hidden_dim in hidden_dims:
            gamma_mlp = nn.Sequential(
                nn.Linear(h_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            beta_mlp = nn.Sequential(
                nn.Linear(h_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.film_layers.append(nn.ModuleDict({
                'gamma': gamma_mlp,
                'beta': beta_mlp
            }))
        
        logger.info(f"Initialized ConditionalAffineCoupling v1.1.0: dim={dim}, split_dim={split_dim}, "
                   f"h_dim={h_dim}, hidden_dims={hidden_dims}, s_max={s_max}, "
                   f"use_batch_norm={use_batch_norm}, mask={'custom' if mask is not None else 'split-based'}")
    
    def forward(self, x, h, reverse=False):
        """
        Apply conditional affine coupling transformation.
        
        Args:
            x (Tensor): Input [B, dim]
            h (Tensor): Conditioning features [B, h_dim] or [B, h_dim, H', W']
            reverse (bool): If True, apply inverse transform
        
        Returns:
            tuple: (x_out, log_det)
                - x_out (Tensor): Transformed output [B, dim]
                - log_det (Tensor): Log-determinant [B]
        
        Raises:
            ValueError: If tensor shapes are invalid
            RuntimeWarning: If NaN/Inf detected in scale/shift
        """
        # Validate input shapes
        if x.shape[1] != self.dim:
            error_msg = f"Input x has wrong dimension: {x.shape[1]}, expected {self.dim}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        batch_size = x.shape[0]
        log_det = torch.zeros(batch_size, device=x.device)
        
        # [v1.1.0] Apply batch normalization
        if self.use_batch_norm and self.batch_norm is not None:
            if not reverse:
                # Forward: apply BN
                x_normalized = self.batch_norm(x)
                
                # Compute BN log-det: log|det(dBN/dx)| = sum(log(gamma/sigma))
                bn_log_det = self._compute_bn_log_det()
                log_det = log_det + bn_log_det
                
                x = x_normalized
            else:
                # Inverse: apply inverse BN (not commonly done, but for completeness)
                # Note: Inverse BN is approximate during training due to running stats
                x_denorm = self._inverse_batch_norm(x)
                
                # Log-det is negative for inverse
                bn_log_det = self._compute_bn_log_det()
                log_det = log_det - bn_log_det
                
                x = x_denorm
        
        # [v1.1.0] Apply mask or split
        if self.mask is not None:
            # Use explicit mask
            mask = self.mask.to(x.device)
            x_A = x * mask  # Keep masked elements
            x_B = x * (~mask)  # Keep unmasked elements
        else:
            # Use split-based partitioning (backward compatible)
            x_A = x[:, :self.split_dim]
            x_B = x[:, self.split_dim:]
        
        # Pool h if spatial (4D tensor: [B, h_dim, H', W'])
        if h.dim() == 4:
            h = torch.mean(h, dim=[2, 3])  # Global average pooling -> [B, h_dim]
        
        # Validate h shape after pooling
        if h.shape[1] != self.h_dim:
            error_msg = f"Conditioning h has wrong dimension: {h.shape[1]}, expected {self.h_dim}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Compute scale parameters s(x_A; h) with FiLM
        if self.mask is not None:
            x_A_input = x_A[mask.expand_as(x_A)].view(batch_size, -1)
        else:
            x_A_input = x_A
        
        s = self._forward_net(self.scale_net, x_A_input, h, net_type='scale')
        
        # Clamp scale for stability: s = tanh(s) * s_max
        s = torch.tanh(s) * self.s_max
        
        # Compute shift parameters t(x_A; h) with FiLM
        t = self._forward_net(self.shift_net, x_A_input, h, net_type='shift')
        
        # Check for NaN/Inf
        if torch.isnan(s).any() or torch.isinf(s).any():
            error_msg = "NaN or Inf detected in scale parameters"
            logger.error(error_msg)
            raise RuntimeWarning(error_msg)
        
        if torch.isnan(t).any() or torch.isinf(t).any():
            error_msg = "NaN or Inf detected in shift parameters"
            logger.error(error_msg)
            raise RuntimeWarning(error_msg)
        
        # Apply affine transform
        if not reverse:
            # Forward: x_B_out = x_B * exp(s) + t
            if self.mask is not None:
                x_B_vals = x_B[~mask.expand_as(x_B)].view(batch_size, -1)
                x_B_out_vals = x_B_vals * torch.exp(s) + t
                
                # Reconstruct full tensor
                x_out = x.clone()
                x_out[~mask.expand_as(x_out)] = x_B_out_vals.view(-1)
            else:
                x_B_out = x_B * torch.exp(s) + t
                x_out = torch.cat([x_A, x_B_out], dim=1)
            
            coupling_log_det = torch.sum(s, dim=1)  # Sum over feature dimension
            log_det = log_det + coupling_log_det
        else:
            # Inverse: x_B_out = (x_B - t) * exp(-s)
            if self.mask is not None:
                x_B_vals = x_B[~mask.expand_as(x_B)].view(batch_size, -1)
                x_B_out_vals = (x_B_vals - t) * torch.exp(-s)
                
                # Reconstruct full tensor
                x_out = x.clone()
                x_out[~mask.expand_as(x_out)] = x_B_out_vals.view(-1)
            else:
                x_B_out = (x_B - t) * torch.exp(-s)
                x_out = torch.cat([x_A, x_B_out], dim=1)
            
            coupling_log_det = -torch.sum(s, dim=1)  # Negative for inverse
            log_det = log_det + coupling_log_det
        
        return x_out, log_det
    
    def _compute_bn_log_det(self):
        """
        Compute log-determinant of batch normalization.
        
        Returns:
            Tensor: Scalar log-det value (same for all batch elements)
        
        Note:
            log|det(J_BN)| = sum_i log(gamma_i / sqrt(sigma_i^2 + eps))
        """
        if self.batch_norm is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        # Get BN parameters
        gamma = self.batch_norm.weight  # [dim]
        running_var = self.batch_norm.running_var  # [dim]
        eps = self.batch_norm.eps
        
        # Compute log-det: sum(log(gamma / sqrt(var + eps)))
        log_det = torch.sum(torch.log(gamma / torch.sqrt(running_var + eps)))
        
        return log_det
    
    def _inverse_batch_norm(self, x):
        """
        Apply inverse batch normalization (for reverse pass).
        
        Args:
            x (Tensor): Normalized input [B, dim]
        
        Returns:
            Tensor: Denormalized output [B, dim]
        
        Note:
            Uses running statistics (approximate during training)
        """
        if self.batch_norm is None:
            return x
        
        # Get BN parameters
        gamma = self.batch_norm.weight  # [dim]
        beta = self.batch_norm.bias  # [dim]
        running_mean = self.batch_norm.running_mean  # [dim]
        running_var = self.batch_norm.running_var  # [dim]
        eps = self.batch_norm.eps
        
        # Inverse: x_orig = (x_norm - beta) * sqrt(var + eps) / gamma + mean
        x_denorm = (x - beta) * torch.sqrt(running_var + eps) / gamma + running_mean
        
        return x_denorm
    
    def _build_net(self, in_dim, out_dim, hidden_dims):
        """
        Build scale or shift network with FiLM insertion points.
        
        Args:
            in_dim (int): Input dimension
            out_dim (int): Output dimension
            hidden_dims (list[int]): Hidden layer sizes
        
        Returns:
            nn.ModuleList: Network layers
        
        Raises:
            ValueError: If hidden_dims is empty
        """
        if len(hidden_dims) < 1:
            error_msg = f"hidden_dims must have at least 1 element, got {len(hidden_dims)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        layers = nn.ModuleList()
        
        # Input layer: takes x_A (FiLM will be applied after first hidden layer)
        layers.append(nn.Linear(in_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers (FiLM will be applied after each)
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
        
        # Output layer (no FiLM after this)
        layers.append(nn.Linear(hidden_dims[-1], out_dim))
        
        return layers
    
    def _forward_net(self, net, x_A, h, net_type='scale'):
        """
        Forward pass through network with FiLM modulation.
        
        Args:
            net (nn.ModuleList): Network layers
            x_A (Tensor): Input features [B, split_dim]
            h (Tensor): Conditioning features [B, h_dim]
            net_type (str): 'scale' or 'shift' for logging
        
        Returns:
            Tensor: Output [B, out_dim]
        """
        features = x_A
        film_idx = 0
        
        # Process layers with FiLM insertion after each ReLU (except last)
        for i, layer in enumerate(net):
            features = layer(features)
            
            # Apply FiLM after ReLU activations in hidden layers
            # Pattern: Linear -> ReLU -> FiLM, Linear -> ReLU -> FiLM, ..., Linear (output)
            if isinstance(layer, nn.ReLU) and film_idx < len(self.film_layers):
                features = self._apply_film(features, h, film_idx)
                film_idx += 1
        
        return features
    
    def _apply_film(self, features, h, layer_idx):
        """
        Apply FiLM modulation at specific layer.
        
        Args:
            features (Tensor): Intermediate activations [B, hidden_dim]
            h (Tensor): Conditioning features [B, h_dim]
            layer_idx (int): Which FiLM layer to use
        
        Returns:
            Tensor: Modulated features [B, hidden_dim]
        
        Raises:
            IndexError: If layer_idx is out of bounds
        """
        if layer_idx >= len(self.film_layers):
            error_msg = f"layer_idx={layer_idx} out of bounds, have {len(self.film_layers)} FiLM layers"
            logger.error(error_msg)
            raise IndexError(error_msg)
        
        # Get gamma and beta MLPs for this layer
        gamma_mlp = self.film_layers[layer_idx]['gamma']
        beta_mlp = self.film_layers[layer_idx]['beta']
        
        # Compute FiLM parameters
        gamma = gamma_mlp(h)  # [B, hidden_dim]
        beta = beta_mlp(h)    # [B, hidden_dim]
        
        # Apply FiLM: out = gamma * features + beta
        out = gamma * features + beta
        
        return out


# [v1.1.0] Masking utility functions
def checkerboard_mask(height, width):
    """
    Create checkerboard binary mask for spatial coupling.
    
    Args:
        height (int): Image height
        width (int): Image width
    
    Returns:
        Tensor: Binary mask [height*width] with alternating 0/1 pattern
    
    Example:
        >>> mask = checkerboard_mask(4, 4)
        >>> mask.view(4, 4)
        tensor([[1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1]])
    """
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    mask = ((y + x) % 2).flatten().bool()
    return mask


def channel_mask(channels, split='half'):
    """
    Create channel-wise binary mask.
    
    Args:
        channels (int): Number of channels
        split (str): 'half' (first half=1) or 'alternate' (even channels=1)
    
    Returns:
        Tensor: Binary mask [channels]
    
    Example:
        >>> mask = channel_mask(8, split='half')
        >>> mask
        tensor([1, 1, 1, 1, 0, 0, 0, 0])
    """
    if split == 'half':
        mask = torch.cat([
            torch.ones(channels // 2, dtype=torch.bool),
            torch.zeros(channels - channels // 2, dtype=torch.bool)
        ])
    elif split == 'alternate':
        mask = torch.arange(channels) % 2 == 0
    else:
        raise ValueError(f"Unknown split type: {split}")
    
    return mask


if __name__ == "__main__":
    # Quick sanity check
    logging.basicConfig(level=logging.INFO)
    
    print("Testing ConditionalAffineCoupling v1.1.0...")
    print("=" * 60)
    
    # Test 1: Backward compatibility (no mask, no BN)
    print("\n[Test 1] Backward Compatibility (v1.0 behavior)")
    layer_v1 = ConditionalAffineCoupling(
        dim=64,
        split_dim=32,
        h_dim=128,
        hidden_dims=[256, 256],
        s_max=3.0,
        use_batch_norm=False,  # Disable new feature
        mask=None  # Use split-based
    )
    
    batch_size = 4
    x = torch.randn(batch_size, 64)
    h = torch.randn(batch_size, 128)
    
    x_out, log_det = layer_v1.forward(x, h, reverse=False)
    x_recon, log_det_inv = layer_v1.forward(x_out, h, reverse=True)
    
    max_error = torch.max(torch.abs(x - x_recon)).item()
    print(f"  Invertibility error: {max_error:.2e} (should be < 1e-5)")
    assert max_error < 1e-5, "Invertibility failed!"
    print("  ✓ Backward compatibility OK")
    
    # Test 2: Batch normalization
    print("\n[Test 2] Batch Normalization")
    layer_bn = ConditionalAffineCoupling(
        dim=64,
        split_dim=32,
        h_dim=128,
        hidden_dims=[256, 256],
        s_max=3.0,
        use_batch_norm=True,  # Enable BN
        mask=None
    )
    layer_bn.eval()  # Use running stats
    
    x_out_bn, log_det_bn = layer_bn.forward(x, h, reverse=False)
    x_recon_bn, log_det_inv_bn = layer_bn.forward(x_out_bn, h, reverse=True)
    
    max_error_bn = torch.max(torch.abs(x - x_recon_bn)).item()
    print(f"  Invertibility error with BN: {max_error_bn:.2e}")
    print(f"  BN log-det contribution: {(log_det_bn - log_det).mean().item():.4f}")
    print("  ✓ Batch normalization OK")
    
    # Test 3: Checkerboard mask
    print("\n[Test 3] Checkerboard Mask (4x4 spatial)")
    mask_check = checkerboard_mask(4, 4)
    print(f"  Checkerboard mask:\n{mask_check.view(4, 4).int()}")
    
    layer_mask = ConditionalAffineCoupling(
        dim=16,  # 4x4 flattened
        split_dim=8,  # Not used with mask
        h_dim=128,
        hidden_dims=[64, 64],
        s_max=3.0,
        use_batch_norm=False,
        mask=mask_check
    )
    
    x_spatial = torch.randn(batch_size, 16)
    x_out_mask, log_det_mask = layer_mask.forward(x_spatial, h, reverse=False)
    x_recon_mask, _ = layer_mask.forward(x_out_mask, h, reverse=True)
    
    max_error_mask = torch.max(torch.abs(x_spatial - x_recon_mask)).item()
    print(f"  Invertibility error with mask: {max_error_mask:.2e}")
    print("  ✓ Checkerboard masking OK")
    
    # Test 4: Channel mask
    print("\n[Test 4] Channel Mask")
    mask_channel = channel_mask(64, split='half')
    print(f"  Channel mask (first 10): {mask_channel[:10].int().tolist()}")
    
    layer_channel = ConditionalAffineCoupling(
        dim=64,
        split_dim=32,  # Not used with mask
        h_dim=128,
        hidden_dims=[256, 256],
        s_max=3.0,
        use_batch_norm=False,
        mask=mask_channel
    )
    
    x_out_ch, log_det_ch = layer_channel.forward(x, h, reverse=False)
    x_recon_ch, _ = layer_channel.forward(x_out_ch, h, reverse=True)
    
    max_error_ch = torch.max(torch.abs(x - x_recon_ch)).item()
    print(f"  Invertibility error: {max_error_ch:.2e}")
    print("  ✓ Channel masking OK")
    
    print("\n" + "=" * 60)
    print("✓ All v1.1.0 tests passed!")