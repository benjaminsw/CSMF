"""
Conditional Affine Coupling Layers for CSMF

Version: WP0.2-Coupling-v1.3
Last Modified: 2025-02-04
Changelog:
  v1.3 (2025-02-04): Added comprehensive FiLM and s/t debug logging for conditioning diagnosis
  v1.2 (2025-02-04): Added mask debugging for extraction, reconstruction, inverse consistency
  v1.1 (2025-10-25): Added batch normalization and explicit masking support
  v1.0 (2025-10-25): Initial conditional coupling with FiLM modulation
Dependencies: torch>=2.0
"""

from networkx import reverse
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
    
    [v1.3.0] Comprehensive FiLM and s/t debug logging
    [v1.2.0] Debug mode for mask extraction, reconstruction, inverse consistency
    [v1.1.0] Batch normalization applied before coupling transform
    [v1.1.0] Supports explicit binary masks for spatial/channel partitioning
    """
    
    def __init__(self, dim, split_dim, h_dim, hidden_dims=[256, 256], s_max=3.0,
                 use_batch_norm=True, bn_momentum=0.9, mask=None, debug=False):
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
            debug (bool): Enable comprehensive debugging [v1.3.0]
        
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
        self.debug = debug  # [v1.3.0]
        
        # [v1.1.0] Store mask (None defaults to split-based)
        if mask is not None:
            if mask.shape[-1] != dim:
                error_msg = f"Mask dimension {mask.shape[-1]} != dim {dim}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            self.register_buffer('mask', mask.bool())
            # Determine coupling dims (split-based vs custom mask)
            if self.mask is not None:
                # mask is stored as buffer, could be [dim] or [B, dim]
                mask_vec = self.mask[0] if self.mask.dim() == 2 else self.mask  # [dim]
                num_masked = int(mask_vec.sum().item())
                num_unmasked = dim - num_masked

                self.masked_dim = num_masked
                self.unmasked_dim = num_unmasked

                in_dim_for_nets = num_masked
                out_dim_for_nets = num_unmasked
            else:
                self.masked_dim = split_dim
                self.unmasked_dim = dim - split_dim

                in_dim_for_nets = split_dim
                out_dim_for_nets = dim - split_dim

            
            if self.debug:
                logger.info(f"[v1.3 MASK DEBUG] Registered mask: shape={mask.shape}, "
                           f"num_True={mask.sum()}, num_False={(~mask).sum()}")
        else:
            self.mask = None
            
            
        # ------------------------------------------------------------
        # Determine coupling dimensions for scale/shift networks
        # ------------------------------------------------------------

        # Default: split-based coupling
        in_dim_for_nets = self.split_dim
        out_dim_for_nets = self.dim - self.split_dim

        # Override for custom mask
        if self.mask is not None:
            # mask may be [dim] or [B, dim] → reduce to [dim]
            mask_vec = self.mask[0] if self.mask.dim() == 2 else self.mask
            num_masked = int(mask_vec.sum().item())
            num_unmasked = self.dim - num_masked

            in_dim_for_nets = num_masked
            out_dim_for_nets = num_unmasked

            if self.debug:
                logger.info(
                    f"[MASK NET DIMS] masked={num_masked}, unmasked={num_unmasked}"
                )


        # If using a custom mask, disable BatchNorm for exact invertibility
        if self.mask is not None and use_batch_norm:
            if self.debug:
                logger.warning("Disabling BatchNorm for masked coupling to preserve exact invertibility.")
            self.use_batch_norm = False
            use_batch_norm = False

        
        
        # [v1.1.0] Batch normalization
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(dim, momentum=bn_momentum)
            logger.info(f"Batch normalization enabled with momentum={bn_momentum}")
        else:
            self.batch_norm = None
        

        # Build scale network s(x_A; h)
        self.scale_net = self._build_net(
            in_dim=in_dim_for_nets,
            out_dim=out_dim_for_nets,
            hidden_dims=hidden_dims
        )

        # Build shift network t(x_A; h)
        self.shift_net = self._build_net(
            in_dim=in_dim_for_nets,
            out_dim=out_dim_for_nets,
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
        
        logger.info(f"Initialized ConditionalAffineCoupling v1.3.0: dim={dim}, split_dim={split_dim}, "
                   f"h_dim={h_dim}, hidden_dims={hidden_dims}, s_max={s_max}, "
                   f"use_batch_norm={use_batch_norm}, mask={'custom' if mask is not None else 'split-based'}, "
                   f"debug={debug}")
    
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
        # [v1.2.0] Store original input for inverse consistency check
        if self.debug:
            x_original = x.clone()
            logger.debug("=" * 70)
            logger.debug(f"[COUPLING DEBUG] Forward pass started (reverse={reverse})")
            logger.debug(f"  Input x: shape={x.shape}, norm={x.norm().item():.6f}")
        
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
                x_normalized = self.batch_norm(x)
                bn_log_det = self._compute_bn_log_det()
                log_det = log_det + bn_log_det
                x = x_normalized
            else:
                x_denorm = self._inverse_batch_norm(x)
                bn_log_det = self._compute_bn_log_det()
                log_det = log_det - bn_log_det
                x = x_denorm
        
        # ========== DEBUG 1: MASK EXTRACTION ==========
        if self.mask is not None:
            if self.debug:
                logger.debug("=" * 70)
                logger.debug("[DEBUG 1] MASK EXTRACTION")
                logger.debug(f"  Using custom mask: shape={self.mask.shape}")
                logger.debug(f"  Mask stats: True={self.mask.sum()}, False={(~self.mask).sum()}")
            
            # Use explicit mask
            #mask = self.mask.to(x.device)
            mask = self.mask.to(x.device) if self.mask is not None else None

            
            # Current approach: multiply by mask (creates zeros)
            x_A = x * mask  # Keep masked elements, zero out others
            x_B = x * (~mask)  # Keep unmasked elements, zero out others
            
            if self.debug:
                logger.debug(f"  x_A (after x * mask): shape={x_A.shape}")
                logger.debug(f"    Non-zero elements: {(x_A != 0).sum()} / {x_A.numel()}")
                logger.debug(f"    Norm: {x_A.norm().item():.6f}")
                logger.debug(f"  x_B (after x * ~mask): shape={x_B.shape}")
                logger.debug(f"    Non-zero elements: {(x_B != 0).sum()} / {x_B.numel()}")
                logger.debug(f"    Norm: {x_B.norm().item():.6f}")
                
                # Check reconstruction of original
                x_recon_check = x_A + x_B
                recon_error = (x - x_recon_check).abs().max().item()
                logger.debug(f"  Mask partitioning check: x_A + x_B vs x, error={recon_error:.2e}")
                if recon_error > 1e-6:
                    logger.error(f"  ❌ Mask partitioning FAILED! x_A + x_B != x")
        else:
            # Use split-based partitioning (backward compatible)
            x_A = x[:, :self.split_dim]
            x_B = x[:, self.split_dim:]
            
            if self.debug:
                logger.debug("=" * 70)
                logger.debug("[DEBUG 1] SPLIT-BASED EXTRACTION")
                logger.debug(f"  Split at dim={self.split_dim}")
                logger.debug(f"  x_A: shape={x_A.shape}, norm={x_A.norm().item():.6f}")
                logger.debug(f"  x_B: shape={x_B.shape}, norm={x_B.norm().item():.6f}")
        
        # Pool h if spatial (4D tensor: [B, h_dim, H', W'])
        if h.dim() == 4:
            h = torch.mean(h, dim=[2, 3])  # Global average pooling -> [B, h_dim]
            
        if self.debug:
            logger.debug(f"  Conditioning h: shape={h.shape}, norm={h.norm().item():.6f}")
        
        # Validate h shape after pooling
        if h.shape[1] != self.h_dim:
            error_msg = f"Conditioning h has wrong dimension: {h.shape[1]}, expected {self.h_dim}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Compute scale parameters s(x_A; h) with FiLM
       #if self.mask is not None:
            # Extract actual values (non-zero elements)
        #    x_A_input = x[:, mask]
        #else:
        #    x_A_input = x_A
        if self.mask is not None:
            x_A_input = x[:, mask]   # shape [B, num_masked] which equals split_dim=8 for checkerboard
        else:
            x_A_input = x_A

                
        # [v1.3 DEBUG] Log input to scale/shift networks
        if self.debug:
            logger.debug("=" * 70)
            logger.debug("[DEBUG 2] SCALE/SHIFT NETWORK INPUT")
            logger.debug(f"  x_A_input: shape={x_A_input.shape}, norm={x_A_input.norm().item():.6f}")
            logger.debug(f"  h: shape={h.shape}, norm={h.norm().item():.6f}")
            logger.debug(f"  Computing scale s(x_A; h)...")
        
        s = self._forward_net(self.scale_net, x_A_input, h, net_type='scale')
        
        # Clamp scale for stability: s = tanh(s) * s_max
        s_unclamped = s.clone() if self.debug else None
        s = torch.tanh(s) * self.s_max
        
        if self.debug:
            logger.debug(f"  Computing shift t(x_A; h)...")
        
        # Compute shift parameters t(x_A; h) with FiLM
        t = self._forward_net(self.shift_net, x_A_input, h, net_type='shift')
        

                
        # [v1.3 DEBUG] Log scale/shift output
        if self.debug:
            logger.debug("=" * 70)
            logger.debug("[DEBUG 3] SCALE/SHIFT OUTPUT")
            logger.debug(f"  Scale s (unclamped): norm={s_unclamped.norm().item():.6f}, "
                        f"range=[{s_unclamped.min().item():.4f}, {s_unclamped.max().item():.4f}]")
            logger.debug(f"  Scale s (clamped):   norm={s.norm().item():.6f}, "
                        f"range=[{s.min().item():.4f}, {s.max().item():.4f}]")
            logger.debug(f"  Shift t: norm={t.norm().item():.6f}, "
                        f"range=[{t.min().item():.4f}, {t.max().item():.4f}]")
            logger.debug(f"  exp(s): range=[{torch.exp(s).min().item():.4f}, {torch.exp(s).max().item():.4f}]")
        
        # Check for NaN/Inf
        if torch.isnan(s).any() or torch.isinf(s).any():
            error_msg = "NaN or Inf detected in scale parameters"
            logger.error(error_msg)
            raise RuntimeWarning(error_msg)
        
        if torch.isnan(t).any() or torch.isinf(t).any():
            error_msg = "NaN or Inf detected in shift parameters"
            logger.error(error_msg)
            raise RuntimeWarning(error_msg)
        
        # ========== DEBUG 4: TRANSFORMATION APPLICATION ==========
        if self.debug:
            logger.debug("=" * 70)
            logger.debug("[DEBUG 4] TRANSFORMATION")
        
        # Make sure mask is 1D [dim]
        #mask = self.mask.to(x.device)
        mask = None
        if self.mask is not None:
            mask = self.mask.to(x.device)
            if mask.dim() == 2:
                mask = mask[0]

        # Apply affine transform
        if mask is not None:
            if not reverse:
                x_out = x.clone()
                x_out[:, ~mask] = x[:, ~mask] * torch.exp(s) + t
                log_det = log_det + s.sum(dim=1)
            else:
                x_out = x.clone()
                x_out[:, ~mask] = (x[:, ~mask] - t) * torch.exp(-s)
                log_det = log_det - s.sum(dim=1)
        else:
            if not reverse:
                x_B_out = x_B * torch.exp(s) + t
                x_out = torch.cat([x_A, x_B_out], dim=1)
                log_det = log_det + s.sum(dim=1)
            else:
                x_B_out = (x_B - t) * torch.exp(-s)
                x_out = torch.cat([x_A, x_B_out], dim=1)
                log_det = log_det - s.sum(dim=1)


        
        # ========== DEBUG 5: INVERSE CONSISTENCY (only on forward pass) ==========
        if self.debug and not reverse:
            logger.debug("=" * 70)
            logger.debug("[DEBUG 5] INVERSE CONSISTENCY CHECK")
            logger.debug(f"  Testing: forward(x) then inverse(forward(x)) == x")
            
            # Apply inverse on the output
            x_reconstructed, _ = self.forward(x_out, h, reverse=True)
            
            # Compare with original
            inv_error = (x_original - x_reconstructed).abs().max().item()
            inv_mean_error = (x_original - x_reconstructed).abs().mean().item()
            
            logger.debug(f"  Reconstruction error:")
            logger.debug(f"    Max abs error: {inv_error:.2e} (threshold: 1e-5)")
            logger.debug(f"    Mean abs error: {inv_mean_error:.2e}")
            
            if inv_error > 1e-5:
                logger.error(f"  ❌ INVERSE CONSISTENCY FAILED!")
                logger.error(f"     Max error {inv_error:.2e} > threshold 1e-5")
                
                # Additional diagnostics
                logger.error(f"  Diagnostic info:")
                logger.error(f"    Original x norm: {x_original.norm().item():.6f}")
                logger.error(f"    Reconstructed x norm: {x_reconstructed.norm().item():.6f}")
                logger.error(f"    Forward output norm: {x_out.norm().item():.6f}")
            else:
                logger.debug(f"  ✓ Inverse consistency OK (error={inv_error:.2e})")
        
        if self.debug:
            logger.debug("=" * 70)
        
        return x_out, log_det
    
    def _compute_bn_log_det(self):
        """Compute log-determinant of batch normalization."""
        if self.batch_norm is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        gamma = self.batch_norm.weight
        running_var = self.batch_norm.running_var
        eps = self.batch_norm.eps
        
        log_det = torch.sum(torch.log(gamma / torch.sqrt(running_var + eps)))
        return log_det
    
    def _inverse_batch_norm(self, x):
        """Apply inverse batch normalization (for reverse pass)."""
        if self.batch_norm is None:
            return x
        
        gamma = self.batch_norm.weight
        beta = self.batch_norm.bias
        running_mean = self.batch_norm.running_mean
        running_var = self.batch_norm.running_var
        eps = self.batch_norm.eps
        
        x_denorm = (x - beta) * torch.sqrt(running_var + eps) / gamma + running_mean
        return x_denorm
    
    def _build_net(self, in_dim, out_dim, hidden_dims):
        """Build scale or shift network with FiLM insertion points."""
        if len(hidden_dims) < 1:
            error_msg = f"hidden_dims must have at least 1 element, got {len(hidden_dims)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        layers = nn.ModuleList()
        layers.append(nn.Linear(in_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dims[-1], out_dim))
        return layers
    
    def _forward_net(self, net, x_A, h, net_type='scale'):
        """Forward pass through network with FiLM modulation."""
        features = x_A
        film_idx = 0
        
        if self.debug:
            logger.debug(f"  [{net_type} network] Starting forward pass")
            logger.debug(f"    Input features: shape={features.shape}, norm={features.norm().item():.6f}")
        
        for i, layer in enumerate(net):
            features = layer(features)
            
            # Apply FiLM after ReLU activations
            if isinstance(layer, nn.ReLU) and film_idx < len(self.film_layers):
                if self.debug:
                    logger.debug(f"    After layer {i} (ReLU): norm={features.norm().item():.6f}")
                features = self._apply_film(features, h, film_idx)
                film_idx += 1
        
        if self.debug:
            logger.debug(f"    Final output: shape={features.shape}, norm={features.norm().item():.6f}")
        
        return features
    
    def _apply_film(self, features, h, layer_idx):
        """Apply FiLM modulation at specific layer."""
        if layer_idx >= len(self.film_layers):
            error_msg = f"layer_idx={layer_idx} out of bounds, have {len(self.film_layers)} FiLM layers"
            logger.error(error_msg)
            raise IndexError(error_msg)
        
        gamma_mlp = self.film_layers[layer_idx]['gamma']
        beta_mlp = self.film_layers[layer_idx]['beta']
        
        gamma = gamma_mlp(h)
        beta = beta_mlp(h)
        
        # [v1.3 DEBUG] Log FiLM parameters and effect
        if self.debug:
            features_before = features.clone()
            logger.debug(f"      [FiLM Layer {layer_idx}]")
            logger.debug(f"        h: norm={h.norm().item():.6f}")
            logger.debug(f"        gamma: norm={gamma.norm().item():.6f}, range=[{gamma.min().item():.4f}, {gamma.max().item():.4f}]")
            logger.debug(f"        beta: norm={beta.norm().item():.6f}, range=[{beta.min().item():.4f}, {beta.max().item():.4f}]")
            logger.debug(f"        features (before): norm={features_before.norm().item():.6f}")
        
        out = gamma * features + beta
        
        # [v1.3 DEBUG] Log effect of FiLM
        if self.debug:
            logger.debug(f"        features (after):  norm={out.norm().item():.6f}")
            feature_change = (out - features_before).norm().item()
            feature_baseline = features_before.norm().item()
            relative_change = feature_change / (feature_baseline + 1e-8)
            logger.debug(f"        FiLM effect: abs_change={feature_change:.6f}, relative={relative_change:.4f}")
            if relative_change < 0.01:
                logger.warning(f"        ⚠️  FiLM has minimal effect (relative change < 1%)")
        
        return out


# [v1.1.0] Masking utility functions
def checkerboard_mask(height, width):
    """Create checkerboard binary mask for spatial coupling."""
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    mask = ((y + x) % 2).flatten().bool()
    return mask


def channel_mask(channels, split='half'):
    """Create channel-wise binary mask."""
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