"""
ConditionalRealNVP for MNIST Inverse Problems - Multi-Scale Architecture

Version: WP0.3-CondRNVP-v2.0
Abbr: COND-RNVP
Last Modified: 2025-12-09
Changelog:
  v1.0 (2025-10-25): Initial flat architecture with 8 sequential couplings
  v2.0 (2025-10-25): Multi-scale with squeeze/unsqueeze and variable factoring
Dependencies: torch>=2.0, coupling_layers WP0.2-Coupling-v1.1+

Purpose: Conditional RealNVP expert with multi-scale architecture, variable factoring,
         and hierarchical 3+squeeze+3 structure for MNIST deblurring/inverse problems

Architecture (MNIST 28×28):
    Scale 1: [B,1,28,28] → 3 coupling → squeeze → [B,4,14,14] → factor 50%
    Scale 2: [B,2,14,14] → 3 coupling → squeeze → [B,8,7,7] → factor 50%
    Scale 3: [B,4,7,7] → 3 coupling → [B,4,7,7] (final latent)
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional
import logging

# W0.1 Level 0: Configuration
from configs.mnist_config import MNIST_CONFIG

# W0.1 Level 1: Conditioning components
from models.conditioning.conditioning_networks import MNISTConditioner

# W0.1 Level 2: Coupling layers
from models.flows.coupling_layers import ConditionalAffineCoupling

# Setup logging
logger = logging.getLogger(__name__)


class ScaleBlock(nn.Module):
    """
    Multi-scale block: N coupling layers with optional squeeze operation.
    
    Args:
        n_layers (int): Number of coupling layers in this block
        channels (int): Number of input channels
        spatial_dim (int): Spatial dimension (H=W for square images)
        h_dim (int): Conditioning feature dimension
        hidden_dims (list): MLP hidden dimensions for coupling nets
        apply_squeeze (bool): Whether to apply squeeze at end of block
    """
    
    def __init__(
        self,
        n_layers: int,
        channels: int,
        spatial_dim: int,
        h_dim: int,
        hidden_dims: List[int],
        apply_squeeze: bool = False
    ):
        super().__init__()
        
        self.n_layers = n_layers
        self.channels = channels
        self.spatial_dim = spatial_dim
        self.h_dim = h_dim
        self.apply_squeeze = apply_squeeze
        
        # Create coupling layers for this scale
        dim = channels * spatial_dim * spatial_dim
        self.coupling_layers = nn.ModuleList()
        
        for i in range(n_layers):
            split_dim = dim // 2
            layer = ConditionalAffineCoupling(
                dim=dim,
                split_dim=split_dim,
                h_dim=h_dim,
                hidden_dims=hidden_dims
            )
            self.coupling_layers.append(layer)
            logger.debug(f"ScaleBlock: added coupling layer {i+1}/{n_layers} for dim={dim}")
    
    def forward(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        reverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward through coupling layers, with optional squeeze.
        
        Args:
            z: Input [B, C, H, W]
            h: Conditioning features [B, h_dim] or [B, h_dim, H', W']
            reverse: If True, run in inverse direction
        
        Returns:
            z_out: Output (squeezed if apply_squeeze=True)
            log_det: Log-determinant [B]
        """
        B, C, H, W = z.shape
        log_det = torch.zeros(B, device=z.device)
        
        # Flatten for coupling layers
        z_flat = z.reshape(B, -1)
        
        if not reverse:
            # Forward: apply coupling layers then squeeze
            for i, layer in enumerate(self.coupling_layers):
                z_flat, ld = layer.forward(z_flat, h, reverse=False)
                log_det = log_det + ld
            
            # Reshape back to spatial
            z_out = z_flat.reshape(B, C, H, W)
            
            # Apply squeeze if configured
            if self.apply_squeeze:
                z_out = self._squeeze(z_out)
                # Squeeze is volume-preserving (no log-det change)
        else:
            # Inverse: unsqueeze first (if needed), then apply coupling layers in reverse
            z_out = z_flat.reshape(B, C, H, W)
            
            if self.apply_squeeze:
                z_out = self._unsqueeze(z_out)
            
            z_flat = z_out.reshape(B, -1)
            
            # Apply coupling layers in reverse order
            for i, layer in enumerate(reversed(self.coupling_layers)):
                z_flat, ld = layer.forward(z_flat, h, reverse=True)
                log_det = log_det + ld
            
            z_out = z_flat.reshape(B, C, H, W)
        
        return z_out, log_det
    
    def _squeeze(self, z: torch.Tensor) -> torch.Tensor:
        """
        Squeeze operation: trade spatial for channel dimensions.
        [B, C, H, W] → [B, 4C, H/2, W/2]
        """
        B, C, H, W = z.shape
        
        # Reshape: [B, C, H, W] → [B, C, H/2, 2, W/2, 2]
        z = z.reshape(B, C, H // 2, 2, W // 2, 2)
        
        # Permute: [B, C, H/2, 2, W/2, 2] → [B, C, 2, 2, H/2, W/2]
        z = z.permute(0, 1, 3, 5, 2, 4)
        
        # Merge: [B, C, 2, 2, H/2, W/2] → [B, 4C, H/2, W/2]
        z = z.reshape(B, C * 4, H // 2, W // 2)
        
        return z
    
    def _unsqueeze(self, z: torch.Tensor) -> torch.Tensor:
        """
        Unsqueeze operation: trade channel for spatial dimensions.
        [B, 4C, H, W] → [B, C, 2H, 2W]
        """
        B, C4, H, W = z.shape
        C = C4 // 4
        
        # Split: [B, 4C, H, W] → [B, C, 2, 2, H, W]
        z = z.reshape(B, C, 2, 2, H, W)
        
        # Permute: [B, C, 2, 2, H, W] → [B, C, H, 2, W, 2]
        z = z.permute(0, 1, 4, 2, 5, 3)
        
        # Merge: [B, C, H, 2, W, 2] → [B, C, 2H, 2W]
        z = z.reshape(B, C, H * 2, W * 2)
        
        return z


class ConditionalRealNVP(nn.Module):
    """
    Conditional RealNVP flow with multi-scale architecture for inverse problems.
    
    Architecture: Hierarchical 3+squeeze+3 structure with variable factoring.
    - Scale 1: 3 coupling on [1,28,28] → squeeze → [4,14,14] → factor 50%
    - Scale 2: 3 coupling on [2,14,14] → squeeze → [8,7,7] → factor 50%
    - Scale 3: 3 coupling on [4,7,7] → final latent
    
    Args:
        h_dim (int): Conditioning feature channels (default: 64)
        hidden_dims (list): MLP hidden dimensions (default: [256, 256])
        config (dict): Configuration dictionary (default: MNIST_CONFIG)
    
    Example:
        >>> model = ConditionalRealNVP(h_dim=64)
        >>> x = torch.randn(32, 1, 28, 28)  # Clean data
        >>> y = torch.randn(32, 1, 28, 28)  # Degraded observation
        >>> z, z_factored, log_det, log_prob = model.forward(x, y)
        >>> x_recon = model.inverse(z, z_factored, y)
    """
    
    def __init__(
        self,
        h_dim: int = 64,
        hidden_dims: List[int] = None,
        config: dict = None
    ):
        """Initialize ConditionalRealNVP with multi-scale architecture."""
        super().__init__()
        
        # Input validation
        if h_dim <= 0:
            raise ValueError(f"h_dim must be positive, got {h_dim}")
        
        # Set defaults
        if hidden_dims is None:
            hidden_dims = [256, 256]
        if config is None:
            config = MNIST_CONFIG
        
        # Store attributes
        self.h_dim = h_dim
        self.hidden_dims = hidden_dims
        self.config = config
        self._cached_h = None
        
        logger.info(f"Initializing ConditionalRealNVP v2.0: h_dim={h_dim}, multi-scale with factoring")
        
        # Create conditioning network (Level 1)
        self.conditioner = MNISTConditioner(h_dim=h_dim)
        logger.info(f"Created MNISTConditioner with h_dim={h_dim}")
        
        # Multi-scale architecture: 3+squeeze+3 per scale
        # Scale 1: [B,1,28,28] → 3 coupling → squeeze → [B,4,14,14]
        self.scale1 = ScaleBlock(
            n_layers=3,
            channels=1,
            spatial_dim=28,
            h_dim=h_dim,
            hidden_dims=hidden_dims,
            apply_squeeze=True
        )
        logger.info("Created Scale 1: [1,28,28] → 3 coupling → squeeze → [4,14,14]")
        
        # Scale 2: [B,2,14,14] → 3 coupling → squeeze → [B,8,7,7]
        self.scale2 = ScaleBlock(
            n_layers=3,
            channels=2,
            spatial_dim=14,
            h_dim=h_dim,
            hidden_dims=hidden_dims,
            apply_squeeze=True
        )
        logger.info("Created Scale 2: [2,14,14] → 3 coupling → squeeze → [8,7,7]")
        
        # Scale 3: [B,4,7,7] → 3 coupling → [B,4,7,7]
        self.scale3 = ScaleBlock(
            n_layers=3,
            channels=4,
            spatial_dim=7,
            h_dim=h_dim,
            hidden_dims=hidden_dims,
            apply_squeeze=False
        )
        logger.info("Created Scale 3: [4,7,7] → 3 coupling → [4,7,7]")
        
        # Register constant buffer for log(2π)
        self.register_buffer('log_2pi', torch.log(torch.tensor(2.0 * torch.pi)))
        
        # Store dimensions for each scale
        self.scale_dims = [
            (1, 28, 28),   # Scale 1 input
            (4, 14, 14),   # Scale 1 output (after squeeze)
            (2, 14, 14),   # Scale 2 input (after factoring)
            (8, 7, 7),     # Scale 2 output (after squeeze)
            (4, 7, 7),     # Scale 3 input (after factoring)
            (4, 7, 7)      # Scale 3 output (final)
        ]
        
        logger.info("ConditionalRealNVP v2.0 initialization complete")
    
    def _factor_out(
        self,
        z: torch.Tensor,
        factor_ratio: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Factor out half of the channels.
        
        Args:
            z: Input [B, C, H, W]
            factor_ratio: Fraction of channels to factor out (default: 0.5)
        
        Returns:
            z_kept: Kept channels [B, C*(1-ratio), H, W]
            z_factored: Factored channels [B, C*ratio, H, W]
        """
        B, C, H, W = z.shape
        n_factor = int(C * factor_ratio)
        n_keep = C - n_factor
        
        z_kept = z[:, :n_keep, :, :]
        z_factored = z[:, n_keep:, :, :]
        
        logger.debug(f"Factored: [{C},{H},{W}] → kept [{n_keep},{H},{W}] + factored [{n_factor},{H},{W}]")
        
        return z_kept, z_factored
    
    def _unfactor(
        self,
        z_kept: torch.Tensor,
        z_factored: torch.Tensor
    ) -> torch.Tensor:
        """
        Reverse factoring: concatenate kept and factored channels.
        
        Args:
            z_kept: Kept channels [B, C_keep, H, W]
            z_factored: Factored channels [B, C_factor, H, W]
        
        Returns:
            z: Combined [B, C_keep+C_factor, H, W]
        """
        z = torch.cat([z_kept, z_factored], dim=1)
        return z
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        compute_h: bool = True
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward transform: x → z with log-determinant and log-probability.
        
        Args:
            x: Clean data [B, 1, 28, 28]
            y: Degraded observation [B, 1, 28, 28]
            compute_h: Whether to compute fresh h or use cache
        
        Returns:
            z_final: Final latent code [B, 4, 7, 7]
            z_factored_list: List of factored variables at intermediate scales
            log_det: Log-determinant [B]
            log_prob: Total log-probability [B]
        
        Raises:
            ValueError: If input shapes are invalid
            RuntimeError: If forward pass fails
        """
        # Input validation
        if x.ndim != 4:
            raise ValueError(f"x must be 4D [batch, C, H, W], got shape {x.shape}")
        if y.ndim != 4:
            raise ValueError(f"y must be 4D [batch, C, H, W], got shape {y.shape}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Batch size mismatch: x={x.shape[0]}, y={y.shape[0]}")
        if x.shape != (x.shape[0], 1, 28, 28):
            raise ValueError(f"x must be [B, 1, 28, 28], got {x.shape}")
        
        logger.debug(f"Forward: x.shape={x.shape}, y.shape={y.shape}")
        
        # Extract conditioning features
        if compute_h:
            h = self.conditioner(y)
            self._cached_h = h
            logger.debug("Computed fresh h from y")
        else:
            if self._cached_h is None:
                raise RuntimeError("No cached h available, set compute_h=True")
            h = self._cached_h
            logger.debug("Using cached h")
        
        B = x.shape[0]
        log_det_total = torch.zeros(B, device=x.device)
        z_factored_list = []
        
        try:
            # Scale 1: [B,1,28,28] → 3 coupling → squeeze → [B,4,14,14]
            z = x.clone()
            z, log_det = self.scale1.forward(z, h, reverse=False)
            log_det_total = log_det_total + log_det
            logger.debug(f"Scale 1: {x.shape} → {z.shape}, log_det mean={log_det.mean():.4f}")
            
            # Factor out 50% channels: [B,4,14,14] → [B,2,14,14] + factored [B,2,14,14]
            z, z_factor1 = self._factor_out(z, factor_ratio=0.5)
            z_factored_list.append(z_factor1)
            logger.debug(f"Factored scale 1: kept {z.shape}, factored {z_factor1.shape}")
            
            # Scale 2: [B,2,14,14] → 3 coupling → squeeze → [B,8,7,7]
            z, log_det = self.scale2.forward(z, h, reverse=False)
            log_det_total = log_det_total + log_det
            logger.debug(f"Scale 2: → {z.shape}, log_det mean={log_det.mean():.4f}")
            
            # Factor out 50% channels: [B,8,7,7] → [B,4,7,7] + factored [B,4,7,7]
            z, z_factor2 = self._factor_out(z, factor_ratio=0.5)
            z_factored_list.append(z_factor2)
            logger.debug(f"Factored scale 2: kept {z.shape}, factored {z_factor2.shape}")
            
            # Scale 3: [B,4,7,7] → 3 coupling → [B,4,7,7]
            z, log_det = self.scale3.forward(z, h, reverse=False)
            log_det_total = log_det_total + log_det
            logger.debug(f"Scale 3: → {z.shape}, log_det mean={log_det.mean():.4f}")
            
            z_final = z
            
            # Numerical stability check
            if torch.any(torch.isnan(z_final)) or torch.any(torch.isinf(z_final)):
                logger.warning("NaN/Inf detected in z_final, applying clamp")
                z_final = torch.clamp(z_final, min=-1e6, max=1e6)
            
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            raise RuntimeError(f"Forward pass failed: {e}")
        
        # Compute base distribution log-probability for all latent variables
        log_pz_list = []
        
        # Final latent z
        z_final_flat = z_final.reshape(B, -1)
        z_squared = torch.sum(z_final_flat ** 2, dim=1)
        z_squared = torch.clamp(z_squared, max=400.0)
        dim_final = z_final_flat.shape[1]
        log_pz_final = -0.5 * (dim_final * self.log_2pi + z_squared)
        log_pz_list.append(log_pz_final)
        
        # Factored variables
        for z_fact in z_factored_list:
            z_fact_flat = z_fact.reshape(B, -1)
            z_squared = torch.sum(z_fact_flat ** 2, dim=1)
            z_squared = torch.clamp(z_squared, max=400.0)
            dim_fact = z_fact_flat.shape[1]
            log_pz_fact = -0.5 * (dim_fact * self.log_2pi + z_squared)
            log_pz_list.append(log_pz_fact)
        
        # Total log-probability: sum all Gaussian log-probs + log-determinant
        log_pz_total = sum(log_pz_list)
        log_prob_total = log_pz_total + log_det_total
        
        # Final NaN/Inf check
        nan_mask = torch.isnan(log_prob_total) | torch.isinf(log_prob_total)
        if torch.any(nan_mask):
            logger.warning(f"NaN/Inf in log_prob for {nan_mask.sum()} samples, replacing with -100.0")
            log_prob_total[nan_mask] = -100.0
        
        logger.debug(f"Forward complete: mean log_prob={log_prob_total.mean():.4f}")
        
        return z_final, z_factored_list, log_det_total, log_prob_total
    
    def inverse(
        self,
        z: torch.Tensor,
        z_factored_list: List[torch.Tensor],
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Inverse transform: z → x (sampling/reconstruction).
        
        Args:
            z: Final latent code [B, 4, 7, 7]
            z_factored_list: List of factored variables [z_factor2, z_factor1]
            y: Degraded observation [B, 1, 28, 28]
        
        Returns:
            x: Reconstructed data [B, 1, 28, 28]
        
        Raises:
            ValueError: If input shapes are invalid
            RuntimeError: If inverse pass fails
        """
        # Input validation
        if z.ndim != 4:
            raise ValueError(f"z must be 4D [batch, C, H, W], got shape {z.shape}")
        if y.ndim != 4:
            raise ValueError(f"y must be 4D [batch, C, H, W], got shape {y.shape}")
        if z.shape[0] != y.shape[0]:
            raise ValueError(f"Batch size mismatch: z={z.shape[0]}, y={y.shape[0]}")
        if len(z_factored_list) != 2:
            raise ValueError(f"Expected 2 factored variables, got {len(z_factored_list)}")
        
        logger.debug(f"Inverse: z.shape={z.shape}, y.shape={y.shape}")
        
        # Extract conditioning features
        h = self.conditioner(y)
        logger.debug("Extracted h for inverse pass")
        
        try:
            # Start from final latent
            x = z.clone()
            
            # Scale 3 inverse: [B,4,7,7] → 3 coupling → [B,4,7,7]
            x, _ = self.scale3.forward(x, h, reverse=True)
            logger.debug(f"Scale 3 inverse: → {x.shape}")
            
            # Unfactor scale 2: [B,4,7,7] + factored [B,4,7,7] → [B,8,7,7]
            z_factor2 = z_factored_list[1]
            x = self._unfactor(x, z_factor2)
            logger.debug(f"Unfactored scale 2: {x.shape}")
            
            # Scale 2 inverse: [B,8,7,7] → unsqueeze → [B,2,14,14] → 3 coupling
            x, _ = self.scale2.forward(x, h, reverse=True)
            logger.debug(f"Scale 2 inverse: → {x.shape}")
            
            # Unfactor scale 1: [B,2,14,14] + factored [B,2,14,14] → [B,4,14,14]
            z_factor1 = z_factored_list[0]
            x = self._unfactor(x, z_factor1)
            logger.debug(f"Unfactored scale 1: {x.shape}")
            
            # Scale 1 inverse: [B,4,14,14] → unsqueeze → [B,1,28,28] → 3 coupling
            x, _ = self.scale1.forward(x, h, reverse=True)
            logger.debug(f"Scale 1 inverse: → {x.shape}")
            
            # Numerical stability check
            if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
                logger.warning("NaN/Inf in inverse reconstruction, applying clamp")
                x = torch.clamp(x, min=-1e6, max=1e6)
            
        except Exception as e:
            logger.error(f"Inverse pass failed: {e}")
            raise RuntimeError(f"Inverse pass failed: {e}")
        
        # Final validation
        assert x.shape == (z.shape[0], 1, 28, 28), f"Shape mismatch: x={x.shape}"
        logger.debug("Inverse pass complete")
        
        return x
    
    def sample(
        self,
        n_samples: int,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate samples from the flow conditioned on observation y.
        
        Args:
            n_samples (int): Number of samples to generate
            y: Degraded observation [B, 1, 28, 28]
        
        Returns:
            x: Generated samples [n_samples, 1, 28, 28]
        
        Raises:
            ValueError: If n_samples invalid or doesn't match y batch size
        """
        # Input validation
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        if y.ndim != 4:
            raise ValueError(f"y must be 4D [batch, C, H, W], got shape {y.shape}")
        if n_samples != y.shape[0]:
            raise ValueError(
                f"n_samples ({n_samples}) must equal y.shape[0] ({y.shape[0]}). "
                "Multi-sample per observation not supported in v2.0"
            )
        
        logger.debug(f"Sampling {n_samples} samples")
        
        device = y.device
        
        # Sample from base distribution N(0, I) for all latent variables
        # Final latent: [B, 4, 7, 7]
        z_final = torch.randn(n_samples, 4, 7, 7, device=device)
        
        # Factored variables
        z_factor2 = torch.randn(n_samples, 4, 7, 7, device=device)  # Scale 2 factored
        z_factor1 = torch.randn(n_samples, 2, 14, 14, device=device)  # Scale 1 factored
        z_factored_list = [z_factor1, z_factor2]
        
        logger.debug("Sampled all z ~ N(0, I)")
        
        # Transform through inverse flow
        x = self.inverse(z_final, z_factored_list, y)
        
        assert x.shape == (n_samples, 1, 28, 28), f"Sample shape mismatch: {x.shape}"
        logger.debug(f"Generated {n_samples} samples")
        
        return x
    
    def log_prob(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log-probability of data x given observation y.
        
        Convenience wrapper around forward() that returns only log_prob.
        
        Args:
            x: Data to evaluate [B, 1, 28, 28]
            y: Degraded observation [B, 1, 28, 28]
        
        Returns:
            log_prob: Log-probability values [B]
        """
        logger.debug("Computing log_prob via forward pass")
        
        try:
            z_final, z_factored_list, log_det, log_prob_total = self.forward(x, y, compute_h=True)
            logger.debug(f"log_prob computed: mean={log_prob_total.mean():.4f}")
            return log_prob_total
            
        except Exception as e:
            logger.error(f"log_prob computation failed: {e}")
            raise RuntimeError(f"Cannot compute log_prob: {e}")


# Version Changelog:
# v2.0 (2025-10-25): Multi-scale architecture with variable factoring
#   - Added ScaleBlock: hierarchical 3+squeeze+3 structure
#   - Implemented squeeze/unsqueeze: trade spatial ↔ channel dimensions
#   - Added variable factoring: split channels at scale transitions
#   - Multi-scale forward: Scale1→factor→Scale2→factor→Scale3
#   - Multi-scale inverse: unfactor and reconstruct through scales
#   - Modified sampling: sample all latent variables (final + factored)
#   - Maintains 2D spatial structure [B,C,H,W] instead of flattening
#   - Architecture: [1,28,28]→[4,14,14]→[2,14,14]→[8,7,7]→[4,7,7]
#   - MNIST-specific (28×28) - hardcoded dimensions for v2.0
#   - Fixed 50% factoring ratio - not configurable
#
# v1.0 (2025-10-25): Initial minimal implementation
#   - Flat architecture: 8 sequential coupling layers on flattened 784-dim
#   - Basic conditional flow with FiLM modulation
#   - Feature caching via compute_h flag
#   - No multi-scale, no factoring, no spatial structure

# End of ConditionalRealNVP
# W0.1-RNVP-v2.0
