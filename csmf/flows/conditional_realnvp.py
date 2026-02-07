"""
ConditionalRealNVP for MNIST Inverse Problems - Multi-Scale Architecture

Version: WP0.3-CondRNVP-v2.1.0
Abbr: COND-RNVP
Last Modified: 2025-02-04
Changelog:
  v2.1.0 (2025-02-04): Added comprehensive debugging for invertibility and log-det issues
  v2.0.1 (2025-02-03): Fixed reshape bug in ScaleBlock inverse
  v2.0 (2025-10-25): Multi-scale with squeeze/unsqueeze and variable factoring
  v1.0 (2025-10-25): Initial flat architecture
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional
import logging

try:
    from configs.mnist_config import MNIST_CONFIG
    from csmf.conditioning.conditioning_networks import MNISTConditioner
    from csmf.flows.coupling_layers import ConditionalAffineCoupling
except ImportError as e:
    logging.error(f"Failed to import dependencies: {e}")
    raise

logger = logging.getLogger(__name__)


class ScaleBlock(nn.Module):
    """Multi-scale block: N coupling layers with optional squeeze operation."""
    
    def __init__(
        self,
        n_layers: int,
        channels: int,
        spatial_dim: int,
        h_dim: int,
        hidden_dims: List[int],
        apply_squeeze: bool = False,
        debug: bool = False
    ):
        super().__init__()
        
        self.n_layers = n_layers
        self.channels = channels
        self.spatial_dim = spatial_dim
        self.h_dim = h_dim
        self.apply_squeeze = apply_squeeze
        self.debug = debug
        
        # Create coupling layers
        dim = channels * spatial_dim * spatial_dim
        self.coupling_layers = nn.ModuleList()
        
        for i in range(n_layers):
            split_dim = dim // 2
            layer = ConditionalAffineCoupling(
                dim=dim,
                split_dim=split_dim,
                h_dim=h_dim,
                hidden_dims=hidden_dims,
                use_batch_norm=False,  # Disable BN for debugging
                s_max=10.0,
                debug=self.debug
            )
            self.coupling_layers.append(layer)
            if self.debug:
                logger.debug(f"ScaleBlock: layer {i+1}/{n_layers}, dim={dim}, split={split_dim}")
    
    def forward(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        reverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward through coupling layers with optional squeeze."""
        
        if self.debug:
            logger.debug("=" * 70)
            logger.debug(f"[SCALE DEBUG] Block forward (reverse={reverse})")
            logger.debug(f"  Input: shape={z.shape}, norm={z.norm().item():.6f}")
        
        B, C, H, W = z.shape
        log_det = torch.zeros(B, device=z.device)
        
        # Store input for invertibility check
        if self.debug and not reverse:
            z_input_original = z.clone()
        
        # Flatten for coupling layers
        z_flat = z.reshape(B, -1)
        
        if not reverse:
            # ========== FORWARD: coupling → squeeze ==========
            for i, coupling in enumerate(self.coupling_layers):
                if self.debug:
                    z_before = z_flat.clone()
                
                z_flat, ld = coupling.forward(z_flat, h, reverse=False)
                log_det = log_det + ld
                
                if self.debug:
                    logger.debug(f"  Coupling {i+1}/{self.n_layers}:")
                    logger.debug(f"    Output norm: {z_flat.norm().item():.6f}")
                    logger.debug(f"    Log-det: mean={ld.mean().item():.4f}, sum={log_det.mean().item():.4f}")
                    
                    # Per-layer invertibility check
                    z_test_inv, _ = coupling.forward(z_flat, h, reverse=True)
                    inv_err = (z_before - z_test_inv).abs().max().item()
                    if inv_err > 1e-4:
                        logger.error(f"    ❌ Layer {i+1} invertibility FAILED: error={inv_err:.2e}")
                    else:
                        logger.debug(f"    ✓ Layer {i+1} invertible: error={inv_err:.2e}")
            
            # Reshape back to spatial
            z_out = z_flat.reshape(B, C, H, W)
            
            # Apply squeeze
            if self.apply_squeeze:
                if self.debug:
                    #logger.debug(f"  Applying squeeze: {z_out.shape} → ", end="")
                    shape_before = z_out.shape
                
                z_out = self._squeeze(z_out)
                
                if self.debug:
                    #logger.debug(f"{z_out.shape}")
                    logger.debug(f"  Applying squeeze: {shape_before} → {z_out.shape}")
                    
                    # Squeeze invertibility check
                    z_unsqueeze_test = self._unsqueeze(z_out)
                    squeeze_err = (z_flat.reshape(B, C, H, W) - z_unsqueeze_test).abs().max().item()
                    if squeeze_err > 1e-6:
                        logger.error(f"  ❌ Squeeze NOT invertible: error={squeeze_err:.2e}")
                    else:
                        logger.debug(f"  ✓ Squeeze invertible: error={squeeze_err:.2e}")
        
        else:
            # ========== INVERSE: unsqueeze → coupling (reversed) ==========
            z_out = z_flat.reshape(B, C, H, W)
            
            if self.apply_squeeze:
                if self.debug:
                    #logger.debug(f"  Applying unsqueeze: {z_out.shape} → ", end="")
                    shape_before = z_out.shape
                
                z_out = self._unsqueeze(z_out)
                
                if self.debug:
                    #logger.debug(f"{z_out.shape}")
                    logger.debug(f"  Applying unsqueeze: {shape_before} → {z_out.shape}")
                    
            
            # Re-flatten with correct shape
            z_flat = z_out.reshape(B, -1)
            
            if self.debug:
                logger.debug(f"  Flattened for coupling: shape={z_flat.shape}")
            
            # Apply couplings in reverse order
            for i, coupling in enumerate(reversed(self.coupling_layers)):
                z_flat, ld = coupling.forward(z_flat, h, reverse=True)
                log_det = log_det + ld
                
                if self.debug:
                    logger.debug(f"  Coupling {self.n_layers-i}/{self.n_layers} (reversed):")
                    logger.debug(f"    Output norm: {z_flat.norm().item():.6f}")
                    logger.debug(f"    Log-det contribution: {ld.mean().item():.4f}")
            
            # Reshape to match output
            z_out = z_flat.reshape(B, z_out.shape[1], z_out.shape[2], z_out.shape[3])
        
        if self.debug:
            logger.debug(f"  Block output: shape={z_out.shape}, total log_det={log_det.mean().item():.4f}")
            logger.debug("=" * 70)
        
        return z_out, log_det
    
    def _squeeze(self, z: torch.Tensor) -> torch.Tensor:
        """Squeeze: [B, C, H, W] → [B, 4C, H/2, W/2]"""
        B, C, H, W = z.shape
        z = z.reshape(B, C, H // 2, 2, W // 2, 2)
        z = z.permute(0, 1, 3, 5, 2, 4)
        z = z.reshape(B, C * 4, H // 2, W // 2)
        return z
    
    def _unsqueeze(self, z: torch.Tensor) -> torch.Tensor:
        """Unsqueeze: [B, 4C, H, W] → [B, C, 2H, 2W]"""
        B, C4, H, W = z.shape
        C = C4 // 4
        z = z.reshape(B, C, 2, 2, H, W)
        z = z.permute(0, 1, 4, 2, 5, 3)
        z = z.reshape(B, C, H * 2, W * 2)
        return z


class ConditionalRealNVP(nn.Module):
    """Conditional RealNVP flow with multi-scale architecture."""
    
    def __init__(
        self,
        h_dim: int = 64,
        hidden_dims: List[int] = None,
        config: dict = None,
        debug: bool = False
    ):
        """Initialize ConditionalRealNVP with debug mode."""
        super().__init__()
        
        if h_dim <= 0:
            raise ValueError(f"h_dim must be positive, got {h_dim}")
        
        if hidden_dims is None:
            hidden_dims = [256, 256]
        if config is None:
            config = MNIST_CONFIG
        
        self.h_dim = h_dim
        self.hidden_dims = hidden_dims
        self.config = config
        self._cached_h = None
        self.debug = debug
        
        logger.info(f"Initializing ConditionalRealNVP v2.1.0: h_dim={h_dim}, debug={debug}")
        
        # Conditioning network
        self.conditioner = MNISTConditioner(h_dim=h_dim, debug=debug)
        
        # Multi-scale blocks
        self.scale1 = ScaleBlock(3, 1, 28, h_dim, hidden_dims, apply_squeeze=True, debug=debug)
        self.scale2 = ScaleBlock(3, 2, 14, h_dim, hidden_dims, apply_squeeze=True, debug=debug)
        self.scale3 = ScaleBlock(3, 4, 7, h_dim, hidden_dims, apply_squeeze=False, debug=debug)
        
        self.register_buffer('log_2pi', torch.log(torch.tensor(2.0 * torch.pi)))
        
        logger.info("ConditionalRealNVP v2.1.0 initialization complete")
    
    def _factor_out(self, z: torch.Tensor, factor_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Factor out channels."""
        B, C, H, W = z.shape
        n_factor = int(C * factor_ratio)
        n_keep = C - n_factor
        
        z_kept = z[:, :n_keep, :, :]
        z_factored = z[:, n_keep:, :, :]
        
        if self.debug:
            logger.debug(f"[FACTOR] Split [{C},{H},{W}] → kept [{n_keep},{H},{W}] + factored [{n_factor},{H},{W}]")
        
        return z_kept, z_factored
    
    def _unfactor(self, z_kept: torch.Tensor, z_factored: torch.Tensor) -> torch.Tensor:
        """Unfactor: concatenate channels."""
        z = torch.cat([z_kept, z_factored], dim=1)
        
        if self.debug:
            logger.debug(f"[UNFACTOR] Combined {z_kept.shape} + {z_factored.shape} → {z.shape}")
        
        return z
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        compute_h: bool = True
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Forward transform with debugging."""
        
        # Validation
        if x.ndim != 4:
            raise ValueError(f"x must be 4D, got {x.shape}")
        if x.shape != (x.shape[0], 1, 28, 28):
            raise ValueError(f"x must be [B,1,28,28], got {x.shape}")
        
        if self.debug:
            logger.debug("=" * 70)
            logger.debug("[REALNVP FORWARD]")
            logger.debug(f"  Input x: shape={x.shape}, norm={x.norm().item():.6f}")
            x_original = x.clone()  # For final invertibility check
        
        # Conditioning
        if compute_h:
            h = self.conditioner(y)
            self._cached_h = h
            if self.debug:
                logger.debug(f"  Conditioning h: shape={h.shape}, norm={h.norm().item():.6f}")
        else:
            if self._cached_h is None:
                raise RuntimeError("No cached h")
            h = self._cached_h
        
        B = x.shape[0]
        log_det_total = torch.zeros(B, device=x.device)
        z_factored_list = []
        
        try:
            # Scale 1
            if self.debug:
                logger.debug("\n[SCALE 1] Processing...")
            
            z = x.clone()
            z, log_det = self.scale1.forward(z, h, reverse=False)
            log_det_total = log_det_total + log_det
            
            if self.debug:
                logger.debug(f"[SCALE 1] Output: shape={z.shape}, log_det={log_det.mean().item():.4f}")
            
            # Factor
            z, z_factor1 = self._factor_out(z, factor_ratio=0.5)
            z_factored_list.append(z_factor1)
            
            # Scale 2
            if self.debug:
                logger.debug("\n[SCALE 2] Processing...")
            
            z, log_det = self.scale2.forward(z, h, reverse=False)
            log_det_total = log_det_total + log_det
            
            if self.debug:
                logger.debug(f"[SCALE 2] Output: shape={z.shape}, log_det={log_det.mean().item():.4f}")
            
            # Factor
            z, z_factor2 = self._factor_out(z, factor_ratio=0.5)
            z_factored_list.append(z_factor2)
            
            # Scale 3
            if self.debug:
                logger.debug("\n[SCALE 3] Processing...")
            
            z, log_det = self.scale3.forward(z, h, reverse=False)
            log_det_total = log_det_total + log_det
            
            if self.debug:
                logger.debug(f"[SCALE 3] Output: shape={z.shape}, log_det={log_det.mean().item():.4f}")
            
            z_final = z
            
            # Stability check
            if torch.any(torch.isnan(z_final)) or torch.any(torch.isinf(z_final)):
                logger.warning("NaN/Inf in z_final")
                z_final = torch.clamp(z_final, min=-1e6, max=1e6)
            
        except Exception as e:
            logger.error(f"Forward failed: {e}")
            raise RuntimeError(f"Forward failed: {e}")
        
        # Compute log-probabilities
        log_pz_list = []
        
        # Final latent
        z_final_flat = z_final.reshape(B, -1)
        z_squared = torch.clamp(torch.sum(z_final_flat ** 2, dim=1), max=400.0)
        dim_final = z_final_flat.shape[1]
        log_pz_final = -0.5 * (dim_final * self.log_2pi + z_squared)
        log_pz_list.append(log_pz_final)
        
        if self.debug:
            logger.debug(f"\n[LOG-PROB] Final latent: dim={dim_final}, log_pz={log_pz_final.mean().item():.4f}")
        
        # Factored variables
        for i, z_fact in enumerate(z_factored_list):
            z_fact_flat = z_fact.reshape(B, -1)
            z_squared = torch.clamp(torch.sum(z_fact_flat ** 2, dim=1), max=400.0)
            dim_fact = z_fact_flat.shape[1]
            log_pz_fact = -0.5 * (dim_fact * self.log_2pi + z_squared)
            log_pz_list.append(log_pz_fact)
            
            if self.debug:
                logger.debug(f"[LOG-PROB] Factored {i+1}: dim={dim_fact}, log_pz={log_pz_fact.mean().item():.4f}")
        
        # Total
        log_pz_total = sum(log_pz_list)
        log_prob_total = log_pz_total + log_det_total
        
        if self.debug:
            logger.debug(f"\n[TOTAL]")
            logger.debug(f"  log_pz: {log_pz_total.mean().item():.4f}")
            logger.debug(f"  log_det: {log_det_total.mean().item():.4f}")
            logger.debug(f"  log_prob: {log_prob_total.mean().item():.4f}")
            
            # End-to-end invertibility check
            logger.debug("\n[INVERTIBILITY CHECK]")
            x_recon = self.inverse(z_final, z_factored_list, y)
            inv_err = (x_original - x_recon).abs().max().item()
            inv_mean = (x_original - x_recon).abs().mean().item()
            logger.debug(f"  Max error: {inv_err:.2e} (threshold: 1e-4)")
            logger.debug(f"  Mean error: {inv_mean:.2e}")
            if inv_err > 1e-4:
                logger.error(f"  ❌ END-TO-END INVERTIBILITY FAILED!")
            else:
                logger.debug(f"  ✓ End-to-end invertible")
        
        # NaN check
        nan_mask = torch.isnan(log_prob_total) | torch.isinf(log_prob_total)
        if torch.any(nan_mask):
            logger.warning(f"NaN/Inf in {nan_mask.sum()} samples")
            log_prob_total[nan_mask] = -100.0
        
        if self.debug:
            logger.debug("=" * 70)
        
        return z_final, z_factored_list, log_det_total, log_prob_total
    
    def inverse(
        self,
        z: torch.Tensor,
        z_factored_list: List[torch.Tensor],
        y: torch.Tensor
    ) -> torch.Tensor:
        """Inverse transform with debugging."""
        
        if self.debug:
            logger.debug("=" * 70)
            logger.debug("[REALNVP INVERSE]")
            logger.debug(f"  Input z: shape={z.shape}")
        
        # Validation
        if len(z_factored_list) != 2:
            raise ValueError(f"Expected 2 factored variables, got {len(z_factored_list)}")
        
        h = self.conditioner(y)
        
        try:
            x = z.clone()
            
            # Scale 3 inverse
            if self.debug:
                logger.debug("\n[SCALE 3 INVERSE]")
            x, _ = self.scale3.forward(x, h, reverse=True)
            if self.debug:
                logger.debug(f"  Output: {x.shape}")
            
            # Unfactor scale 2
            z_factor2 = z_factored_list[1]
            x = self._unfactor(x, z_factor2)
            
            # Scale 2 inverse
            if self.debug:
                logger.debug("\n[SCALE 2 INVERSE]")
            x, _ = self.scale2.forward(x, h, reverse=True)
            if self.debug:
                logger.debug(f"  Output: {x.shape}")
            
            # Unfactor scale 1
            z_factor1 = z_factored_list[0]
            x = self._unfactor(x, z_factor1)
            
            # Scale 1 inverse
            if self.debug:
                logger.debug("\n[SCALE 1 INVERSE]")
            x, _ = self.scale1.forward(x, h, reverse=True)
            if self.debug:
                logger.debug(f"  Output: {x.shape}")
            
            # Stability
            if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
                logger.warning("NaN/Inf in reconstruction")
                x = torch.clamp(x, min=-1e6, max=1e6)
            
        except Exception as e:
            logger.error(f"Inverse failed: {e}")
            raise RuntimeError(f"Inverse failed: {e}")
        
        assert x.shape == (z.shape[0], 1, 28, 28), f"Shape mismatch: {x.shape}"
        
        if self.debug:
            logger.debug("=" * 70)
        
        return x
    
    def sample(self, n_samples: int, y: torch.Tensor) -> torch.Tensor:
        """Generate samples."""
        if n_samples != y.shape[0]:
            raise ValueError(f"n_samples ({n_samples}) != y.shape[0] ({y.shape[0]})")
        
        device = y.device
        z_final = torch.randn(n_samples, 4, 7, 7, device=device)
        z_factor2 = torch.randn(n_samples, 4, 7, 7, device=device)
        z_factor1 = torch.randn(n_samples, 2, 14, 14, device=device)
        z_factored_list = [z_factor1, z_factor2]
        
        x = self.inverse(z_final, z_factored_list, y)
        return x
    
    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute log-probability."""
        _, _, _, log_prob_total = self.forward(x, y, compute_h=True)
        return log_prob_total