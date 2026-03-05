"""
Conditioning Networks
Extract features h = c_η(y) from degraded images

Version: WP0.1-CondNet-v1.2.1
Last Modified: 2026-02-22
Changelog:
  v1.2.1 (2026-02-22): [F1] Fixed false-positive A2 error — h.norm() replaced with h.norm(dim=1).mean()
                       for per-sample threshold check; batch norm [B,h_dim]~85 != per-sample norm ~8.
  v1.2 (2026-02-22): [C1] Added LayerNorm(h_dim) on both paths — fixes h norm ~5400 explosion;
                     [A1] Xavier uniform (gain=0.1) + zero bias on fc layers to prevent early-training spike;
                     [A2] h.norm() monitoring for first 50 forward calls via _fwd_log_count;
                     perf path now global-avg-pools spatial output to flat [B,h_dim] before LayerNorm.
  v1.1 (2025-12-09): Added spec_compliant flag for validation (pooling+flatten)
  v1.0 (2025-12-01): Initial implementation with stride-2 convs
Dependencies: torch>=2.0
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MNISTConditioner(nn.Module):
    """
    CNN-based conditioner for MNIST images
    Extracts spatial features from degraded images for FiLM modulation
    
    Args:
        in_channels: Number of input channels (1 for MNIST)
        h_dim: Output feature dimension
        config: Configuration dictionary (from CONDITIONING_NET_CONFIG)
        spec_compliant: If True, use explicit pooling + flatten (matches WP0 spec)
                       If False, use stride-2 convs (better performance, default)
    """
    
    def __init__(self, in_channels=1, h_dim=64, config=None, spec_compliant=False, debug=False):
        super().__init__()
        self.debug = debug
        
        self.h_dim = h_dim
        self.spec_compliant = spec_compliant
        
        # Parse config or use defaults
        if config is not None:
            channels = config.get('channels', [32, 64, 128, 64])
            kernel_sizes = config.get('kernel_sizes', [3, 3, 3, 3])
            normalization = config.get('normalization', 'batchnorm')
        else:
            # Default configuration
            channels = [32, 64, 128, 64]
            kernel_sizes = [3, 3, 3, 3]
            normalization = 'batchnorm'
        
        if spec_compliant:
            # WP0 spec architecture: Conv→ReLU→Pool→Conv→ReLU→Pool→Flatten→FC
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            
            if normalization == 'batchnorm':
                self.bn1 = nn.BatchNorm2d(32)
                self.bn2 = nn.BatchNorm2d(64)
            else:
                self.bn1 = nn.Identity()
                self.bn2 = nn.Identity()
            
            # After 2x pooling: 28→14→7, so flattened size is 64*7*7
            self.fc = nn.Linear(64 * 7 * 7, h_dim)
            
            # [C1] Output normalization — prevents h norm explosion (~5400 → ~sqrt(h_dim))
            self.output_norm = nn.LayerNorm(h_dim)
            
            # [A1] Xavier init with small gain — prevents norm spike from epoch 1
            nn.init.xavier_uniform_(self.fc.weight, gain=0.1)
            nn.init.zeros_(self.fc.bias)
            
        else:
            # Performance-optimized: stride-2 convs (no explicit pooling)
            layers = []
            in_ch = in_channels
            
            # Use stride-2 for downsampling in first two layers
            strides = [2, 2, 1, 1]
            paddings = [1, 1, 1, 1]
            
            for i, out_ch in enumerate(channels):
                # Convolutional layer
                layers.append(nn.Conv2d(
                    in_ch, 
                    out_ch, 
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i]
                ))
                
                # Activation
                layers.append(nn.ReLU())
                
                # Batch normalization (except for last layer)
                if normalization == 'batchnorm' and i < len(channels) - 1:
                    layers.append(nn.BatchNorm2d(out_ch))
                
                in_ch = out_ch
            
            # Final layer to h_dim
            layers.append(nn.Conv2d(
                channels[-1],
                h_dim,
                kernel_size=kernel_sizes[-1],
                stride=1,
                padding=paddings[-1]
            ))
            layers.append(nn.ReLU())
            
            self.encoder = nn.Sequential(*layers)
            
            # [C1] Global avg pool + FC to produce flat [B, h_dim] + LayerNorm
            # Fixes shape inconsistency (spatial [B,h_dim,H',W'] → flat [B,h_dim])
            # and prevents h norm explosion
            self.final_fc = nn.Linear(h_dim, h_dim)
            self.output_norm = nn.LayerNorm(h_dim)
            
            # [A1] Xavier init with small gain on final_fc
            nn.init.xavier_uniform_(self.final_fc.weight, gain=0.1)
            nn.init.zeros_(self.final_fc.bias)
        
        # [A2] Counter for h.norm() monitoring — first 50 forward calls
        self._fwd_log_count = 0
    
    def forward(self, y):
        """
        Extract features from degraded image
        
        Args:
            y: Degraded image [B, 1, 28, 28]
        
        Returns:
            If spec_compliant=True: h [B, h_dim] (flattened)
            If spec_compliant=False: h [B, h_dim, H', W'] (spatial, default [B, h_dim, 4, 4])
        """
        
        # ADD HERE - Start
        if self.debug:
            logger.debug(f"[CONDITIONER] Input y: shape={y.shape}, norm={torch.norm(y).item():.4f}")

        if self.spec_compliant:
            # WP0 spec path: explicit pooling + flatten
            h = F.relu(self.bn1(self.conv1(y)))
            h = F.max_pool2d(h, 2)  # 28→14
            h = F.relu(self.bn2(self.conv2(h)))
            h = F.max_pool2d(h, 2)  # 14→7
            h = h.flatten(1)  # [B, 64*7*7]
            h = self.fc(h)  # [B, h_dim]
            h = self.output_norm(h)  # [C1] LayerNorm — normalise output scale
            
            # [A2] Monitor h stats for first 50 calls
            if self._fwd_log_count < 50:
                per_sample_norm = h.norm(dim=1).mean().item()  # [F1] per-sample, not batch norm
                logger.info(
                    f"[A2] MNISTConditioner forward spec (call {self._fwd_log_count + 1}/50): "
                    f"per_sample_norm={per_sample_norm:.4f}, std={h.std().item():.4f}, "
                    f"max_abs={h.abs().max().item():.4f}"
                )
                if per_sample_norm > 50.0:
                    logger.error(
                        f"[A2] per-sample h norm={per_sample_norm:.4f} still elevated post-LayerNorm. "
                        f"Check Xavier init was applied to self.fc."
                    )
                self._fwd_log_count += 1
            
            # ADD HERE - Before return (spec path)
            if self.debug:
                logger.debug(f"[CONDITIONER] Output h (spec): shape={h.shape}, norm={torch.norm(h).item():.4f}")
            
            return h
        else:
            # Performance path: stride-2 convs → global avg pool → flat [B, h_dim]
            h = self.encoder(y)
            h = h.mean(dim=[2, 3])      # Global avg pool: [B, h_dim, H', W'] → [B, h_dim]
            h = self.final_fc(h)         # [B, h_dim]
            h = self.output_norm(h)      # [C1] LayerNorm — normalise output scale
            
            # [A2] Monitor h stats for first 50 calls
            if self._fwd_log_count < 50:
                per_sample_norm = h.norm(dim=1).mean().item()  # [F1] per-sample, not batch norm
                logger.info(
                    f"[A2] MNISTConditioner forward perf (call {self._fwd_log_count + 1}/50): "
                    f"per_sample_norm={per_sample_norm:.4f}, std={h.std().item():.4f}, "
                    f"max_abs={h.abs().max().item():.4f}"
                )
                if per_sample_norm > 50.0:
                    logger.error(
                        f"[A2] per-sample h norm={per_sample_norm:.4f} still elevated post-LayerNorm. "
                        f"Check Xavier init was applied to self.final_fc."
                    )
                self._fwd_log_count += 1
            
            # ADD HERE - Before return (performance path)
            if self.debug:
                logger.debug(f"[CONDITIONER] Output h (spatial): shape={h.shape}, norm={torch.norm(h).item():.4f}")
            
            return h
