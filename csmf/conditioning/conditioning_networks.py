"""
Conditioning Networks
Extract features h = c_η(y) from degraded images

Version: WP0.1-CondNet-v1.1
Last Modified: 2025-12-09
Changelog:
  v1.0 (2025-12-01): Initial implementation with stride-2 convs
  v1.1 (2025-12-09): Added spec_compliant flag for validation (pooling+flatten)
Dependencies: torch>=2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    
    def __init__(self, in_channels=1, h_dim=64, config=None, spec_compliant=False):
        super().__init__()
        
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
    
    def forward(self, y):
        """
        Extract features from degraded image
        
        Args:
            y: Degraded image [B, 1, 28, 28]
        
        Returns:
            If spec_compliant=True: h [B, h_dim] (flattened)
            If spec_compliant=False: h [B, h_dim, H', W'] (spatial, default [B, h_dim, 4, 4])
        """
        if self.spec_compliant:
            # WP0 spec path: explicit pooling + flatten
            h = F.relu(self.bn1(self.conv1(y)))
            h = F.max_pool2d(h, 2)  # 28→14
            h = F.relu(self.bn2(self.conv2(h)))
            h = F.max_pool2d(h, 2)  # 14→7
            h = h.flatten(1)  # [B, 64*7*7]
            h = self.fc(h)  # [B, h_dim]
            return h
        else:
            # Performance path: stride-2 convs (spatial output)
            h = self.encoder(y)
            return h
