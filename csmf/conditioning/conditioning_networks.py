"""
Conditioning Networks
Extract features h = c_η(y) from degraded images
"""

import torch
import torch.nn as nn


class MNISTConditioner(nn.Module):
    """
    CNN-based conditioner for MNIST images
    Extracts spatial features from degraded images for FiLM modulation
    
    Args:
        in_channels: Number of input channels (1 for MNIST)
        h_dim: Output feature dimension
        config: Configuration dictionary (from CONDITIONING_NET_CONFIG)
    """
    
    def __init__(self, in_channels=1, h_dim=64, config=None):
        super().__init__()
        
        self.h_dim = h_dim
        
        # Parse config or use defaults
        if config is not None:
            channels = config.get('channels', [32, 64, 128, 64])
            kernel_sizes = config.get('kernel_sizes', [3, 3, 3, 3])
            strides = config.get('strides', [2, 2, 2, 1])
            paddings = config.get('paddings', [1, 1, 1, 1])
            normalization = config.get('normalization', 'batchnorm')
        else:
            # Default configuration
            channels = [32, 64, 128, 64]
            kernel_sizes = [3, 3, 3, 3]
            strides = [2, 2, 2, 1]
            paddings = [1, 1, 1, 1]
            normalization = 'batchnorm'
        
        # Build CNN encoder
        layers = []
        in_ch = in_channels
        
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
            stride=strides[-1],
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
            h: Extracted features [B, h_dim, H', W']
               With default config: [B, h_dim, 4, 4]
        """
        h = self.encoder(y)
        return h
