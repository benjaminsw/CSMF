"""
MNIST Inverse Problem Dataset
Load MNIST → apply degradation → return (x_clean, y_degraded)

Version: WP0.1-MNISTInv-v1.0
Last Modified: 2025-12-09
Changelog:
  v1.0 (2025-12-09): Initial implementation with blur+downsample+noise degradation
                     Added normalization options [0,1] or [-1,1]
Dependencies: torch>=2.0, torchvision>=0.10
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MNISTInverseDataset(Dataset):
    """
    MNIST dataset with forward degradation for inverse problems
    
    Degradation pipeline: x_clean → blur → downsample → + noise → y_degraded
    
    Args:
        root: Path to MNIST data directory
        train: If True, use training set; else test set
        download: If True, download MNIST if not present
        blur_kernel_size: Size of Gaussian blur kernel (default: 5)
        blur_sigma: Standard deviation of Gaussian blur (default: 1.0)
        downsample_factor: Downsampling factor (default: 2)
        noise_std: Standard deviation of additive Gaussian noise (default: 0.1)
        normalize: Normalization range - '[0,1]' or '[-1,1]' (default: '[0,1]')
        transform: Additional transforms to apply (optional)
    """
    
    def __init__(
        self,
        root='./data/mnist',
        train=True,
        download=True,
        blur_kernel_size=5,
        blur_sigma=1.0,
        downsample_factor=2,
        noise_std=0.1,
        normalize='[0,1]',
        transform=None
    ):
        super().__init__()
        
        # Load base MNIST dataset
        try:
            self.mnist = torchvision.datasets.MNIST(
                root=root,
                train=train,
                download=download,
                transform=torchvision.transforms.ToTensor()  # Convert to [0,1] tensor
            )
        except Exception as e:
            logger.error(f"Failed to load MNIST dataset: {e}")
            raise
        
        # Store degradation parameters
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.downsample_factor = downsample_factor
        self.noise_std = noise_std
        self.normalize = normalize
        self.transform = transform
        
        # Validate parameters
        if blur_kernel_size % 2 == 0:
            logger.warning(f"blur_kernel_size={blur_kernel_size} is even, using {blur_kernel_size+1} instead")
            self.blur_kernel_size = blur_kernel_size + 1
        
        if normalize not in ['[0,1]', '[-1,1]']:
            logger.error(f"Invalid normalize option: {normalize}. Must be '[0,1]' or '[-1,1]'")
            raise ValueError(f"normalize must be '[0,1]' or '[-1,1]', got {normalize}")
        
        # Create Gaussian blur kernel
        self.blur_kernel = self._create_gaussian_kernel(
            self.blur_kernel_size, 
            self.blur_sigma
        )
        
        logger.info(
            f"MNISTInverseDataset initialized: "
            f"train={train}, blur_k={self.blur_kernel_size}, "
            f"sigma={blur_sigma}, down={downsample_factor}, "
            f"noise_std={noise_std}, normalize={normalize}"
        )
    
    def _create_gaussian_kernel(self, kernel_size, sigma):
        """
        Create 2D Gaussian kernel for blurring
        
        Args:
            kernel_size: Size of kernel (odd number)
            sigma: Standard deviation
        
        Returns:
            kernel: (1, 1, kernel_size, kernel_size) tensor
        """
        # Create 1D Gaussian
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords -= kernel_size // 2
        
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        
        # Create 2D kernel via outer product
        kernel_2d = g[:, None] * g[None, :]
        kernel_2d /= kernel_2d.sum()
        
        # Shape for conv2d: (out_channels, in_channels, H, W)
        kernel = kernel_2d.view(1, 1, kernel_size, kernel_size)
        
        return kernel
    
    def _normalize_image(self, img):
        """
        Normalize image to specified range
        
        Args:
            img: Image tensor in [0, 1] range
        
        Returns:
            Normalized image tensor
        """
        if self.normalize == '[-1,1]':
            return 2 * img - 1  # [0,1] → [-1,1]
        else:
            return img  # Keep [0,1]
    
    def _denormalize_image(self, img):
        """
        Denormalize image back to [0, 1] for degradation operations
        
        Args:
            img: Image tensor in normalized range
        
        Returns:
            Image tensor in [0, 1] range
        """
        if self.normalize == '[-1,1]':
            return (img + 1) / 2  # [-1,1] → [0,1]
        else:
            return img
    
    def _degrade(self, x_clean):
        """
        Apply degradation: blur → downsample → add noise
        
        Args:
            x_clean: Clean image [1, 28, 28] in [0, 1] range
        
        Returns:
            y_degraded: Degraded image [1, H', W'] where H'=W'=28//downsample_factor
        """
        try:
            # Ensure batch dimension
            if x_clean.dim() == 3:
                x_clean = x_clean.unsqueeze(0)  # [1, 1, 28, 28]
            
            # Step 1: Gaussian blur
            padding = self.blur_kernel_size // 2
            x_blurred = F.conv2d(
                x_clean, 
                self.blur_kernel, 
                padding=padding
            )
            
            # Step 2: Downsample
            target_size = (
                x_clean.shape[2] // self.downsample_factor,
                x_clean.shape[3] // self.downsample_factor
            )
            y_downsampled = F.interpolate(
                x_blurred,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
            
            # Step 3: Add white Gaussian noise
            noise = torch.randn_like(y_downsampled) * self.noise_std
            y_degraded = y_downsampled + noise
            
            # Clip to valid range [0, 1] before normalization
            y_degraded = torch.clamp(y_degraded, 0.0, 1.0)
            
            # Remove batch dimension
            y_degraded = y_degraded.squeeze(0)  # [1, H', W']
            
            return y_degraded
            
        except Exception as e:
            logger.error(f"Degradation failed: {e}")
            raise
    
    def __len__(self):
        """Return dataset size"""
        return len(self.mnist)
    
    def __getitem__(self, idx):
        """
        Get item at index
        
        Args:
            idx: Index
        
        Returns:
            x_clean: Clean image [1, 28, 28] in normalized range
            y_degraded: Degraded image [1, H', W'] in normalized range
        """
        # Load clean image and label (we ignore label for inverse problems)
        x_clean, _ = self.mnist[idx]  # x_clean is [1, 28, 28] in [0, 1]
        
        # Apply degradation (operates in [0, 1] range)
        y_degraded = self._degrade(x_clean)
        
        # Normalize both images to specified range
        x_clean = self._normalize_image(x_clean)
        y_degraded = self._normalize_image(y_degraded)
        
        # Apply optional additional transforms
        if self.transform is not None:
            x_clean = self.transform(x_clean)
            y_degraded = self.transform(y_degraded)
        
        return x_clean, y_degraded


def create_mnist_inverse_dataloaders(
    root='./data/mnist',
    batch_size=128,
    blur_kernel_size=5,
    blur_sigma=1.0,
    downsample_factor=2,
    noise_std=0.1,
    normalize='[0,1]',
    num_workers=4
):
    """
    Convenience function to create train and test dataloaders
    
    Args:
        root: Path to MNIST data
        batch_size: Batch size for dataloaders
        blur_kernel_size: Gaussian blur kernel size
        blur_sigma: Gaussian blur sigma
        downsample_factor: Downsampling factor
        noise_std: Noise standard deviation
        normalize: Normalization range '[0,1]' or '[-1,1]'
        num_workers: Number of dataloader workers
    
    Returns:
        train_loader: Training dataloader
        test_loader: Test dataloader
    """
    # Create datasets
    train_dataset = MNISTInverseDataset(
        root=root,
        train=True,
        download=True,
        blur_kernel_size=blur_kernel_size,
        blur_sigma=blur_sigma,
        downsample_factor=downsample_factor,
        noise_std=noise_std,
        normalize=normalize
    )
    
    test_dataset = MNISTInverseDataset(
        root=root,
        train=False,
        download=True,
        blur_kernel_size=blur_kernel_size,
        blur_sigma=blur_sigma,
        downsample_factor=downsample_factor,
        noise_std=noise_std,
        normalize=normalize
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(
        f"Created dataloaders: train={len(train_dataset)}, "
        f"test={len(test_dataset)}, batch_size={batch_size}"
    )
    
    return train_loader, test_loader


if __name__ == '__main__':
    """Test the dataset"""
    print("Testing MNISTInverseDataset...")
    
    # Test with [0,1] normalization
    print("\n=== Testing with [0,1] normalization ===")
    dataset_01 = MNISTInverseDataset(
        root='./data/mnist',
        train=True,
        download=True,
        blur_kernel_size=5,
        blur_sigma=1.0,
        downsample_factor=2,
        noise_std=0.1,
        normalize='[0,1]'
    )
    
    x_clean, y_degraded = dataset_01[0]
    print(f"Clean image shape: {x_clean.shape}, range: [{x_clean.min():.3f}, {x_clean.max():.3f}]")
    print(f"Degraded image shape: {y_degraded.shape}, range: [{y_degraded.min():.3f}, {y_degraded.max():.3f}]")
    
    # Test with [-1,1] normalization
    print("\n=== Testing with [-1,1] normalization ===")
    dataset_11 = MNISTInverseDataset(
        root='./data/mnist',
        train=True,
        download=True,
        blur_kernel_size=5,
        blur_sigma=1.0,
        downsample_factor=2,
        noise_std=0.1,
        normalize='[-1,1]'
    )
    
    x_clean, y_degraded = dataset_11[0]
    print(f"Clean image shape: {x_clean.shape}, range: [{x_clean.min():.3f}, {x_clean.max():.3f}]")
    print(f"Degraded image shape: {y_degraded.shape}, range: [{y_degraded.min():.3f}, {y_degraded.max():.3f}]")
    
    # Test dataloaders
    print("\n=== Testing dataloaders ===")
    train_loader, test_loader = create_mnist_inverse_dataloaders(
        batch_size=32,
        normalize='[0,1]'
    )
    
    x_batch, y_batch = next(iter(train_loader))
    print(f"Batch clean shape: {x_batch.shape}")
    print(f"Batch degraded shape: {y_batch.shape}")
    
    print("\n✓ All tests passed!")
