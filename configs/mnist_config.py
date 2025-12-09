"""
MNIST Configuration for WP0-WP3

Version: WP0.1-Config-v1.1
Last Modified: 2025-12-09
Changelog:
  v1.0 (2025-12-01): Initial configuration
  v1.1 (2025-12-09): Fixed version format, added optimizer field
"""

MNIST_CONFIG = {
    # Dataset parameters
    'dataset': {
        'name': 'MNIST',
        'root': './data/mnist',
        'download': True,
        'image_size': (28, 28),
        'num_channels': 1,
    },
    
    # Forward model (inverse problem)
    'forward_model': {
        'type': 'blur_downsample',
        'blur_kernel_size': 5,
        'blur_sigma': 1.0,
        'downsample_factor': 2,
        'noise_std': 0.1,
    },
    
    # Conditioning network (CNN encoder)
    'conditioner': {
        'type': 'cnn',
        'num_layers': 4,
        'channels': [1, 32, 64, 128],
        'kernel_size': 3,
        'activation': 'relu',
        'output_dim': 64,  # h dimension
    },
    
    # FiLM parameters
    'film': {
        'hidden_dims': [64, 64],
        'activation': 'relu',
    },
    
    # Flow architecture
    'flow': {
        'type': 'realnvp',  # or 'maf'
        'num_blocks': 8,
        'hidden_dims': [256, 256],
        'num_experts': 3,
    },
    
    # Training
    'training': {
        'batch_size': 128,
        'learning_rate': 1e-3,
        'num_epochs': 50,
        'optimizer': 'Adam',
        'weight_decay': 1e-5,
        'grad_clip': 1.0,
    },
}
