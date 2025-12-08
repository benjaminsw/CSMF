"""
Super-Resolution Configuration for WP4.1
Version: W4.1-SR-v1.0
"""

SR_CONFIG = {
    # Dataset parameters
    'dataset': {
        'name': 'DIV2K',
        'root': './data/div2k',
        'patch_size': 64,
        'num_channels': 1,  # Grayscale
    },
    
    # Forward model (SR)
    'forward_model': {
        'type': 'blur_downsample',
        'blur_sigma_range': (0.8, 1.8),
        'downsample_factor': 4,  # 2x or 4x
        'noise_std': 0.05,
    },
    
    # Conditioning network (UNet encoder)
    'conditioner': {
        'type': 'unet_encoder',
        'channels': [1, 32, 64, 128],
        'output_dim': 64,
    },
    
    # Physics-based corrections
    'physics': {
        'use_prox': True,
        'prox_steps': 1,
        'prox_lambda': 0.1,
        'use_pcg': True,
        'pcg_iters': 3,
    },
    
    # Training
    'training': {
        'batch_size': 32,
        'learning_rate': 5e-4,
        'num_epochs': 100,
    },
}
