"""
MNIST Configuration for WP0-WP3

Version: WP0.1-Config-v1.2
Last Modified: 2026-02-22
Changelog:
  v1.2 (2026-02-22): Added flat constants required by train_csmf.py and
                     experiment scripts; nested MNIST_CONFIG preserved for
                     backward compatibility
  v1.1 (2025-12-09): Fixed version format, added optimizer field
  v1.0 (2025-12-01): Initial configuration
"""

# =============================================================================
# Nested config (preserved for backward compatibility)
# =============================================================================
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
        'output_dim': 64,   # h dimension
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

# =============================================================================
# Flat constants — derived from MNIST_CONFIG + additional fields
# Required by train_csmf.py and all experiment scripts
# =============================================================================

# Paths
DATA_ROOT   = MNIST_CONFIG['dataset']['root']       # './data/mnist'
CKPT_DIR    = './checkpoints'
RESULTS_DIR = './results'

# Training
BATCH_SIZE  = MNIST_CONFIG['training']['batch_size']        # 128
EPOCHS      = MNIST_CONFIG['training']['num_epochs']        # 50
LR          = MNIST_CONFIG['training']['learning_rate']     # 1e-3
SEED        = 2026

# Architecture
HIDDEN_DIM  = MNIST_CONFIG['conditioner']['output_dim']     # 64
NUM_LAYERS  = MNIST_CONFIG['flow']['num_blocks']            # 8
LATENT_DIM  = 784   # 28×28 flattened MNIST

# Forward model / degradation
DOWNSAMPLE_FACTOR = MNIST_CONFIG['forward_model']['downsample_factor']   # 2
BLUR_KERNEL       = MNIST_CONFIG['forward_model']['blur_kernel_size']    # 5
NOISE_SIGMA       = MNIST_CONFIG['forward_model']['noise_std']           # 0.1

# Hybrid loss weights
LAMBDA_CONS  = 0.1
LAMBDA_TRANS = 0.01
LAMBDA_CAL   = 0.0

# Active experts for CSMF (subset of registry keys)
ACTIVE_EXPERTS = ['realnvp', 'maf', 'nsf']

# Training protocol
VAL_SPLIT         = 0.1    # fraction of train set used for validation
PATIENCE          = 5      # early stopping patience (epochs)
BLOCKS_TO_UNFREEZE = 1     # Stage C: last N blocks unfrozen per expert
TAU_START         = 1.1    # Stage C gate temperature start
TAU_END           = 1.0    # Stage C gate temperature end
