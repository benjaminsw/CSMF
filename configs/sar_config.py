"""
SAR Despeckling Configuration for WP4.2
Version: W4.2-SAR-v1.0
"""

SAR_CONFIG = {
    # Dataset parameters
    'dataset': {
        'name': 'SAR_simulated',
        'root': './data/sar',
        'patch_size': 128,
        'num_looks': 4,
    },
    
    # Forward model (multiplicative noise in log domain)
    'forward_model': {
        'type': 'gamma_speckle',
        'num_looks': 4,
        'work_in_log_domain': True,
    },
    
    # Conditioning network
    'conditioner': {
        'type': 'cnn',
        'channels': [1, 32, 64, 128],
        'output_dim': 64,
    },
    
    # Physics (log-domain consistency)
    'physics': {
        'use_log_consistency': True,
        'consistency_weight': 0.1,
    },
    
    # Training
    'training': {
        'batch_size': 16,
        'learning_rate': 1e-4,
        'num_epochs': 150,
    },
}
