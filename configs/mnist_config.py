"""
MNIST Configuration for WP0-WP3

Version: WP0.1-Config-v2.4
Last Modified: 2026-04-12
Changelog:
  v2.4 (2026-04-12): Set ACTIVE_EXPERTS to ['realnvp','nice','nsf','csf'] — 4-expert
                     config matching logdet_diag_run2 checkpoint; maf and gcsf removed
                     to align config_hash() with saved Stage A checkpoint and allow
                     Stage B/C to load without hash mismatch and expert count mismatch
                     (Try B): LAMBDA_GAP 0.05→0.02 (gentler penalty to avoid
                     destabilising weaker experts early); MARGIN 1.0→0.5 (z-score
                     units — penalise gaps > 0.5 std, tighter than margin=1.0);
                     penalises winner-too-far-ahead while keeping z-normalisation intact
  v2.2 (2026-04-09): [PATCH-SA-ZN] Replace raw NLL competition with z-score normalised
                     score: ALPHA_Z=0.3 (z_cons weight), TAU_A updated to 1.0 (z-score
                     gaps are O(1-3) so raw-NLL tau of 2000 no longer applies); MARGIN
                     updated to 1.0 (z-score units); ALPHA_Z added to config_hash();
                     EPS_FLOOR removed from competition (softmax guarantees nonzero weights)
  v2.1 (2026-04-09): [PATCH-SA-SCW] Add Stage A soft competition hyperparameters —
                     TAU_A=2000.0 (competition temperature, scaled to MNIST NLL gaps ~4000);
                     LAMBDA_GAP=0.05 (gap penalty weight, 0 for first 3 epochs warmup);
                     MARGIN=200.0 (gap penalty margin); EPS_FLOOR=0.05 (weight floor);
                     TAU_A added to config_hash() for cross-stage drift detection
  v2.0 (2026-04-02): Added 'gcsf' to ACTIVE_EXPERTS — ConditionalGlowCSF (COND-GCSF-v1.0)
                     enabled (6-expert config); added 'gcsf': LR entry to EXPERT_LR;
                     train_csmf.py registry updated separately; config_hash() will
                     invalidate existing checkpoints; version bumped to v2.0 as expert
                     set now includes all implemented flow architectures
  v1.9 (2026-04-02): Added 'csf' to ACTIVE_EXPERTS — ConditionalCSF (COND-CSF-v1.0)
                     enabled alongside realnvp, nice, nsf, maf (5-expert config);
                     added 'csf': LR entry to EXPERT_LR; train_csmf.py registry
                     updated separately; config_hash() will invalidate existing checkpoints
  v1.8 (2026-04-02): Added 'maf' to ACTIVE_EXPERTS — ConditionalMAF re-enabled alongside
                     realnvp, nice, nsf (4-expert config); added 'maf': LR entry to
                     EXPERT_LR dict; train_csmf.py registry already had maf, no changes
                     needed there; config_hash() will invalidate existing checkpoints
  v1.7 (2026-03-31): Added set_seed(seed) — sets Python/NumPy/PyTorch CPU+CUDA seeds
                     and cuDNN deterministic flags; added make_worker_init_fn(seed) —
                     returns a worker_init_fn closure for deterministic DataLoader workers
                     using a per-worker offset; both replace scattered seed calls in
                     train_csmf.py; SEED constant is the single source of truth (2026)
  v1.6 (2026-03-29): BUG FIX — config_hash() fields were all commented out, hashing
                     empty dict; re-enabled all fields for drift detection;
                     EXPERT_LR updated: replaced 'maf' with 'nice' to match ACTIVE_EXPERTS
  v1.5 (2026-03-26): ACTIVE_EXPERTS updated to ['realnvp','nice','nsf'] — replaces maf;
                     LAMBDA_TRANS set to 0.0 — SW2 sampling too slow for Stage B
                     (half-day per epoch); re-enable for Stage C with n_sw2_samples=1;
                     LAMBDA_CONS kept at 0.05; LAMBDA_CAL remains 0.0
  v1.4 (2026-03-26): Added LAMBDA_CONS, LAMBDA_TRANS, LAMBDA_CAL to config_hash() —
                     loss weights now tracked for cross-stage drift detection; changing
                     these values will invalidate existing checkpoints; set LAMBDA_CONS=0.05
                     and LAMBDA_TRANS=0.01 as active defaults (were 0.0)
  v1.3 (2026-02-25): Added PREPROCESSED_DIR for precomputed dataset path
                     Added BLUR_SIGMA constant — single source of truth for preprocess_mnist.py
                     Added config_hash() utility — MD5 of key params for cross-stage drift detection
                     Added EXPERT_LR dict for per-expert learning rates in Stage A
  v1.2 (2026-02-22): Added flat constants required by train_csmf.py and
                     experiment scripts; nested MNIST_CONFIG preserved for
                     backward compatibility
  v1.1 (2025-12-09): Fixed version format, added optimizer field
  v1.0 (2025-12-01): Initial configuration
"""

import logging
logger = logging.getLogger(__name__)

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
DATA_ROOT        = MNIST_CONFIG['dataset']['root']          # './data/mnist'
PREPROCESSED_DIR = './data/preprocessed'                    # v1.3: precomputed .pt files
CKPT_DIR         = './checkpoints'
RESULTS_DIR      = './results'

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
BLUR_SIGMA        = MNIST_CONFIG['forward_model']['blur_sigma']          # 1.0 — v1.3
NOISE_SIGMA       = MNIST_CONFIG['forward_model']['noise_std']           # 0.1

# Hybrid loss weights
LAMBDA_CONS  = 0.05
LAMBDA_TRANS = 0.001   # v1.5: disabled for Stage B speed — re-enable for Stage C
LAMBDA_CAL   = 0.001

# Active experts for CSMF (subset of registry keys)
ACTIVE_EXPERTS = ['realnvp', 'nice', 'nsf', 'csf']  # v2.4: 4-expert config matching logdet_diag_run2 checkpoint

# Training protocol
VAL_SPLIT          = 0.1    # fraction of train set used for validation
PATIENCE           = 10 #5      # early stopping patience (epochs)
BLOCKS_TO_UNFREEZE = 0 #1      # Stage C: last N blocks unfrozen per expert
TAU_START          = 1.1    # Stage C gate temperature start
TAU_END            = 1.0    # Stage C gate temperature end

# [PATCH-SA-SCW] v2.1 / [PATCH-SA-ZN] v2.2: Stage A soft competition hyperparameters
# v2.2: competition now uses z-score normalised score (z_nll + ALPHA_Z * z_cons).
# Z-score gaps are O(1-3), so TAU_A=1.0 and MARGIN=1.0 (z-score units).
TAU_A       = 1.0    # [v2.2] competition tau — z-score scale (was 2000.0 for raw NLL)
LAMBDA_GAP  = 0.02   # [v2.3] gap penalty weight — reduced from 0.05 (Try B)
MARGIN      = 0.5    # [v2.3] gap penalty margin — z-score units, reduced from 1.0 (Try B)
EPS_FLOOR   = 0.05   # retained for backward compat; not used in z-score competition
ALPHA_Z     = 0.3    # [v2.2] weight of z_cons in competition score: z_nll + ALPHA_Z*z_cons

# v1.3: Per-expert learning rates for Stage A (defaults to LR if key missing)
EXPERT_LR = {
    'realnvp': LR,   # 1e-3
    'nice':    LR,   # 1e-3
    'nsf':     LR,   # 1e-3
    'maf':     LR,   # 1e-3 — v1.8
    'csf':     LR,   # 1e-3 — v1.9
    'gcsf':    LR,   # 1e-3 — v2.0
}

# =============================================================================
# v1.3: config_hash() — MD5 of key training params
# Used by save_checkpoint() and load_stage_checkpoint() to detect config drift
# =============================================================================
def config_hash() -> str:
    import hashlib
    import json
    cfg = {
        'DOWNSAMPLE_FACTOR': DOWNSAMPLE_FACTOR,
        'BLUR_KERNEL':       BLUR_KERNEL,
        'BLUR_SIGMA':        BLUR_SIGMA,
        'NOISE_SIGMA':       NOISE_SIGMA,
        'HIDDEN_DIM':        HIDDEN_DIM,
        'LATENT_DIM':        LATENT_DIM,
        'ACTIVE_EXPERTS':    sorted(ACTIVE_EXPERTS),
        'LAMBDA_CONS':  LAMBDA_CONS,
        'LAMBDA_TRANS': LAMBDA_TRANS,
        'LAMBDA_CAL':   LAMBDA_CAL,
        'TAU_A':        TAU_A,      # [PATCH-SA-SCW] v2.1
        'ALPHA_Z':      ALPHA_Z,    # [PATCH-SA-ZN] v2.2
    }
    try:
        return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    except Exception as e:
        logger.error(f"MNIST-CFG | config_hash() failed: {e}")
        raise


# =============================================================================
# v1.7: set_seed() — single call to fix all random sources
# Call before model init and before DataLoader construction
# =============================================================================
def set_seed(seed: int = SEED) -> None:
    """
    Fix all random seeds for reproducibility.

    Sets: Python random, NumPy, PyTorch CPU, PyTorch CUDA (all devices),
    cuDNN deterministic mode, cuDNN benchmark disabled.

    Args:
        seed: Random seed integer. Defaults to SEED (2026).
    """
    import random as _random
    import numpy as _np
    import torch
    try:
        _random.seed(seed)
        _np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"MNIST-CFG | set_seed({seed}) — all random sources fixed")
    except Exception as e:
        logger.error(f"MNIST-CFG | set_seed({seed}) failed: {e}")
        raise


# =============================================================================
# v1.7: make_worker_init_fn() — deterministic DataLoader workers
# Pass the returned fn to DataLoader(worker_init_fn=...)
# =============================================================================
def make_worker_init_fn(seed: int = SEED):
    """
    Return a worker_init_fn for torch DataLoader that seeds each worker
    deterministically using seed + worker_id offset.

    Usage:
        g = torch.Generator()
        g.manual_seed(seed)
        DataLoader(dataset, worker_init_fn=make_worker_init_fn(seed), generator=g)

    Args:
        seed: Base seed. Each worker receives seed + worker_id.

    Returns:
        Callable[[int], None] — suitable for DataLoader worker_init_fn.
    """
    def _worker_init_fn(worker_id: int) -> None:
        import random as _random
        import numpy as _np
        import torch
        worker_seed = seed + worker_id
        try:
            _random.seed(worker_seed)
            _np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)
        except Exception as e:
            # Use print — logging may not be available in worker processes
            print(f"[worker {worker_id}] make_worker_init_fn seed error: {e}")
            raise
    return _worker_init_fn
