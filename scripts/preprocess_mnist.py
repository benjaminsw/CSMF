"""
MNIST Inverse Problem — Preprocessing & Dataset
Merged from mnist_inverse.py (MNIST-INV v1.2) and preprocess_mnist.py (PREP-MNIST v1.0).
mnist_inverse.py is now deleted. All imports should point to this file.

Sections:
  1. MNISTInverseDataset          — on-the-fly degradation (used by preprocessing only)
  2. PrecomputedMNISTDataset      — loads .pt files (used by train_csmf.py)
  3. create_precomputed_dataloaders() — DataLoader factory for training
  4. preprocess_mnist()           — one-time runner, saves .pt + metadata.json
  5. CLI                          — python scripts/preprocess_mnist.py [--force]

Version: WP0.5-PrepMNIST-v2.2
Abbr: PREP-MNIST
Last Modified: 2026-03-31
Changelog:
  v2.2 (2026-03-31): create_precomputed_dataloaders() accepts worker_init_fn and
                     generator kwargs — passed through to DataLoader for deterministic
                     worker seeding; defaults to None (backward compatible); generator
                     applied to train loader only (shuffle=True); val/test loaders
                     receive worker_init_fn only (no shuffle, generator not needed)
  v2.1 (2026-02-25): Import degradation params from mnist_config.py — single source of truth
                     Local DEFAULT_* constants kept as fallback only if mnist_config import fails
                     metadata.json now includes config_hash field for cross-stage drift detection
                     _METADATA_VALIDATE_KEYS extended with config_hash
  v2.0 (2026-02-24): Merged mnist_inverse.py into this file — single source of truth
                     MNISTInverseDataset kept for degradation during preprocessing only
                     PrecomputedMNISTDataset + create_precomputed_dataloaders() moved here
                     mnist_inverse.py deleted — train_csmf.py now imports from here
                     create_mnist_inverse_dataloaders() removed (on-the-fly no longer used)
  v1.0 (2026-02-24): Initial preprocessing script — SHA-256 checksums, --force flag,
                     metadata.json, seed=2026 fixed
Dependencies: torch>=2.0, torchvision>=0.10
"""

# WP0.5-PrepMNIST-v2.1 | PREP-MNIST

import os
import json
import hashlib
import logging
import argparse
from datetime import datetime, timezone

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger('PREP-MNIST')

# ---------------------------------------------------------------------------
# Defaults — v2.1: imported from mnist_config; local values as fallback only
# ---------------------------------------------------------------------------
DEFAULT_DATA_ROOT = './data/mnist'
DEFAULT_OUT_DIR   = './data/preprocessed'
DEFAULT_NORMALIZE = '[0,1]'

try:
    from configs.mnist_config import (
        BLUR_KERNEL       as DEFAULT_BLUR_KERNEL_SIZE,
        BLUR_SIGMA        as DEFAULT_BLUR_SIGMA,
        DOWNSAMPLE_FACTOR as DEFAULT_DOWNSAMPLE,
        NOISE_SIGMA       as DEFAULT_NOISE_STD,
        VAL_SPLIT         as DEFAULT_VAL_SPLIT,
        SEED              as DEFAULT_SEED,
        config_hash,
    )
    logger.info("PREP-MNIST: loaded defaults from mnist_config.py")
except ImportError:
    logger.warning("PREP-MNIST: mnist_config not found — using local defaults")
    DEFAULT_BLUR_KERNEL_SIZE = 5
    DEFAULT_BLUR_SIGMA       = 1.0
    DEFAULT_DOWNSAMPLE       = 2
    DEFAULT_NOISE_STD        = 0.1
    DEFAULT_VAL_SPLIT        = 0.2
    DEFAULT_SEED             = 2026   # Fixed — do not change without re-running preprocessing
    config_hash = lambda: "no-config-hash"

# Keys validated by PrecomputedMNISTDataset against metadata.json
_METADATA_VALIDATE_KEYS = [
    'blur_kernel_size', 'blur_sigma', 'downsample_factor',
    'noise_std', 'normalize', 'val_split', 'seed', 'config_hash'  # v2.1: added config_hash
]


# ===========================================================================
# Section 1: MNISTInverseDataset — on-the-fly degradation
# Used internally by preprocess_mnist() only. Not imported by train_csmf.py.
# ===========================================================================

class MNISTInverseDataset(Dataset):
    """
    Wraps MNIST and applies degradation on-the-fly.
    Used only during preprocessing — not at training time.

    Degradation pipeline: x_clean → Gaussian blur → bilinear downsample → AWGN → clamp [0,1]
    """

    def __init__(
        self,
        root=DEFAULT_DATA_ROOT,
        train=True,
        download=True,
        blur_kernel_size=DEFAULT_BLUR_KERNEL_SIZE,
        blur_sigma=DEFAULT_BLUR_SIGMA,
        downsample_factor=DEFAULT_DOWNSAMPLE,
        noise_std=DEFAULT_NOISE_STD,
        normalize=DEFAULT_NORMALIZE,
    ):
        super().__init__()

        try:
            self.mnist = torchvision.datasets.MNIST(
                root=root, train=train, download=download,
                transform=torchvision.transforms.ToTensor()
            )
        except Exception as e:
            logger.error(f"Failed to load MNIST: {e}")
            raise

        if blur_kernel_size % 2 == 0:
            logger.warning(f"blur_kernel_size={blur_kernel_size} is even, using {blur_kernel_size + 1}")
            blur_kernel_size += 1

        if normalize not in ('[0,1]', '[-1,1]'):
            logger.error(f"Invalid normalize='{normalize}'")
            raise ValueError(f"normalize must be '[0,1]' or '[-1,1]', got {normalize}")

        self.blur_kernel_size  = blur_kernel_size
        self.blur_sigma        = blur_sigma
        self.downsample_factor = downsample_factor
        self.noise_std         = noise_std
        self.normalize         = normalize
        self.blur_kernel       = self._create_gaussian_kernel(blur_kernel_size, blur_sigma)

        logger.info(
            f"MNISTInverseDataset: train={train}, blur_k={blur_kernel_size}, "
            f"sigma={blur_sigma}, down={downsample_factor}, "
            f"noise_std={noise_std}, normalize={normalize}"
        )

    def _create_gaussian_kernel(self, kernel_size, sigma):
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        k2d = g[:, None] * g[None, :]
        k2d /= k2d.sum()
        return k2d.view(1, 1, kernel_size, kernel_size)

    def _normalize_image(self, img):
        if self.normalize == '[-1,1]':
            return 2 * img - 1
        return img

    def _degrade(self, x_clean):
        try:
            x = x_clean.unsqueeze(0)                                              # [1,1,28,28]
            x = F.conv2d(x, self.blur_kernel, padding=self.blur_kernel_size // 2) # blur
            h = x.shape[2] // self.downsample_factor
            w = x.shape[3] // self.downsample_factor
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)  # downsample
            x = x + torch.randn_like(x) * self.noise_std                          # AWGN
            x = torch.clamp(x, 0.0, 1.0)                                          # clamp
            return x.squeeze(0)
        except Exception as e:
            logger.error(f"_degrade() failed: {e}")
            raise

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        x_clean, _ = self.mnist[idx]
        y_degraded  = self._degrade(x_clean)
        return self._normalize_image(x_clean), self._normalize_image(y_degraded)


# ===========================================================================
# Section 2: PrecomputedMNISTDataset — loads .pt files for training
# Imported by train_csmf.py via create_precomputed_dataloaders()
# ===========================================================================

class PrecomputedMNISTDataset(Dataset):
    """
    Loads precomputed (x_clean, y_degraded) pairs from a .pt file.
    Produced by preprocess_mnist() — no degradation at training time.

    Args:
        pt_path:       Path to .pt file (e.g. data/preprocessed/mnist_train.pt)
        metadata_path: Path to metadata.json
        config_params: Dict of current config values to validate against metadata.
                       Validated keys: blur_kernel_size, blur_sigma, downsample_factor,
                       noise_std, normalize, val_split, seed.
                       Pass None to skip (not recommended for training runs).
    """

    def __init__(self, pt_path, metadata_path, config_params=None):
        super().__init__()

        if not os.path.exists(metadata_path):
            logger.error(f"metadata.json not found: {metadata_path}")
            raise FileNotFoundError(f"metadata.json not found: {metadata_path}")

        try:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read metadata.json: {e}")
            raise

        if config_params is not None:
            self._validate_metadata(config_params)

        if 'checksum_sha256' in self.metadata:
            expected = self.metadata['checksum_sha256'].get(os.path.basename(pt_path))
            self._verify_checksum(pt_path, expected)

        if not os.path.exists(pt_path):
            logger.error(f".pt file not found: {pt_path}")
            raise FileNotFoundError(
                f".pt file not found: {pt_path} — run: python scripts/preprocess_mnist.py"
            )

        try:
            data = torch.load(pt_path, weights_only=True)
        except Exception as e:
            logger.error(f"Failed to load {pt_path}: {e}")
            raise

        self.x_clean    = data['x_clean']
        self.y_degraded = data['y_degraded']

        if self.x_clean.shape[0] != self.y_degraded.shape[0]:
            logger.error(
                f"Shape mismatch: x_clean N={self.x_clean.shape[0]} "
                f"!= y_degraded N={self.y_degraded.shape[0]}"
            )
            raise ValueError("x_clean and y_degraded sample counts do not match")

        logger.info(
            f"PrecomputedMNISTDataset: {os.path.basename(pt_path)} | "
            f"N={len(self.x_clean)} | "
            f"x:{tuple(self.x_clean.shape[1:])} y:{tuple(self.y_degraded.shape[1:])}"
        )

    def _validate_metadata(self, config_params):
        mismatches = []
        for key in _METADATA_VALIDATE_KEYS:
            if key not in config_params:
                continue
            meta_val = self.metadata.get(key)
            cfg_val  = config_params[key]
            if meta_val != cfg_val:
                mismatches.append(f"  {key}: metadata={meta_val!r} vs config={cfg_val!r}")

        if mismatches:
            msg = "Metadata mismatch — re-run preprocess_mnist.py:\n" + "\n".join(mismatches)
            logger.error(msg)
            raise ValueError(msg)

        logger.info("Metadata validation passed.")

    def _verify_checksum(self, pt_path, expected_sha256):
        if expected_sha256 is None:
            logger.warning(f"No checksum for {os.path.basename(pt_path)} — skipping.")
            return

        sha256 = hashlib.sha256()
        try:
            with open(pt_path, 'rb') as f:
                for chunk in iter(lambda: f.read(65536), b''):
                    sha256.update(chunk)
            actual = sha256.hexdigest()
        except Exception as e:
            logger.error(f"Checksum failed for {pt_path}: {e}")
            raise

        if actual != expected_sha256:
            logger.error(f"Checksum FAIL {pt_path}: expected={expected_sha256} actual={actual}")
            raise RuntimeError(
                f"Corrupt .pt file: {pt_path} — re-run: python scripts/preprocess_mnist.py --force"
            )

        logger.info(f"Checksum OK: {os.path.basename(pt_path)}")

    def __len__(self):
        return self.x_clean.shape[0]

    def __getitem__(self, idx):
        return self.x_clean[idx], self.y_degraded[idx]


# ===========================================================================
# Section 3: create_precomputed_dataloaders — called by train_csmf.py
# ===========================================================================

def create_precomputed_dataloaders(
    preprocessed_dir=DEFAULT_OUT_DIR,
    batch_size=128,
    num_workers=4,
    config_params=None,
    worker_init_fn=None,
    generator=None,
):
    """
    Build train/val/test DataLoaders from precomputed .pt files.
    Validates metadata.json against config_params before any data is loaded.

    Args:
        preprocessed_dir: Directory with mnist_train.pt, mnist_val.pt,
                          mnist_test.pt, metadata.json
        batch_size:        Batch size for all loaders
        num_workers:       DataLoader worker count
        config_params:     Dict with keys: blur_kernel_size, blur_sigma,
                           downsample_factor, noise_std, normalize, val_split, seed
        worker_init_fn:    Optional callable(worker_id) for deterministic worker seeds
                           (v2.2 — use make_worker_init_fn() from mnist_config)
        generator:         Optional torch.Generator for deterministic shuffling in
                           train loader (v2.2)

    Returns:
        train_loader, val_loader, test_loader
    """
    metadata_path = os.path.join(preprocessed_dir, 'metadata.json')

    splits = {
        'train': (os.path.join(preprocessed_dir, 'mnist_train.pt'), True),
        'val':   (os.path.join(preprocessed_dir, 'mnist_val.pt'),   False),
        'test':  (os.path.join(preprocessed_dir, 'mnist_test.pt'),  False),
    }

    loaders = {}
    for split_name, (pt_path, shuffle) in splits.items():
        try:
            dataset = PrecomputedMNISTDataset(
                pt_path=pt_path,
                metadata_path=metadata_path,
                config_params=config_params
            )
        except Exception as e:
            logger.error(f"Failed to create {split_name} dataset: {e}")
            raise

        loaders[split_name] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            generator=generator if shuffle else None,  # v2.2: generator only for train (shuffle=True)
        )

    logger.info(
        f"Loaders ready — train:{len(loaders['train'].dataset)}, "
        f"val:{len(loaders['val'].dataset)}, "
        f"test:{len(loaders['test'].dataset)}, "
        f"batch_size={batch_size}"
    )

    return loaders['train'], loaders['val'], loaders['test']


# ===========================================================================
# Section 4: preprocess_mnist — one-time runner
# ===========================================================================

def _sha256_of_file(path):
    sha256 = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        logger.error(f"Checksum failed for {path}: {e}")
        raise


def _dataset_to_tensors(dataset, split_label):
    x_list, y_list = [], []
    n = len(dataset)
    logger.info(f"  Processing {split_label}: {n} samples...")

    for i, (x, y) in enumerate(dataset):
        x_list.append(x)
        y_list.append(y)
        if (i + 1) % 10000 == 0:
            logger.info(f"    {split_label}: {i+1}/{n}")

    try:
        x_clean    = torch.stack(x_list)
        y_degraded = torch.stack(y_list)
    except Exception as e:
        logger.error(f"torch.stack failed for {split_label}: {e}")
        raise

    if torch.isnan(x_clean).any() or torch.isinf(x_clean).any():
        logger.error(f"{split_label} x_clean has NaN/Inf")
        raise ValueError(f"{split_label} x_clean has NaN/Inf")

    if torch.isnan(y_degraded).any() or torch.isinf(y_degraded).any():
        logger.error(f"{split_label} y_degraded has NaN/Inf")
        raise ValueError(f"{split_label} y_degraded has NaN/Inf")

    logger.info(
        f"  {split_label}: x{tuple(x_clean.shape)} "
        f"[{x_clean.min():.3f},{x_clean.max():.3f}] | "
        f"y{tuple(y_degraded.shape)} "
        f"[{y_degraded.min():.3f},{y_degraded.max():.3f}]"
    )
    return x_clean, y_degraded


def preprocess_mnist(
    data_root=DEFAULT_DATA_ROOT,
    out_dir=DEFAULT_OUT_DIR,
    blur_kernel_size=DEFAULT_BLUR_KERNEL_SIZE,
    blur_sigma=DEFAULT_BLUR_SIGMA,
    downsample_factor=DEFAULT_DOWNSAMPLE,
    noise_std=DEFAULT_NOISE_STD,
    normalize=DEFAULT_NORMALIZE,
    val_split=DEFAULT_VAL_SPLIT,
    seed=DEFAULT_SEED,
    force=False
):
    """
    Precompute degraded MNIST pairs and save as .pt files.

    Outputs:
        {out_dir}/mnist_train.pt   — {'x_clean': Tensor, 'y_degraded': Tensor}
        {out_dir}/mnist_val.pt
        {out_dir}/mnist_test.pt
        {out_dir}/metadata.json    — all params + split sizes + SHA-256 checksums
    """
    if not (0.0 < val_split < 1.0):
        logger.error(f"val_split={val_split} out of range (0,1)")
        raise ValueError(f"val_split must be in (0,1), got {val_split}")

    if normalize not in ('[0,1]', '[-1,1]'):
        logger.error(f"normalize='{normalize}' invalid")
        raise ValueError(f"normalize must be '[0,1]' or '[-1,1]'")

    os.makedirs(out_dir, exist_ok=True)

    out_paths = {
        'train': os.path.join(out_dir, 'mnist_train.pt'),
        'val':   os.path.join(out_dir, 'mnist_val.pt'),
        'test':  os.path.join(out_dir, 'mnist_test.pt'),
    }
    metadata_path = os.path.join(out_dir, 'metadata.json')

    all_exist = all(os.path.exists(p) for p in out_paths.values()) and os.path.exists(metadata_path)
    if all_exist and not force:
        logger.info(f"Files exist in '{out_dir}'. Use --force to overwrite.")
        return

    torch.manual_seed(seed)
    logger.info(f"Seed fixed to {seed}")

    logger.info("Building MNISTInverseDataset (train)...")
    train_full = MNISTInverseDataset(
        root=data_root, train=True, download=True,
        blur_kernel_size=blur_kernel_size, blur_sigma=blur_sigma,
        downsample_factor=downsample_factor, noise_std=noise_std,
        normalize=normalize
    )

    n_total = len(train_full)
    n_val   = int(n_total * val_split)
    n_train = n_total - n_val

    try:
        train_subset, val_subset = torch.utils.data.random_split(
            train_full, [n_train, n_val],
            generator=torch.Generator().manual_seed(seed)
        )
    except Exception as e:
        logger.error(f"random_split failed: {e}")
        raise

    logger.info(f"Split: train={n_train}, val={n_val}")

    logger.info("Building MNISTInverseDataset (test)...")
    test_dataset = MNISTInverseDataset(
        root=data_root, train=False, download=True,
        blur_kernel_size=blur_kernel_size, blur_sigma=blur_sigma,
        downsample_factor=downsample_factor, noise_std=noise_std,
        normalize=normalize
    )

    checksums   = {}
    split_sizes = {}

    for split_name, dataset in [('train', train_subset), ('val', val_subset), ('test', test_dataset)]:
        x_clean, y_degraded = _dataset_to_tensors(dataset, split_name)

        try:
            torch.save({'x_clean': x_clean, 'y_degraded': y_degraded}, out_paths[split_name])
            logger.info(f"Saved: {out_paths[split_name]}")
        except Exception as e:
            logger.error(f"Failed to save {out_paths[split_name]}: {e}")
            raise

        checksums[f'mnist_{split_name}.pt'] = _sha256_of_file(out_paths[split_name])
        split_sizes[split_name] = len(dataset)

    metadata = {
        'version':           'WP0.5-PrepMNIST-v2.1',
        'created_at':        datetime.now(timezone.utc).isoformat(),
        'blur_kernel_size':  blur_kernel_size,
        'blur_sigma':        blur_sigma,
        'downsample_factor': downsample_factor,
        'noise_std':         noise_std,
        'normalize':         normalize,
        'val_split':         val_split,
        'seed':              seed,
        'split_sizes':       split_sizes,
        'x_clean_shape':     [1, 28, 28],
        'y_degraded_shape':  [1, 28 // downsample_factor, 28 // downsample_factor],
        'checksum_sha256':   checksums,
        'config_hash':       config_hash(),   # v2.1: cross-stage drift detection
    }

    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved: {metadata_path}")
    except Exception as e:
        logger.error(f"Failed to save metadata.json: {e}")
        raise

    logger.info(
        f"\n{'='*50}\n"
        f"Preprocessing complete.\n"
        f"  out_dir : {out_dir}\n"
        f"  train   : {split_sizes['train']} samples\n"
        f"  val     : {split_sizes['val']} samples\n"
        f"  test    : {split_sizes['test']} samples\n"
        f"  metadata: {metadata_path}\n"
        f"{'='*50}"
    )


# ===========================================================================
# Section 5: CLI
# ===========================================================================

def _parse_args():
    parser = argparse.ArgumentParser(description='WP0.5-PrepMNIST-v2.1 | PREP-MNIST')
    parser.add_argument('--data-root',        default=DEFAULT_DATA_ROOT,        type=str)
    parser.add_argument('--out-dir',          default=DEFAULT_OUT_DIR,          type=str)
    parser.add_argument('--blur-kernel-size', default=DEFAULT_BLUR_KERNEL_SIZE, type=int)
    parser.add_argument('--blur-sigma',       default=DEFAULT_BLUR_SIGMA,       type=float)
    parser.add_argument('--downsample',       default=DEFAULT_DOWNSAMPLE,       type=int)
    parser.add_argument('--noise-std',        default=DEFAULT_NOISE_STD,        type=float)
    parser.add_argument('--normalize',        default=DEFAULT_NORMALIZE,        type=str,
                        choices=['[0,1]', '[-1,1]'])
    parser.add_argument('--val-split',        default=DEFAULT_VAL_SPLIT,        type=float)
    parser.add_argument('--seed',             default=DEFAULT_SEED,             type=int)
    parser.add_argument('--force',            action='store_true',
                        help='Overwrite existing .pt files')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    preprocess_mnist(
        data_root=args.data_root,
        out_dir=args.out_dir,
        blur_kernel_size=args.blur_kernel_size,
        blur_sigma=args.blur_sigma,
        downsample_factor=args.downsample,
        noise_std=args.noise_std,
        normalize=args.normalize,
        val_split=args.val_split,
        seed=args.seed,
        force=args.force
    )
