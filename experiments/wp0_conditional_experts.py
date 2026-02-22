# =============================================================================
# Version: WP3.3-ExpWP0-v1.1 | Abbr: EXP-WP0
# Description: WP0 experiment — train & evaluate individual conditional experts
# Changelog:
#   v1.1 (2025-02-21): Added learning curve plots per expert, qualitative
#                      sample grid (LR / per-expert reconstructions),
#                      default seed changed to 2026
#   v1.0 (2025-02-21): Initial — NLL, invertibility, conditioning quality, runtime
# Dependencies: COND-RNVP, COND-MAF, COND-NICE, COND-NSF, MNIST-INV, MNIST-CFG
# =============================================================================

import os
import csv
import time
import random
import logging
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display required
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from configs.mnist_config import (
    DATA_ROOT, RESULTS_DIR,
    BATCH_SIZE, NUM_LAYERS, HIDDEN_DIM, LATENT_DIM,
    DOWNSAMPLE_FACTOR, BLUR_KERNEL, NOISE_SIGMA,
    VAL_SPLIT, ACTIVE_EXPERTS,
)
from data.mnist_inverse import MNISTInverseDataset
from csmf.conditioning.conditioning_networks import MNISTConditioner
from csmf.flows.conditional_realnvp import ConditionalRealNVP
from csmf.flows.conditional_maf import ConditionalMAF
from csmf.flows.conditional_nice import ConditionalNICE
from csmf.flows.conditional_nsf import ConditionalNSF

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_SEED    = 2026
EPOCHS_PER_EXPERT = 30          # standalone training epochs (NLL only)
LR_EXPERT         = 1e-3
RESULTS_SUBDIR    = "wp0"
N_QUAL_SAMPLES    = 8           # number of images in qualitative grid

EXPERT_REGISTRY = {
    "realnvp": ConditionalRealNVP,
    "maf":     ConditionalMAF,
    "nice":    ConditionalNICE,
    "nsf":     ConditionalNSF,
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(results_dir: str) -> logging.Logger:
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "wp0_conditional_experts.log")
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("EXP-WP0")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WP0: Conditional expert comparison")
    p.add_argument("--experts", nargs="+", choices=list(EXPERT_REGISTRY.keys()),
                   default=None, help="Experts to evaluate (default: ACTIVE_EXPERTS from config)")
    p.add_argument("--epochs",  type=int,   default=EPOCHS_PER_EXPERT,
                   help=f"Training epochs per expert (default: {EPOCHS_PER_EXPERT})")
    p.add_argument("--lr",      type=float, default=LR_EXPERT,
                   help=f"Learning rate (default: {LR_EXPERT})")
    p.add_argument("--seed",    type=int,   default=DEFAULT_SEED,
                   help=f"Random seed (default: {DEFAULT_SEED})")
    p.add_argument("--results-dir", type=str, default=None,
                   help="Override results directory")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_dataloaders(
    data_root: str,
    batch_size: int,
    val_split: float,
    logger: logging.Logger,
) -> tuple:
    """Build train / val / test DataLoaders."""
    train_full = MNISTInverseDataset(
        root=data_root, train=True,
        blur_k=BLUR_KERNEL, down=DOWNSAMPLE_FACTOR, sigma=NOISE_SIGMA,
    )
    test_ds = MNISTInverseDataset(
        root=data_root, train=False,
        blur_k=BLUR_KERNEL, down=DOWNSAMPLE_FACTOR, sigma=NOISE_SIGMA,
    )
    n_val   = int(len(train_full) * val_split)
    n_train = len(train_full) - n_val
    train_ds, val_ds = random_split(
        train_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(DEFAULT_SEED),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    logger.info(
        f"Data | train={n_train} | val={n_val} | test={len(test_ds)} | "
        f"batch={batch_size}"
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Single-expert training (NLL only — no gate, no hybrid loss)
# ---------------------------------------------------------------------------

def train_expert(
    name: str,
    expert: nn.Module,
    conditioner: nn.Module,
    base_dist: torch.distributions.Distribution,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    logger: logging.Logger,
) -> list:
    """
    Train one expert with NLL loss only.

    Returns:
        train_nll_curve: list of per-epoch mean train NLL (length = epochs)
    """
    optimizer = torch.optim.Adam(
        list(expert.parameters()) + list(conditioner.parameters()), lr=lr
    )
    train_nll_curve = []

    for epoch in range(epochs):
        expert.train()
        conditioner.train()
        total_nll = 0.0
        n_batches = 0

        for x_clean, y_deg in train_loader:
            optimizer.zero_grad()

            h          = conditioner(y_deg)
            z, log_det = expert.forward(x_clean, h)

            if torch.any(torch.isnan(log_det)):
                logger.error(
                    f"[{name}] Epoch {epoch} | NaN log_det — skipping batch"
                )
                continue

            log_p_z = base_dist.log_prob(z).sum(dim=1)
            nll     = -(log_p_z + log_det).mean()

            if torch.isnan(nll):
                logger.error(
                    f"[{name}] Epoch {epoch} | NaN NLL — skipping batch"
                )
                continue

            nll.backward()
            torch.nn.utils.clip_grad_norm_(
                list(expert.parameters()) + list(conditioner.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            total_nll += nll.item()
            n_batches += 1

        if n_batches == 0:
            logger.error(f"[{name}] Epoch {epoch} | All batches skipped")
            train_nll_curve.append(float("nan"))
            continue

        avg_nll = total_nll / n_batches
        train_nll_curve.append(avg_nll)
        logger.info(f"[{name}] Epoch {epoch+1}/{epochs} | TrainNLL={avg_nll:.4f}")

    return train_nll_curve


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_nll(
    name: str,
    expert: nn.Module,
    conditioner: nn.Module,
    base_dist: torch.distributions.Distribution,
    test_loader: DataLoader,
    logger: logging.Logger,
) -> float:
    """[M1] Mean NLL on test set."""
    expert.eval()
    conditioner.eval()
    total = 0.0
    n     = 0

    for x_clean, y_deg in test_loader:
        h          = conditioner(y_deg)
        z, log_det = expert.forward(x_clean, h)
        log_p_z    = base_dist.log_prob(z).sum(dim=1)
        nll        = -(log_p_z + log_det).mean()

        if torch.isnan(nll):
            logger.warning(f"[{name}] eval_nll: NaN in test batch — skipping")
            continue

        total += nll.item()
        n     += 1

    if n == 0:
        logger.error(f"[{name}] eval_nll: all batches NaN")
        return float("nan")

    result = total / n
    logger.info(f"[{name}] NLL = {result:.4f}")
    return result


@torch.no_grad()
def eval_invertibility(
    name: str,
    expert: nn.Module,
    conditioner: nn.Module,
    test_loader: DataLoader,
    logger: logging.Logger,
) -> float:
    """[M2] Invertibility error: mean ||f^{-1}(f(x, h), h) - x||^2."""
    if not callable(getattr(expert, "inverse", None)):
        logger.error(f"[{name}] .inverse() not implemented")
        raise NotImplementedError(
            f"Expert '{name}' does not implement .inverse(). "
            f"Cannot compute invertibility error."
        )

    expert.eval()
    conditioner.eval()
    total = 0.0
    n     = 0

    for x_clean, y_deg in test_loader:
        h          = conditioner(y_deg)
        z, _       = expert.forward(x_clean, h)
        x_recon    = expert.inverse(z, h)
        err        = torch.mean((x_recon - x_clean) ** 2)

        if torch.isnan(err):
            logger.warning(f"[{name}] eval_invertibility: NaN in batch — skipping")
            continue

        total += err.item()
        n     += 1

    if n == 0:
        logger.error(f"[{name}] eval_invertibility: all batches NaN")
        return float("nan")

    result = total / n
    logger.info(f"[{name}] InvertibilityError = {result:.6f}")
    return result


@torch.no_grad()
def eval_conditioning_quality(
    name: str,
    expert: nn.Module,
    conditioner: nn.Module,
    base_dist: torch.distributions.Distribution,
    test_loader: DataLoader,
    logger: logging.Logger,
) -> float:
    """
    [M3] Conditioning quality: sample z ~ N(0,I), x_hat = inverse(z, h).
    Measures ||x_hat - x_clean||^2 — lower = conditioning signal is effective.
    """
    expert.eval()
    conditioner.eval()
    total = 0.0
    n     = 0

    for x_clean, y_deg in test_loader:
        h     = conditioner(y_deg)
        z     = base_dist.sample((x_clean.shape[0], expert.dim)).to(y_deg.device)
        x_hat = expert.inverse(z, h)
        err   = torch.mean((x_hat - x_clean) ** 2)

        if torch.isnan(err):
            logger.warning(
                f"[{name}] eval_conditioning_quality: NaN in batch — skipping"
            )
            continue

        total += err.item()
        n     += 1

    if n == 0:
        logger.error(f"[{name}] eval_conditioning_quality: all batches NaN")
        return float("nan")

    result = total / n
    if result > 1.0:
        logger.warning(
            f"[{name}] Conditioning quality high ({result:.4f} > 1.0) — "
            f"check FiLM init or conditioner capacity"
        )
    logger.info(f"[{name}] ConditioningQuality = {result:.4f}")
    return result


@torch.no_grad()
def eval_runtime(
    name: str,
    expert: nn.Module,
    conditioner: nn.Module,
    base_dist: torch.distributions.Distribution,
    test_loader: DataLoader,
    logger: logging.Logger,
    n_warmup: int = 5,
) -> float:
    """
    [M4] Mean inference runtime (ms/batch) for expert.inverse().
    Runs n_warmup batches before timing to avoid cold-start bias.
    """
    expert.eval()
    conditioner.eval()
    times   = []
    batches = 0

    for x_clean, y_deg in test_loader:
        h = conditioner(y_deg)
        z = base_dist.sample((x_clean.shape[0], expert.dim)).to(y_deg.device)

        if batches < n_warmup:
            expert.inverse(z, h)  # warmup — not timed
            batches += 1
            continue

        t0 = time.perf_counter()
        expert.inverse(z, h)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms
        batches += 1

    if len(times) == 0:
        logger.error(f"[{name}] eval_runtime: no batches timed")
        return float("nan")

    result = float(np.mean(times))
    logger.info(f"[{name}] Runtime = {result:.2f} ms/batch")
    return result


# ---------------------------------------------------------------------------
# Plotting — learning curves
# ---------------------------------------------------------------------------

def plot_learning_curves(
    curves: dict,
    results_dir: str,
    logger: logging.Logger,
) -> None:
    """
    Plot NLL vs epoch for each expert on the same axes.

    Args:
        curves: {expert_name: [nll_epoch_0, nll_epoch_1, ...]}
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "D"]

    for i, (name, nll_curve) in enumerate(curves.items()):
        clean = [v for v in nll_curve if not (isinstance(v, float) and v != v)]
        epochs = list(range(1, len(nll_curve) + 1))
        nll_plot = [
            v if not (isinstance(v, float) and v != v) else None
            for v in nll_curve
        ]
        ax.plot(
            epochs, nll_curve,
            label=name,
            marker=markers[i % len(markers)],
            markevery=max(1, len(epochs) // 10),
            linewidth=1.8,
        )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Train NLL", fontsize=12)
    ax.set_title("WP0: Learning Curves — Conditional Experts", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_path = os.path.join(results_dir, "wp0_learning_curves.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Learning curves saved: {save_path}")


# ---------------------------------------------------------------------------
# Plotting — qualitative sample grid
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_sample_grid(
    experts_dict: dict,
    conditioners_dict: dict,
    base_dist: torch.distributions.Distribution,
    test_loader: DataLoader,
    results_dir: str,
    logger: logging.Logger,
    n_samples: int = N_QUAL_SAMPLES,
) -> None:
    """
    Qualitative grid: rows = [LR input | expert_1 | expert_2 | ...]
                      cols = N_QUAL_SAMPLES images

    Args:
        experts_dict:     {name: expert_module}
        conditioners_dict:{name: conditioner_module}
        base_dist:        base distribution for sampling
        test_loader:      test DataLoader
        results_dir:      directory to save PNG
        n_samples:        number of columns (images)
    """
    for name, expert in experts_dict.items():
        expert.eval()
    for name, cond in conditioners_dict.items():
        cond.eval()

    # Grab one batch
    x_clean, y_deg = next(iter(test_loader))
    x_clean = x_clean[:n_samples]
    y_deg   = y_deg[:n_samples]

    expert_names = list(experts_dict.keys())
    n_rows = 1 + len(expert_names) + 1   # LR | experts... | GT clean
    n_cols = n_samples

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.8))

    # Helper to safely display a tensor image
    def show(ax, img_tensor, title=""):
        img = img_tensor.squeeze().detach().cpu().numpy()
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=7)
        ax.axis("off")

    # Row 0: LR (degraded) input
    for j in range(n_cols):
        t = "LR input" if j == 0 else ""
        show(axes[0, j], y_deg[j], title=t)

    # Rows 1..: per-expert samples
    for r, name in enumerate(expert_names, start=1):
        expert     = experts_dict[name]
        conditioner = conditioners_dict[name]
        h = conditioner(y_deg)
        z = base_dist.sample((n_samples, expert.dim)).to(y_deg.device)

        try:
            x_hat = expert.inverse(z, h)           # (N, d)
        except NotImplementedError:
            logger.error(
                f"plot_sample_grid: expert '{name}' .inverse() not implemented — "
                f"skipping row"
            )
            for j in range(n_cols):
                axes[r, j].set_visible(False)
            continue

        if torch.any(torch.isnan(x_hat)):
            logger.warning(
                f"plot_sample_grid: NaN in x_hat for expert '{name}' — "
                f"displaying blank"
            )

        for j in range(n_cols):
            t = name if j == 0 else ""
            show(axes[r, j], x_hat[j].reshape(28, 28), title=t)

    # Last row: GT clean
    for j in range(n_cols):
        t = "GT clean" if j == 0 else ""
        show(axes[n_rows - 1, j], x_clean[j].reshape(28, 28), title=t)

    fig.suptitle("WP0: Expert Reconstructions vs Ground Truth", fontsize=12, y=1.01)
    fig.tight_layout()

    save_path = os.path.join(results_dir, "wp0_samples.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Sample grid saved: {save_path}")


# ---------------------------------------------------------------------------
# CSV save
# ---------------------------------------------------------------------------

def save_csv(rows: list, results_dir: str, logger: logging.Logger) -> None:
    """Save per-expert results to CSV."""
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, "wp0_expert_comparison.csv")
    fieldnames = ["expert", "nll", "inv_error", "cond_quality", "runtime_ms"]

    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Results saved: {save_path}")


# ---------------------------------------------------------------------------
# Comparison table log
# ---------------------------------------------------------------------------

def log_comparison_table(rows: list, logger: logging.Logger) -> None:
    """Log point-by-point comparison table."""
    header = f"{'Expert':<12} {'NLL':>10} {'InvErr':>12} {'CondQ':>12} {'RT(ms)':>10}"
    sep    = "-" * len(header)
    logger.info("")
    logger.info("=== WP0: Expert Comparison Summary ===")
    logger.info(sep)
    logger.info(header)
    logger.info(sep)
    for r in rows:
        logger.info(
            f"{r['expert']:<12} "
            f"{r['nll']:>10.4f} "
            f"{r['inv_error']:>12.6f} "
            f"{r['cond_quality']:>12.4f} "
            f"{r['runtime_ms']:>10.2f}"
        )
    logger.info(sep)

    # Point-by-point best/worst
    def best(key, lower_is_better=True):
        valid = [r for r in rows if not (isinstance(r[key], float) and r[key] != r[key])]
        if not valid:
            return "N/A"
        fn = min if lower_is_better else max
        return fn(valid, key=lambda r: r[key])["expert"]

    logger.info(f"  Best NLL          : {best('nll')}")
    logger.info(f"  Best InvErr       : {best('inv_error')}")
    logger.info(f"  Best CondQuality  : {best('cond_quality')}")
    logger.info(f"  Fastest Inference : {best('runtime_ms')}")
    logger.info("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    seed        = args.seed
    epochs      = args.epochs
    lr          = args.lr
    expert_names = args.experts or ACTIVE_EXPERTS
    results_dir  = os.path.join(args.results_dir or RESULTS_DIR, RESULTS_SUBDIR)

    logger = setup_logging(results_dir)
    logger.info("=" * 60)
    logger.info(f"WP0: Conditional Expert Comparison | EXP-WP0 | v1.1")
    logger.info(f"Experts : {expert_names}")
    logger.info(f"Epochs  : {epochs} | LR: {lr} | Seed: {seed}")
    logger.info("=" * 60)

    fix_seed(seed)
    logger.info(f"Seed fixed: {seed}")

    # Validate expert names
    unknown = [n for n in expert_names if n not in EXPERT_REGISTRY]
    if unknown:
        logger.error(
            f"Unknown expert(s): {unknown} | "
            f"Valid: {list(EXPERT_REGISTRY.keys())}"
        )
        raise KeyError(f"Unknown expert(s): {unknown}")

    # Data
    train_loader, val_loader, test_loader = build_dataloaders(
        data_root  = DATA_ROOT,
        batch_size = BATCH_SIZE,
        val_split  = VAL_SPLIT,
        logger     = logger,
    )

    # Base distribution
    base_dist = torch.distributions.Normal(
        torch.zeros(1), torch.ones(1)
    )

    # Storage
    nll_curves       = {}   # {name: [epoch_nll, ...]}
    experts_trained  = {}   # {name: expert module}
    conditioners     = {}   # {name: conditioner module}
    csv_rows         = []

    # =========================================================================
    # Per-expert: train + evaluate
    # =========================================================================
    for name in expert_names:
        logger.info(f"{'='*40}")
        logger.info(f"Expert: {name}")
        logger.info(f"{'='*40}")

        # Fresh expert + conditioner per run (no cross-contamination)
        cls    = EXPERT_REGISTRY[name]
        expert = cls(
            dim=LATENT_DIM, cond_dim=HIDDEN_DIM, num_layers=NUM_LAYERS
        )
        conditioner = MNISTConditioner(out_dim=HIDDEN_DIM)

        # --- Train ---
        curve = train_expert(
            name        = name,
            expert      = expert,
            conditioner = conditioner,
            base_dist   = base_dist,
            train_loader= train_loader,
            val_loader  = val_loader,
            epochs      = epochs,
            lr          = lr,
            logger      = logger,
        )
        nll_curves[name]      = curve
        experts_trained[name] = expert
        conditioners[name]    = conditioner

        # --- Evaluate ---
        nll = eval_nll(
            name, expert, conditioner, base_dist, test_loader, logger
        )
        inv_err = eval_invertibility(
            name, expert, conditioner, test_loader, logger
        )
        cond_q = eval_conditioning_quality(
            name, expert, conditioner, base_dist, test_loader, logger
        )
        rt = eval_runtime(
            name, expert, conditioner, base_dist, test_loader, logger
        )

        csv_rows.append({
            "expert":       name,
            "nll":          round(nll,     4),
            "inv_error":    round(inv_err, 6),
            "cond_quality": round(cond_q,  4),
            "runtime_ms":   round(rt,      2),
        })

    # =========================================================================
    # Save results
    # =========================================================================
    save_csv(csv_rows, results_dir, logger)
    log_comparison_table(csv_rows, logger)

    # =========================================================================
    # Additional: learning curves
    # =========================================================================
    plot_learning_curves(nll_curves, results_dir, logger)

    # =========================================================================
    # Additional: qualitative sample grid
    # =========================================================================
    plot_sample_grid(
        experts_dict      = experts_trained,
        conditioners_dict = conditioners,
        base_dist         = base_dist,
        test_loader       = test_loader,
        results_dir       = results_dir,
        logger            = logger,
        n_samples         = N_QUAL_SAMPLES,
    )

    logger.info("WP0 experiment complete.")
    logger.info(f"Results in: {results_dir}")


if __name__ == "__main__":
    main()
