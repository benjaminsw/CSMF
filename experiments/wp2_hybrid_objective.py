# =============================================================================
# Version: WP3.3-ExpWP2-v1.1 | Abbr: EXP-WP2
# Description: WP2 experiment — hybrid loss lambda ablation, Pareto analysis,
#              annotated Pareto frontier, best config JSON save
# Changelog:
#   v1.1 (2025-02-21): Added annotated Pareto points, Pareto-optimal frontier
#                      (convex hull of non-dominated points), best config JSON,
#                      NLL + residual bar charts, default seed 2026
#   v1.0 (2025-02-21): Initial — lambda ablation, Pareto plot, CSV
# Dependencies: HYBRID, CSMF-MAIN, TRAIN-MAIN, MNIST-INV, MNIST-CFG
# =============================================================================

import os
import csv
import json
import random
import logging
import argparse
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from configs.mnist_config import (
    DATA_ROOT, RESULTS_DIR,
    BATCH_SIZE, EPOCHS, LR, HIDDEN_DIM, NUM_LAYERS, LATENT_DIM,
    DOWNSAMPLE_FACTOR, BLUR_KERNEL, NOISE_SIGMA,
    VAL_SPLIT, ACTIVE_EXPERTS, BLOCKS_TO_UNFREEZE,
)
from data.mnist_inverse import MNISTInverseDataset
from csmf.conditioning.conditioning_networks import MNISTConditioner
from csmf.flows.conditional_realnvp import ConditionalRealNVP
from csmf.flows.conditional_maf import ConditionalMAF
from csmf.flows.conditional_nice import ConditionalNICE
from csmf.flows.conditional_nsf import ConditionalNSF
from csmf.models.csmf import CSMF
from csmf.physics.forward_models import SRForwardModel
from csmf.losses.hybrid_loss import HybridLoss

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_SEED   = 2026
RESULTS_SUBDIR = "wp2"

# Lambda sweep values
LAMBDA_CONS_VALUES  = [0.0, 0.05, 0.1, 0.2]
LAMBDA_TRANS_VALUES = [0.0, 0.01, 0.05]
LAMBDA_CAL_VALUES   = [0.0, 0.01, 0.05]

# Fixed values when not being swept
LAMBDA_CONS_FIXED   = 0.1
LAMBDA_TRANS_FIXED  = 0.01
LAMBDA_CAL_FIXED    = 0.0

# Quick-train: Stage A + B only (skip C for ablation speed)
QUICK_EPOCHS_A = max(1, EPOCHS // 6)
QUICK_EPOCHS_B = max(1, EPOCHS // 6)

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
    log_path = os.path.join(results_dir, "wp2_hybrid_objective.log")
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("EXP-WP2")


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
    p = argparse.ArgumentParser(description="WP2: Hybrid loss lambda ablation")
    p.add_argument("--seed",        type=int,   default=DEFAULT_SEED)
    p.add_argument("--batch",       type=int,   default=BATCH_SIZE)
    p.add_argument("--results-dir", type=str,   default=None)
    p.add_argument("--epochs-a",    type=int,   default=QUICK_EPOCHS_A,
                   help="Stage A epochs per run (quick train)")
    p.add_argument("--epochs-b",    type=int,   default=QUICK_EPOCHS_B,
                   help="Stage B epochs per run (quick train)")
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
# Model / loss builders
# ---------------------------------------------------------------------------

def _build_fresh_csmf(
    active_experts: list,
    logger: logging.Logger,
) -> CSMF:
    """
    Build a fresh CSMF from config — called before each lambda run.
    No weight sharing between runs.
    """
    unknown = [n for n in active_experts if n not in EXPERT_REGISTRY]
    if unknown:
        logger.error(f"Unknown experts: {unknown}")
        raise KeyError(f"Unknown experts: {unknown}")

    experts = [
        EXPERT_REGISTRY[name](
            dim=LATENT_DIM, cond_dim=HIDDEN_DIM, num_layers=NUM_LAYERS
        )
        for name in active_experts
    ]
    K           = len(experts)
    conditioner = MNISTConditioner(out_dim=HIDDEN_DIM)
    gate        = nn.Sequential(
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM // 2, K),
    )
    return CSMF(experts, conditioner, gate)


def _build_loss(
    lambda_cons:  float,
    lambda_trans: float,
    lambda_cal:   float,
) -> HybridLoss:
    fwd_model = SRForwardModel(blur_sigma=1.0, downsample=DOWNSAMPLE_FACTOR)
    return HybridLoss(
        fwd_model,
        lambda_cons=lambda_cons,
        lambda_trans=lambda_trans,
        lambda_cal=lambda_cal,
    )


# ---------------------------------------------------------------------------
# Quick train: Stage A + B only (Stage C skipped intentionally)
# ---------------------------------------------------------------------------

def _quick_train(
    csmf: CSMF,
    hybrid_loss: HybridLoss,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs_a: int,
    epochs_b: int,
    lr: float,
    seed: int,
    label: str,
    logger: logging.Logger,
) -> None:
    """
    Run Stage A then Stage B. Stage C intentionally skipped for ablation speed.
    Seed is fixed before each call for reproducible per-run comparison.
    """
    fix_seed(seed)
    logger.info(
        f"[{label}] Quick train | Stage A ({epochs_a} ep) + "
        f"Stage B ({epochs_b} ep) | Stage C skipped (ablation mode)"
    )

    # Stage A
    optimizer_A = torch.optim.Adam(
        [p for expert in csmf.experts for p in expert.parameters()],
        lr=lr,
    )
    csmf.train_stage_A(
        dataloader  = train_loader,
        optimizer   = optimizer_A,
        hybrid_loss = hybrid_loss,
        epochs      = epochs_a,
        lambda_cons = hybrid_loss.lambda_cons,
        val_loader  = val_loader,
        patience    = 3,
        ckpt_path   = f"/tmp/csmf_wp2_stageA_{label}.pth",
    )

    # Stage B
    optimizer_B = torch.optim.Adam(
        csmf.gate.parameters(), lr=lr / 10,
    )
    csmf.train_stage_B(
        dataloader  = train_loader,
        optimizer   = optimizer_B,
        hybrid_loss = hybrid_loss,
        epochs      = epochs_b,
        val_loader  = val_loader,
        patience    = 3,
        ckpt_path   = f"/tmp/csmf_wp2_stageB_{label}.pth",
    )

    logger.info(f"[{label}] Quick train complete.")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_nll_and_residual(
    csmf: CSMF,
    hybrid_loss: HybridLoss,
    test_loader: DataLoader,
    label: str,
    logger: logging.Logger,
) -> tuple:
    """
    Evaluate mixture NLL and measurement residual ||A(x_hat) - y||^2
    on the test set.

    Returns:
        (mean_nll, mean_residual)
    """
    csmf.eval()
    total_nll      = 0.0
    total_residual = 0.0
    n_batches      = 0

    for x_clean, y_deg in test_loader:
        log_q, _ = csmf.forward(x_clean, y_deg)
        nll      = -log_q.mean()

        if torch.isnan(nll):
            logger.warning(f"[{label}] NaN NLL in test batch — skipping")
            continue

        # Sample for residual
        x_samples, _ = csmf.sample(y_deg, num_samples=1)
        x_hat         = x_samples[:, 0, :]
        Ax            = hybrid_loss.A.forward(x_hat)
        residual      = torch.mean((Ax - y_deg) ** 2)

        if torch.isnan(residual):
            logger.warning(f"[{label}] NaN residual in test batch — skipping")
            continue

        total_nll      += nll.item()
        total_residual += residual.item()
        n_batches      += 1

    csmf.train()

    if n_batches == 0:
        logger.error(f"[{label}] All test batches NaN — returning nan")
        return float("nan"), float("nan")

    mean_nll = total_nll      / n_batches
    mean_res = total_residual / n_batches
    return mean_nll, mean_res


# ---------------------------------------------------------------------------
# Single ablation run
# ---------------------------------------------------------------------------

def run_single(
    sweep:        str,
    lambda_name:  str,
    lambda_val:   float,
    lambda_cons:  float,
    lambda_trans: float,
    lambda_cal:   float,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    test_loader:  DataLoader,
    epochs_a:     int,
    epochs_b:     int,
    lr:           float,
    seed:         int,
    logger:       logging.Logger,
) -> dict:
    """
    Build fresh CSMF + loss, quick-train, eval, return result dict.
    """
    label = f"{lambda_name}={lambda_val:.3f}"

    csmf        = _build_fresh_csmf(ACTIVE_EXPERTS, logger)
    hybrid_loss = _build_loss(lambda_cons, lambda_trans, lambda_cal)

    try:
        _quick_train(
            csmf, hybrid_loss, train_loader, val_loader,
            epochs_a, epochs_b, lr, seed, label, logger,
        )
    except Exception as e:
        logger.error(f"[{label}] Training failed: {e}")
        return {
            "sweep": sweep, "lambda_name": lambda_name,
            "lambda_val": lambda_val, "nll": float("nan"),
            "residual": float("nan"),
            "lambda_cons": lambda_cons, "lambda_trans": lambda_trans,
            "lambda_cal": lambda_cal,
        }

    nll, residual = eval_nll_and_residual(
        csmf, hybrid_loss, test_loader, label, logger
    )

    if nll != nll:  # isnan check
        logger.warning(f"[{label}] NaN NLL — Pareto point will be skipped")
    else:
        logger.info(
            f"[{label}] NLL={nll:.4f} | Residual={residual:.6f}"
        )

    return {
        "sweep":        sweep,
        "lambda_name":  lambda_name,
        "lambda_val":   lambda_val,
        "nll":          nll,
        "residual":     residual,
        "lambda_cons":  lambda_cons,
        "lambda_trans": lambda_trans,
        "lambda_cal":   lambda_cal,
    }


# ---------------------------------------------------------------------------
# Pareto frontier (non-dominated points)
# ---------------------------------------------------------------------------

def pareto_frontier(points: list) -> list:
    """
    Find non-dominated (Pareto-optimal) points minimising both NLL and residual.

    Args:
        points: list of (nll, residual, label) tuples

    Returns:
        List of indices into `points` that are Pareto-optimal.
    """
    n = len(points)
    dominated = [False] * n

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j is better or equal in both objectives
            if (points[j][0] <= points[i][0] and
                    points[j][1] <= points[i][1] and
                    (points[j][0] < points[i][0] or points[j][1] < points[i][1])):
                dominated[i] = True
                break

    return [i for i in range(n) if not dominated[i]]


# ---------------------------------------------------------------------------
# CSV save
# ---------------------------------------------------------------------------

def save_csv(rows: list, results_dir: str, logger: logging.Logger) -> None:
    path = os.path.join(results_dir, "wp2_lambda_ablation.csv")
    fieldnames = [
        "sweep", "lambda_name", "lambda_val",
        "nll", "residual",
        "lambda_cons", "lambda_trans", "lambda_cal",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in r.items()
            })
    logger.info(f"CSV saved: {path}")


# ---------------------------------------------------------------------------
# Best config JSON
# ---------------------------------------------------------------------------

def save_best_config(rows: list, results_dir: str, logger: logging.Logger) -> None:
    """
    Save best lambda config: best NLL, best residual, best trade-off
    (minimise NLL + residual sum) to wp2_best_config.json.
    """
    valid = [r for r in rows if r["nll"] == r["nll"] and r["residual"] == r["residual"]]
    if not valid:
        logger.error("save_best_config: no valid rows — skipping")
        return

    best_nll      = min(valid, key=lambda r: r["nll"])
    best_residual = min(valid, key=lambda r: r["residual"])
    best_tradeoff = min(valid, key=lambda r: r["nll"] + r["residual"])

    cfg = {
        "timestamp": datetime.now().isoformat(),
        "best_nll": {
            "sweep": best_nll["sweep"],
            "lambda_name": best_nll["lambda_name"],
            "lambda_val": best_nll["lambda_val"],
            "nll": round(best_nll["nll"], 6),
            "residual": round(best_nll["residual"], 6),
            "lambda_cons": best_nll["lambda_cons"],
            "lambda_trans": best_nll["lambda_trans"],
            "lambda_cal": best_nll["lambda_cal"],
        },
        "best_residual": {
            "sweep": best_residual["sweep"],
            "lambda_name": best_residual["lambda_name"],
            "lambda_val": best_residual["lambda_val"],
            "nll": round(best_residual["nll"], 6),
            "residual": round(best_residual["residual"], 6),
            "lambda_cons": best_residual["lambda_cons"],
            "lambda_trans": best_residual["lambda_trans"],
            "lambda_cal": best_residual["lambda_cal"],
        },
        "best_tradeoff": {
            "sweep": best_tradeoff["sweep"],
            "lambda_name": best_tradeoff["lambda_name"],
            "lambda_val": best_tradeoff["lambda_val"],
            "nll": round(best_tradeoff["nll"], 6),
            "residual": round(best_tradeoff["residual"], 6),
            "lambda_cons": best_tradeoff["lambda_cons"],
            "lambda_trans": best_tradeoff["lambda_trans"],
            "lambda_cal": best_tradeoff["lambda_cal"],
        },
    }

    path = os.path.join(results_dir, "wp2_best_config.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    logger.info(f"Best config saved: {path}")
    logger.info(f"  Best NLL      : {best_nll['lambda_name']}={best_nll['lambda_val']:.3f} | NLL={best_nll['nll']:.4f}")
    logger.info(f"  Best Residual : {best_residual['lambda_name']}={best_residual['lambda_val']:.3f} | Res={best_residual['residual']:.6f}")
    logger.info(f"  Best Trade-off: {best_tradeoff['lambda_name']}={best_tradeoff['lambda_val']:.3f}")


# ---------------------------------------------------------------------------
# Comparison table log
# ---------------------------------------------------------------------------

def log_comparison_table(rows: list, logger: logging.Logger) -> None:
    header = f"{'Sweep':<12} {'Lambda':<12} {'Value':>8} {'NLL':>10} {'Residual':>12}"
    sep    = "-" * len(header)
    logger.info("")
    logger.info("=== WP2: Lambda Ablation Summary ===")
    logger.info(sep)
    logger.info(header)
    logger.info(sep)

    current_sweep = None
    for r in rows:
        if r["sweep"] != current_sweep:
            if current_sweep is not None:
                logger.info(sep)
            current_sweep = r["sweep"]
        nll_str = f"{r['nll']:>10.4f}" if r["nll"] == r["nll"] else "       NaN"
        res_str = f"{r['residual']:>12.6f}" if r["residual"] == r["residual"] else "         NaN"
        logger.info(
            f"{r['sweep']:<12} {r['lambda_name']:<12} "
            f"{r['lambda_val']:>8.3f} {nll_str} {res_str}"
        )

    logger.info(sep)
    valid = [r for r in rows if r["nll"] == r["nll"]]
    if valid:
        bn = min(valid, key=lambda r: r["nll"])
        br = min(valid, key=lambda r: r["residual"])
        bt = min(valid, key=lambda r: r["nll"] + r["residual"])
        logger.info(f"  Best NLL       : {bn['lambda_name']}={bn['lambda_val']:.3f} → NLL={bn['nll']:.4f}")
        logger.info(f"  Best Residual  : {br['lambda_name']}={br['lambda_val']:.3f} → Res={br['residual']:.6f}")
        logger.info(f"  Best Trade-off : {bt['lambda_name']}={bt['lambda_val']:.3f} → NLL={bt['nll']:.4f} | Res={bt['residual']:.6f}")
    logger.info("")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

SWEEP_STYLES = {
    "lambda_cons":  {"color": "steelblue",   "marker": "o", "label": "λ_cons sweep"},
    "lambda_trans": {"color": "darkorange",  "marker": "s", "label": "λ_trans sweep"},
    "lambda_cal":   {"color": "seagreen",    "marker": "^", "label": "λ_cal sweep"},
}


def plot_pareto(rows: list, results_dir: str, logger: logging.Logger) -> None:
    """
    Annotated Pareto scatter: NLL vs Residual.
    - Each sweep has its own colour + marker
    - Each point labelled with its lambda value
    - Pareto-optimal frontier highlighted with dashed convex hull line
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    valid_rows   = [r for r in rows if r["nll"] == r["nll"] and r["residual"] == r["residual"]]
    pareto_input = [(r["nll"], r["residual"], r) for r in valid_rows]
    pareto_idx   = pareto_frontier(pareto_input)
    pareto_rows  = {id(valid_rows[i]) for i in pareto_idx}

    # Plot per sweep
    plotted_labels = set()
    for r in valid_rows:
        sweep  = r["sweep"]
        style  = SWEEP_STYLES.get(sweep, {"color": "gray", "marker": "x", "label": sweep})
        lbl    = style["label"] if style["label"] not in plotted_labels else ""
        plotted_labels.add(style["label"])

        is_pareto = id(r) in pareto_rows
        ax.scatter(
            r["nll"], r["residual"],
            color=style["color"],
            marker=style["marker"],
            s=120 if is_pareto else 70,
            zorder=5 if is_pareto else 4,
            edgecolors="black" if is_pareto else "none",
            linewidths=1.2,
            label=lbl,
        )
        # Annotate with lambda value
        ax.annotate(
            f"{r['lambda_val']:.3f}",
            xy=(r["nll"], r["residual"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color=style["color"],
        )

    # Pareto frontier line (sorted by NLL)
    if pareto_idx:
        pf_points = sorted(
            [pareto_input[i] for i in pareto_idx], key=lambda p: p[0]
        )
        pf_nll = [p[0] for p in pf_points]
        pf_res = [p[1] for p in pf_points]
        ax.plot(
            pf_nll, pf_res,
            linestyle="--", color="black", linewidth=1.5,
            label="Pareto frontier", zorder=6,
        )

    ax.set_xlabel("NLL (↓ better)", fontsize=12)
    ax.set_ylabel("Residual ||Ax−y||² (↓ better)", fontsize=12)
    ax.set_title("WP2: Pareto Frontier — NLL vs Measurement Residual", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = os.path.join(results_dir, "wp2_pareto.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Pareto plot saved: {path}")


def plot_nll_bars(rows: list, results_dir: str, logger: logging.Logger) -> None:
    """Bar chart: NLL per lambda value, grouped by sweep."""
    sweeps  = ["lambda_cons", "lambda_trans", "lambda_cal"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)

    for ax, sweep in zip(axes, sweeps):
        sweep_rows = [r for r in rows if r["sweep"] == sweep and r["nll"] == r["nll"]]
        if not sweep_rows:
            ax.set_visible(False)
            continue
        lam_vals = [r["lambda_val"] for r in sweep_rows]
        nlls     = [r["nll"]        for r in sweep_rows]
        style    = SWEEP_STYLES.get(sweep, {"color": "gray"})
        bars = ax.bar(
            [str(round(v, 3)) for v in lam_vals],
            nlls,
            color=style["color"], alpha=0.8,
        )
        ax.set_xlabel(sweep.replace("_", " "), fontsize=10)
        ax.set_ylabel("NLL" if ax == axes[0] else "", fontsize=10)
        ax.set_title(f"{sweep}", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, nlls):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8,
            )

    fig.suptitle("WP2: NLL per Lambda Sweep", fontsize=13)
    fig.tight_layout()
    path = os.path.join(results_dir, "wp2_nll_bars.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"NLL bar chart saved: {path}")


def plot_residual_bars(rows: list, results_dir: str, logger: logging.Logger) -> None:
    """Bar chart: Residual per lambda value, grouped by sweep."""
    sweeps = ["lambda_cons", "lambda_trans", "lambda_cal"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)

    for ax, sweep in zip(axes, sweeps):
        sweep_rows = [
            r for r in rows
            if r["sweep"] == sweep and r["residual"] == r["residual"]
        ]
        if not sweep_rows:
            ax.set_visible(False)
            continue
        lam_vals  = [r["lambda_val"] for r in sweep_rows]
        residuals = [r["residual"]   for r in sweep_rows]
        style     = SWEEP_STYLES.get(sweep, {"color": "gray"})
        bars = ax.bar(
            [str(round(v, 3)) for v in lam_vals],
            residuals,
            color=style["color"], alpha=0.8,
        )
        ax.set_xlabel(sweep.replace("_", " "), fontsize=10)
        ax.set_ylabel("Residual" if ax == axes[0] else "", fontsize=10)
        ax.set_title(f"{sweep}", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, residuals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1e-5,
                f"{val:.5f}", ha="center", va="bottom", fontsize=7,
            )

    fig.suptitle("WP2: Residual per Lambda Sweep", fontsize=13)
    fig.tight_layout()
    path = os.path.join(results_dir, "wp2_residual_bars.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Residual bar chart saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args        = parse_args()
    seed        = args.seed
    batch_size  = args.batch
    epochs_a    = args.epochs_a
    epochs_b    = args.epochs_b
    results_dir = os.path.join(args.results_dir or RESULTS_DIR, RESULTS_SUBDIR)

    logger = setup_logging(results_dir)
    logger.info("=" * 60)
    logger.info("WP2: Hybrid Loss Lambda Ablation | EXP-WP2 | v1.1")
    logger.info(f"Seed={seed} | epochs_A={epochs_a} | epochs_B={epochs_b}")
    logger.info(f"λ_cons sweep : {LAMBDA_CONS_VALUES}")
    logger.info(f"λ_trans sweep: {LAMBDA_TRANS_VALUES}")
    logger.info(f"λ_cal sweep  : {LAMBDA_CAL_VALUES}")
    logger.info("Note: Stage C skipped in all runs (ablation mode)")
    logger.info("=" * 60)

    fix_seed(seed)

    train_loader, val_loader, test_loader = build_dataloaders(
        DATA_ROOT, batch_size, VAL_SPLIT, logger
    )

    all_rows = []

    # =========================================================================
    # [A] λ_cons sweep — fix λ_trans=0, λ_cal=0
    # =========================================================================
    logger.info("--- [A] λ_cons sweep ---")
    for lc in LAMBDA_CONS_VALUES:
        row = run_single(
            sweep        = "lambda_cons",
            lambda_name  = "lambda_cons",
            lambda_val   = lc,
            lambda_cons  = lc,
            lambda_trans = 0.0,
            lambda_cal   = 0.0,
            train_loader = train_loader,
            val_loader   = val_loader,
            test_loader  = test_loader,
            epochs_a     = epochs_a,
            epochs_b     = epochs_b,
            lr           = LR,
            seed         = seed,
            logger       = logger,
        )
        all_rows.append(row)

    # =========================================================================
    # [B] λ_trans sweep — fix λ_cons=0.1, λ_cal=0
    # =========================================================================
    logger.info("--- [B] λ_trans sweep ---")
    for lt in LAMBDA_TRANS_VALUES:
        row = run_single(
            sweep        = "lambda_trans",
            lambda_name  = "lambda_trans",
            lambda_val   = lt,
            lambda_cons  = LAMBDA_CONS_FIXED,
            lambda_trans = lt,
            lambda_cal   = 0.0,
            train_loader = train_loader,
            val_loader   = val_loader,
            test_loader  = test_loader,
            epochs_a     = epochs_a,
            epochs_b     = epochs_b,
            lr           = LR,
            seed         = seed,
            logger       = logger,
        )
        all_rows.append(row)

    # =========================================================================
    # [C] λ_cal sweep — fix λ_cons=0.1, λ_trans=0.01
    # =========================================================================
    logger.info("--- [C] λ_cal sweep ---")
    for lk in LAMBDA_CAL_VALUES:
        row = run_single(
            sweep        = "lambda_cal",
            lambda_name  = "lambda_cal",
            lambda_val   = lk,
            lambda_cons  = LAMBDA_CONS_FIXED,
            lambda_trans = LAMBDA_TRANS_FIXED,
            lambda_cal   = lk,
            train_loader = train_loader,
            val_loader   = val_loader,
            test_loader  = test_loader,
            epochs_a     = epochs_a,
            epochs_b     = epochs_b,
            lr           = LR,
            seed         = seed,
            logger       = logger,
        )
        all_rows.append(row)

    # =========================================================================
    # Save CSV + log table
    # =========================================================================
    save_csv(all_rows, results_dir, logger)
    log_comparison_table(all_rows, logger)

    # =========================================================================
    # Additional: best config JSON
    # =========================================================================
    save_best_config(all_rows, results_dir, logger)

    # =========================================================================
    # Plots
    # =========================================================================
    plot_pareto(all_rows, results_dir, logger)       # annotated + frontier
    plot_nll_bars(all_rows, results_dir, logger)
    plot_residual_bars(all_rows, results_dir, logger)

    logger.info("WP2 experiment complete.")
    logger.info(f"Results in: {results_dir}")


if __name__ == "__main__":
    main()
