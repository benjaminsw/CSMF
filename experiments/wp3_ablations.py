# =============================================================================
# Version: WP3.3-ExpWP3-v1.1 | Abbr: EXP-WP3
# Description: WP3 ablations — gate mechanism, consistency, geometry loss,
#              calibration; compiles WP4 defaults from results
# Changelog:
#   v1.1 (2025-02-21): Added Neff std band, consistency heatmap with PSNR
#                      contours, SW2 convergence curve vs L, PIT histogram
#                      with uniform reference, default seed 2026
#   v1.0 (2025-02-21): Initial — 4 ablations, 4 CSVs, summary table
# Dependencies: CSMF-MAIN, HYBRID, SW2, CALIB, PROX, FWD-MOD, MNIST-CFG
# =============================================================================

import os
import csv
import json
import random
import logging
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from configs.mnist_config import (
    DATA_ROOT, RESULTS_DIR, CKPT_DIR,
    BATCH_SIZE, EPOCHS, LR, HIDDEN_DIM, NUM_LAYERS, LATENT_DIM,
    DOWNSAMPLE_FACTOR, BLUR_KERNEL, NOISE_SIGMA,
    VAL_SPLIT, ACTIVE_EXPERTS, BLOCKS_TO_UNFREEZE,
    LAMBDA_CONS, LAMBDA_TRANS, LAMBDA_CAL,
)
from data.mnist_inverse import MNISTInverseDataset
from csmf.conditioning.conditioning_networks import MNISTConditioner
from csmf.flows.conditional_realnvp import ConditionalRealNVP
from csmf.flows.conditional_maf import ConditionalMAF
from csmf.flows.conditional_nice import ConditionalNICE
from csmf.flows.conditional_nsf import ConditionalNSF
from csmf.models.csmf import CSMF
from csmf.physics.forward_models import SRForwardModel
from csmf.physics.proximal import ProximalOperator
from csmf.losses.hybrid_loss import HybridLoss
from csmf.losses.sliced_wasserstein import sliced_wasserstein_distance
from csmf.losses.calibration import energy_score, crps

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_SEED   = 2026
RESULTS_SUBDIR = "wp3"

# [A] Gate ablation
TAU_VALUES   = [0.8, 1.0, 1.1, 1.5]
TOPK_VALUES  = [1, 2, None]          # None = soft (all experts)

# [B] Consistency ablation
T_VALUES         = [0, 1, 2, 3]
LAMBDA_CONS_VALS = [0.05, 0.1, 0.2]

# [C] Geometry ablation
SW2_L_VALUES  = [128, 256, 512]
MMD_B_VALUES  = [3, 5]
N_GEO_SAMPLES = 512                  # samples per geometry eval

# [D] Calibration ablation
TEMP_VALUES   = [0.5, 1.0, 1.5, 2.0]
N_CAL_SAMPLES = 50                   # samples per observation for calibration

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
    log_path = os.path.join(results_dir, "wp3_ablations.log")
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("EXP-WP3")


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
    p = argparse.ArgumentParser(description="WP3: Full ablation study")
    p.add_argument("--seed",        type=int, default=DEFAULT_SEED)
    p.add_argument("--batch",       type=int, default=BATCH_SIZE)
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument("--ckpt",        type=str, default=None,
                   help="Path to Stage C checkpoint (default: CKPT_DIR/csmf_stage_C.pth)")
    p.add_argument("--skip",        nargs="+", choices=["A","B","C","D"], default=[],
                   help="Skip specific ablations")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_dataloaders(
    data_root: str, batch_size: int,
    val_split: float, logger: logging.Logger,
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
        f"Data | train={n_train} | val={n_val} | test={len(test_ds)} | batch={batch_size}"
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Model loading / quick-train fallback
# ---------------------------------------------------------------------------

def _build_fresh_csmf(logger: logging.Logger) -> CSMF:
    unknown = [n for n in ACTIVE_EXPERTS if n not in EXPERT_REGISTRY]
    if unknown:
        logger.error(f"Unknown experts: {unknown}")
        raise KeyError(f"Unknown experts: {unknown}")
    experts = [
        EXPERT_REGISTRY[n](dim=LATENT_DIM, cond_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
        for n in ACTIVE_EXPERTS
    ]
    K           = len(experts)
    conditioner = MNISTConditioner(out_dim=HIDDEN_DIM)
    gate        = nn.Sequential(
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM // 2, K),
    )
    return CSMF(experts, conditioner, gate)


def build_trained_csmf(
    ckpt_path: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    logger: logging.Logger,
) -> CSMF:
    """
    Load Stage C checkpoint if available.
    Falls back to quick-train (A+B) with WARNING if not found.
    """
    csmf = _build_fresh_csmf(logger)

    if os.path.exists(ckpt_path):
        meta = csmf.load_checkpoint(ckpt_path)
        logger.info(f"Loaded Stage C checkpoint: {ckpt_path} | meta={meta}")
        return csmf

    logger.warning(
        f"Stage C checkpoint not found at '{ckpt_path}' — "
        f"falling back to quick-train (Stage A+B). "
        f"Results may differ from full training."
    )

    fwd_model   = SRForwardModel(blur_sigma=1.0, downsample=DOWNSAMPLE_FACTOR)
    hybrid_loss = HybridLoss(fwd_model, lambda_cons=LAMBDA_CONS,
                             lambda_trans=LAMBDA_TRANS, lambda_cal=LAMBDA_CAL)
    quick_ep    = max(1, EPOCHS // 6)

    optimizer_A = torch.optim.Adam(
        [p for exp in csmf.experts for p in exp.parameters()], lr=LR
    )
    csmf.train_stage_A(
        train_loader, optimizer_A, hybrid_loss, epochs=quick_ep,
        val_loader=val_loader, patience=3,
        ckpt_path="/tmp/wp3_fallback_stageA.pth",
    )
    optimizer_B = torch.optim.Adam(csmf.gate.parameters(), lr=LR / 10)
    csmf.train_stage_B(
        train_loader, optimizer_B, hybrid_loss, epochs=quick_ep,
        val_loader=val_loader, patience=3,
        ckpt_path="/tmp/wp3_fallback_stageB.pth",
    )
    logger.info("Quick-train (A+B) fallback complete.")
    return csmf


# ---------------------------------------------------------------------------
# Helper: PSNR / SSIM
# ---------------------------------------------------------------------------

def batch_psnr(x_hat: torch.Tensor, x_ref: torch.Tensor) -> float:
    mse = torch.mean((x_hat - x_ref) ** 2)
    if mse < 1e-10:
        return 100.0
    return (10.0 * torch.log10(torch.tensor(1.0) / mse)).item()


def batch_ssim(
    x_hat: torch.Tensor, x_ref: torch.Tensor,
    C1: float = 0.01 ** 2, C2: float = 0.03 ** 2,
) -> float:
    mu1    = x_hat.mean()
    mu2    = x_ref.mean()
    sigma1 = x_hat.var()
    sigma2 = x_ref.var()
    sigma12 = ((x_hat - mu1) * (x_ref - mu2)).mean()
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
    return ssim.item()


# ---------------------------------------------------------------------------
# Helper: topk gate masking
# ---------------------------------------------------------------------------

def topk_gate_weights(
    logits: torch.Tensor, k: int
) -> torch.Tensor:
    """Zero non-top-k logits then softmax."""
    if k is None or k >= logits.shape[1]:
        return torch.softmax(logits, dim=1)
    topk_vals, _ = torch.topk(logits, k, dim=1)
    threshold     = topk_vals[:, -1:].expand_as(logits)
    masked_logits = logits.masked_fill(logits < threshold, float("-inf"))
    return torch.softmax(masked_logits, dim=1)


# ---------------------------------------------------------------------------
# Helper: MMD with median heuristic
# ---------------------------------------------------------------------------

def mmd_median(
    X: torch.Tensor, Y: torch.Tensor, n_bandwidths: int
) -> torch.Tensor:
    """
    MMD with multi-bandwidth RBF kernel.
    Bandwidths determined by median heuristic on the batch.
    """
    # Compute pairwise distances for bandwidth selection
    XY   = torch.cat([X, Y], dim=0)
    dists = torch.cdist(XY, XY, p=2)
    median_dist = dists.median().item()
    if median_dist < 1e-6:
        median_dist = 1.0
        logger_ref = logging.getLogger("EXP-WP3")
        logger_ref.warning("MMD: median distance near zero — using bandwidth=1.0")

    # Log-spaced bandwidths around median
    sigmas = torch.logspace(
        np.log10(median_dist / n_bandwidths),
        np.log10(median_dist * n_bandwidths),
        steps=n_bandwidths,
        device=X.device,
    )

    def rbf_kernel(A, B, sigma):
        d = torch.cdist(A, B, p=2) ** 2
        return torch.exp(-d / (2 * sigma ** 2))

    mmd_val = torch.tensor(0.0, device=X.device)
    for sigma in sigmas:
        Kxx = rbf_kernel(X, X, sigma).mean()
        Kyy = rbf_kernel(Y, Y, sigma).mean()
        Kxy = rbf_kernel(X, Y, sigma).mean()
        mmd_val = mmd_val + Kxx + Kyy - 2 * Kxy

    return mmd_val / n_bandwidths


# ---------------------------------------------------------------------------
# [A] Gate ablation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_gate_ablation(
    csmf: CSMF,
    test_loader: DataLoader,
    logger: logging.Logger,
) -> list:
    """
    Sub-A1: temperature τ ∈ TAU_VALUES
    Sub-A2: top-k masking k ∈ TOPK_VALUES

    Metrics per config: mean Neff, std Neff, mean entropy, churn rate.
    Churn = fraction of consecutive batch pairs where argmax expert changes.

    Returns list of row dicts.
    """
    csmf.eval()
    rows = []

    def _eval_gate_config(sub, param_name, param_val, tau, k):
        neffs      = []
        entropies  = []
        prev_argmax = None
        churn_events = 0
        churn_total  = 0

        for x_clean, y_deg in test_loader:
            h      = csmf.conditioner(y_deg)
            logits = csmf.gate(h) / max(tau, 1e-6)
            w      = topk_gate_weights(logits, k)             # (B, K)

            neff    = csmf._compute_neff(w).mean().item()
            w_safe  = w.clamp(min=1e-8)
            entropy = -(w_safe * w_safe.log()).sum(dim=1).mean().item()
            argmax  = w.argmax(dim=1)                         # (B,)

            if prev_argmax is not None:
                churn_events += (argmax != prev_argmax).sum().item()
                churn_total  += argmax.shape[0]
            prev_argmax = argmax

            neffs.append(neff)
            entropies.append(entropy)

            if neff < 1.1:
                logger.warning(
                    f"[A] Gate collapse: Neff={neff:.3f} < 1.1 | "
                    f"{param_name}={param_val}"
                )

        mean_neff = float(np.mean(neffs))
        std_neff  = float(np.std(neffs))
        mean_ent  = float(np.mean(entropies))
        churn     = churn_events / churn_total if churn_total > 0 else float("nan")

        logger.info(
            f"[A] {param_name}={param_val} | "
            f"Neff={mean_neff:.3f}±{std_neff:.3f} | "
            f"Entropy={mean_ent:.4f} | Churn={churn:.3f}"
        )
        return {
            "sub": sub, "param_name": param_name, "param_val": str(param_val),
            "neff_mean": round(mean_neff, 4), "neff_std": round(std_neff, 4),
            "entropy":   round(mean_ent, 4),  "churn":    round(churn, 4),
        }

    # A1: temperature sweep (soft gate, no top-k)
    for tau in TAU_VALUES:
        rows.append(_eval_gate_config("A1_tau", "tau", tau, tau=tau, k=None))

    # A2: top-k sweep (fixed tau=1.1 from WP2 defaults)
    for k in TOPK_VALUES:
        k_label = k if k is not None else "soft"
        rows.append(_eval_gate_config("A2_topk", "top_k", k_label, tau=1.1, k=k))

    csmf.train()
    return rows


# ---------------------------------------------------------------------------
# [B] Consistency ablation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_consistency_ablation(
    csmf: CSMF,
    A: SRForwardModel,
    test_loader: DataLoader,
    logger: logging.Logger,
) -> list:
    """
    For each (T, λ_cons) pair:
      - Sample x_hat from CSMF
      - Apply T proximal steps
      - Compute NLL, residual, PSNR, SSIM

    Returns list of row dicts.
    """
    csmf.eval()
    rows = []

    base_dist = torch.distributions.Normal(torch.zeros(1), torch.ones(1))

    for T in T_VALUES:
        for lam in LAMBDA_CONS_VALS:
            prox = ProximalOperator(A, sigma=NOISE_SIGMA, lam=lam)

            total_nll      = 0.0
            total_residual = 0.0
            total_psnr     = 0.0
            total_ssim     = 0.0
            n_batches      = 0

            for x_clean, y_deg in test_loader:
                # NLL
                log_q, _ = csmf.forward(x_clean, y_deg)
                nll      = -log_q.mean()
                if torch.isnan(nll):
                    logger.warning(
                        f"[B] T={T} | λ={lam} | NaN NLL — skipping batch"
                    )
                    continue

                # Sample + prox steps
                x_samples, _ = csmf.sample(y_deg, num_samples=1)
                x_hat         = x_samples[:, 0, :]          # (B, d)

                y_flat = y_deg.flatten(1)
                for _ in range(T):
                    x_hat = prox.solve(x_hat, y_flat, method="closed_form")
                    if torch.any(torch.isnan(x_hat)):
                        logger.error(
                            f"[B] T={T} | λ={lam} | NaN after prox — stopping steps"
                        )
                        break

                Ax       = A.forward(x_hat)
                residual = torch.mean((Ax - y_flat) ** 2)
                psnr     = batch_psnr(x_hat, x_clean.flatten(1))
                ssim     = batch_ssim(x_hat, x_clean.flatten(1))

                total_nll      += nll.item()
                total_residual += residual.item()
                total_psnr     += psnr
                total_ssim     += ssim
                n_batches      += 1

            if n_batches == 0:
                logger.error(f"[B] T={T} | λ={lam} | All batches failed")
                mean_nll = mean_res = mean_psnr = mean_ssim = float("nan")
            else:
                mean_nll  = total_nll      / n_batches
                mean_res  = total_residual / n_batches
                mean_psnr = total_psnr     / n_batches
                mean_ssim = total_ssim     / n_batches

            logger.info(
                f"[B] T={T} | λ={lam:.3f} | NLL={mean_nll:.4f} | "
                f"Res={mean_res:.6f} | PSNR={mean_psnr:.2f} | SSIM={mean_ssim:.4f}"
            )
            rows.append({
                "T": T, "lambda_cons": lam,
                "nll":      round(mean_nll,  4),
                "residual": round(mean_res,  6),
                "psnr":     round(mean_psnr, 3),
                "ssim":     round(mean_ssim, 4),
            })

    csmf.train()
    return rows


# ---------------------------------------------------------------------------
# [C] Geometry ablation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_geometry_ablation(
    csmf: CSMF,
    test_loader: DataLoader,
    logger: logging.Logger,
) -> list:
    """
    SW2 at L ∈ {128, 256, 512} projections.
    MMD at B ∈ {3, 5} bandwidths.
    Also record NLL for each config.

    Returns list of row dicts.
    """
    csmf.eval()
    rows = []

    # Collect a fixed test batch for geometry comparisons
    x_ref_list, y_ref_list = [], []
    for x_clean, y_deg in test_loader:
        x_ref_list.append(x_clean.flatten(1))
        y_ref_list.append(y_deg.flatten(1))
        if len(x_ref_list) * BATCH_SIZE >= N_GEO_SAMPLES:
            break

    x_ref = torch.cat(x_ref_list, dim=0)[:N_GEO_SAMPLES]    # (N, d)
    y_ref = torch.cat(y_ref_list, dim=0)[:N_GEO_SAMPLES]

    # Sample from CSMF
    x_samples, _ = csmf.sample(y_ref, num_samples=1)
    x_hat         = x_samples[:, 0, :]                       # (N, d)

    # NLL on reference
    log_q, _ = csmf.forward(x_ref, y_ref)
    mean_nll  = -log_q.mean().item()

    # SW2 at different L
    for L in SW2_L_VALUES:
        sw2_val = sliced_wasserstein_distance(x_hat, x_ref, num_projections=L).item()
        if torch.tensor(sw2_val).isnan():
            logger.warning(f"[C] SW2(L={L}) NaN")
        logger.info(f"[C] SW2(L={L}) = {sw2_val:.4f} | NLL={mean_nll:.4f}")
        rows.append({
            "method": "SW2", "param_name": "L", "param_val": L,
            "sw2":    round(sw2_val, 6), "mmd": float("nan"),
            "nll":    round(mean_nll, 4),
        })

    # MMD at different B (bandwidths)
    for B in MMD_B_VALUES:
        mmd_val = mmd_median(x_hat, x_ref, n_bandwidths=B).item()
        if torch.tensor(mmd_val).isnan():
            logger.warning(f"[C] MMD(B={B}) NaN")
        logger.info(f"[C] MMD(B={B}) = {mmd_val:.4f} | NLL={mean_nll:.4f}")
        rows.append({
            "method": "MMD", "param_name": "B", "param_val": B,
            "sw2":    float("nan"), "mmd": round(mmd_val, 6),
            "nll":    round(mean_nll, 4),
        })

    csmf.train()
    return rows


# ---------------------------------------------------------------------------
# [D] Calibration ablation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_calibration_ablation(
    csmf: CSMF,
    test_loader: DataLoader,
    logger: logging.Logger,
) -> tuple:
    """
    D1: ES vs CRPS (fixed temp=1.0)
    D2: Temperature scaling τ ∈ TEMP_VALUES
    D3: PIT coverage (collect empirical quantile levels)

    Returns:
        rows:      list of row dicts for CSV
        pit_vals:  (N_quantiles,) array for PIT histogram
    """
    csmf.eval()
    rows     = []
    pit_all  = []

    for tau in TEMP_VALUES:
        total_es   = 0.0
        total_crps = 0.0
        total_pit  = []
        n_batches  = 0

        for x_clean, y_deg in test_loader:
            B  = x_clean.shape[0]
            # Sample S posterior samples at temperature tau
            x_samples, _ = csmf.sample(
                y_deg, num_samples=N_CAL_SAMPLES, temperature=tau
            )                                                  # (B, S, d)

            batch_es   = 0.0
            batch_crps = 0.0

            for b in range(B):
                samples_b = x_samples[b]                      # (S, d)
                ref_b     = x_clean[b].flatten().unsqueeze(0) # (1, d)

                # Energy Score
                es_val = energy_score(samples_b, ref_b)
                if not torch.isnan(es_val):
                    batch_es += es_val.item()

                # CRPS: average over dimensions (1D marginals)
                crps_vals = []
                for dim_i in range(min(samples_b.shape[1], 50)):  # cap at 50 dims
                    crps_dim = crps(
                        samples_b[:, dim_i],
                        x_clean[b].flatten()[dim_i],
                    )
                    if not torch.isnan(crps_dim):
                        crps_vals.append(crps_dim.item())
                if crps_vals:
                    batch_crps += float(np.mean(crps_vals))

                # PIT: for each dim, compute empirical quantile of reference
                for dim_i in range(min(samples_b.shape[1], 50)):
                    s_sorted = torch.sort(samples_b[:, dim_i])[0].cpu().numpy()
                    ref_val  = x_clean[b].flatten()[dim_i].item()
                    q_level  = float(np.searchsorted(s_sorted, ref_val)) / len(s_sorted)
                    total_pit.append(q_level)

            total_es   += batch_es   / B
            total_crps += batch_crps / B
            n_batches  += 1

        if n_batches == 0:
            logger.error(f"[D] tau={tau} | All batches failed")
            mean_es = mean_crps = pit_err = float("nan")
        else:
            mean_es   = total_es   / n_batches
            mean_crps = total_crps / n_batches

            # PIT coverage error: KS distance from uniform
            if total_pit:
                pit_arr  = np.array(total_pit)
                n_bins   = 20
                hist, _  = np.histogram(pit_arr, bins=n_bins, range=(0, 1), density=True)
                pit_err  = float(np.mean(np.abs(hist - 1.0)))  # deviation from uniform
            else:
                pit_err = float("nan")

        logger.info(
            f"[D] tau={tau:.1f} | ES={mean_es:.4f} | "
            f"CRPS={mean_crps:.4f} | PIT_err={pit_err:.4f}"
        )
        rows.append({
            "method": "ES+CRPS", "temp": tau,
            "es":           round(mean_es,   4),
            "crps":         round(mean_crps, 4),
            "pit_cov_err":  round(pit_err,   4),
        })

        if tau == 1.0 and total_pit:
            pit_all = total_pit   # store tau=1.0 PIT for histogram

    csmf.train()
    return rows, np.array(pit_all) if pit_all else np.array([])


# ---------------------------------------------------------------------------
# CSV saves
# ---------------------------------------------------------------------------

def save_gate_csv(rows, results_dir, logger):
    path = os.path.join(results_dir, "wp3_gate_ablation.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    logger.info(f"Saved: {path}")


def save_consistency_csv(rows, results_dir, logger):
    path = os.path.join(results_dir, "wp3_consistency_ablation.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    logger.info(f"Saved: {path}")


def save_geometry_csv(rows, results_dir, logger):
    path = os.path.join(results_dir, "wp3_geometry_ablation.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    logger.info(f"Saved: {path}")


def save_calibration_csv(rows, results_dir, logger):
    path = os.path.join(results_dir, "wp3_calibration_ablation.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    logger.info(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Summary CSV + WP4 defaults
# ---------------------------------------------------------------------------

def compile_summary(
    gate_rows:  list,
    cons_rows:  list,
    geo_rows:   list,
    cal_rows:   list,
    results_dir: str,
    logger:     logging.Logger,
) -> dict:
    """
    Select best value per ablation and write wp3_ablation_summary.csv.
    Returns selected_defaults dict (feeds WP4 config).
    """
    summary_rows    = []
    selected        = {}

    # Gate: best tau (highest Neff, target > 1.5)
    tau_rows  = [r for r in gate_rows if r["sub"] == "A1_tau"
                 and r["neff_mean"] == r["neff_mean"]]
    topk_rows = [r for r in gate_rows if r["sub"] == "A2_topk"
                 and r["neff_mean"] == r["neff_mean"]]

    if tau_rows:
        best_tau = max(tau_rows, key=lambda r: r["neff_mean"])
        selected["tau"]    = best_tau["param_val"]
        selected["tau_neff"] = best_tau["neff_mean"]
        summary_rows.append({
            "ablation": "gate_tau", "selected_param": "tau",
            "selected_val": best_tau["param_val"],
            "metric": "neff", "value": best_tau["neff_mean"],
        })

    if topk_rows:
        best_topk = max(topk_rows, key=lambda r: r["neff_mean"])
        selected["top_k"] = best_topk["param_val"]
        summary_rows.append({
            "ablation": "gate_topk", "selected_param": "top_k",
            "selected_val": best_topk["param_val"],
            "metric": "neff", "value": best_topk["neff_mean"],
        })

    # Consistency: best (T, λ) by lowest residual
    valid_cons = [r for r in cons_rows if r["residual"] == r["residual"]]
    if valid_cons:
        best_cons = min(valid_cons, key=lambda r: r["residual"])
        selected["T"]           = best_cons["T"]
        selected["lambda_cons"] = best_cons["lambda_cons"]
        summary_rows.append({
            "ablation": "consistency_T", "selected_param": "T",
            "selected_val": best_cons["T"],
            "metric": "residual", "value": best_cons["residual"],
        })
        summary_rows.append({
            "ablation": "consistency_lambda", "selected_param": "lambda_cons",
            "selected_val": best_cons["lambda_cons"],
            "metric": "residual", "value": best_cons["residual"],
        })

    # Geometry: best SW2 method (lowest SW2 at L=256), best MMD
    sw2_rows = [r for r in geo_rows if r["method"] == "SW2"
                and r["sw2"] == r["sw2"]]
    mmd_rows = [r for r in geo_rows if r["method"] == "MMD"
                and r["mmd"] == r["mmd"]]
    if sw2_rows:
        best_sw2 = min(sw2_rows, key=lambda r: r["sw2"])
        selected["sw2_L"] = best_sw2["param_val"]
        summary_rows.append({
            "ablation": "geometry", "selected_param": "SW2_L",
            "selected_val": best_sw2["param_val"],
            "metric": "sw2", "value": best_sw2["sw2"],
        })
    if mmd_rows:
        best_mmd = min(mmd_rows, key=lambda r: r["mmd"])
        selected["mmd_B"] = best_mmd["param_val"]
        summary_rows.append({
            "ablation": "geometry_mmd", "selected_param": "MMD_B",
            "selected_val": best_mmd["param_val"],
            "metric": "mmd", "value": best_mmd["mmd"],
        })

    # Calibration: best temp by lowest CRPS
    valid_cal = [r for r in cal_rows if r["crps"] == r["crps"]]
    if valid_cal:
        best_cal = min(valid_cal, key=lambda r: r["crps"])
        selected["cal_temp"] = best_cal["temp"]
        summary_rows.append({
            "ablation": "calibration", "selected_param": "temp",
            "selected_val": best_cal["temp"],
            "metric": "crps", "value": best_cal["crps"],
        })

    # Write summary CSV
    path = os.path.join(results_dir, "wp3_ablation_summary.csv")
    if summary_rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["ablation","selected_param","selected_val","metric","value"]
            )
            writer.writeheader(); writer.writerows(summary_rows)
        logger.info(f"Summary CSV saved: {path}")

    logger.info(f"WP4 selected defaults: {selected}")
    return selected


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_gate_ablation(gate_rows: list, results_dir: str, logger: logging.Logger) -> None:
    """
    Panel 1: Neff ± std vs τ (line with shaded band) — Additional
    Panel 2: Neff vs top-k (bar)
    """
    tau_rows  = [r for r in gate_rows if r["sub"] == "A1_tau"]
    topk_rows = [r for r in gate_rows if r["sub"] == "A2_topk"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Panel 1: Neff ± std vs τ
    if tau_rows:
        taus      = [float(r["param_val"]) for r in tau_rows]
        neff_mean = [r["neff_mean"] for r in tau_rows]
        neff_std  = [r["neff_std"]  for r in tau_rows]
        neff_arr  = np.array(neff_mean)
        std_arr   = np.array(neff_std)

        ax1.plot(taus, neff_mean, "o-", color="steelblue", linewidth=2.0, label="Mean Neff")
        ax1.fill_between(
            taus, neff_arr - std_arr, neff_arr + std_arr,
            alpha=0.25, color="steelblue", label="±1 std",
        )
        ax1.axhline(1.5, linestyle="--", color="firebrick", linewidth=1.2,
                    label="Target Neff=1.5")
        ax1.axhline(1.1, linestyle=":", color="darkorange", linewidth=1.0,
                    label="Collapse threshold=1.1")
        ax1.set_xlabel("Temperature τ", fontsize=11)
        ax1.set_ylabel("Neff = exp(H(w))", fontsize=11)
        ax1.set_title("A1: Neff vs Gate Temperature", fontsize=12)
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)

    # Panel 2: Neff vs top-k
    if topk_rows:
        labels = [str(r["param_val"]) for r in topk_rows]
        neffs  = [r["neff_mean"] for r in topk_rows]
        colors = ["steelblue" if n >= 1.5 else "darkorange" for n in neffs]
        bars   = ax2.bar(labels, neffs, color=colors, alpha=0.8)
        ax2.axhline(1.5, linestyle="--", color="firebrick", linewidth=1.2,
                    label="Target=1.5")
        ax2.axhline(1.1, linestyle=":", color="darkorange", linewidth=1.0,
                    label="Collapse=1.1")
        for bar, val in zip(bars, neffs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        ax2.set_xlabel("Top-k (None = soft)", fontsize=11)
        ax2.set_ylabel("Neff", fontsize=11)
        ax2.set_title("A2: Neff vs Top-k Masking", fontsize=12)
        ax2.legend(fontsize=9)
        ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("WP3 [A]: Gate Mechanism Ablation", fontsize=13)
    fig.tight_layout()
    path = os.path.join(results_dir, "wp3_gate_ablation.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_consistency_heatmap(
    cons_rows: list, results_dir: str, logger: logging.Logger
) -> None:
    """
    Heatmap: rows=T, cols=λ_cons, cell=residual.
    Additional: PSNR contour overlay.
    """
    T_vals   = sorted(set(r["T"]           for r in cons_rows))
    lam_vals = sorted(set(r["lambda_cons"] for r in cons_rows))

    res_grid  = np.zeros((len(T_vals), len(lam_vals)))
    psnr_grid = np.zeros((len(T_vals), len(lam_vals)))

    for r in cons_rows:
        i = T_vals.index(r["T"])
        j = lam_vals.index(r["lambda_cons"])
        res_grid[i, j]  = r["residual"] if r["residual"] == r["residual"] else 0.0
        psnr_grid[i, j] = r["psnr"]     if r["psnr"]     == r["psnr"]     else 0.0

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(res_grid, aspect="auto", cmap="YlOrRd_r", origin="lower")
    plt.colorbar(im, ax=ax, label="Residual ||Ax−y||²")

    # PSNR contour overlay (additional)
    X, Y = np.meshgrid(range(len(lam_vals)), range(len(T_vals)))
    cs = ax.contour(X, Y, psnr_grid, colors="black", linewidths=0.8, alpha=0.6)
    ax.clabel(cs, fmt="%.1f dB", fontsize=7)

    ax.set_xticks(range(len(lam_vals)))
    ax.set_xticklabels([str(l) for l in lam_vals])
    ax.set_yticks(range(len(T_vals)))
    ax.set_yticklabels([str(t) for t in T_vals])
    ax.set_xlabel("λ_cons", fontsize=11)
    ax.set_ylabel("T (proximal steps)", fontsize=11)
    ax.set_title("WP3 [B]: Residual Heatmap (T × λ)\nContours = PSNR (dB)", fontsize=12)
    fig.tight_layout()

    path = os.path.join(results_dir, "wp3_consistency_heatmap.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_geometry_ablation(
    geo_rows: list, results_dir: str, logger: logging.Logger
) -> None:
    """
    Panel 1: SW2 vs L (convergence curve) — Additional
    Panel 2: MMD vs B (bar)
    """
    sw2_rows = [r for r in geo_rows if r["method"] == "SW2" and r["sw2"] == r["sw2"]]
    mmd_rows = [r for r in geo_rows if r["method"] == "MMD" and r["mmd"] == r["mmd"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Panel 1: SW2 convergence vs L (additional)
    if sw2_rows:
        Ls   = [r["param_val"] for r in sw2_rows]
        sw2s = [r["sw2"]       for r in sw2_rows]
        ax1.plot(Ls, sw2s, "o-", color="steelblue", linewidth=2.0, markersize=7)
        for L, s in zip(Ls, sw2s):
            ax1.annotate(f"{s:.4f}", (L, s), textcoords="offset points",
                         xytext=(4, 4), fontsize=8)
        ax1.set_xlabel("Number of projections (L)", fontsize=11)
        ax1.set_ylabel("SW2", fontsize=11)
        ax1.set_title("C1: SW2 Convergence vs L", fontsize=12)
        ax1.grid(alpha=0.3)

    # Panel 2: MMD vs B
    if mmd_rows:
        Bs   = [str(r["param_val"]) for r in mmd_rows]
        mmds = [r["mmd"]            for r in mmd_rows]
        bars = ax2.bar(Bs, mmds, color="darkorange", alpha=0.8)
        for bar, val in zip(bars, mmds):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1e-4,
                     f"{val:.5f}", ha="center", va="bottom", fontsize=9)
        ax2.set_xlabel("Number of bandwidths (B)", fontsize=11)
        ax2.set_ylabel("MMD", fontsize=11)
        ax2.set_title("C2: MMD vs Bandwidth Count", fontsize=12)
        ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("WP3 [C]: Geometry Loss Ablation", fontsize=13)
    fig.tight_layout()
    path = os.path.join(results_dir, "wp3_geometry_ablation.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_pit_histogram(
    pit_vals: np.ndarray, results_dir: str, logger: logging.Logger
) -> None:
    """
    PIT histogram with ideal uniform reference line (additional).
    Perfect calibration → flat histogram at density=1.0.
    """
    if len(pit_vals) == 0:
        logger.warning("plot_pit_histogram: no PIT values to plot — skipping")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    n_bins = 20
    ax.hist(pit_vals, bins=n_bins, range=(0, 1), density=True,
            color="steelblue", alpha=0.75, label="Empirical PIT")
    ax.axhline(1.0, color="firebrick", linestyle="--", linewidth=1.5,
               label="Ideal uniform (density=1.0)")
    ax.set_xlabel("Probability Integral Transform (quantile level)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("WP3 [D]: PIT Histogram (τ=1.0) — Calibration Diagnostic", fontsize=12)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = os.path.join(results_dir, "wp3_pit_histogram.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    logger.info(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Comparison table log
# ---------------------------------------------------------------------------

def log_summary_table(selected: dict, logger: logging.Logger) -> None:
    logger.info("")
    logger.info("=== WP3: Ablation Summary — WP4 Defaults ===")
    logger.info(f"  {'Parameter':<20} {'Selected Value'}")
    logger.info("  " + "-" * 36)
    for k, v in selected.items():
        logger.info(f"  {k:<20} {v}")
    logger.info("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args        = parse_args()
    seed        = args.seed
    batch_size  = args.batch
    results_dir = os.path.join(args.results_dir or RESULTS_DIR, RESULTS_SUBDIR)
    ckpt_path   = args.ckpt or os.path.join(CKPT_DIR, "csmf_stage_C.pth")
    skip        = set(args.skip)

    logger = setup_logging(results_dir)
    logger.info("=" * 60)
    logger.info("WP3: Full Ablation Study | EXP-WP3 | v1.1")
    logger.info(f"Seed={seed} | Checkpoint={ckpt_path}")
    logger.info(f"Skipping: {skip if skip else 'none'}")
    logger.info("=" * 60)

    fix_seed(seed)

    train_loader, val_loader, test_loader = build_dataloaders(
        DATA_ROOT, batch_size, VAL_SPLIT, logger
    )

    csmf = build_trained_csmf(ckpt_path, train_loader, val_loader, logger)

    A    = SRForwardModel(blur_sigma=1.0, downsample=DOWNSAMPLE_FACTOR)
    logger.info(f"SRForwardModel | blur_sigma=1.0 | downsample={DOWNSAMPLE_FACTOR}")

    gate_rows = cons_rows = geo_rows = cal_rows = []
    pit_vals  = np.array([])

    # =========================================================================
    # [A] Gate ablation
    # =========================================================================
    if "A" not in skip:
        logger.info("--- [A] Gate Mechanism Ablation ---")
        gate_rows = run_gate_ablation(csmf, test_loader, logger)
        save_gate_csv(gate_rows, results_dir, logger)

    # =========================================================================
    # [B] Consistency ablation
    # =========================================================================
    if "B" not in skip:
        logger.info("--- [B] Consistency Ablation ---")
        cons_rows = run_consistency_ablation(csmf, A, test_loader, logger)
        save_consistency_csv(cons_rows, results_dir, logger)

    # =========================================================================
    # [C] Geometry ablation
    # =========================================================================
    if "C" not in skip:
        logger.info("--- [C] Geometry Loss Ablation ---")
        geo_rows = run_geometry_ablation(csmf, test_loader, logger)
        save_geometry_csv(geo_rows, results_dir, logger)

    # =========================================================================
    # [D] Calibration ablation
    # =========================================================================
    if "D" not in skip:
        logger.info("--- [D] Calibration Ablation ---")
        cal_rows, pit_vals = run_calibration_ablation(csmf, test_loader, logger)
        save_calibration_csv(cal_rows, results_dir, logger)

    # =========================================================================
    # Summary + WP4 defaults
    # =========================================================================
    selected = compile_summary(
        gate_rows, cons_rows, geo_rows, cal_rows, results_dir, logger
    )
    log_summary_table(selected, logger)

    # =========================================================================
    # Plots
    # =========================================================================
    if gate_rows:
        plot_gate_ablation(gate_rows, results_dir, logger)
    if cons_rows:
        plot_consistency_heatmap(cons_rows, results_dir, logger)
    if geo_rows:
        plot_geometry_ablation(geo_rows, results_dir, logger)
    if len(pit_vals) > 0:
        plot_pit_histogram(pit_vals, results_dir, logger)

    logger.info("WP3 ablation study complete.")
    logger.info(f"Results in: {results_dir}")


if __name__ == "__main__":
    main()
