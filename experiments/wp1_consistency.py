# =============================================================================
# Version: WP3.3-ExpWP1-v1.1 | Abbr: EXP-WP1
# Description: WP1 experiment — proximal step ablation, Fourier vs PCG solver
#              comparison, lambda sensitivity analysis
# Changelog:
#   v1.1 (2025-02-21): Added convergence curves per T, lambda sensitivity plot,
#                      default seed 2026, per-step residual tracking
#   v1.0 (2025-02-21): Initial — T ablation, solver comparison, CSV + plots
# Dependencies: PROX, FWD-MOD, MNIST-INV, MNIST-CFG
# =============================================================================

import os
import csv
import time
import random
import logging
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from configs.mnist_config import (
    DATA_ROOT, RESULTS_DIR,
    BATCH_SIZE, DOWNSAMPLE_FACTOR, BLUR_KERNEL, NOISE_SIGMA,
)
from data.mnist_inverse import MNISTInverseDataset
from csmf.physics.forward_models import SRForwardModel
from csmf.physics.proximal import ProximalOperator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_SEED   = 2026
RESULTS_SUBDIR = "wp1"

T_VALUES       = [0, 1, 2, 3]           # proximal step counts to ablate
LAMBDA_VALUES  = [0.01, 0.05, 0.1, 0.5] # lambda sensitivity sweep
LAMBDA_FIXED   = 0.1                     # fixed lambda for T ablation
T_FIXED        = 1                       # fixed T for lambda sensitivity
SIGMA_NOISE    = 0.1                     # measurement noise level
PCG_MAX_ITER   = 50                      # PCG convergence budget
PCG_TOL        = 1e-4                    # PCG stopping tolerance
N_WARMUP       = 3                       # timing warmup batches

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(results_dir: str) -> logging.Logger:
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "wp1_consistency.log")
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("EXP-WP1")


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
    p = argparse.ArgumentParser(description="WP1: Consistency layer experiments")
    p.add_argument("--seed",        type=int,   default=DEFAULT_SEED)
    p.add_argument("--batch",       type=int,   default=BATCH_SIZE)
    p.add_argument("--results-dir", type=str,   default=None)
    p.add_argument("--sigma",       type=float, default=SIGMA_NOISE,
                   help="Noise sigma for proximal operator")
    p.add_argument("--pcg-max-iter",type=int,   default=PCG_MAX_ITER)
    p.add_argument("--pcg-tol",     type=float, default=PCG_TOL)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_test_loader(
    data_root: str,
    batch_size: int,
    logger: logging.Logger,
) -> DataLoader:
    test_ds = MNISTInverseDataset(
        root=data_root, train=False,
        blur_k=BLUR_KERNEL, down=DOWNSAMPLE_FACTOR, sigma=NOISE_SIGMA,
    )
    loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    logger.info(f"Test set: {len(test_ds)} samples | batch={batch_size}")
    return loader


# ---------------------------------------------------------------------------
# Residual computation
# ---------------------------------------------------------------------------

def compute_residual(
    A: SRForwardModel,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """Compute mean ||A(x) - y||^2 over batch."""
    Ax  = A.forward(x)
    res = torch.mean((Ax - y) ** 2)
    if torch.isnan(res):
        return float("nan")
    return res.item()


# ---------------------------------------------------------------------------
# [A] T ablation — proximal steps {0, 1, 2, 3}
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_T_ablation(
    A: SRForwardModel,
    prox: ProximalOperator,
    test_loader: DataLoader,
    t_values: list,
    logger: logging.Logger,
) -> dict:
    """
    For each T in t_values, apply T proximal steps and measure residual.

    T=0: x_init = y_deg upsampled (no prox) — baseline.
    T>0: iterate prox.solve(x, y, method='closed_form') T times.

    Returns:
        results: {T: {"mean": float, "std": float,
                       "per_step_curves": [[res_step_0..T] per batch]}}
    """
    results = {}

    for T in t_values:
        batch_residuals     = []
        per_step_all        = []   # list of lists: per_step_all[batch][step]

        for b_idx, (x_clean, y_deg) in enumerate(test_loader):
            # Upsample y to x space as initialisation (T=0 baseline)
            x = torch.nn.functional.interpolate(
                y_deg,
                scale_factor=DOWNSAMPLE_FACTOR,
                mode="bilinear",
                align_corners=False,
            )
            # Flatten if needed to match prox interface
            x_flat = x.flatten(1)
            y_flat = y_deg.flatten(1)

            step_residuals = []

            # Record T=0 residual (before any prox)
            res0 = compute_residual(A, x_flat, y_flat)
            if T == 0:
                step_residuals.append(res0)

            # Apply T proximal steps
            for t in range(T):
                x_flat = prox.solve(x_flat, y_flat, method="closed_form")

                if torch.any(torch.isnan(x_flat)):
                    logger.error(
                        f"T={T} | batch={b_idx} | step={t} | "
                        f"NaN after prox.solve — stopping steps"
                    )
                    break

                res_t = compute_residual(A, x_flat, y_flat)
                step_residuals.append(res_t)

            # Final residual after T steps
            final_res = step_residuals[-1] if step_residuals else float("nan")
            if not (isinstance(final_res, float) and final_res != final_res):
                batch_residuals.append(final_res)

            per_step_all.append(step_residuals)

        if len(batch_residuals) == 0:
            logger.error(f"T={T} | All batches produced NaN")
            mean_r, std_r = float("nan"), float("nan")
        else:
            mean_r = float(np.mean(batch_residuals))
            std_r  = float(np.std(batch_residuals))

        results[T] = {
            "mean":            mean_r,
            "std":             std_r,
            "per_step_curves": per_step_all,
        }
        logger.info(
            f"T={T} | Residual = {mean_r:.6f} ± {std_r:.6f}"
        )

    return results


# ---------------------------------------------------------------------------
# [B] Solver comparison — Fourier vs PCG
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_solver_comparison(
    A: SRForwardModel,
    test_loader: DataLoader,
    sigma: float,
    lam: float,
    pcg_max_iter: int,
    pcg_tol: float,
    logger: logging.Logger,
) -> dict:
    """
    Compare Fourier (closed_form) vs PCG solve for T=1 proximal step.

    Returns:
        {solver: {"mean_residual", "std_residual", "mean_time_ms",
                  "std_time_ms", "mean_iters"}}
    """
    fourier_res   = []
    fourier_times = []
    pcg_res       = []
    pcg_times     = []
    pcg_iters_all = []

    prox_fourier = ProximalOperator(A, sigma=sigma, lam=lam)
    prox_pcg     = ProximalOperator(A, sigma=sigma, lam=lam)

    for b_idx, (x_clean, y_deg) in enumerate(test_loader):
        x_init = torch.nn.functional.interpolate(
            y_deg,
            scale_factor=DOWNSAMPLE_FACTOR,
            mode="bilinear",
            align_corners=False,
        ).flatten(1)
        y_flat = y_deg.flatten(1)

        # --- Fourier solve ---
        if b_idx < N_WARMUP:
            prox_fourier.solve(x_init, y_flat, method="closed_form")
        else:
            t0      = time.perf_counter()
            x_f     = prox_fourier.solve(x_init, y_flat, method="closed_form")
            t1      = time.perf_counter()

            if torch.any(torch.isnan(x_f)):
                logger.warning(f"Fourier | batch={b_idx} | NaN result — skipping")
            else:
                fourier_res.append(compute_residual(A, x_f, y_flat))
                fourier_times.append((t1 - t0) * 1000.0)

        # --- PCG solve ---
        if b_idx < N_WARMUP:
            prox_pcg.solve(
                x_init, y_flat, method="pcg",
                max_iter=pcg_max_iter, tol=pcg_tol,
            )
        else:
            t0 = time.perf_counter()
            x_p, n_iters, converged = prox_pcg.solve(
                x_init, y_flat, method="pcg",
                max_iter=pcg_max_iter, tol=pcg_tol,
                return_info=True,
            )
            t1 = time.perf_counter()

            if not converged:
                res_pcg = compute_residual(A, x_p, y_flat)
                logger.warning(
                    f"PCG | batch={b_idx} | did not converge | "
                    f"iters={n_iters} | residual={res_pcg:.6f} > tol={pcg_tol}"
                )

            if torch.any(torch.isnan(x_p)):
                logger.warning(f"PCG | batch={b_idx} | NaN result — skipping")
            else:
                pcg_res.append(compute_residual(A, x_p, y_flat))
                pcg_times.append((t1 - t0) * 1000.0)
                pcg_iters_all.append(n_iters)

    def safe_stats(vals):
        if not vals:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals))

    f_res_mean,  f_res_std  = safe_stats(fourier_res)
    f_time_mean, f_time_std = safe_stats(fourier_times)
    p_res_mean,  p_res_std  = safe_stats(pcg_res)
    p_time_mean, p_time_std = safe_stats(pcg_times)
    p_iter_mean             = float(np.mean(pcg_iters_all)) if pcg_iters_all else float("nan")

    results = {
        "fourier": {
            "mean_residual": f_res_mean,
            "std_residual":  f_res_std,
            "mean_time_ms":  f_time_mean,
            "std_time_ms":   f_time_std,
            "mean_iters":    float("nan"),   # N/A for direct solve
        },
        "pcg": {
            "mean_residual": p_res_mean,
            "std_residual":  p_res_std,
            "mean_time_ms":  p_time_mean,
            "std_time_ms":   p_time_std,
            "mean_iters":    p_iter_mean,
        },
    }

    logger.info(
        f"Fourier | Residual={f_res_mean:.6f}±{f_res_std:.6f} | "
        f"Time={f_time_mean:.2f}ms"
    )
    logger.info(
        f"PCG     | Residual={p_res_mean:.6f}±{p_res_std:.6f} | "
        f"Time={p_time_mean:.2f}ms | Iters={p_iter_mean:.1f}"
    )
    return results


# ---------------------------------------------------------------------------
# [C] Lambda sensitivity — λ ∈ {0.01, 0.05, 0.1, 0.5} at fixed T=1
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_lambda_sensitivity(
    A: SRForwardModel,
    test_loader: DataLoader,
    sigma: float,
    lambda_values: list,
    T: int,
    logger: logging.Logger,
) -> dict:
    """
    For each lambda at fixed T=1, measure mean residual.

    Returns:
        {lam: {"mean_residual": float, "std_residual": float}}
    """
    results = {}

    for lam in lambda_values:
        prox          = ProximalOperator(A, sigma=sigma, lam=lam)
        batch_residuals = []

        for x_clean, y_deg in test_loader:
            x = torch.nn.functional.interpolate(
                y_deg,
                scale_factor=DOWNSAMPLE_FACTOR,
                mode="bilinear",
                align_corners=False,
            ).flatten(1)
            y_flat = y_deg.flatten(1)

            for _ in range(T):
                x = prox.solve(x, y_flat, method="closed_form")
                if torch.any(torch.isnan(x)):
                    logger.error(
                        f"Lambda={lam:.3f} | NaN after prox — stopping"
                    )
                    break

            res = compute_residual(A, x, y_flat)
            if not (isinstance(res, float) and res != res):
                batch_residuals.append(res)

        mean_r = float(np.mean(batch_residuals)) if batch_residuals else float("nan")
        std_r  = float(np.std(batch_residuals))  if batch_residuals else float("nan")
        results[lam] = {"mean_residual": mean_r, "std_residual": std_r}
        logger.info(f"λ={lam:.3f} | T={T} | Residual={mean_r:.6f}±{std_r:.6f}")

    return results


# ---------------------------------------------------------------------------
# CSV saves
# ---------------------------------------------------------------------------

def save_residuals_csv(
    T_results: dict,
    lambda_results: dict,
    results_dir: str,
    logger: logging.Logger,
) -> None:
    path = os.path.join(results_dir, "wp1_residuals.csv")
    fieldnames = ["experiment", "T", "lambda", "mean_residual", "std_residual"]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # T ablation rows
        for T, v in T_results.items():
            writer.writerow({
                "experiment":    "T_ablation",
                "T":             T,
                "lambda":        LAMBDA_FIXED,
                "mean_residual": round(v["mean"], 6),
                "std_residual":  round(v["std"],  6),
            })
        # Lambda sensitivity rows
        for lam, v in lambda_results.items():
            writer.writerow({
                "experiment":    "lambda_sensitivity",
                "T":             T_FIXED,
                "lambda":        lam,
                "mean_residual": round(v["mean_residual"], 6),
                "std_residual":  round(v["std_residual"],  6),
            })

    logger.info(f"Residuals CSV saved: {path}")


def save_solver_csv(
    solver_results: dict,
    results_dir: str,
    logger: logging.Logger,
) -> None:
    path = os.path.join(results_dir, "wp1_solver_comparison.csv")
    fieldnames = [
        "solver", "mean_residual", "std_residual",
        "mean_time_ms", "std_time_ms", "mean_iters",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for solver, v in solver_results.items():
            writer.writerow({
                "solver":        solver,
                "mean_residual": round(v["mean_residual"], 6),
                "std_residual":  round(v["std_residual"],  6),
                "mean_time_ms":  round(v["mean_time_ms"],  3),
                "std_time_ms":   round(v["std_time_ms"],   3),
                "mean_iters":    round(v["mean_iters"],    2)
                                 if v["mean_iters"] == v["mean_iters"] else "N/A",
            })
    logger.info(f"Solver comparison CSV saved: {path}")


# ---------------------------------------------------------------------------
# Comparison table log
# ---------------------------------------------------------------------------

def log_solver_table(solver_results: dict, logger: logging.Logger) -> None:
    header = (
        f"{'Solver':<10} {'Residual':>12} {'Std':>10} "
        f"{'Time(ms)':>10} {'Iters':>8}"
    )
    sep = "-" * len(header)
    logger.info("")
    logger.info("=== WP1: Fourier vs PCG Comparison ===")
    logger.info(sep)
    logger.info(header)
    logger.info(sep)
    for solver, v in solver_results.items():
        iters_str = (
            f"{v['mean_iters']:>8.1f}"
            if v["mean_iters"] == v["mean_iters"]
            else "     N/A"
        )
        logger.info(
            f"{solver:<10} "
            f"{v['mean_residual']:>12.6f} "
            f"{v['std_residual']:>10.6f} "
            f"{v['mean_time_ms']:>10.2f} "
            f"{iters_str}"
        )
    logger.info(sep)

    # Point-by-point winner
    solvers = list(solver_results.keys())
    if len(solvers) == 2:
        s0, s1 = solvers
        r0 = solver_results[s0]["mean_residual"]
        r1 = solver_results[s1]["mean_residual"]
        t0 = solver_results[s0]["mean_time_ms"]
        t1 = solver_results[s1]["mean_time_ms"]
        logger.info(
            f"  Better residual : {s0 if r0 < r1 else s1}"
        )
        logger.info(
            f"  Faster inference: {s0 if t0 < t1 else s1}"
        )
    logger.info("")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_residual_vs_T(
    T_results: dict,
    results_dir: str,
    logger: logging.Logger,
) -> None:
    """Bar + line chart: residual vs number of proximal steps."""
    T_vals   = sorted(T_results.keys())
    means    = [T_results[T]["mean"] for T in T_vals]
    stds     = [T_results[T]["std"]  for T in T_vals]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        T_vals, means, yerr=stds,
        color="steelblue", alpha=0.75, capsize=5,
        label="Mean residual ± std",
    )
    ax.plot(T_vals, means, "o-", color="navy", linewidth=1.8, zorder=5)
    ax.set_xlabel("Number of proximal steps (T)", fontsize=12)
    ax.set_ylabel("Mean ||A(x) - y||²", fontsize=12)
    ax.set_title("WP1: Residual vs Proximal Steps (T)", fontsize=13)
    ax.set_xticks(T_vals)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(results_dir, "wp1_residual_vs_T.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Plot saved: {path}")


def plot_solver_comparison(
    solver_results: dict,
    results_dir: str,
    logger: logging.Logger,
) -> None:
    """Grouped bar chart: Fourier vs PCG — residual and time side-by-side."""
    solvers   = list(solver_results.keys())
    residuals = [solver_results[s]["mean_residual"] for s in solvers]
    res_stds  = [solver_results[s]["std_residual"]  for s in solvers]
    times     = [solver_results[s]["mean_time_ms"]  for s in solvers]
    time_stds = [solver_results[s]["std_time_ms"]   for s in solvers]

    x     = np.arange(len(solvers))
    width = 0.35
    colors = ["steelblue", "darkorange"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar(x, residuals, width, yerr=res_stds, capsize=5,
            color=colors, alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(solvers)
    ax1.set_ylabel("Mean ||A(x) - y||²", fontsize=11)
    ax1.set_title("Residual", fontsize=12)
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, times, width, yerr=time_stds, capsize=5,
            color=colors, alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(solvers)
    ax2.set_ylabel("Mean time (ms/batch)", fontsize=11)
    ax2.set_title("Inference Time", fontsize=12)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("WP1: Fourier vs PCG Solver Comparison", fontsize=13)
    fig.tight_layout()

    path = os.path.join(results_dir, "wp1_solver_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Plot saved: {path}")


def plot_convergence_curves(
    T_results: dict,
    results_dir: str,
    logger: logging.Logger,
) -> None:
    """
    Additional: per-step residual convergence curves for each T.
    Shows mean residual at each proximal iteration step.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "D"]
    colors  = ["steelblue", "darkorange", "seagreen", "firebrick"]

    for i, T in enumerate(sorted(T_results.keys())):
        if T == 0:
            continue  # no curve for T=0 (single point)

        per_step = T_results[T]["per_step_curves"]
        # per_step: list over batches of list over steps
        max_steps = max(len(s) for s in per_step)

        # Mean residual at each step across batches
        step_means = []
        step_stds  = []
        for step in range(max_steps):
            vals = [
                batch[step]
                for batch in per_step
                if step < len(batch)
                   and not (isinstance(batch[step], float)
                            and batch[step] != batch[step])
            ]
            if vals:
                step_means.append(float(np.mean(vals)))
                step_stds.append(float(np.std(vals)))
            else:
                step_means.append(float("nan"))
                step_stds.append(0.0)

        steps = list(range(1, max_steps + 1))
        ax.errorbar(
            steps, step_means, yerr=step_stds,
            label=f"T={T}",
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            capsize=4, linewidth=1.8,
        )

    ax.set_xlabel("Proximal step", fontsize=12)
    ax.set_ylabel("Mean ||A(x) - y||²", fontsize=12)
    ax.set_title("WP1: Residual Convergence per Proximal Step", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xticks([1, 2, 3])
    fig.tight_layout()

    path = os.path.join(results_dir, "wp1_convergence_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Plot saved: {path}")


def plot_lambda_sensitivity(
    lambda_results: dict,
    results_dir: str,
    logger: logging.Logger,
) -> None:
    """
    Additional: residual vs lambda at fixed T=1.
    Line + shaded std band.
    """
    lams  = sorted(lambda_results.keys())
    means = [lambda_results[l]["mean_residual"] for l in lams]
    stds  = [lambda_results[l]["std_residual"]  for l in lams]

    means_arr = np.array(means)
    stds_arr  = np.array(stds)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lams, means, "o-", color="steelblue", linewidth=2.0,
            label=f"Mean residual (T={T_FIXED})")
    ax.fill_between(
        lams,
        means_arr - stds_arr,
        means_arr + stds_arr,
        alpha=0.25, color="steelblue", label="±1 std",
    )
    ax.set_xlabel("Lambda (λ)", fontsize=12)
    ax.set_ylabel("Mean ||A(x) - y||²", fontsize=12)
    ax.set_title(f"WP1: Lambda Sensitivity (T={T_FIXED})", fontsize=13)
    ax.set_xscale("log")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = os.path.join(results_dir, "wp1_lambda_sensitivity.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Plot saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args        = parse_args()
    seed        = args.seed
    batch_size  = args.batch
    sigma       = args.sigma
    pcg_max_iter = args.pcg_max_iter
    pcg_tol     = args.pcg_tol
    results_dir = os.path.join(args.results_dir or RESULTS_DIR, RESULTS_SUBDIR)

    logger = setup_logging(results_dir)
    logger.info("=" * 60)
    logger.info("WP1: Consistency Layer Experiment | EXP-WP1 | v1.1")
    logger.info(f"Seed={seed} | Sigma={sigma} | Lambda_fixed={LAMBDA_FIXED} | T_fixed={T_FIXED}")
    logger.info(f"T_values={T_VALUES} | Lambda_values={LAMBDA_VALUES}")
    logger.info(f"PCG max_iter={pcg_max_iter} | tol={pcg_tol}")
    logger.info("=" * 60)

    fix_seed(seed)
    logger.info(f"Seed fixed: {seed}")

    # Build forward model and proximal operator
    A    = SRForwardModel(blur_sigma=1.0, downsample=DOWNSAMPLE_FACTOR)
    prox = ProximalOperator(A, sigma=sigma, lam=LAMBDA_FIXED)
    logger.info(
        f"SRForwardModel | blur_sigma=1.0 | downsample={DOWNSAMPLE_FACTOR}"
    )
    logger.info(
        f"ProximalOperator | sigma={sigma} | lam={LAMBDA_FIXED}"
    )

    # Data
    test_loader = build_test_loader(DATA_ROOT, batch_size, logger)

    # =========================================================================
    # [A] T ablation
    # =========================================================================
    logger.info("--- [A] T ablation ---")
    T_results = run_T_ablation(A, prox, test_loader, T_VALUES, logger)

    # =========================================================================
    # [B] Solver comparison
    # =========================================================================
    logger.info("--- [B] Solver comparison ---")
    solver_results = run_solver_comparison(
        A, test_loader, sigma, LAMBDA_FIXED,
        pcg_max_iter, pcg_tol, logger,
    )

    # =========================================================================
    # [C] Lambda sensitivity
    # =========================================================================
    logger.info("--- [C] Lambda sensitivity ---")
    lambda_results = run_lambda_sensitivity(
        A, test_loader, sigma, LAMBDA_VALUES, T_FIXED, logger,
    )

    # =========================================================================
    # Save CSVs
    # =========================================================================
    save_residuals_csv(T_results, lambda_results, results_dir, logger)
    save_solver_csv(solver_results, results_dir, logger)

    # =========================================================================
    # Log comparison table
    # =========================================================================
    log_solver_table(solver_results, logger)

    # =========================================================================
    # Core plots
    # =========================================================================
    plot_residual_vs_T(T_results, results_dir, logger)
    plot_solver_comparison(solver_results, results_dir, logger)

    # =========================================================================
    # Additional plots
    # =========================================================================
    plot_convergence_curves(T_results, results_dir, logger)
    plot_lambda_sensitivity(lambda_results, results_dir, logger)

    logger.info("WP1 experiment complete.")
    logger.info(f"Results in: {results_dir}")


if __name__ == "__main__":
    main()
