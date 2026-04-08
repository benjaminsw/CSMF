# =============================================================================
# Version: WP3.2-TrainMain-v1.6 | Abbr: TRAIN-MAIN
# Description: Main CSMF training script — 3-stage protocol with expert registry
# Changelog:
#   v1.6 (2026-03-01): Added --skip-sanity CLI flag; after Stage A calls
#                      run_expert_sanity() from EXP-SANITY for diagnostic checks
#                      and plots (Core 1-3 + A/D/F); epoch_logs captured from
#                      train_stage_A() return value (v1.3.7); output to
#                      results/expert_sanity/
#   v1.5 (2026-02-28): BUG FIX — SR forward model kernel was on CPU while x_hat was
#                      on CUDA; added hybrid_loss.A.to(device) after build_loss();
#                      also moves hybrid_loss to device if it is nn.Module
#   v1.4 (2026-02-28): GPU support — auto-detect cuda/cpu device after fix_seed();
#                      model.to(device) after build_model(); eval_final() moves
#                      x_clean/y_deg to device in dataloader loop; device logged
#                      at INFO level; no API changes to stage methods
#   v1.3 (2026-02-25): Replaced build_dataloaders() + MNISTInverseDataset with
#                      create_precomputed_dataloaders() from PREP-MNIST
#                      Stage A optimizer changed from shared optimizer_A to optimizer_fn
#                      callable for per-expert LR support
#                      load_stage_checkpoint() extended with config_hash verification —
#                      raises ValueError on drift
#                      Added --preprocessed-dir and --expert-lr CLI args; config_params
#                      dict built from mnist_config and passed for metadata validation
#   v1.2 (2025-02-23): Separate-stage training: auto-load prior-stage checkpoint
#                      when a stage is skipped; added --ckpt-A/B/C CLI overrides
#                      for per-stage checkpoint paths; added metadata validation
#                      to verify expert config matches before loading
#   v1.1 (2025-02-21): Added NICE/NSF to expert registry, ACTIVE_EXPERTS config
#                      toggle, per-expert NLL breakdown in final eval, gate usage
#                      stats, argparse CLI overrides, seed fixing, JSON eval save
#   v1.0 (2025-02-21): Initial 3-stage training script
# Dependencies: CSMF-MAIN, HYBRID, PREP-MNIST, MNIST-CFG
# =============================================================================

import os
import json
import random
import logging
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from configs.mnist_config import (
    DATA_ROOT, PREPROCESSED_DIR, CKPT_DIR, RESULTS_DIR,
    BATCH_SIZE, EPOCHS, LR, SEED,
    HIDDEN_DIM, NUM_LAYERS, LATENT_DIM,
    DOWNSAMPLE_FACTOR, BLUR_KERNEL, BLUR_SIGMA, NOISE_SIGMA,
    LAMBDA_CONS, LAMBDA_TRANS, LAMBDA_CAL,
    ACTIVE_EXPERTS,
    VAL_SPLIT, PATIENCE,
    BLOCKS_TO_UNFREEZE, TAU_START, TAU_END,
    config_hash,
)
from scripts.preprocess_mnist import create_precomputed_dataloaders
from csmf.conditioning.conditioning_networks import MNISTConditioner
from csmf.flows.conditional_realnvp import ConditionalRealNVP
from csmf.flows.conditional_maf import ConditionalMAF
from csmf.flows.conditional_nice import ConditionalNICE
from csmf.flows.conditional_nsf import ConditionalNSF
from csmf.models.csmf import CSMF
from csmf.physics.forward_models import SRForwardModel
from csmf.losses.hybrid_loss import HybridLoss
from csmf.evaluation.expert_sanity import run_expert_sanity  # v1.6: EXP-SANITY

# ---------------------------------------------------------------------------
# Expert registry — add/remove entries to test combinations
# ---------------------------------------------------------------------------
EXPERT_REGISTRY = {
    "realnvp": ConditionalRealNVP,
    "maf":     ConditionalMAF,
    "nice":    ConditionalNICE,
    "nsf":     ConditionalNSF,
}

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_path: str) -> logging.Logger:
    """Configure file + stream logging."""
    os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("train_csmf")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def fix_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """CLI overrides for config values."""
    p = argparse.ArgumentParser(description="Train CSMF — 3-stage protocol")

    # Training
    p.add_argument("--lr",      type=float, default=None,
                   help="Learning rate (overrides config LR)")
    p.add_argument("--epochs",  type=int,   default=None,
                   help="Total epochs split across A/B/C (overrides config EPOCHS)")
    p.add_argument("--batch",   type=int,   default=None,
                   help="Batch size (overrides config BATCH_SIZE)")
    p.add_argument("--seed",    type=int,   default=None,
                   help="Random seed (overrides config SEED)")

    # Stage selection — run only specific stages (default: all)
    p.add_argument("--stages",  nargs="+", choices=["A", "B", "C"],
                   default=["A", "B", "C"],
                   help="Which stages to run, e.g. --stages A B")

    # Expert selection — override ACTIVE_EXPERTS from config
    p.add_argument("--experts", nargs="+",
                   choices=list(EXPERT_REGISTRY.keys()),
                   default=None,
                   help="Expert subset, e.g. --experts realnvp maf")

    # Paths
    p.add_argument("--ckpt-dir",    type=str, default=None,
                   help="Checkpoint directory (overrides config CKPT_DIR)")
    p.add_argument("--results-dir", type=str, default=None,
                   help="Results directory (overrides config RESULTS_DIR)")

    # Resume from checkpoint (general)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")

    # Per-stage checkpoint path overrides (for separate-stage training)
    p.add_argument("--ckpt-A", type=str, default=None,
                   help="Path to Stage A checkpoint to load when skipping Stage A "
                        "(default: <ckpt-dir>/csmf_stage_A.pth)")
    p.add_argument("--ckpt-B", type=str, default=None,
                   help="Path to Stage B checkpoint to load when skipping Stage B "
                        "(default: <ckpt-dir>/csmf_stage_B.pth)")
    p.add_argument("--ckpt-C", type=str, default=None,
                   help="Path to Stage C checkpoint to load when skipping Stage C "
                        "(default: <ckpt-dir>/csmf_stage_C.pth)")

    # v1.3: precomputed data path
    p.add_argument("--preprocessed-dir", type=str, default=None,
                   help="Path to precomputed .pt files (overrides config PREPROCESSED_DIR)")

    # v1.3: per-expert learning rates e.g. --expert-lr realnvp=1e-3 maf=5e-4
    p.add_argument("--expert-lr", nargs="+", default=None,
                   metavar="NAME=LR",
                   help="Per-expert LR as name=value pairs, e.g. realnvp=1e-3 maf=5e-4")

    # v1.6: skip expert sanity checks after Stage A
    p.add_argument("--skip-sanity", action="store_true", default=False,
                   help="Skip EXP-SANITY diagnostic checks/plots after Stage A")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint loader with metadata validation
# ---------------------------------------------------------------------------

def load_stage_checkpoint(
    model: "CSMF",
    ckpt_path: str,
    stage_label: str,
    experts_cfg: list,
    logger: logging.Logger,
) -> None:
    """
    Load a stage checkpoint into *model* with optional expert-config validation.

    Validation logic:
      - If checkpoint contains 'active_experts' metadata, verify it matches
        the current experts_cfg.  Mismatch → ERROR + raise.
      - If checkpoint has no metadata (older format), log a warning and
        continue loading state_dict only.

    Args:
        model:        CSMF instance to load weights into.
        ckpt_path:    Path to the .pth checkpoint file.
        stage_label:  Human-readable stage name, e.g. "A".
        experts_cfg:  Current active expert list (for validation).
        logger:       Logger instance.

    Raises:
        FileNotFoundError: checkpoint file does not exist.
        ValueError:         expert config mismatch detected.
        RuntimeError:       state_dict load fails.
    """
    if not os.path.isfile(ckpt_path):
        logger.error(
            f"Stage {stage_label} checkpoint not found: {ckpt_path} | "
            f"Run Stage {stage_label} first or provide --ckpt-{stage_label}."
        )
        raise FileNotFoundError(
            f"Stage {stage_label} checkpoint not found: {ckpt_path}"
        )

    try:
        meta = model.load_checkpoint(ckpt_path)
        logger.info(f"Loaded Stage {stage_label} checkpoint: {ckpt_path} | meta={meta}")
    except Exception as e:
        logger.error(f"Failed to load Stage {stage_label} checkpoint {ckpt_path}: {e}")
        raise RuntimeError(
            f"Stage {stage_label} checkpoint load failed: {e}"
        ) from e

    # Metadata validation — expert config check
    if meta and "active_experts" in meta:
        saved_experts = meta["active_experts"]
        if sorted(saved_experts) != sorted(experts_cfg):
            logger.error(
                f"Expert config mismatch for Stage {stage_label} checkpoint | "
                f"checkpoint={saved_experts} | current={experts_cfg}"
            )
            raise ValueError(
                f"Expert config mismatch: checkpoint has {saved_experts}, "
                f"current run has {experts_cfg}. "
                f"Use matching --experts or provide a compatible --ckpt-{stage_label}."
            )
        logger.info(
            f"Stage {stage_label} metadata validated | active_experts={saved_experts}"
        )
    else:
        logger.warning(
            f"Stage {stage_label} checkpoint has no 'active_experts' metadata — "
            f"skipping expert config validation. Ensure checkpoints match current run."
        )

    # v1.3: config_hash drift check
    if 'config_hash' in meta:
        current_hash = config_hash()
        if meta['config_hash'] != current_hash:
            logger.error(
                f"Config hash mismatch on Stage {stage_label} checkpoint | "
                f"saved={meta['config_hash']} | current={current_hash} | "
                f"Re-run from Stage A with current config."
            )
            raise ValueError(f"Config hash mismatch on Stage {stage_label} checkpoint")
    else:
        logger.warning(
            f"Stage {stage_label} checkpoint has no config_hash — "
            f"skipping config drift check (pre-v1.3 checkpoint)"
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(
    active_experts: list,
    hidden_dim: int,
    num_layers: int,
    latent_dim: int,
    logger: logging.Logger,
) -> CSMF:
    """
    Build CSMF from active expert names.

    Args:
        active_experts: list of strings from EXPERT_REGISTRY keys
        hidden_dim:     conditioner output dim (= cond_dim for experts)
        num_layers:     number of flow layers per expert
        latent_dim:     data/latent dimensionality
        logger:         logger instance

    Returns:
        CSMF model
    """
    # Validate expert names
    unknown = [n for n in active_experts if n not in EXPERT_REGISTRY]
    if unknown:
        logger.error(
            f"Unknown expert(s): {unknown} | "
            f"Valid options: {list(EXPERT_REGISTRY.keys())}"
        )
        raise KeyError(
            f"Unknown expert(s): {unknown}. "
            f"Valid: {list(EXPERT_REGISTRY.keys())}"
        )

    # Instantiate experts
    experts = []
    for name in active_experts:
        cls = EXPERT_REGISTRY[name]
        try:
            if name == "realnvp":
                # ConditionalRealNVP uses h_dim; hardcodes 28x28 MNIST dims internally
                expert = cls(h_dim=hidden_dim)
                expert.dim = latent_dim  # required by CSMF for gate/sampling
            else:
                # ConditionalMAF, ConditionalNICE, ConditionalNSF use dim + cond_dim
                expert = cls(dim=latent_dim, cond_dim=hidden_dim)
        except TypeError as e:
            logger.error(
                f"Failed to instantiate expert '{name}' | {e}"
            )
            raise
        experts.append(expert)
        logger.info(
            f"Expert added: '{name}' ({cls.__name__}) | "
            f"dim={latent_dim} | cond_dim/h_dim={hidden_dim}"
        )

    K = len(experts)

    # Conditioner: y -> h
    conditioner = MNISTConditioner(h_dim=hidden_dim)
    logger.info(f"Conditioner: MNISTConditioner | h_dim={hidden_dim}")

    # Gate: h -> logits (K,)
    gate = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Linear(hidden_dim // 2, K),
    )
    logger.info(f"Gate: Linear({hidden_dim}) -> ReLU -> Linear({K})")

    model = CSMF(experts, conditioner, gate)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"CSMF built | K={K} | total params={n_params:,}")
    return model


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def build_loss(
    downsample_factor: int,
    lambda_cons: float,
    lambda_trans: float,
    lambda_cal: float,
    logger: logging.Logger,
) -> HybridLoss:
    """Build HybridLoss with SR forward model."""
    fwd_model = SRForwardModel(blur_sigma=1.0, downsample_factor=downsample_factor)
    loss_fn   = HybridLoss(
        fwd_model,
        lambda_cons=lambda_cons,
        lambda_trans=lambda_trans,
        lambda_cal=lambda_cal,
    )
    logger.info(
        f"HybridLoss | lambda_cons={lambda_cons} | "
        f"lambda_trans={lambda_trans} | lambda_cal={lambda_cal}"
    )
    return loss_fn


# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_final(
    model: CSMF,
    hybrid_loss: HybridLoss,
    test_loader: DataLoader,
    active_experts: list,
    results_dir: str,
    logger: logging.Logger,
) -> dict:
    """
    Full evaluation on test set after Stage C.

    Computes:
        - Mixture NLL
        - Per-expert NLL
        - Mean gate weights per expert (usage stats)
        - Mean Neff
        - Measurement residual ||Ax - y||

    Saves results to results_dir/train_csmf_final_eval.json.

    Returns:
        eval_dict
    """
    model.eval()

    total_nll      = 0.0
    per_expert_nll = torch.zeros(model.K)
    total_neff     = 0.0
    total_residual = 0.0
    mean_w_accum   = torch.zeros(model.K)
    n_batches      = 0

    for x_clean, y_deg in test_loader:
        x_clean = x_clean.to(next(model.parameters()).device)
        y_deg   = y_deg.to(next(model.parameters()).device)
        # Mixture NLL
        log_q, log_q_experts = model.forward(x_clean, y_deg)
        nll = -log_q.mean()
        if torch.isnan(nll):
            logger.warning("eval_final: NaN NLL in test batch — skipping")
            continue

        # Per-expert NLL
        for k in range(model.K):
            per_expert_nll[k] += -log_q_experts[:, k].mean().item()

        # Gate stats
        h      = model.conditioner(y_deg)
        logits = model.gate(h)
        w      = torch.softmax(logits, dim=1)           # (B, K)
        neff   = model._compute_neff(w).mean().item()
        mean_w_accum += w.mean(dim=0).cpu()

        # Residual ||Ax - y||
        x_samples, _ = model.sample(y_deg, num_samples=1)
        x_hat         = x_samples[:, 0, :]             # (B, d)
        x_hat_4d      = x_hat.view(x_hat.shape[0], 1, 28, 28) # reshape for SR model
        #Ax            = hybrid_loss.A.forward(x_hat)
        Ax            = hybrid_loss.A.forward(x_hat_4d) 
        residual      = torch.mean((Ax - y_deg) ** 2).item()

        total_nll      += nll.item()
        total_neff     += neff
        total_residual += residual
        n_batches      += 1

    if n_batches == 0:
        logger.error("eval_final: all test batches had NaN — evaluation failed")
        return {}

    avg_nll      = total_nll      / n_batches
    avg_neff     = total_neff     / n_batches
    avg_residual = total_residual / n_batches
    avg_w        = (mean_w_accum  / n_batches).tolist()
    avg_per_nll  = (per_expert_nll / n_batches).tolist()

    # Build result dict
    eval_dict = {
        "timestamp":       datetime.now().isoformat(),
        "active_experts":  active_experts,
        "K":               model.K,
        "nll":             round(avg_nll,      6),
        "neff":            round(avg_neff,     4),
        "residual":        round(avg_residual, 6),
        "per_expert_nll":  {
            name: round(nll, 6)
            for name, nll in zip(active_experts, avg_per_nll)
        },
        "gate_usage_mean_w": {
            name: round(w, 4)
            for name, w in zip(active_experts, avg_w)
        },
    }

    # Log summary
    logger.info(f"=== Final Eval ===")
    logger.info(f"  NLL      : {avg_nll:.4f}")
    logger.info(f"  Neff     : {avg_neff:.3f}")
    logger.info(f"  Residual : {avg_residual:.6f}")
    logger.info(f"  Per-expert NLL  : {eval_dict['per_expert_nll']}")
    logger.info(f"  Gate usage w_k  : {eval_dict['gate_usage_mean_w']}")

    # Save JSON
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, "train_csmf_final_eval.json")
    with open(save_path, "w") as f:
        json.dump(eval_dict, f, indent=2)
    logger.info(f"Final eval saved: {save_path}")

    model.train()
    return eval_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Parse args ---
    args = parse_args()

    # --- Apply CLI overrides ---
    lr          = args.lr      or LR
    epochs      = args.epochs  or EPOCHS
    batch_size  = args.batch   or BATCH_SIZE
    seed        = args.seed    or SEED
    ckpt_dir    = args.ckpt_dir    or CKPT_DIR
    results_dir = args.results_dir or RESULTS_DIR
    experts_cfg = args.experts     or ACTIVE_EXPERTS
    stages      = args.stages
    preprocessed_dir = args.preprocessed_dir or PREPROCESSED_DIR  # v1.3

    # --- Logging ---
    log_path = os.path.join(results_dir, "train_csmf.log")
    logger   = setup_logging(log_path)
    logger.info("=" * 60)
    logger.info("CSMF Training | WP3.2-TrainMain-v1.3 | TRAIN-MAIN")
    logger.info("=" * 60)

    # --- Config summary ---
    cfg_summary = {
        "lr": lr, "epochs": epochs, "batch_size": batch_size,
        "seed": seed, "stages": stages, "active_experts": experts_cfg,
        "ckpt_dir": ckpt_dir, "results_dir": results_dir,
        "latent_dim": LATENT_DIM, "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS, "lambda_cons": LAMBDA_CONS,
        "lambda_trans": LAMBDA_TRANS, "lambda_cal": LAMBDA_CAL,
    }
    logger.info(f"Config: {json.dumps(cfg_summary, indent=2)}")

    # --- Seed ---
    fix_seed(seed)
    logger.info(f"Seed fixed: {seed}")

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}" + (
        f" | {torch.cuda.get_device_name(0)}" if device.type == "cuda" else " (CPU — no GPU detected)"
    ))

    # --- Dirs ---
    os.makedirs(ckpt_dir,    exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # --- Resolve per-stage checkpoint paths ---
    # CLI --ckpt-A/B/C override; otherwise fall back to default paths in ckpt_dir
    ckpt_path_A = args.ckpt_A or os.path.join(ckpt_dir, "csmf_stage_A.pth")
    ckpt_path_B = args.ckpt_B or os.path.join(ckpt_dir, "csmf_stage_B.pth")
    ckpt_path_C = args.ckpt_C or os.path.join(ckpt_dir, "csmf_stage_C.pth")
    logger.info(
        f"Stage checkpoint paths | A={ckpt_path_A} | B={ckpt_path_B} | C={ckpt_path_C}"
    )

    # --- Data ---
    config_params = {
        'blur_kernel_size':  BLUR_KERNEL,
        'blur_sigma':        BLUR_SIGMA,
        'downsample_factor': DOWNSAMPLE_FACTOR,
        'noise_std':         NOISE_SIGMA,
        'normalize':         '[0,1]',
        'val_split':         VAL_SPLIT,
        'seed':              seed,
    }
    train_loader, val_loader, test_loader = create_precomputed_dataloaders(
        preprocessed_dir = preprocessed_dir,
        batch_size       = batch_size,
        config_params    = config_params,   # validates metadata.json on load
    )

    # --- Model ---
    model = build_model(
        active_experts = experts_cfg,
        hidden_dim     = HIDDEN_DIM,
        num_layers     = NUM_LAYERS,
        latent_dim     = LATENT_DIM,
        logger         = logger,
    )
    model = model.to(device)
    logger.info(f"Model moved to {device}")

    # --- General resume (overrides all stages if provided) ---
    if args.resume:
        try:
            meta = model.load_checkpoint(args.resume)
            logger.info(f"Resumed from: {args.resume} | meta={meta}")
        except FileNotFoundError as e:
            logger.error(f"Resume failed: {e}")
            raise

    # --- Loss ---
    hybrid_loss = build_loss(
        downsample_factor = DOWNSAMPLE_FACTOR,
        lambda_cons       = LAMBDA_CONS,
        lambda_trans      = LAMBDA_TRANS,
        lambda_cal        = LAMBDA_CAL,
        logger            = logger,
    )
    # Move forward model (SR kernel) to same device as model
    try:
        hybrid_loss.A = hybrid_loss.A.to(device)
        logger.info(f"SRForwardModel moved to {device}")
    except Exception as e:
        logger.error(f"Failed to move hybrid_loss.A to {device}: {e}")
        raise
    if isinstance(hybrid_loss, nn.Module):
        hybrid_loss = hybrid_loss.to(device)
        logger.info(f"HybridLoss moved to {device}")

    epochs_per_stage = epochs // 3

    # =========================================================================
    # Stage A — Expert training
    # =========================================================================
    if "A" in stages:
        logger.info("=" * 40)
        logger.info("Stage A: Expert Training")
        logger.info("=" * 40)
        try:
            # v1.3: per-expert optimizer_fn callable; supports --expert-lr per expert
            if args.expert_lr:
                expert_lr_map = dict(pair.split('=') for pair in args.expert_lr)
                optimizer_fn = lambda expert: torch.optim.Adam(
                    expert.parameters(),
                    lr=float(expert_lr_map.get(
                        type(expert).__name__.replace('Conditional', '').lower(), lr
                    ))
                )
            else:
                optimizer_fn = lambda expert: torch.optim.Adam(
                    expert.parameters(), lr=lr
                )
            epoch_logs = model.train_stage_A(
                dataloader   = train_loader,
                optimizer_fn = optimizer_fn,
                hybrid_loss  = hybrid_loss,
                epochs       = epochs_per_stage,
                lambda_cons  = LAMBDA_CONS,
                val_loader   = val_loader,
                patience     = PATIENCE,
                ckpt_dir     = ckpt_dir,
                fwd_model    = hybrid_loss.A,
            )
            logger.info("Stage A complete.")

            # v1.6: EXP-SANITY — diagnostic checks and plots after Stage A
            if not args.skip_sanity and val_loader is not None:
                try:
                    sanity_dir = os.path.join(results_dir, "expert_sanity")
                    sanity_summary = run_expert_sanity(
                        csmf_model     = model,
                        val_loader     = val_loader,
                        fwd_model      = hybrid_loss.A,
                        epoch_logs     = epoch_logs,
                        output_dir     = sanity_dir,
                        plots          = ["1", "2", "3", "A", "D", "F"],
                    )
                    logger.info(f"EXP-SANITY complete | summary: {sanity_summary}")
                except ValueError as e:
                    logger.error(f"EXP-SANITY fatal check failed — aborting: {e}")
                    raise
                except Exception as e:
                    logger.warning(f"EXP-SANITY non-fatal error (continuing): {e}")
            elif args.skip_sanity:
                logger.info("EXP-SANITY skipped (--skip-sanity)")
            else:
                logger.info("EXP-SANITY skipped (no val_loader)")

        except Exception as e:
            logger.error(f"Stage A failed: {e}")
            raise
    else:
        logger.info("Stage A skipped — loading checkpoint for downstream stages.")
        # Auto-load Stage A checkpoint so experts are properly initialised
        # before Stage B (gate training) can begin.
        if not args.resume:
            load_stage_checkpoint(
                model       = model,
                ckpt_path   = ckpt_path_A,
                stage_label = "A",
                experts_cfg = experts_cfg,
                logger      = logger,
            )
        else:
            logger.info(
                "Stage A auto-load skipped — general --resume checkpoint already loaded."
            )
        # Warn if any expert still has trainable params (should be frozen in saved ckpt)
        for k, expert in enumerate(model.experts):
            n = sum(p.requires_grad for p in expert.parameters())
            if n > 0:
                logger.warning(
                    f"After loading Stage A ckpt, expert {k} has {n} trainable params — "
                    f"Stage A checkpoint may not have frozen experts correctly."
                )

    # =========================================================================
    # Stage B — Gate training
    # =========================================================================
    if "B" in stages:
        logger.info("=" * 40)
        logger.info("Stage B: Gate Training")
        logger.info("=" * 40)
        try:
            optimizer_B = torch.optim.Adam(
                model.gate.parameters(),
                lr=lr / 10,
            )
            model.train_stage_B(
                dataloader  = train_loader,
                optimizer   = optimizer_B,
                hybrid_loss = hybrid_loss,
                epochs      = epochs_per_stage,
                val_loader  = val_loader,
                patience    = PATIENCE,
                ckpt_path   = ckpt_path_B,
            )
            logger.info("Stage B complete.")
        except Exception as e:
            logger.error(f"Stage B failed: {e}")
            raise
    else:
        logger.info("Stage B skipped — loading checkpoint for downstream stages.")
        # Auto-load Stage B checkpoint so gate weights are properly initialised
        # before Stage C (joint fine-tuning) can begin.
        if not args.resume:
            load_stage_checkpoint(
                model       = model,
                ckpt_path   = ckpt_path_B,
                stage_label = "B",
                experts_cfg = experts_cfg,
                logger      = logger,
            )
        else:
            logger.info(
                "Stage B auto-load skipped — general --resume checkpoint already loaded."
            )

    # =========================================================================
    # Stage C — Joint fine-tuning
    # =========================================================================
    if "C" in stages:
        logger.info("=" * 40)
        logger.info("Stage C: Joint Fine-Tuning")
        logger.info("=" * 40)
        try:
            optimizer_C = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad],
                lr=lr / 100,
            )
            model.train_stage_C(
                dataloader         = train_loader,
                optimizer          = optimizer_C,
                hybrid_loss        = hybrid_loss,
                epochs             = epochs_per_stage,
                blocks_to_unfreeze = BLOCKS_TO_UNFREEZE,
                tau_start          = TAU_START,
                tau_end            = TAU_END,
                val_loader         = val_loader,
                patience           = PATIENCE,
                ckpt_path          = ckpt_path_C,
            )
            logger.info("Stage C complete.")
        except Exception as e:
            logger.error(f"Stage C failed: {e}")
            raise
    else:
        logger.info("Stage C skipped (not in --stages)")

    # =========================================================================
    # Final evaluation
    # =========================================================================
    logger.info("=" * 40)
    logger.info("Final Evaluation on Test Set")
    logger.info("=" * 40)
    try:
        eval_dict = eval_final(
            model          = model,
            hybrid_loss    = hybrid_loss,
            test_loader    = test_loader,
            active_experts = experts_cfg,
            results_dir    = results_dir,
            logger         = logger,
        )
        if eval_dict:
            logger.info("Training and evaluation complete.")
        else:
            logger.error("Final evaluation returned empty result.")
    except Exception as e:
        logger.error(f"Final evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
