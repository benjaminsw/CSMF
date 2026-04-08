# Version: WP2.1-HybridLoss-v1.8
# Abbr: HYBRID
# File: csmf/losses/hybrid_loss.py
# Description: Hybrid training objective for CSMF
#              L = NLL + λ_cons·‖Ax−y‖² + λ_trans·SW2 + λ_cal·ES
# Changelog:
#   v1.8 (2026-02-28): BUG FIX — x_sample [B,784] passed to SRForwardModel which requires
#                      4D [B,1,28,28]; added x_sample_4d = x_sample.view(B,1,28,28) before
#                      self.A.forward(); flat x_sample still used for SW2/ES calculations
#   v1.7 (2026-02-28): BUG FIX — flow.sample() returns 2-tuple; fixed all 3 call sites
#   v1.1 - Added three-stage training schedule (StageConfig + run_stage_A/B/C)
#   v1.1 - Added gate freeze/unfreeze helpers (freeze_experts, unfreeze_last_blocks)
#   v1.1 - Added checkpoint save/load per stage
#   v1.0 - Core HybridLoss: NLL + consistency + SW2 + ES with lambda annealing
# =============================================================================

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from .sliced_wasserstein import sliced_wasserstein_distance
from .calibration import energy_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage Configuration Dataclass
# ---------------------------------------------------------------------------

@dataclass
class StageConfig:
    """
    Configuration for one training stage (A, B, or C).

    Stage A — experts only:   NLL + weak consistency, gate frozen to uniform
    Stage B — gate only:      full hybrid loss, experts frozen, τ≈1.1, top-k=2-3
    Stage C — joint finetune: unfreeze last n_unfreeze_blocks per expert + gate,
                               small LR, consistency active, early stop on val NLL+residual
    """
    name: str                          # "A", "B", or "C"
    max_epochs: int                    # maximum epochs for this stage
    lr: float                          # learning rate
    lambda_cons: float = 0.05          # consistency weight
    lambda_trans: float = 0.0          # transport (SW2) weight
    lambda_cal: float = 0.0            # calibration (ES) weight
    anneal_schedule: dict = field(default_factory=dict)
    # Stage B / C extras
    gate_temperature: float = 1.1      # τ for gate softmax
    topk: Optional[int] = None         # top-k masking for gate (None = soft)
    n_unfreeze_blocks: int = 0         # Stage C: blocks to unfreeze per expert
    # Early stopping
    early_stop_patience: int = 10      # epochs without improvement before stopping
    early_stop_metric: str = "nll"     # "nll" | "residual" | "nll+residual"


# ---------------------------------------------------------------------------
# Core: HybridLoss Module
# ---------------------------------------------------------------------------

class HybridLoss(nn.Module):
    """
    Hybrid objective for CSMF training.

    L = NLL + λ_cons·‖Ax−y‖² + λ_trans·SW2(x_samples, x_clean) + λ_cal·ES

    Args:
        forward_model: physics forward operator A (must implement .forward(x))
        lambda_cons:   weight for measurement consistency term
        lambda_trans:  weight for transport (SW2) term
        lambda_cal:    weight for calibration (ES) term
        anneal_schedule: dict of per-term annealing params
                         e.g. {'cons': {'warmup': 5, 'rampup': 20}}
        n_sw2_samples: number of samples drawn for SW2 transport term
        sw2_projections: number of random projections for SW2
    """

    def __init__(
        self,
        forward_model: nn.Module,
        lambda_cons: float = 0.1,
        lambda_trans: float = 0.05,
        lambda_cal: float = 0.01,
        anneal_schedule: Optional[dict] = None,
        n_sw2_samples: int = 4,
        sw2_projections: int = 32,

        
    ):
        super().__init__()
        self.A = forward_model
        self.lambda_cons = lambda_cons
        self.lambda_trans = lambda_trans
        self.lambda_cal = lambda_cal
        self.anneal_schedule = anneal_schedule or {}
        self.n_sw2_samples = n_sw2_samples
        self.sw2_projections = sw2_projections
        self.sw2_every = 10            # compute SW2 every N batches
        self._batch_counter = 0        # internal counter



    # ------------------------------------------------------------------
    def forward(
        self,
        flow: nn.Module,
        x_clean: torch.Tensor,
        y_degraded: torch.Tensor,
        epoch: int = 0,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute hybrid loss for one batch.

        Args:
            flow:        conditional flow model with .conditioner(), .forward(),
                         .base_log_prob(), .sample()
            x_clean:     (B, d)  clean ground-truth samples
            y_degraded:  (B, d') degraded observations
            epoch:       current training epoch (used for annealing)

        Returns:
            loss:      scalar total loss
            loss_dict: dict with individual component values (detached)
        """
        B = x_clean.shape[0]
        # Do NOT flatten x_clean — CSMF handles per-expert flattening internally via
        # _prepare_x_for_expert(). RealNVP needs [B,1,28,28]; MAF/NSF/NICE need [B,784].
        # Only flatten here for SW2/ES shape calculations (d = flat dim).
        d = x_clean.flatten(1).shape[1]   # 784 — used for SW2 ref_flat and energy_score only

        # Anneal weights
        lam_cons  = self._anneal(self.lambda_cons,  epoch, "cons")
        lam_trans = self._anneal(self.lambda_trans, epoch, "trans")
        lam_cal   = self._anneal(self.lambda_cal,   epoch, "cal")

        # ---- 1. NLL -------------------------------------------------------
        # CSMF.forward() returns (log_q [B,], log_q_experts [B,K])
        # log_q is already the mixture log-probability log q(x|y) — use directly
        log_q, _ = flow.forward(x_clean, y_degraded)

        if torch.any(torch.isnan(log_q)) or torch.any(torch.isinf(log_q)):
            logger.error(
                "HybridLoss: NaN/Inf in log_q at epoch %d — "
                "NaN count: %d, Inf count: %d",
                epoch,
                torch.isnan(log_q).sum().item(),
                torch.isinf(log_q).sum().item(),
            )
            raise RuntimeError("NaN/Inf in log_q — check flow numerical stability")

        nll = -log_q.mean()

        if torch.isnan(nll):
            logger.error("HybridLoss: NaN in NLL at epoch %d", epoch)
            raise RuntimeError("NaN in NLL loss")

        # ---- 2. Consistency: ‖A(x) − y‖² --------------------------------
        # CSMF.sample() returns (x_samples [B,S,d], expert_ids [B,S]) — unpack first
        if lam_cons > 0:
            x_sample, _ = flow.sample(y_degraded, num_samples=1)
            x_sample = x_sample.squeeze(1)
            x_sample_4d = x_sample.view(B, 1, 28, 28)
            Ax = self.A.forward(x_sample_4d)
            consistency = torch.mean((Ax - y_degraded) ** 2)

            if torch.isnan(consistency):
                logger.error("HybridLoss: NaN in consistency at epoch %d", epoch)
                raise RuntimeError("NaN in consistency loss")
        else:
            consistency = torch.tensor(0.0, device=x_clean.device)

        if torch.isnan(consistency):
            logger.error("HybridLoss: NaN in consistency at epoch %d", epoch)
            raise RuntimeError("NaN in consistency loss")

        # ---- 3. Transport: SW2(samples, x_clean) -------------------------
        #if lam_trans > 0:
        self._batch_counter += 1
        if lam_trans > 0 and (self._batch_counter % self.sw2_every == 0):
            x_multi, _ = flow.sample(y_degraded, num_samples=self.n_sw2_samples)  # (B, S, d)
            x_flat    = x_multi.reshape(-1, d)                           # (B*S, d)
            ref_flat  = x_clean.flatten(1).repeat(self.n_sw2_samples, 1)  # (B*S, d)
            transport = sliced_wasserstein_distance(
                x_flat, ref_flat, num_projections=self.sw2_projections
            )
            if torch.isnan(transport):
                logger.error("HybridLoss: NaN in SW2 transport at epoch %d", epoch)
                raise RuntimeError("NaN in SW2 transport loss")
        else:
            transport = torch.tensor(0.0, device=x_clean.device)

        # ---- 4. Calibration: Energy Score (mean over batch) --------------
        if lam_cal > 0:
            x_multi_cal = (
                x_multi if lam_trans > 0
                else flow.sample(y_degraded, num_samples=self.n_sw2_samples)[0]
            )                                                              # (B, S, d)
            cal_terms = []
            for i in range(B):
                es = energy_score(x_multi_cal[i], x_clean.flatten(1)[i])  # scalar
                cal_terms.append(es)
            calibration = torch.stack(cal_terms).mean()

            if torch.isnan(calibration):
                logger.error("HybridLoss: NaN in calibration (ES) at epoch %d", epoch)
                raise RuntimeError("NaN in calibration loss")
        else:
            calibration = torch.tensor(0.0, device=x_clean.device)

        # ---- Total --------------------------------------------------------
        loss = (
            nll
            + lam_cons  * consistency
            + lam_trans * transport
            + lam_cal   * calibration
        )

        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(
                "HybridLoss: total loss is NaN/Inf at epoch %d — "
                "nll=%.4f, cons=%.4f, trans=%.4f, cal=%.4f",
                epoch, nll.item(), consistency.item(),
                transport.item(), calibration.item(),
            )
            raise RuntimeError("NaN/Inf in total hybrid loss")

        loss_dict = {
            "loss":        loss.item(),
            "nll":         nll.item(),
            "consistency": consistency.item(),
            "transport":   transport.item(),
            "calibration": calibration.item(),
            "lam_cons":    lam_cons,
            "lam_trans":   lam_trans,
            "lam_cal":     lam_cal,
        }

        return loss, loss_dict

    # ------------------------------------------------------------------
    def _anneal(self, lam: float, epoch: int, name: str) -> float:
        """Linear annealing: warmup (0) → rampup (linear) → full λ."""
        if name not in self.anneal_schedule:
            return lam
        sched   = self.anneal_schedule[name]
        warmup  = sched.get("warmup", 0)
        rampup  = sched.get("rampup", warmup + 1)
        if epoch < warmup:
            return 0.0
        if epoch < rampup:
            return lam * (epoch - warmup) / max(rampup - warmup, 1)
        return lam


# ---------------------------------------------------------------------------
# [Additional / FATAL] Gate Freeze / Unfreeze Helpers
# ---------------------------------------------------------------------------

def freeze_experts(model: nn.Module) -> None:
    """
    Stage B: freeze all expert flow parameters.
    Expects model to have a .experts attribute (list/ModuleList of flows).
    """
    if not hasattr(model, "experts"):
        logger.error("freeze_experts: model has no 'experts' attribute")
        raise AttributeError("model must have an 'experts' attribute (ModuleList)")

    for k, expert in enumerate(model.experts):
        for param in expert.parameters():
            param.requires_grad_(False)
        logger.info("freeze_experts: expert %d frozen (%d params)",
                    k, sum(p.numel() for p in expert.parameters()))


def unfreeze_experts(model: nn.Module) -> None:
    """Fully unfreeze all expert parameters (used when entering Stage C full)."""
    if not hasattr(model, "experts"):
        logger.error("unfreeze_experts: model has no 'experts' attribute")
        raise AttributeError("model must have an 'experts' attribute (ModuleList)")

    for k, expert in enumerate(model.experts):
        for param in expert.parameters():
            param.requires_grad_(True)
        logger.info("unfreeze_experts: expert %d fully unfrozen", k)


def unfreeze_last_blocks(model: nn.Module, n_blocks: int = 1) -> None:
    """
    Stage C: unfreeze only the last n_blocks coupling/flow blocks per expert.
    Expects each expert to have a .blocks attribute (list/ModuleList).
    All other expert params remain frozen.
    """
    if not hasattr(model, "experts"):
        logger.error("unfreeze_last_blocks: model has no 'experts' attribute")
        raise AttributeError("model must have an 'experts' attribute (ModuleList)")

    for k, expert in enumerate(model.experts):
        if not hasattr(expert, "blocks"):
            logger.error(
                "unfreeze_last_blocks: expert %d has no 'blocks' attribute", k
            )
            raise AttributeError(f"expert {k} must have a 'blocks' attribute (ModuleList)")

        blocks = list(expert.blocks)
        n_total = len(blocks)
        
        # ✅ NEW: if asking for >= all blocks, unfreeze the whole expert
        if n_blocks >= n_total:
            for param in expert.parameters():
                param.requires_grad_(True)
            logger.info(
                "unfreeze_last_blocks: expert %d — full unfreeze (%d blocks)",
                k, n_total
            )
            continue
        
        unfreeze_idx = max(0, n_total - n_blocks)

        for i, block in enumerate(blocks):
            requires_grad = i >= unfreeze_idx
            for param in block.parameters():
                param.requires_grad_(requires_grad)

        logger.info(
            "unfreeze_last_blocks: expert %d — unfroze last %d of %d blocks",
            k, min(n_blocks, n_total), n_total,
        )


def freeze_gate(model: nn.Module) -> None:
    """Freeze gate network parameters (used in Stage A)."""
    if not hasattr(model, "gate"):
        logger.error("freeze_gate: model has no 'gate' attribute")
        raise AttributeError("model must have a 'gate' attribute")
    for param in model.gate.parameters():
        param.requires_grad_(False)
    logger.info("freeze_gate: gate frozen (%d params)",
                sum(p.numel() for p in model.gate.parameters()))


def unfreeze_gate(model: nn.Module) -> None:
    """Unfreeze gate network parameters (used in Stage B/C)."""
    if not hasattr(model, "gate"):
        logger.error("unfreeze_gate: model has no 'gate' attribute")
        raise AttributeError("model must have a 'gate' attribute")
    for param in model.gate.parameters():
        param.requires_grad_(True)
    logger.info("unfreeze_gate: gate unfrozen (%d params)",
                sum(p.numel() for p in model.gate.parameters()))


# ---------------------------------------------------------------------------
# [Additional / FATAL] Checkpoint Save / Load
# ---------------------------------------------------------------------------

def save_checkpoint(
    stage: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_dict: dict,
    checkpoint_dir: str,
) -> str:
    """
    Save a per-stage checkpoint.

    Args:
        stage:          "A", "B", or "C"
        epoch:          current epoch number
        model:          CSMF model
        optimizer:      stage optimizer
        loss_dict:      latest loss component values
        checkpoint_dir: directory to save into

    Returns:
        path: full path of the saved checkpoint file
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = f"stage_{stage}_epoch_{epoch:04d}.pt"
    path = os.path.join(checkpoint_dir, filename)

    state = {
        "stage":      stage,
        "epoch":      epoch,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "loss_dict":  loss_dict,
    }

    try:
        torch.save(state, path)
        logger.info("save_checkpoint: Stage %s epoch %d → %s", stage, epoch, path)
    except Exception as exc:
        logger.error("save_checkpoint: failed to save to %s — %s", path, exc)
        raise

    return path


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    """
    Load a stage checkpoint into model (and optionally optimizer).

    Args:
        path:      full path to .pt checkpoint file
        model:     CSMF model (weights loaded in-place)
        optimizer: if provided, optimizer state is also restored

    Returns:
        state dict (contains 'stage', 'epoch', 'loss_dict')
    """
    if not os.path.isfile(path):
        logger.error("load_checkpoint: file not found — %s", path)
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state["model"])
        if optimizer is not None:
            optimizer.load_state_dict(state["optimizer"])
        logger.info(
            "load_checkpoint: Stage %s epoch %d loaded from %s",
            state.get("stage", "?"), state.get("epoch", -1), path,
        )
    except Exception as exc:
        logger.error("load_checkpoint: failed to load from %s — %s", path, exc)
        raise

    return state


# ---------------------------------------------------------------------------
# [Additional / FATAL] Three-Stage Training Runners
# ---------------------------------------------------------------------------

def run_stage_A(
    model: nn.Module,
    loss_fn: HybridLoss,
    train_loader,
    cfg: StageConfig,
    checkpoint_dir: str,
    device: torch.device,
) -> dict:
    """
    Stage A — Train each expert independently with NLL + weak consistency.
    Gate is frozen to uniform. Transport and calibration terms are off.

    Args:
        model:           CSMF model with .experts and .gate
        loss_fn:         HybridLoss instance (λ_trans and λ_cal ignored here)
        train_loader:    DataLoader yielding (x_clean, y_degraded) batches
        cfg:             StageConfig for Stage A
        checkpoint_dir:  directory for checkpoints
        device:          torch device

    Returns:
        dict with 'best_epoch', 'best_nll', 'last_loss_dict'
    """
    logger.info("=== Stage A: Expert-only training (max %d epochs) ===", cfg.max_epochs)

    freeze_gate(model)
    unfreeze_experts(model)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr
    )

    best_nll = float("inf")
    best_epoch = 0
    no_improve = 0
    last_loss_dict = {}

    for epoch in range(cfg.max_epochs):
        model.train()
        epoch_losses = _run_epoch(
            model, loss_fn, train_loader, optimizer, epoch, device,
            override_lambdas={"cons": cfg.lambda_cons, "trans": 0.0, "cal": 0.0},
        )

        nll = epoch_losses["nll"]
        logger.info("Stage A | epoch %d | nll=%.4f | cons=%.4f",
                    epoch, nll, epoch_losses["consistency"])

        # Early stopping on NLL
        if nll < best_nll - 1e-5:
            best_nll = nll
            best_epoch = epoch
            no_improve = 0
            save_checkpoint("A", epoch, model, optimizer, epoch_losses, checkpoint_dir)
        else:
            no_improve += 1

        last_loss_dict = epoch_losses

        if no_improve >= cfg.early_stop_patience:
            logger.info("Stage A: early stop at epoch %d (patience=%d)",
                        epoch, cfg.early_stop_patience)
            break

    logger.info("Stage A complete. Best NLL=%.4f at epoch %d", best_nll, best_epoch)
    return {"best_epoch": best_epoch, "best_nll": best_nll, "last_loss_dict": last_loss_dict}


def run_stage_B(
    model: nn.Module,
    loss_fn: HybridLoss,
    train_loader,
    cfg: StageConfig,
    checkpoint_dir: str,
    device: torch.device,
    stage_A_checkpoint: Optional[str] = None,
) -> dict:
    """
    Stage B — Freeze experts, train gate with full hybrid loss.
    Temperature τ≈1.1 and optional top-k masking applied to gate.

    Args:
        model:                CSMF model
        loss_fn:              HybridLoss instance
        train_loader:         DataLoader
        cfg:                  StageConfig for Stage B
        checkpoint_dir:       directory for checkpoints
        device:               torch device
        stage_A_checkpoint:   optional path to best Stage A checkpoint to load

    Returns:
        dict with 'best_epoch', 'best_metric', 'last_loss_dict'
    """
    logger.info("=== Stage B: Gate-only training (max %d epochs) ===", cfg.max_epochs)

    if stage_A_checkpoint:
        load_checkpoint(stage_A_checkpoint, model)

    freeze_experts(model)
    unfreeze_gate(model)

    # Apply gate temperature and top-k if model supports it
    if hasattr(model, "gate"):
        if hasattr(model.gate, "temperature"):
            model.gate.temperature = cfg.gate_temperature
            logger.info("Stage B: gate temperature set to %.2f", cfg.gate_temperature)
        if cfg.topk is not None and hasattr(model.gate, "topk"):
            model.gate.topk = cfg.topk
            logger.info("Stage B: gate top-k set to %d", cfg.topk)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr
    )

    best_metric = float("inf")
    best_epoch = 0
    no_improve = 0
    last_loss_dict = {}

    for epoch in range(cfg.max_epochs):
        model.train()
        epoch_losses = _run_epoch(
            model, loss_fn, train_loader, optimizer, epoch, device,
            override_lambdas={
                "cons":  cfg.lambda_cons,
                "trans": cfg.lambda_trans,
                "cal":   cfg.lambda_cal,
            },
        )

        metric = _early_stop_metric(epoch_losses, cfg.early_stop_metric)
        logger.info(
            "Stage B | epoch %d | loss=%.4f | nll=%.4f | cons=%.4f | trans=%.4f | cal=%.4f",
            epoch, epoch_losses["loss"], epoch_losses["nll"],
            epoch_losses["consistency"], epoch_losses["transport"],
            epoch_losses["calibration"],
        )

        if metric < best_metric - 1e-5:
            best_metric = metric
            best_epoch = epoch
            no_improve = 0
            save_checkpoint("B", epoch, model, optimizer, epoch_losses, checkpoint_dir)
        else:
            no_improve += 1

        last_loss_dict = epoch_losses

        if no_improve >= cfg.early_stop_patience:
            logger.info("Stage B: early stop at epoch %d (patience=%d)",
                        epoch, cfg.early_stop_patience)
            break

    logger.info("Stage B complete. Best metric=%.4f at epoch %d", best_metric, best_epoch)
    return {"best_epoch": best_epoch, "best_metric": best_metric, "last_loss_dict": last_loss_dict}


def run_stage_C(
    model: nn.Module,
    loss_fn: HybridLoss,
    train_loader,
    cfg: StageConfig,
    checkpoint_dir: str,
    device: torch.device,
    stage_B_checkpoint: Optional[str] = None,
) -> dict:
    """
    Stage C — Light joint fine-tuning.
    Unfreeze last n_unfreeze_blocks per expert + gate. Small LR.
    Early stop on combined NLL + residual.

    Args:
        model:                CSMF model
        loss_fn:              HybridLoss instance
        train_loader:         DataLoader
        cfg:                  StageConfig for Stage C
        checkpoint_dir:       directory for checkpoints
        device:               torch device
        stage_B_checkpoint:   optional path to best Stage B checkpoint to load

    Returns:
        dict with 'best_epoch', 'best_metric', 'last_loss_dict'
    """
    logger.info("=== Stage C: Joint fine-tuning (max %d epochs, LR=%.1e) ===",
                cfg.max_epochs, cfg.lr)

    if stage_B_checkpoint:
        load_checkpoint(stage_B_checkpoint, model)

    freeze_experts(model)                                  # re-freeze first
    unfreeze_last_blocks(model, n_blocks=cfg.n_unfreeze_blocks)
    unfreeze_gate(model)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr
    )

    best_metric = float("inf")
    best_epoch = 0
    no_improve = 0
    last_loss_dict = {}

    for epoch in range(cfg.max_epochs):
        model.train()
        epoch_losses = _run_epoch(
            model, loss_fn, train_loader, optimizer, epoch, device,
            override_lambdas={
                "cons":  cfg.lambda_cons,
                "trans": cfg.lambda_trans,
                "cal":   cfg.lambda_cal,
            },
        )

        metric = _early_stop_metric(epoch_losses, cfg.early_stop_metric)
        logger.info(
            "Stage C | epoch %d | loss=%.4f | nll=%.4f | cons=%.4f",
            epoch, epoch_losses["loss"], epoch_losses["nll"], epoch_losses["consistency"],
        )

        if metric < best_metric - 1e-5:
            best_metric = metric
            best_epoch = epoch
            no_improve = 0
            save_checkpoint("C", epoch, model, optimizer, epoch_losses, checkpoint_dir)
        else:
            no_improve += 1

        last_loss_dict = epoch_losses

        if no_improve >= cfg.early_stop_patience:
            logger.info("Stage C: early stop at epoch %d (patience=%d)",
                        epoch, cfg.early_stop_patience)
            break

    logger.info("Stage C complete. Best metric=%.4f at epoch %d", best_metric, best_epoch)
    return {"best_epoch": best_epoch, "best_metric": best_metric, "last_loss_dict": last_loss_dict}


# ---------------------------------------------------------------------------
# Internal: Single Epoch Runner
# ---------------------------------------------------------------------------

def _run_epoch(
    model: nn.Module,
    loss_fn: HybridLoss,
    loader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: torch.device,
    override_lambdas: Optional[dict] = None,
) -> dict:
    """
    Run one full training epoch. Returns averaged loss dict.

    override_lambdas: if provided, temporarily override loss_fn lambda values
                      keys: 'cons', 'trans', 'cal'
    """
    # Temporarily override lambdas for this stage
    original = {}
    if override_lambdas:
        for key, val in override_lambdas.items():
            attr = f"lambda_{key}"
            original[attr] = getattr(loss_fn, attr)
            setattr(loss_fn, attr, val)

    totals = {}
    n_batches = 0

    for x_clean, y_degraded in loader:
        x_clean    = x_clean.to(device)
        y_degraded = y_degraded.to(device)

        optimizer.zero_grad()

        try:
            loss, loss_dict = loss_fn(model, x_clean, y_degraded, epoch=epoch)
        except RuntimeError as exc:
            logger.error("_run_epoch: loss computation failed at epoch %d — %s", epoch, exc)
            raise

        loss.backward()

        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), max_norm=5.0
        )
        optimizer.step()

        for k, v in loss_dict.items():
            totals[k] = totals.get(k, 0.0) + v
        n_batches += 1

    # Restore overridden lambdas
    for attr, val in original.items():
        setattr(loss_fn, attr, val)

    if n_batches == 0:
        logger.error("_run_epoch: no batches processed at epoch %d", epoch)
        raise RuntimeError("Empty data loader — no batches in epoch")

    return {k: v / n_batches for k, v in totals.items()}


def _early_stop_metric(loss_dict: dict, metric_name: str) -> float:
    """Compute scalar early-stopping metric from loss_dict."""
    if metric_name == "nll":
        return loss_dict["nll"]
    if metric_name == "residual":
        return loss_dict["consistency"]
    if metric_name == "nll+residual":
        return loss_dict["nll"] + loss_dict["consistency"]
    logger.error("_early_stop_metric: unknown metric '%s'", metric_name)
    raise ValueError(f"Unknown early_stop_metric: '{metric_name}'")
