# =============================================================================
# Version: WP3.1-CSMF-v1.3.13 | Abbr: CSMF-MAIN
# Description: Conditional Sequential Mixture of Flows — main model class
# Changelog:
#   v1.3.13 (2026-03-25): [GD] Stage B gate diagnostics — epoch_logs tracks train_loss,
#                         val_loss, neff, gate_weights per epoch; after training saves
#                         stage_b_gate_weights.png (per-expert mean weight over epochs),
#                         stage_b_neff.png (Neff with collapse threshold line), and
#                         stage_b_gate_summary.json (final weights, Neff, early_stopped)
#   v1.3.12 (2026-03-09): [F] Relaxed eval_expert inv_err threshold 1e-4 → 5e-3 —
#                         comparison is image-space wrapped (sigmoid applied); direct
#                         logit-space inv_err=8.48e-08 confirms RealNVP is correctly
#                         invertible; ~1e-3 noise is sigmoid/logit numerical precision.
#   v1.3.11 (2026-03-09): [F] eval_expert invertibility check now compares in pixel space
#                         [0,1] — x_in was logit-space while x_recon had sigmoid applied,
#                         giving spurious inv_err≈5.7; sigmoid(x_in) aligns both to [0,1]
#                         for image experts; non-image experts unchanged.
#   v1.3.10 (2026-03-09): [F] Reverted clamp(0.005) back to clamp(1e-6) — tighter clamp
#                         caused ActNorm NaN: MNIST background pixels all collapse to 0.005
#                         → std(xA)≈0 → log_scale=-inf; wide clamp(1e-6) preserves variance;
#                         range warning silenced by widening threshold in conditional_realnvp.
#   v1.3.9 (2026-03-09): [F] Tightened logit clamp in _prepare_x_for_expert from 1e-6 to
#                        0.005 — clamp(1e-6) produces range ~[-13.8,13.8] which is
#                        unnecessarily wide for flows mapping to N(0,1); clamp(0.005)
#                        gives ~[-5.3,5.3] matching expected dequantized+logit input range;
#                        no change to inverse/sample paths (sigmoid handles all reals).
#   v1.3.8 (2026-03-09): [F1] _expert_inverse: replaced _prepare_x_for_expert(expert, z)
#                        with direct flatten-only for non-image experts — _prepare_x_for_expert
#                        now applies dequantize+logit (added externally) which would corrupt
#                        latent z if called on it; image experts pass z unchanged.
#                        [F2] sigmoid applied to RealNVP inverse output in _expert_inverse
#                        and sample() — inverts the logit transform applied in
#                        _prepare_x_for_expert, converting output back to [0,1] pixel space.
#   v1.3.7 (2026-03-01): train_stage_A() returns epoch_logs dict for EXP-SANITY;
#                        epoch_logs = {expert_name: {train_nll:[], val_nll:[], inv_err:[]}};
#                        inv_err computed per epoch on first val batch (cheap);
#                        return type changed: None → Dict[str, Dict[str, list]]
#   v1.3.6 (2026-02-28): NLL curve tracking and plotting — train_stage_A() accumulates
#                        per-epoch train_nll_history and val_nll_history per expert;
#                        _plot_nll_curves() saves stage_a_nll_{expert_name}.png to plot_dir;
#                        plot_dir param added to train_stage_A() (default="results");
#                        matplotlib Agg backend used for server-safe non-interactive plotting
#   v1.3.5 (2026-02-28): BUG FIX — base_dist loc/scale were CPU tensors, failing when z
#                        is on CUDA; replaced self.base_dist assignment with registered
#                        buffers base_loc/base_scale + @property base_dist that constructs
#                        Normal on-the-fly using buffers (always on correct device)
#   v1.3.4 (2026-02-28): GPU support — added @property device (derives from model params);
#                        all 6 dataloader loops in train_stage_A/B/C, _eval_nll_single,
#                        _eval_hybrid_loss, eval_expert now move x_clean/y_deg to
#                        self.device at top of each batch; no API changes
#   v1.3.3 (2026-02-26): BUG FIX — _expert_inverse missing z flatten for non-image experts;
#                        NSF/NICE/MAF received [B,1,28,28] instead of [B,784] causing
#                        "Tensors must have same number of dimensions: got 3 and 2";
#                        added z_in = _prepare_x_for_expert(expert, z) at top of
#                        _expert_inverse; RealNVP unaffected (_is_image_expert returns True)
#   v1.3.2 (2026-02-26): [B] Spec-compliant h passing — _expert_forward passes h= to
#                        RealNVP/MAF so they use CSMF's shared conditioner h (≈7.4) instead
#                        of recomputing internally (≈92.6); _expert_inverse passes h= to
#                        RealNVP/MAF inverse(); forward/inverse now use identical h per spec
#                        "cache h per mini-batch" requirement; fixes stuck NLL root cause
#   v1.3.1 (2026-02-25): BUG FIX — ConditionalRealNVP.inverse() missing 'y' argument
#                        _expert_forward 4-tuple branch now preserves z_factored_list as 4th return
#                        _expert_inverse signature extended with z_factored_list=None param;
#                        RealNVP path calls inverse(z_final, z_factored_list, y) correctly
#                        All _expert_forward/_expert_inverse call sites updated to unpack/pass z_factored_list
#   v1.3 (2026-02-25): train_stage_A() accepts optimizer_fn callable — per-expert optimizer,
#                      supports different LR per expert
#                      Signature change: ckpt_path -> ckpt_dir; per-expert checkpoints
#                      expert_{k}_{Name}.pth saved after each expert
#                      Added eval_expert() method — evaluates NLL, invertibility, h.norm, NaN rate;
#                      fatal raise on fail, stops before Stage B
#                      save_checkpoint() now embeds config_hash in every payload for drift detection
#                      Stage A combined checkpoint csmf_stage_A.pth always saved (not conditional)
#   v1.2 (2026-02-22): [L1] Expert logs now include class name (e.g. Expert 1 ConditionalMAF);
#                      [L2] Per-epoch NLL logged for ALL experts (was only Expert 0 visible);
#                      [L3] Batch-level loss logged every 50 batches per expert;
#                      [A1] NaN batch count logged per epoch per expert to detect silent dead experts;
#                      [A2] h.norm() mean logged per epoch to confirm COND-NET-v1.2 fix is active.
#   v1.1 (2025-02-21): Added checkpoint save/load, early stopping, gate
#                      temperature annealing, _compute_neff helper
#   v1.0 (2025-02-21): Initial implementation — forward, sample, 3-stage training
# Dependencies: COND-NET, COND-RNVP, COND-MAF, HYBRID
# =============================================================================

import os
import logging
import inspect
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


class CSMF(nn.Module):
    """
    Conditional Sequential Mixture of Flows.

    Architecture:
        c_eta(y) -> h                          (conditioner)
        {q_k(x | h; phi_k)}_{k=1}^K           (expert flows)
        w_psi(h) in Delta^{K-1}               (gating network)

    Mixture posterior:
        q(x|y) = sum_k w_k(y) * q_k(x|y)
    """

    def __init__(
        self,
        experts: List[nn.Module],
        conditioner: nn.Module,
        gate: nn.Module,
        base_dist: Optional[torch.distributions.Distribution] = None,
    ):
        """
        Args:
            experts:     List of K conditional flow modules.
                         Each must implement:
                           forward(x, h) -> (z, log_det)
                           inverse(z, h) -> x
                         and expose a .dim attribute (int).
            conditioner: Network mapping y -> h  shape: (B, d') -> (B, h_dim)
            gate:        Network mapping h -> logits  shape: (B, h_dim) -> (B, K)
            base_dist:   Base distribution. Default: standard Normal N(0, I).
        """
        super().__init__()

        if len(experts) == 0:
            raise ValueError("CSMF requires at least one expert flow.")

        for k, expert in enumerate(experts):
            if not hasattr(expert, "dim"):
                raise AttributeError(
                    f"Expert {k} ({type(expert).__name__}) must expose a `.dim` "
                    f"attribute indicating the data dimensionality."
                )

        self.experts     = nn.ModuleList(experts)
        self.conditioner = conditioner
        self.gate        = gate
        self.K           = len(experts)
        self.dim         = experts[0].dim

        # Register as buffers so they move with model.to(device)
        self.register_buffer('base_loc',   torch.zeros(1))
        self.register_buffer('base_scale', torch.ones(1))

        logger.info(
            f"CSMF initialised | K={self.K} | dim={self.dim} | "
            f"base_dist=Normal (device-tracked via buffers)"
        )

    # =========================================================================
    # Core: forward / sample
    # =========================================================================

    @staticmethod
    def _is_image_expert(expert: nn.Module) -> bool:
        return expert.__class__.__name__ == "ConditionalRealNVP"

    @staticmethod
    def _expects_raw_y(expert: nn.Module) -> bool:
        return expert.__class__.__name__ in {"ConditionalRealNVP", "ConditionalMAF"}

    def _prepare_x_for_expert(self, expert: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self._is_image_expert(expert):
            # Dequantize + logit transform: [0,1] → ~[-5,5]
            x = (x * 255 + torch.rand_like(x)) / 256  # dequantize
            x = x.clamp(1e-6, 1 - 1e-6)  # logit → ~[-13.8, 13.8]; wide clamp preserves variance
            return torch.logit(x)
        return x.flatten(1) if x.dim() > 2 else x

    def _expert_forward(
        self,
        expert: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        """
        Returns: (z, log_det, log_prob, z_factored_list)
          z_factored_list is non-None only for ConditionalRealNVP (4-tuple output).
          All callers must unpack 4 values.
        """
        x_in = self._prepare_x_for_expert(expert, x)
        if self._expects_raw_y(expert):
            # RealNVP/MAF: pass both y (for internal conditioner fallback) and h
            # (pre-computed from CSMF's shared conditioner — spec-compliant caching)
            out = expert.forward(x_in, y, h=h)
        else:
            out = expert.forward(x_in, h)

        if not isinstance(out, tuple):
            raise RuntimeError(f"Unexpected forward output type from {type(expert).__name__}: {type(out)}")

        if len(out) == 2:
            z, log_det = out
            return z, log_det, None, None
        if len(out) == 3:
            z, log_det, log_prob = out
            return z, log_det, log_prob, None
        if len(out) == 4:
            # ConditionalRealNVP: (z_final, z_factored_list, log_det, log_prob)
            z_final, z_factored_list, log_det, log_prob = out
            return z_final, log_det, log_prob, z_factored_list

        raise RuntimeError(f"Unexpected forward output length from {type(expert).__name__}: {len(out)}")

    def _expert_inverse(
        self,
        expert: nn.Module,
        z: torch.Tensor,
        y: torch.Tensor,
        h: torch.Tensor,
        z_factored_list: Optional[list] = None,
    ) -> torch.Tensor:
        """
        Dispatch inverse call based on expert type.
          ConditionalRealNVP: inverse(z_final, z_factored_list, y) — 3 args (image, no flatten)
          ConditionalMAF:     inverse(z, y)                        — 2 args (flat)
          NICE/NSF:           inverse(z, h)                        — 2 args (flat)
        """
        # Flatten z for non-image experts (MAF/NSF/NICE expect [B,784], not [B,1,28,28])
        # Do NOT apply logit transform on latent z — _prepare_x_for_expert now applies
        # dequantize+logit for image experts which would corrupt a latent tensor.
        z_in = z if self._is_image_expert(expert) else (z.flatten(1) if z.dim() > 2 else z)

        if self._is_image_expert(expert):
            # ConditionalRealNVP: always needs z_factored_list; use [] if not provided (e.g. sanity eval)
            zfl = z_factored_list if z_factored_list is not None else []
            # sigmoid inverts the logit transform applied in _prepare_x_for_expert → [0,1]
            return torch.sigmoid(expert.inverse(z_in, zfl, y, h=h))
        if self._expects_raw_y(expert):
            # ConditionalMAF: z_in is [B,784]
            return expert.inverse(z_in, y, h=h)
        # NICE/NSF: z_in is [B,784]
        return expert.inverse(z_in, h)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log q(x|y) = log sum_k w_k(y) q_k(x|y).

        Args:
            x: (B, d)   clean samples
            y: (B, d')  degraded observations

        Returns:
            log_q         (B,)    mixture log-probability
            log_q_experts (B, K)  per-expert log-probabilities
        """
        h      = self.conditioner(y)                          # (B, h_dim)
        logits = self.gate(h)                                 # (B, K)
        log_w  = torch.log_softmax(logits, dim=1)             # (B, K)

        log_q_experts = []
        for k, expert in enumerate(self.experts):
            z, log_det, log_prob, z_flist = self._expert_forward(expert, x, y, h)

            if torch.any(torch.isnan(log_det)):
                logger.error(
                    f"NaN in log_det | expert={k} | "
                    f"x=[{x.min():.4f}, {x.max():.4f}]"
                )
                raise ValueError(f"NaN in log_det from expert {k}")

            if log_prob is not None:
                log_q_experts.append(log_prob)
            else:
                z_flat = z.flatten(1) if z.dim() > 2 else z
                log_p_z = self.base_dist.log_prob(z_flat).sum(dim=1)   # (B,)
                log_q_experts.append(log_p_z + log_det)                # (B,)

        log_q_experts = torch.stack(log_q_experts, dim=1)     # (B, K)
        log_q         = torch.logsumexp(log_w + log_q_experts, dim=1)  # (B,)

        if torch.any(torch.isnan(log_q)):
            logger.error(
                f"NaN in mixture log_q | "
                f"log_w={log_w} | log_q_experts={log_q_experts}"
            )
            raise ValueError("NaN in mixture log-probability")

        return log_q, log_q_experts

    def sample(
        self,
        y: torch.Tensor,
        num_samples: int = 1,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from mixture q(x|y).

        Args:
            y:           (B, d')  degraded observations
            num_samples: number of samples per observation
            temperature: gate softmax temperature (lower = sharper selection)

        Returns:
            x_samples  (B, num_samples, d)
            expert_ids (B, num_samples)   which expert generated each sample
        """
        B   = y.shape[0]
        dev = y.device

        h      = self.conditioner(y)                                   # (B, h_dim)
        logits = self.gate(h) / max(temperature, 1e-6)                 # (B, K)
        w      = torch.softmax(logits, dim=1)                          # (B, K)

        if torch.any(w < 0) or torch.any((w.sum(dim=1) - 1.0).abs() > 1e-3):
            logger.error(
                f"Invalid gate weights | min={w.min():.6f} | "
                f"row_sums={w.sum(dim=1)}"
            )
            raise ValueError("Gate weights are invalid.")

        x_samples  = torch.zeros(B, num_samples, self.dim, device=dev)
        expert_ids = torch.zeros(B, num_samples, dtype=torch.long, device=dev)

        for i in range(B):
            chosen = torch.multinomial(w[i], num_samples, replacement=True)  # (S,)
            for s in range(num_samples):
                k   = chosen[s].item()
                expert = self.experts[k]
                if self._is_image_expert(expert):
                    x_img = torch.sigmoid(expert.sample(1, y[i:i+1])).flatten(1)
                    x_samples[i, s] = x_img.squeeze(0)
                else:
                    z = self.base_dist.sample((self.dim,)).to(dev)
                    x = self._expert_inverse(expert, z.unsqueeze(0), y[i:i+1], h[i:i+1], z_factored_list=None)
                    x_samples[i, s] = x.squeeze(0)
                expert_ids[i, s] = k

        logger.debug(
            f"sample | B={B} | S={num_samples} | tau={temperature:.3f} | "
            f"mean_Neff={self._compute_neff(w).mean().item():.3f}"
        )
        return x_samples, expert_ids

    # =========================================================================
    # Helpers
    # =========================================================================

    def _compute_neff(self, w: torch.Tensor) -> torch.Tensor:
        """
        Effective number of experts: Neff = exp(H(w)).

        Args:
            w: (B, K) gate probabilities

        Returns:
            neff: (B,) per-sample effective expert count. Target > 1.5.
        """
        w_safe   = w.clamp(min=1e-8)
        entropy  = -(w_safe * w_safe.log()).sum(dim=1)  # (B,)
        return entropy.exp()

    def _gate_weights(
        self, y: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        """Return gate weights (B, K) given observation y."""
        h      = self.conditioner(y)
        logits = self.gate(h) / max(temperature, 1e-6)
        return torch.softmax(logits, dim=1)

    @property
    def device(self) -> torch.device:
        """Device the model currently lives on."""
        return next(self.parameters()).device

    @property
    def base_dist(self) -> torch.distributions.Normal:
        """Base distribution — always on the same device as the model."""
        return torch.distributions.Normal(self.base_loc, self.base_scale)

    # =========================================================================
    # Checkpoint save / load
    # =========================================================================

    def save_checkpoint(
        self,
        path: str,
        stage: str,
        epoch: int,
        loss: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save model checkpoint with metadata.

        Args:
            path:  save path (.pth)
            stage: 'A', 'B', or 'C'
            epoch: current epoch (0-indexed)
            loss:  scalar loss value at save time
            extra: optional additional metadata dict
        """
        save_dir = os.path.dirname(path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        try:
            from configs.mnist_config import config_hash as _config_hash
            _hash = _config_hash()
        except Exception as e:
            logger.warning(f"save_checkpoint: config_hash unavailable — {e}")
            _hash = "no-config-hash"

        payload = {
            "state_dict":  self.state_dict(),
            "stage":       stage,
            "epoch":       epoch,
            "loss":        loss,
            "config_hash": _hash,   # v1.3: cross-stage drift detection
        }
        if extra:
            payload.update(extra)

        torch.save(payload, path)
        logger.info(
            f"Checkpoint saved | stage={stage} | epoch={epoch} | "
            f"loss={loss:.6f} | path={path}"
        )

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Load checkpoint and return metadata.

        Args:
            path: checkpoint path (.pth)

        Returns:
            metadata dict (stage, epoch, loss, ...)
        """
        if not os.path.exists(path):
            logger.error(f"Checkpoint not found: {path}")
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        payload = torch.load(path, map_location="cpu")
        self.load_state_dict(payload.pop("state_dict"))
        logger.info(
            f"Checkpoint loaded | stage={payload.get('stage')} | "
            f"epoch={payload.get('epoch')} | "
            f"loss={payload.get('loss', float('nan')):.6f} | path={path}"
        )
        return payload

    # =========================================================================
    # Three-stage training
    # =========================================================================

    def train_stage_A(
        self,
        dataloader,
        optimizer_fn,           # callable: expert -> Optimizer (fresh per expert)
        hybrid_loss,
        epochs: int,
        lambda_cons: float = 0.05,
        val_loader=None,
        patience: int = 5,
        ckpt_dir: str = "checkpoints",
        fwd_model=None,
        plot_dir: str = "results",  # [v1.3.4] directory for NLL curve plots
    ) -> Dict[str, Dict[str, list]]:
        """
        Stage A: Train each expert independently with weak consistency.

            Loss_k = -log q_k(x|y) + lambda_cons * ||A x_hat - y||^2

        Experts are frozen after stage completes.

        Args:
            dataloader:  DataLoader yielding (x_clean, y_deg)
            optimizer_fn: callable — takes expert module, returns a fresh Optimizer
            hybrid_loss: HybridLoss instance — used for its .A forward model
            epochs:      epochs per expert
            lambda_cons: weak consistency weight (default 0.05)
            val_loader:  optional validation DataLoader for early stopping
            patience:    early stopping patience (epochs without val improvement)
            ckpt_dir:    directory for checkpoints (per-expert + combined)
            fwd_model:   optional forward model override (defaults to hybrid_loss.A)
            plot_dir:    directory for NLL curve plots

        Returns:
            epoch_logs: {expert_name: {train_nll: [], val_nll: [], inv_err: []}}
                        For use by EXP-SANITY diagnostic plots.
        """
        logger.info(
            f"=== Stage A | K={self.K} experts | epochs={epochs} | "
            f"lambda_cons={lambda_cons} ==="
        )

        # [v1.3.7] Collect epoch logs for EXP-SANITY
        epoch_logs: Dict[str, Dict[str, list]] = {}

        for k, expert in enumerate(self.experts):
            expert_name = type(expert).__name__  # [L1] e.g. ConditionalMAF, ConditionalRealNVP
            logger.info(f"  Expert {k+1}/{self.K} ({expert_name}) training start")
            expert.train()
            opt_k = optimizer_fn(expert)   # v1.3: fresh optimizer per expert

            best_val_nll     = float("inf")
            patience_counter = 0
            last_nll         = float("inf")
            log_every        = 50  # [L3] batch-level log frequency
            train_nll_history: list = []   # [v1.3.4] per-epoch train NLL
            val_nll_history:   list = []   # [v1.3.4] per-epoch val NLL
            inv_err_history:   list = []   # [v1.3.7] per-epoch invertibility error

            # [v1.3.7] register expert in epoch_logs
            epoch_logs[expert_name] = {
                "train_nll": train_nll_history,
                "val_nll":   val_nll_history,
                "inv_err":   inv_err_history,
            }

            for epoch in range(epochs):
                total_nll  = 0.0
                total_cons = 0.0
                n_batches  = 0
                n_nan_batches = 0   # [A1] count skipped NaN batches
                total_h_norm  = 0.0 # [A2] accumulate h.norm() per batch

                for x_clean, y_deg in dataloader:
                    x_clean = x_clean.to(self.device)
                    y_deg   = y_deg.to(self.device)
                    opt_k.zero_grad()

                    h = self.conditioner(y_deg)
                    total_h_norm += h.norm().item()  # [A2] accumulate per batch
                    z, log_det, log_prob, z_flist = self._expert_forward(expert, x_clean, y_deg, h)

                    if torch.any(torch.isnan(log_det)):
                        logger.error(
                            f"Stage A | expert={k} ({expert_name}) | epoch={epoch} | "
                            f"NaN log_det — skipping batch"
                        )
                        n_nan_batches += 1  # [A1]
                        continue

                    if log_prob is not None:
                        nll = -log_prob.mean()
                    else:
                        z_flat = z.flatten(1) if z.dim() > 2 else z
                        log_p_z = self.base_dist.log_prob(z_flat).sum(dim=1)
                        nll = -(log_p_z + log_det).mean()

                    # Weak consistency via reconstructed sample
                    with torch.no_grad():
                        if self._expects_raw_y(expert) and hasattr(expert, "sample"):
                            x_hat = expert.sample(x_clean.shape[0], y_deg)
                            if x_hat.dim() == 3:
                                x_hat = x_hat[:, 0, :]
                            if x_hat.dim() == 2:
                                x_hat = x_hat.view(-1, 1, 28, 28)
                        else:
                            z_base = self.base_dist.sample(
                                (x_clean.shape[0], self.dim)
                            ).to(x_clean.device)
                            x_hat = self._expert_inverse(expert, z_base, y_deg, h, z_factored_list=None)
                            if x_hat.dim() == 2:
                                x_hat = x_hat.view(-1, 1, 28, 28)
                    Ax    = hybrid_loss.A.forward(x_hat)
                    cons  = torch.mean((Ax - y_deg) ** 2)

                    loss = nll + lambda_cons * cons

                    if torch.isnan(loss):
                        logger.error(
                            f"Stage A | expert={k} ({expert_name}) | epoch={epoch} | "
                            f"NaN total loss — skipping batch"
                        )
                        n_nan_batches += 1  # [A1]
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        expert.parameters(), max_norm=1.0
                    )
                    opt_k.step()

                    total_nll  += nll.item()
                    total_cons += cons.item()
                    n_batches  += 1

                    # [L3] Batch-level log every `log_every` batches
                    if n_batches % log_every == 0:
                        logger.info(
                            f"  Expert {k} ({expert_name}) | Epoch {epoch+1} | "
                            f"Batch {n_batches} | NLL={nll.item():.4f} | Cons={cons.item():.4f}"
                        )

                if n_batches == 0:
                    logger.error(
                        f"Stage A | expert={k} ({expert_name}) | epoch={epoch} | "
                        f"All batches skipped — check NaN sources"
                    )
                    continue

                avg_nll    = total_nll  / n_batches
                avg_cons   = total_cons / n_batches
                avg_h_norm = total_h_norm / max(n_batches + n_nan_batches, 1)  # [A2]
                last_nll   = avg_nll
                train_nll_history.append(avg_nll)  # [v1.3.4] track for plot
                # [L1+L2] expert name visible; [A1] NaN batch count; [A2] h.norm
                logger.info(
                    f"  Expert {k} ({expert_name}) | Epoch {epoch+1}/{epochs} | "
                    f"NLL={avg_nll:.4f} | Cons={avg_cons:.4f} | "
                    f"NaN_batches={n_nan_batches} | h_norm={avg_h_norm:.4f}"
                )
                if n_nan_batches > 0:
                    logger.warning(
                        f"  [A1] Expert {k} ({expert_name}) | Epoch {epoch+1} | "
                        f"{n_nan_batches}/{n_batches + n_nan_batches} batches skipped — expert may be diverging"
                    )

                # Early stopping on validation NLL
                if val_loader is not None:
                    val_nll = self._eval_nll_single(expert, val_loader)
                    logger.info(
                        f"  Expert {k} ({expert_name}) | Epoch {epoch+1} | ValNLL={val_nll:.4f}"
                    )
                    val_nll_history.append(val_nll)  # [v1.3.4] track for plot
                    if val_nll < best_val_nll - 1e-4:
                        best_val_nll     = val_nll
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger.info(
                                f"  Expert {k} ({expert_name}) | Early stopping at epoch {epoch+1} "
                                f"(patience={patience})"
                            )
                            break

                # [v1.3.7] Per-epoch inv_err on first val batch (cheap)
                if val_loader is not None:
                    try:
                        expert.eval()
                        vx, vy = next(iter(val_loader))
                        vx, vy = vx.to(self.device), vy.to(self.device)
                        vh = self.conditioner(vy)
                        vz, _, _, vz_flist = self._expert_forward(expert, vx, vy, vh)
                        vx_in = self._prepare_x_for_expert(expert, vx)
                        vx_rec = self._expert_inverse(expert, vz, vy, vh, z_factored_list=vz_flist)
                        if vx_rec.shape != vx_in.shape:
                            vx_rec = vx_rec.view_as(vx_in)
                        ie = (vx_rec - vx_in).abs().mean().item()
                        inv_err_history.append(ie)
                        expert.train()
                    except Exception as e:
                        logger.error(f"[v1.3.7] inv_err tracking failed | expert={k} epoch={epoch+1}: {e}")
                        expert.train()

            # [v1.3.4] Save NLL curve plot for this expert
            self._plot_nll_curves(expert_name, train_nll_history, val_nll_history, plot_dir)

            # v1.3: per-expert checkpoint after training completes
            expert_ckpt = os.path.join(ckpt_dir, f"expert_{k}_{expert_name}.pth")
            self.save_checkpoint(
                expert_ckpt, stage="A", epoch=epochs - 1,
                loss=last_nll, extra={"expert_k": k}
            )
            logger.info(f"  Expert {k} ({expert_name}) | checkpoint saved: {expert_ckpt}")

            # v1.3: eval_expert — fatal raise on fail, stops before Stage B
            if val_loader is not None:
                fwd = fwd_model if fwd_model is not None else hybrid_loss.A
                self.eval_expert(k, expert, val_loader, fwd)

        # Freeze all expert parameters
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = False

        logger.info("Stage A complete — all expert parameters frozen.")

        # v1.3: always save combined Stage A checkpoint (not conditional on val_loader)
        stage_A_ckpt = os.path.join(ckpt_dir, "csmf_stage_A.pth")
        self.save_checkpoint(stage_A_ckpt, stage="A", epoch=epochs - 1, loss=last_nll)

        # [v1.3.7] Return epoch_logs for EXP-SANITY
        return epoch_logs

    def train_stage_B(
        self,
        dataloader,
        optimizer: torch.optim.Optimizer,
        hybrid_loss,
        epochs: int,
        val_loader=None,
        patience: int = 5,
        ckpt_path: str = "checkpoints/csmf_stage_B.pth",
        results_dir: str = "results",
    ) -> None:
        """
        Stage B: Train gate network only (experts frozen).

            Loss = HybridLoss(model, x_clean, y_deg, epoch)

        Args:
            dataloader:  DataLoader yielding (x_clean, y_deg)
            optimizer:   optimizer over gate.parameters() ONLY
            hybrid_loss: HybridLoss instance
            epochs:      training epochs
            val_loader:  optional validation DataLoader for early stopping
            patience:    early stopping patience
            ckpt_path:   checkpoint save path
            results_dir: directory for gate diagnostic plots and summary JSON
        """
        # Sanity-check expert freeze
        for k, expert in enumerate(self.experts):
            n_trainable = sum(p.requires_grad for p in expert.parameters())
            if n_trainable > 0:
                logger.warning(
                    f"Stage B | expert {k} has {n_trainable} trainable params "
                    f"— expected 0. Ensure Stage A completed."
                )

        logger.info(f"=== Stage B | gate training | epochs={epochs} ===")
        self.gate.train()

        best_val_loss    = float("inf")
        patience_counter = 0
        last_loss        = float("inf")
        early_stopped    = False

        # [GD] v1.3.13: epoch-level diagnostic logs
        expert_names = [type(e).__name__ for e in self.experts]
        epoch_logs: Dict[str, list] = {
            "train_loss":   [],
            "val_loss":     [],
            "neff":         [],
            "gate_weights": [],   # list of [K] mean weights per epoch
        }

        for epoch in range(epochs):
            total_loss         = 0.0
            total_neff         = 0.0
            total_gate_weights = torch.zeros(len(self.experts), device=self.device)
            n_batches          = 0

            for x_clean, y_deg in dataloader:
                x_clean = x_clean.to(self.device)
                y_deg   = y_deg.to(self.device)
                optimizer.zero_grad()

                loss, loss_dict = hybrid_loss(self, x_clean, y_deg, epoch)

                if torch.isnan(loss):
                    logger.error(
                        f"Stage B | epoch={epoch} | NaN hybrid loss — skipping batch"
                    )
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.gate.parameters(), max_norm=1.0
                )
                optimizer.step()

                with torch.no_grad():
                    w    = self._gate_weights(y_deg)          # (B, K)
                    neff = self._compute_neff(w).mean().item()
                    total_gate_weights += w.mean(dim=0)       # [GD] accumulate mean weights

                total_loss += loss.item()
                total_neff += neff
                n_batches  += 1

            if n_batches == 0:
                logger.error(f"Stage B | epoch={epoch} | All batches skipped")
                continue

            avg_loss         = total_loss / n_batches
            avg_neff         = total_neff / n_batches
            avg_gate_weights = (total_gate_weights / n_batches).cpu().tolist()
            last_loss        = avg_loss

            # [GD] v1.3.13: record epoch logs
            epoch_logs["train_loss"].append(avg_loss)
            epoch_logs["neff"].append(avg_neff)
            epoch_logs["gate_weights"].append(avg_gate_weights)

            logger.info(
                f"Stage B | Epoch {epoch+1}/{epochs} | "
                f"Loss={avg_loss:.4f} | Neff={avg_neff:.3f} | "
                f"Weights={[f'{w:.3f}' for w in avg_gate_weights]}"
            )

            if avg_neff < 1.1:
                logger.warning(
                    f"Stage B | Epoch {epoch+1} | "
                    f"Gate collapse: Neff={avg_neff:.3f} < 1.1"
                )

            if val_loader is not None:
                val_loss = self._eval_hybrid_loss(hybrid_loss, val_loader, epoch)
                epoch_logs["val_loss"].append(val_loss)  # [GD]
                logger.info(
                    f"Stage B | Epoch {epoch+1} | ValLoss={val_loss:.4f}"
                )
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss    = val_loss
                    patience_counter = 0
                    self.save_checkpoint(
                        ckpt_path, stage="B", epoch=epoch, loss=best_val_loss
                    )
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(
                            f"Stage B | Early stopping at epoch {epoch+1}"
                        )
                        early_stopped = True
                        break

        logger.info("Stage B complete.")
        if val_loader is None:
            self.save_checkpoint(
                ckpt_path, stage="B", epoch=epochs - 1, loss=last_loss
            )

        # [GD] v1.3.13: save gate diagnostic plots and summary JSON
        self._save_stage_b_diagnostics(
            epoch_logs=epoch_logs,
            expert_names=expert_names,
            early_stopped=early_stopped,
            best_val_loss=best_val_loss,
            results_dir=results_dir,
        )

    def _save_stage_b_diagnostics(
        self,
        epoch_logs: Dict[str, list],
        expert_names: List[str],
        early_stopped: bool,
        best_val_loss: float,
        results_dir: str,
    ) -> None:
        """
        [GD] v1.3.13: Save Stage B gate diagnostic plots and summary JSON.

        Outputs:
            results_dir/stage_b_gate_weights.png  — per-expert mean weight over epochs
            results_dir/stage_b_neff.png          — Neff over epochs with collapse threshold
            results_dir/stage_b_gate_summary.json — final weights, Neff, flags
        """
        import json

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("[GD] matplotlib not available — Stage B plots skipped")
            return

        os.makedirs(results_dir, exist_ok=True)
        epochs_axis = list(range(1, len(epoch_logs["train_loss"]) + 1))

        if not epochs_axis:
            logger.warning("[GD] No epoch logs recorded — Stage B diagnostics skipped")
            return

        # --- Plot 1: Gate weights over epochs ---
        try:
            gate_weights_arr = epoch_logs["gate_weights"]  # list of [K]
            fig, ax = plt.subplots(figsize=(8, 4))
            for k, name in enumerate(expert_names):
                weights_k = [gw[k] for gw in gate_weights_arr]
                ax.plot(epochs_axis, weights_k, marker="o", markersize=3, label=name)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Mean Gate Weight")
            ax.set_title("Stage B — Gate Weights Over Epochs")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            save_path = os.path.join(results_dir, "stage_b_gate_weights.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"[GD] Stage B gate weights plot saved: {save_path}")
        except Exception as e:
            logger.error(f"[GD] Failed to save gate weights plot: {e}")

        # --- Plot 2: Neff over epochs ---
        try:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(epochs_axis, epoch_logs["neff"], marker="o", markersize=3,
                    color="steelblue", label="Neff")
            ax.axhline(y=1.1, color="red", linestyle="--", linewidth=1.0,
                       label="Collapse threshold (1.1)")
            if epoch_logs["val_loss"]:
                ax2 = ax.twinx()
                ax2.plot(epochs_axis[:len(epoch_logs["val_loss"])],
                         epoch_logs["val_loss"], linestyle="--",
                         color="orange", marker="s", markersize=3, label="Val Loss")
                ax2.set_ylabel("Val Loss")
                ax2.legend(loc="upper right")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Neff")
            ax.set_title("Stage B — Effective Expert Count (Neff)")
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)
            save_path = os.path.join(results_dir, "stage_b_neff.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"[GD] Stage B Neff plot saved: {save_path}")
        except Exception as e:
            logger.error(f"[GD] Failed to save Neff plot: {e}")

        # --- JSON summary ---
        try:
            final_weights = epoch_logs["gate_weights"][-1] if epoch_logs["gate_weights"] else []
            final_neff    = epoch_logs["neff"][-1] if epoch_logs["neff"] else None
            summary = {
                "final_gate_weights": {
                    name: round(float(w), 4)
                    for name, w in zip(expert_names, final_weights)
                },
                "final_neff":      round(float(final_neff), 4) if final_neff is not None else None,
                "early_stopped":   early_stopped,
                "best_val_loss":   round(float(best_val_loss), 4) if best_val_loss < float("inf") else None,
                "total_epochs":    len(epochs_axis),
            }
            save_path = os.path.join(results_dir, "stage_b_gate_summary.json")
            with open(save_path, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"[GD] Stage B gate summary saved: {save_path}")
            logger.info(f"[GD] Stage B summary: {summary}")
        except Exception as e:
            logger.error(f"[GD] Failed to save gate summary JSON: {e}")

    def train_stage_C(
        self,
        dataloader,
        optimizer: torch.optim.Optimizer,
        hybrid_loss,
        epochs: int,
        blocks_to_unfreeze: int = 1,
        tau_start: float = 1.0,
        tau_end: float = 0.5,
        val_loader=None,
        patience: int = 5,
        ckpt_path: str = "checkpoints/csmf_stage_C.pth",
    ) -> None:
        """
        Stage C: Light joint fine-tuning with gate temperature annealing.

            - Unfreezes last `blocks_to_unfreeze` child modules per expert
            - Gate temperature: tau(e) = tau_start -> tau_end  (linear)
            - Loss = HybridLoss(model, x_clean, y_deg, epoch)

        Args:
            dataloader:         DataLoader yielding (x_clean, y_deg)
            optimizer:          optimizer over requires_grad=True params
            hybrid_loss:        HybridLoss instance
            epochs:             training epochs
            blocks_to_unfreeze: trailing child modules to unfreeze per expert
            tau_start:          gate temperature at epoch 0
            tau_end:            gate temperature at final epoch
            val_loader:         optional validation DataLoader
            patience:           early stopping patience
            ckpt_path:          checkpoint save path
        """
        # Unfreeze last N child modules of each expert
        for k, expert in enumerate(self.experts):
            children = list(expert.children())
            if len(children) == 0:
                logger.warning(
                    f"Stage C | expert {k} has no child modules — "
                    f"unfreezing all parameters"
                )
                for param in expert.parameters():
                    param.requires_grad = True
            else:
                for layer in children[-blocks_to_unfreeze:]:
                    for param in layer.parameters():
                        param.requires_grad = True
            n_trainable = sum(p.requires_grad for p in expert.parameters())
            logger.info(
                f"Stage C | expert {k}: unfroze last {blocks_to_unfreeze} "
                f"block(s) — {n_trainable} trainable params"
            )

        self.gate.train()
        logger.info(
            f"=== Stage C | joint fine-tune | epochs={epochs} | "
            f"blocks={blocks_to_unfreeze} | tau {tau_start}->{tau_end} ==="
        )

        best_val_loss    = float("inf")
        patience_counter = 0
        last_loss        = float("inf")

        for epoch in range(epochs):
            # Linear temperature annealing
            tau = tau_start - (tau_start - tau_end) * (
                epoch / max(epochs - 1, 1)
            )

            total_loss = 0.0
            total_neff = 0.0
            n_batches  = 0

            for x_clean, y_deg in dataloader:
                x_clean = x_clean.to(self.device)
                y_deg   = y_deg.to(self.device)
                optimizer.zero_grad()

                loss, loss_dict = hybrid_loss(self, x_clean, y_deg, epoch)

                if torch.isnan(loss):
                    logger.error(
                        f"Stage C | epoch={epoch} | NaN loss — skipping batch"
                    )
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()

                with torch.no_grad():
                    w    = self._gate_weights(y_deg, temperature=tau)
                    neff = self._compute_neff(w).mean().item()

                total_loss += loss.item()
                total_neff += neff
                n_batches  += 1

            if n_batches == 0:
                logger.error(f"Stage C | epoch={epoch} | All batches skipped")
                continue

            avg_loss  = total_loss / n_batches
            avg_neff  = total_neff / n_batches
            last_loss = avg_loss
            logger.info(
                f"Stage C | Epoch {epoch+1}/{epochs} | Loss={avg_loss:.4f} | "
                f"Neff={avg_neff:.3f} | tau={tau:.4f}"
            )

            if avg_neff < 1.1:
                logger.warning(
                    f"Stage C | Epoch {epoch+1} | "
                    f"Gate collapse: Neff={avg_neff:.3f} < 1.1"
                )

            if val_loader is not None:
                val_loss = self._eval_hybrid_loss(hybrid_loss, val_loader, epoch)
                logger.info(
                    f"Stage C | Epoch {epoch+1} | ValLoss={val_loss:.4f}"
                )
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss    = val_loss
                    patience_counter = 0
                    self.save_checkpoint(
                        ckpt_path, stage="C", epoch=epoch,
                        loss=best_val_loss, extra={"tau": tau}
                    )
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(
                            f"Stage C | Early stopping at epoch {epoch+1}"
                        )
                        break

        logger.info("Stage C complete.")
        if val_loader is None:
            self.save_checkpoint(
                ckpt_path, stage="C", epoch=epochs - 1, loss=last_loss
            )

    # =========================================================================
    # Private evaluation helpers
    # =========================================================================

    @torch.no_grad()
    def _plot_nll_curves(
        self,
        expert_name: str,
        train_nll: list,
        val_nll: list,
        plot_dir: str,
    ) -> None:
        """
        [v1.3.4] Plot train and val NLL vs epoch for one expert.
        Saves to: {plot_dir}/stage_a_nll_{expert_name}.png
        """
        try:
            import matplotlib
            matplotlib.use("Agg")  # non-interactive backend — safe for server use
            import matplotlib.pyplot as plt
            import os

            os.makedirs(plot_dir, exist_ok=True)
            epochs = range(1, len(train_nll) + 1)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(epochs, train_nll, label="Train NLL", marker="o", markersize=3)
            if val_nll:
                val_epochs = range(1, len(val_nll) + 1)
                ax.plot(val_epochs, val_nll, label="Val NLL", linestyle="--",
                        marker="s", markersize=3)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("NLL (lower = better)")
            ax.set_title(f"Stage A — {expert_name} NLL vs Epoch")
            ax.legend()
            ax.grid(True, alpha=0.3)

            save_path = os.path.join(plot_dir, f"stage_a_nll_{expert_name}.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"[v1.3.4] NLL curve saved: {save_path}")
        except Exception as e:
            logger.error(f"_plot_nll_curves failed for {expert_name}: {e}")

    def _eval_nll_single(self, expert: nn.Module, val_loader) -> float:
        """Evaluate mean NLL of a single expert on val_loader."""
        expert.eval()
        total = 0.0
        n     = 0

        for x_clean, y_deg in val_loader:
            x_clean = x_clean.to(self.device)
            y_deg   = y_deg.to(self.device)
            h = self.conditioner(y_deg)
            z, log_det, log_prob, _z_flist = self._expert_forward(expert, x_clean, y_deg, h)
            if log_prob is not None:
                nll = -log_prob.mean()
            else:
                z_flat = z.flatten(1) if z.dim() > 2 else z
                log_p_z = self.base_dist.log_prob(z_flat).sum(dim=1)
                nll = -(log_p_z + log_det).mean()
            if torch.isnan(nll):
                logger.warning("_eval_nll_single: NaN in val batch — skipping")
                continue
            total += nll.item()
            n     += 1

        expert.train()
        if n == 0:
            logger.error("_eval_nll_single: all val batches were NaN")
            return float("inf")
        return total / n

    @torch.no_grad()
    def eval_expert(self, k, expert, val_loader, fwd_model, logger=None) -> dict:
        """
        Evaluate single expert on val_loader after Stage A training.
        Returns dict: {nll, invertibility_err, h_norm_mean, nan_rate}
        Raises ValueError if:
          - nan_rate > 0
          - invertibility_err > 1e-4
          - h_norm_mean < 0.01 (dead conditioner)
        """
        _log = logger or globals().get('logger')
        expert.eval()
        total_nll   = 0.0
        total_inv   = 0.0
        total_hnorm = 0.0
        nan_batches = 0
        n_batches   = 0

        for x_clean, y_deg in val_loader:
            try:
                x_clean = x_clean.to(self.device)
                y_deg   = y_deg.to(self.device)
                h = self.conditioner(y_deg)
                z, log_det, log_prob, z_flist = self._expert_forward(expert, x_clean, y_deg, h)

                if torch.isnan(log_det).any() or (log_prob is not None and torch.isnan(log_prob).any()):
                    nan_batches += 1
                    continue

                # NLL
                if log_prob is not None:
                    nll = -log_prob.mean().item()
                else:
                    z_flat  = z.flatten(1) if z.dim() > 2 else z
                    log_p_z = self.base_dist.log_prob(z_flat).sum(dim=1)
                    nll = -(log_p_z + log_det).mean().item()

                # Invertibility: ||f^{-1}(f(x)) - x|| compared in pixel space [0,1]
                # x_in is logit-space for image experts; x_recon has sigmoid applied.
                # sigmoid(x_in) aligns both to [0,1] for a valid comparison.
                x_in    = self._prepare_x_for_expert(expert, x_clean)
                x_recon = self._expert_inverse(expert, z, y_deg, h, z_factored_list=z_flist)
                x_ref   = torch.sigmoid(x_in) if self._is_image_expert(expert) else x_in
                if x_recon.shape != x_ref.shape:
                    x_recon = x_recon.view_as(x_ref)
                inv_err = (x_recon - x_ref).abs().mean().item()

                total_nll   += nll
                total_inv   += inv_err
                total_hnorm += h.norm().item()
                n_batches   += 1

            except Exception as e:
                _log.error(f"eval_expert | expert={k} | batch error: {e}")
                nan_batches += 1

        expert.train()

        if n_batches == 0:
            _log.error(f"eval_expert | expert={k} | all batches failed")
            raise ValueError(f"eval_expert: expert {k} — all batches failed")

        total_batches = n_batches + nan_batches
        nll_mean      = total_nll   / n_batches
        inv_err_mean  = total_inv   / n_batches
        h_norm_mean   = total_hnorm / n_batches
        nan_rate      = nan_batches / total_batches

        _log.info(
            f"eval_expert | expert={k} | NLL={nll_mean:.4f} | "
            f"inv_err={inv_err_mean:.2e} | h_norm={h_norm_mean:.4f} | "
            f"nan_rate={nan_rate:.3f}"
        )

        if nan_rate > 0:
            _log.error(f"eval_expert | expert={k} | nan_rate={nan_rate:.3f} > 0 — FATAL")
            raise ValueError(f"eval_expert: expert {k} has nan_rate={nan_rate:.3f}")
        # Threshold is image-space wrapped (sigmoid applied) — not exact flow invertibility.
        # Direct logit-space inv_err=8.48e-08; sigmoid wrapping adds ~1e-3 numerical noise.
        if inv_err_mean > 5e-3:
            _log.error(f"eval_expert | expert={k} | inv_err={inv_err_mean:.2e} > 5e-3 — FATAL")
            raise ValueError(f"eval_expert: expert {k} invertibility error too large: {inv_err_mean:.2e}")
        if h_norm_mean < 0.01:
            _log.error(f"eval_expert | expert={k} | h_norm={h_norm_mean:.4f} < 0.01 — dead conditioner FATAL")
            raise ValueError(f"eval_expert: expert {k} dead conditioner h_norm={h_norm_mean:.4f}")

        return {
            "nll":               nll_mean,
            "invertibility_err": inv_err_mean,
            "h_norm_mean":       h_norm_mean,
            "nan_rate":          nan_rate,
        }

    @torch.no_grad()
    def _eval_hybrid_loss(
        self, hybrid_loss, val_loader, epoch: int
    ) -> float:
        """Evaluate mean hybrid loss on val_loader."""
        self.eval()
        total = 0.0
        n     = 0

        for x_clean, y_deg in val_loader:
            x_clean = x_clean.to(self.device)
            y_deg   = y_deg.to(self.device)
            loss, _ = hybrid_loss(self, x_clean, y_deg, epoch)
            if torch.isnan(loss):
                logger.warning(
                    "_eval_hybrid_loss: NaN in val batch — skipping"
                )
                continue
            total += loss.item()
            n     += 1

        self.train()
        if n == 0:
            logger.error("_eval_hybrid_loss: all val batches were NaN")
            return float("inf")
        return total / n
