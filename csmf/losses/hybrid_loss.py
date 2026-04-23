# Version: WP2.1-HybridLoss-v1.10.2
# Abbr: HYBRID
# File: csmf/losses/hybrid_loss.py
# Description: Hybrid training objective for CSMF
#              L = NLL + λ_cons·‖Ax−y‖² + λ_trans·SW2 + λ_cal·ES + λ_neff·Neff_penalty
# Changelog:
#   v1.10.2 (2026-04-21): [NLL-ANCHOR] forward_stage_c() gains lambda_nll_c param
#                         (default 0.001); adds λ_nll_c · NLL to Stage C total loss
#                         to prevent catastrophic forgetting of flow distribution when
#                         expert blocks are unfrozen. Without NLL anchor, reconstruction
#                         loss alone causes expert NLL to collapse from -1871 to +8212
#                         in 10 epochs. __init__ gains lambda_nll_c param; nll_anchor
#                         and lam_nll_c logged in loss_dict; NaN-guarded.
#   v1.10.1 (2026-04-20): [PROX-SHAPE-FIX] Flatten y_degraded to (B,196) before
#                         passing to prox_fn in forward_stage_c(); x_hat_mix is flat
#                         (B,784) so A_fn returns flat (B,196), but y_degraded from
#                         dataloader is spatial (B,1,14,14) — subtraction shape-mismatch
#                         caused prox to fail every batch silently since v1.9.1.
#   v1.10.0 (2026-04-19): [NEFF-REG] forward() gains λ_neff · max(0, neff_target − Neff)
#                         entropy regularisation term; prevents winner-take-all gate
#                         collapse in Stage B without stop-gradient (experts receive full
#                         gradient signal through Neff penalty). __init__ gains lambda_neff
#                         (default 0.5) and neff_target (default 1.5). Effective annealed
#                         lambdas (lambda_cons_eff, lambda_trans_eff, lambda_cal_eff) and
#                         neff_reg_loss now included in loss_dict for SB-DIAG.
#                         [PRE-PROX] forward_stage_c() adds λ_cons_pre · ‖A(x̂_mix)−y‖²
#                         computed on attached x̂_mix before prox step — gives experts
#                         direct gradient signal independent of prox correction. No detach
#                         applied: full gradient flows to experts through both pre-prox and
#                         post-prox terms. __init__ gains lambda_cons_pre (default 0.02).
#                         [SC-TRANS-CAL] forward_stage_c() adds λ_trans_c·SW2 and
#                         λ_cal_c·ES on post-prox x̂_corr; __init__ gains lambda_trans_c
#                         (default 0.02) and lambda_cal_c (default 0.005). residual_pre_prox,
#                         residual_post_prox, cons_pre_loss, trans_loss_c, cal_loss_c added
#                         to loss_dict for SC-DIAG P_pre_post_prox and P_trans_cal_c.
#   v1.9.1 (2026-04-17): [PROX-C-ACTIVATE] forward_stage_c() now accepts prox_fn=None.
#   v1.9 (2026-04-15): [SC-RECFIRST] Add forward_stage_c() — reconstruction-first Stage C
#                      loss L_C1 = λ_cons·‖A(x̂_mix)−y‖² + λ_img·‖x̂_mix−x‖² +
#                      λ_alive·max(0,1.5−Neff); no NLL/SW2/calibration; add lambda_img
#                      and lambda_alive params to HybridLoss.__init__().
#                      [SC-RECFIRST] Replace run_stage_C() body with forward_stage_c()
#                      calls via new _run_epoch_stage_c(); add gate-frozen warmup phase
#                      (freeze_gate_epochs=5 default) — gate stays frozen during warmup,
#                      optimizer rebuilt on unfreeze; early stop on cons_c1+img_loss.
#   v1.8.1 (2026-04-12): BUG FIX — UnboundLocalError: x_multi used in lam_cal block
#                        when lam_trans > 0 but batch_counter % sw2_every != 0 (SW2
#                        skipped batch); fixed by initialising x_multi = None before
#                        transport block and checking x_multi is not None in cal ternary
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
        # [SC-RECFIRST] Stage C reconstruction-first loss params
        lambda_img: float = 1.0,       # weight for ||x_hat_mix - x_clean||²
        lambda_alive: float = 0.1,     # weight for alive penalty max(0, 1.5 - Neff)
        # [NEFF-REG] Stage B entropy regularisation
        lambda_neff: float = 0.5,      # weight for Neff penalty max(0, neff_target - Neff)
        neff_target: float = 1.5,      # diversity target; penalty fires below this Neff
        # [PRE-PROX] Stage C pre-prox consistency (keeps experts honest before prox)
        lambda_cons_pre: float = 0.02, # weight for pre-prox ||A(x̂_mix) - y||²
        # [SC-TRANS-CAL] Stage C geometry and calibration terms on post-prox output
        lambda_trans_c: float = 0.02,  # weight for SW2 in Stage C
        lambda_cal_c: float = 0.005,   # weight for ES in Stage C
        # [NLL-ANCHOR] Stage C NLL anchor — prevents catastrophic forgetting of flow dist
        lambda_nll_c: float = 0.01,   # weight for mixture NLL in Stage C (0 = disabled)
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
        self.lambda_img      = lambda_img     # [SC-RECFIRST]
        self.lambda_alive    = lambda_alive   # [SC-RECFIRST]
        self.lambda_neff     = lambda_neff    # [NEFF-REG]
        self.neff_target     = neff_target    # [NEFF-REG]
        self.lambda_cons_pre = lambda_cons_pre  # [PRE-PROX]
        self.lambda_trans_c  = lambda_trans_c   # [SC-TRANS-CAL]
        self.lambda_cal_c    = lambda_cal_c     # [SC-TRANS-CAL]
        self.lambda_nll_c    = lambda_nll_c     # [NLL-ANCHOR]



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
        x_multi = None   # [v1.8.1] initialise before conditional — avoids UnboundLocalError
                         # in lam_cal block when sw2_every skips this batch
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
                x_multi if x_multi is not None   # [v1.8.1] was: x_multi if lam_trans > 0
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

        # ---- 5. Neff entropy regularisation [NEFF-REG] ------------------
        # Penalises gate collapse: λ_neff · max(0, neff_target − Neff)
        # Full gradient flows to experts through gate weights — no stop-grad.
        if self.lambda_neff > 0:
            try:
                w_gate = flow._gate_weights(y_degraded)          # (B, K)
                w_safe = w_gate.clamp(min=1e-8)
                w_mean = w_safe.mean(dim=0)                       # (K,) batch-mean
                w_mean = w_mean / w_mean.sum()
                entropy = -(w_mean * w_mean.log()).sum()
                neff    = torch.exp(entropy)
                neff_reg_loss = torch.clamp(self.neff_target - neff, min=0.0)
                if torch.isnan(neff_reg_loss):
                    logger.error(
                        "HybridLoss: NaN in neff_reg_loss at epoch %d — skipping term",
                        epoch
                    )
                    neff_reg_loss = torch.tensor(0.0, device=x_clean.device)
            except Exception as e:
                logger.error(
                    "HybridLoss: neff_reg computation failed at epoch %d: %s — "
                    "skipping term", epoch, e
                )
                neff_reg_loss = torch.tensor(0.0, device=x_clean.device)
                neff = torch.tensor(0.0)
        else:
            neff_reg_loss = torch.tensor(0.0, device=x_clean.device)
            neff = torch.tensor(0.0)

        # ---- Total --------------------------------------------------------
        loss = (
            nll
            + lam_cons  * consistency
            + lam_trans * transport
            + lam_cal   * calibration
            + self.lambda_neff * neff_reg_loss   # [NEFF-REG]
        )

        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(
                "HybridLoss: total loss is NaN/Inf at epoch %d — "
                "nll=%.4f, cons=%.4f, trans=%.4f, cal=%.4f, neff_reg=%.4f",
                epoch, nll.item(), consistency.item(),
                transport.item(), calibration.item(), neff_reg_loss.item(),
            )
            raise RuntimeError("NaN/Inf in total hybrid loss")

        loss_dict = {
            "loss":             loss.item(),
            "nll":              nll.item(),
            "consistency":      consistency.item(),
            "transport":        transport.item(),
            "calibration":      calibration.item(),
            "neff_reg_loss":    neff_reg_loss.item(),       # [NEFF-REG]
            "lam_cons":         lam_cons,
            "lam_trans":        lam_trans,
            "lam_cal":          lam_cal,
            "lambda_cons_eff":  lam_cons,                   # [NEFF-REG] for SB-DIAG P_anneal
            "lambda_trans_eff": lam_trans,
            "lambda_cal_eff":   lam_cal,
        }

        return loss, loss_dict

    # ------------------------------------------------------------------
    # [SC-RECFIRST] Stage C reconstruction-first loss
    # ------------------------------------------------------------------
    def forward_stage_c(
        self,
        flow: nn.Module,
        x_clean: torch.Tensor,
        y_degraded: torch.Tensor,
        epoch: int = 0,
        prox_fn=None,   # [PROX-C-ACTIVATE] callable (x [B,d], y [B,d']) -> x_corrected [B,d]
    ) -> tuple[torch.Tensor, dict]:
        """
        Reconstruction-first Stage C loss (L_C1) with optional prox correction.

        Pipeline (PROX-USAGE Stage C):
            x̂_mix  = Σ_k w_k · x̂_k          (soft weighted mixture)
            x̂_corr = prox_fn(x̂_mix, y)        (physics correction — if prox_fn provided)
            L_C1   = λ_cons·‖A(x̂_corr)−y‖²
                   + λ_img·‖x̂_corr−x‖²
                   + λ_alive·max(0, 1.5 − N_eff)

        Args:
            flow:        CSMF model (must implement sample_all_experts())
            x_clean:     (B, d) clean ground-truth (flat pixel space)
            y_degraded:  (B, d') degraded observations
            epoch:       current epoch (unused; kept for API symmetry with forward())
            prox_fn:     optional callable (x, y) -> x_corrected from make_prox_fn();
                         if None, loss is computed on x̂_mix directly (logged as warning)

        Returns:
            loss:      scalar total loss
            loss_dict: dict with keys: loss, cons_c1, img_loss, alive_penalty, neff_c1,
                       lam_cons, lam_img, lam_alive, prox_applied
        """
        B = x_clean.shape[0]
        x_clean_flat = x_clean.flatten(1)   # (B, d)
        d = x_clean_flat.shape[1]

        # 1. Per-expert reconstructions + gate weights
        w, x_hats = flow.sample_all_experts(y_degraded)   # (B,K), (B,K,d)

        # 2. Renormalise w over surviving experts (zeros = failed inverse)
        alive_mask = (x_hats.abs().sum(dim=2) > 0).float()        # (B, K)
        if alive_mask.min() < 1.0:
            n_dead = int((1.0 - alive_mask).sum().item())
            logger.warning(
                "forward_stage_c: %d expert slot(s) zeroed in batch at epoch %d — "
                "renormalising w over surviving experts", n_dead, epoch
            )
        w_alive = w * alive_mask
        w_alive = w_alive / w_alive.sum(dim=1, keepdim=True).clamp(min=1e-8)

        # 3. Soft mixture reconstruction over surviving experts
        x_hat_mix = (w_alive.unsqueeze(-1) * x_hats).sum(dim=1)   # (B, d)

        if torch.any(torch.isnan(x_hat_mix)):
            logger.error(
                "forward_stage_c: NaN in x_hat_mix at epoch %d after renorm — "
                "NaN count: %d; skipping batch (returning zero loss)",
                epoch, torch.isnan(x_hat_mix).sum().item()
            )
            zero = torch.tensor(0.0, device=x_clean.device, requires_grad=True)
            return zero, {
                "loss": 0.0, "cons_c1": 0.0, "cons_pre_loss": 0.0,
                "img_loss": 0.0, "alive_penalty": 0.0, "neff_c1": 1.0,
                "trans_loss_c": 0.0, "cal_loss_c": 0.0,
                "residual_pre_prox": 0.0, "residual_post_prox": 0.0,
                "lam_cons": self.lambda_cons, "lam_img": self.lambda_img,
                "lam_alive": self.lambda_alive, "prox_applied": False,
            }

        # [PRE-PROX] 3a. Pre-prox consistency on attached x_hat_mix.
        # Full gradient flows to experts — forces experts to produce
        # measurement-consistent output before prox corrects them.
        x_hat_mix_4d_pre = x_hat_mix.view(B, 1, 28, 28)
        Ax_pre = self.A.forward(x_hat_mix_4d_pre)
        cons_pre_loss = torch.mean((Ax_pre - y_degraded) ** 2)
        residual_pre_prox = cons_pre_loss.item()   # raw, for P_pre_post_prox

        if torch.isnan(cons_pre_loss):
            logger.error(
                "forward_stage_c: NaN in cons_pre_loss at epoch %d", epoch
            )
            raise RuntimeError("NaN in Stage C pre-prox consistency loss")

        # [PROX-C-ACTIVATE] Apply prox correction: x̂_mix → prox(x̂_mix, y) → x̂_corr
        # No detach — full gradient flows through prox step.
        # cons_pre already provides direct expert gradient signal above.
        # [PROX-SHAPE-FIX] v1.10.1: x_hat_mix is flat (B,784); A_fn returns flat (B,196).
        # y_degraded from dataloader may be spatial (B,1,14,14) — flatten to (B,196)
        # so Ax-y subtraction in prox_gradient_step does not shape-mismatch.
        y_for_prox = y_degraded.flatten(1) if y_degraded.dim() > 2 else y_degraded

        prox_applied = False
        x_hat_corr = x_hat_mix   # default: no correction
        if prox_fn is not None:
            try:
                x_hat_corr = prox_fn(x_hat_mix, y_for_prox)   # (B, d)
                prox_applied = True
                if torch.any(torch.isnan(x_hat_corr)):
                    logger.error(
                        "forward_stage_c: NaN in x_hat_corr after prox at epoch %d — "
                        "NaN count: %d",
                        epoch, torch.isnan(x_hat_corr).sum().item()
                    )
                    raise RuntimeError("NaN in x_hat_corr after prox correction")
            except Exception as e:
                logger.error(
                    "forward_stage_c: prox_fn failed at epoch %d — %s. "
                    "Falling back to uncorrected x_hat_mix.", epoch, e
                )
                x_hat_corr   = x_hat_mix
                prox_applied = False
        else:
            logger.warning(
                "forward_stage_c: prox_fn=None at epoch %d — "
                "loss computed on uncorrected x̂_mix (deviates from PROX-USAGE Stage C spec)",
                epoch,
            )

        # 4. Post-prox consistency loss: ‖A(x̂_corr) − y‖²
        x_hat_corr_4d = x_hat_corr.view(B, 1, 28, 28)
        Ax_corr   = self.A.forward(x_hat_corr_4d)
        cons_loss = torch.mean((Ax_corr - y_degraded) ** 2)
        residual_post_prox = cons_loss.item()   # raw, for P_pre_post_prox

        if torch.isnan(cons_loss):
            logger.error("forward_stage_c: NaN in cons_loss at epoch %d", epoch)
            raise RuntimeError("NaN in Stage C post-prox consistency loss")

        # 5. Image reconstruction loss: ‖x̂_corr − x‖²
        img_loss = torch.mean((x_hat_corr - x_clean_flat) ** 2)

        if torch.isnan(img_loss):
            logger.error("forward_stage_c: NaN in img_loss at epoch %d", epoch)
            raise RuntimeError("NaN in Stage C image reconstruction loss")

        # 6. Alive penalty: max(0, 1.5 − N_eff)
        w_mean = w.mean(dim=0).clamp(min=1e-8)
        w_mean = w_mean / w_mean.sum()
        entropy      = -(w_mean * torch.log(w_mean)).sum()
        neff         = torch.exp(entropy)
        alive_penalty = torch.clamp(1.5 - neff, min=0.0)

        if torch.isnan(neff):
            logger.error("forward_stage_c: NaN in Neff at epoch %d", epoch)
            raise RuntimeError("NaN in Stage C Neff computation")

        # [SC-TRANS-CAL] 7. Transport: SW2(x̂_corr, x_clean)
        if self.lambda_trans_c > 0:
            try:
                transport_c = sliced_wasserstein_distance(
                    x_hat_corr.detach(),   # stop-grad: SW2 geometry term only
                    x_clean_flat,
                    num_projections=self.sw2_projections,
                )
                if torch.isnan(transport_c):
                    logger.error(
                        "forward_stage_c: NaN in SW2 at epoch %d — skipping", epoch
                    )
                    transport_c = torch.tensor(0.0, device=x_clean.device)
            except Exception as e:
                logger.error(
                    "forward_stage_c: SW2 failed at epoch %d: %s — skipping", epoch, e
                )
                transport_c = torch.tensor(0.0, device=x_clean.device)
        else:
            transport_c = torch.tensor(0.0, device=x_clean.device)

        # [SC-TRANS-CAL] 8. Calibration: Energy Score on x̂_corr batch
        if self.lambda_cal_c > 0:
            try:
                cal_c = energy_score(
                    x_hat_corr,
                    x_clean_flat.mean(dim=0, keepdim=True),  # (1, d) batch-mean reference
                )
                if torch.isnan(cal_c):
                    logger.error(
                        "forward_stage_c: NaN in ES at epoch %d — skipping", epoch
                    )
                    cal_c = torch.tensor(0.0, device=x_clean.device)
            except Exception as e:
                logger.error(
                    "forward_stage_c: ES failed at epoch %d: %s — skipping", epoch, e
                )
                cal_c = torch.tensor(0.0, device=x_clean.device)
        else:
            cal_c = torch.tensor(0.0, device=x_clean.device)

        # [NLL-ANCHOR] 9a. Mixture NLL anchor — prevents catastrophic forgetting.
        # flow.forward(x_clean, y_degraded) returns (log_q [B,], log_q_experts [B,K]).
        # Even lambda=0.001 is enough to keep experts near their Stage B distribution.
        if self.lambda_nll_c > 0:
            try:
                log_q_c, _ = flow.forward(x_clean, y_degraded)
                nll_anchor = -log_q_c.mean()
                if torch.isnan(nll_anchor) or torch.isinf(nll_anchor):
                    logger.error(
                        "forward_stage_c: NaN/Inf in NLL anchor at epoch %d — skipping",
                        epoch
                    )
                    nll_anchor = torch.tensor(0.0, device=x_clean.device)
            except Exception as e:
                logger.error(
                    "forward_stage_c: NLL anchor failed at epoch %d: %s — skipping",
                    epoch, e
                )
                nll_anchor = torch.tensor(0.0, device=x_clean.device)
        else:
            nll_anchor = torch.tensor(0.0, device=x_clean.device)

        # 9. Total
        loss = (
              self.lambda_cons_pre  * cons_pre_loss    # [PRE-PROX] expert direct signal
            + self.lambda_cons      * cons_loss         # post-prox consistency
            + self.lambda_img       * img_loss          # reconstruction quality
            + self.lambda_alive     * alive_penalty     # gate diversity
            + self.lambda_trans_c   * transport_c       # [SC-TRANS-CAL] geometry
            + self.lambda_cal_c     * cal_c             # [SC-TRANS-CAL] spread
            + self.lambda_nll_c     * nll_anchor        # [NLL-ANCHOR] forgetting prevention
        )

        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(
                "forward_stage_c: total loss is NaN/Inf at epoch %d — "
                "cons_pre=%.4f, cons=%.4f, img=%.4f, alive=%.4f, "
                "trans_c=%.4f, cal_c=%.4f, nll_anchor=%.4f, neff=%.4f",
                epoch, cons_pre_loss.item(), cons_loss.item(), img_loss.item(),
                alive_penalty.item(), transport_c.item(), cal_c.item(),
                nll_anchor.item(), neff.item(),
            )
            raise RuntimeError("NaN/Inf in Stage C reconstruction-first loss")

        loss_dict = {
            "loss":               loss.item(),
            "cons_c1":            cons_loss.item(),
            "cons_pre_loss":      cons_pre_loss.item(),
            "img_loss":           img_loss.item(),
            "alive_penalty":      alive_penalty.item(),
            "neff_c1":            neff.item(),
            "trans_loss_c":       transport_c.item(),
            "cal_loss_c":         cal_c.item(),
            "residual_pre_prox":  residual_pre_prox,
            "residual_post_prox": residual_post_prox,
            "lam_cons":           self.lambda_cons,
            "lam_cons_pre":       self.lambda_cons_pre,
            "lam_img":            self.lambda_img,
            "lam_alive":          self.lambda_alive,
            "lam_trans_c":        self.lambda_trans_c,
            "lam_cal_c":          self.lambda_cal_c,
            "nll_anchor":         nll_anchor.item(),   # [NLL-ANCHOR]
            "lam_nll_c":          self.lambda_nll_c,
            "prox_applied":       prox_applied,
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
    freeze_gate_epochs: int = 5,
) -> dict:
    """
    Stage C — Reconstruction-first joint fine-tuning (L_C1).

    Uses forward_stage_c() instead of forward() — no mixture NLL, no SW2, no
    calibration. Loss = λ_cons·‖A(x̂_mix)−y‖² + λ_img·‖x̂_mix−x‖² + λ_alive·max(0,1.5−Neff)

    Gate-frozen warmup: gate is frozen for the first `freeze_gate_epochs` epochs,
    so expert weights can adapt to the reconstruction objective before gate routing
    is allowed to shift. This prevents early-epoch gate drift toward the
    lowest-reconstruction-error expert before the alive penalty can act.

    Args:
        model:                CSMF model (must implement sample_all_experts())
        loss_fn:              HybridLoss instance (must have lambda_img, lambda_alive)
        train_loader:         DataLoader
        cfg:                  StageConfig for Stage C
        checkpoint_dir:       directory for checkpoints
        device:               torch device
        stage_B_checkpoint:   optional path to best Stage B checkpoint to load
        freeze_gate_epochs:   epochs to keep gate frozen before unfreezing (default 5)

    Returns:
        dict with 'best_epoch', 'best_metric', 'last_loss_dict'
    """
    logger.info(
        "=== Stage C: Reconstruction-first fine-tuning (max %d epochs, LR=%.1e, "
        "gate_warmup=%d) ===",
        cfg.max_epochs, cfg.lr, freeze_gate_epochs,
    )

    if stage_B_checkpoint:
        load_checkpoint(stage_B_checkpoint, model)

    # Phase 1 setup: experts partially unfrozen, gate frozen
    freeze_experts(model)
    unfreeze_last_blocks(model, n_blocks=cfg.n_unfreeze_blocks)
    freeze_gate(model)
    logger.info("Stage C: gate frozen for warmup (%d epochs)", freeze_gate_epochs)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr
    )

    best_metric = float("inf")
    best_epoch  = 0
    no_improve  = 0
    last_loss_dict = {}
    gate_unfrozen  = False

    for epoch in range(cfg.max_epochs):
        # Phase 2: unfreeze gate after warmup
        if epoch == freeze_gate_epochs and not gate_unfrozen:
            unfreeze_gate(model)
            gate_unfrozen = True
            # Rebuild optimizer to include newly unfrozen gate params
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr
            )
            logger.info("Stage C: gate unfrozen at epoch %d", epoch)

        model.train()
        epoch_losses = _run_epoch_stage_c(
            model, loss_fn, train_loader, optimizer, epoch, device,
        )

        metric = epoch_losses["cons_c1"] + epoch_losses["img_loss"]
        logger.info(
            "Stage C | epoch %d | loss=%.4f | cons=%.4f | img=%.4f | "
            "alive=%.4f | neff=%.4f | gate_frozen=%s",
            epoch, epoch_losses["loss"], epoch_losses["cons_c1"],
            epoch_losses["img_loss"], epoch_losses["alive_penalty"],
            epoch_losses["neff_c1"], str(not gate_unfrozen),
        )

        if metric < best_metric - 1e-5:
            best_metric = metric
            best_epoch  = epoch
            no_improve  = 0
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


# ---------------------------------------------------------------------------
# [SC-RECFIRST] Stage C epoch runner — uses forward_stage_c()
# ---------------------------------------------------------------------------

def _run_epoch_stage_c(
    model: nn.Module,
    loss_fn: HybridLoss,
    loader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: torch.device,
) -> dict:
    """
    Run one Stage C epoch using loss_fn.forward_stage_c().
    Returns averaged loss_dict with keys: loss, cons_c1, img_loss,
    alive_penalty, neff_c1, lam_cons, lam_img, lam_alive.
    """
    totals    = {}
    n_batches = 0

    for x_clean, y_degraded in loader:
        x_clean    = x_clean.to(device)
        y_degraded = y_degraded.to(device)

        optimizer.zero_grad()

        try:
            loss, loss_dict = loss_fn.forward_stage_c(
                model, x_clean, y_degraded, epoch=epoch
            )
        except RuntimeError as exc:
            logger.error(
                "_run_epoch_stage_c: loss failed at epoch %d — %s", epoch, exc
            )
            raise

        loss.backward()

        nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), max_norm=5.0
        )
        optimizer.step()

        for k, v in loss_dict.items():
            totals[k] = totals.get(k, 0.0) + v
        n_batches += 1

    if n_batches == 0:
        logger.error("_run_epoch_stage_c: no batches processed at epoch %d", epoch)
        raise RuntimeError("Empty data loader — no batches in Stage C epoch")

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
