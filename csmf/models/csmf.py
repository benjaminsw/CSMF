# =============================================================================
# Version: WP3.1-CSMF-v1.1 | Abbr: CSMF-MAIN
# Description: Conditional Sequential Mixture of Flows — main model class
# Changelog:
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

        if base_dist is None:
            self.base_dist = torch.distributions.Normal(
                torch.zeros(1), torch.ones(1)
            )
        else:
            self.base_dist = base_dist

        logger.info(
            f"CSMF initialised | K={self.K} | dim={self.dim} | "
            f"base_dist={type(self.base_dist).__name__}"
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
            return x
        return x.flatten(1) if x.dim() > 2 else x

    def _expert_forward(
        self,
        expert: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x_in = self._prepare_x_for_expert(expert, x)
        cond = y if self._expects_raw_y(expert) else h
        out = expert.forward(x_in, cond)

        if not isinstance(out, tuple):
            raise RuntimeError(f"Unexpected forward output type from {type(expert).__name__}: {type(out)}")

        if len(out) == 2:
            z, log_det = out
            return z, log_det, None
        if len(out) == 3:
            z, log_det, log_prob = out
            return z, log_det, log_prob
        if len(out) == 4:
            z, _, log_det, log_prob = out
            return z, log_det, log_prob

        raise RuntimeError(f"Unexpected forward output length from {type(expert).__name__}: {len(out)}")

    def _expert_inverse(self, expert: nn.Module, z: torch.Tensor, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if self._expects_raw_y(expert):
            sig = inspect.signature(expert.inverse)
            if len(sig.parameters) <= 2:
                return expert.inverse(z, y)
            return expert.inverse(z, y)
        return expert.inverse(z, h)

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
            z, log_det, log_prob = self._expert_forward(expert, x, y, h)

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
                    x_img = expert.sample(1, y[i:i+1]).flatten(1)
                    x_samples[i, s] = x_img.squeeze(0)
                else:
                    z = self.base_dist.sample((self.dim,)).to(dev)
                    x = self._expert_inverse(expert, z.unsqueeze(0), y[i:i+1], h[i:i+1])
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

        payload = {
            "state_dict": self.state_dict(),
            "stage":      stage,
            "epoch":      epoch,
            "loss":       loss,
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
        optimizer: torch.optim.Optimizer,
        hybrid_loss,
        epochs: int,
        lambda_cons: float = 0.05,
        val_loader=None,
        patience: int = 5,
        ckpt_path: str = "checkpoints/csmf_stage_A.pth",
    ) -> None:
        """
        Stage A: Train each expert independently with weak consistency.

            Loss_k = -log q_k(x|y) + lambda_cons * ||A x_hat - y||^2

        Experts are frozen after stage completes.

        Args:
            dataloader:  DataLoader yielding (x_clean, y_deg)
            optimizer:   optimizer over ALL expert parameters
            hybrid_loss: HybridLoss instance — used for its .A forward model
            epochs:      epochs per expert
            lambda_cons: weak consistency weight (default 0.05)
            val_loader:  optional validation DataLoader for early stopping
            patience:    early stopping patience (epochs without val improvement)
            ckpt_path:   checkpoint save path
        """
        logger.info(
            f"=== Stage A | K={self.K} experts | epochs={epochs} | "
            f"lambda_cons={lambda_cons} ==="
        )

        for k, expert in enumerate(self.experts):
            logger.info(f"  Expert {k+1}/{self.K} training start")
            expert.train()

            best_val_nll     = float("inf")
            patience_counter = 0
            last_nll         = float("inf")

            for epoch in range(epochs):
                total_nll  = 0.0
                total_cons = 0.0
                n_batches  = 0

                for x_clean, y_deg in dataloader:
                    optimizer.zero_grad()

                    h = self.conditioner(y_deg)
                    z, log_det, log_prob = self._expert_forward(expert, x_clean, y_deg, h)

                    if torch.any(torch.isnan(log_det)):
                        logger.error(
                            f"Stage A | expert={k} | epoch={epoch} | "
                            f"NaN log_det — skipping batch"
                        )
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
                            x_hat = self._expert_inverse(expert, z_base, y_deg, h)
                            if x_hat.dim() == 2:
                                x_hat = x_hat.view(-1, 1, 28, 28)
                    Ax    = hybrid_loss.A.forward(x_hat)
                    cons  = torch.mean((Ax - y_deg) ** 2)

                    loss = nll + lambda_cons * cons

                    if torch.isnan(loss):
                        logger.error(
                            f"Stage A | expert={k} | epoch={epoch} | "
                            f"NaN total loss — skipping batch"
                        )
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        expert.parameters(), max_norm=1.0
                    )
                    optimizer.step()

                    total_nll  += nll.item()
                    total_cons += cons.item()
                    n_batches  += 1

                if n_batches == 0:
                    logger.error(
                        f"Stage A | expert={k} | epoch={epoch} | "
                        f"All batches skipped — check NaN sources"
                    )
                    continue

                avg_nll  = total_nll  / n_batches
                avg_cons = total_cons / n_batches
                last_nll = avg_nll
                logger.info(
                    f"  Expert {k} | Epoch {epoch+1}/{epochs} | "
                    f"NLL={avg_nll:.4f} | Cons={avg_cons:.4f}"
                )

                # Early stopping on validation NLL
                if val_loader is not None:
                    val_nll = self._eval_nll_single(expert, val_loader)
                    logger.info(
                        f"  Expert {k} | Epoch {epoch+1} | ValNLL={val_nll:.4f}"
                    )
                    if val_nll < best_val_nll - 1e-4:
                        best_val_nll     = val_nll
                        patience_counter = 0
                        self.save_checkpoint(
                            ckpt_path, stage="A", epoch=epoch,
                            loss=best_val_nll, extra={"expert_k": k}
                        )
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger.info(
                                f"  Expert {k} | Early stopping at epoch {epoch+1} "
                                f"(patience={patience})"
                            )
                            break

        # Freeze all expert parameters
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = False

        logger.info("Stage A complete — all expert parameters frozen.")

        if val_loader is None:
            self.save_checkpoint(
                ckpt_path, stage="A", epoch=epochs - 1, loss=last_nll
            )

    def train_stage_B(
        self,
        dataloader,
        optimizer: torch.optim.Optimizer,
        hybrid_loss,
        epochs: int,
        val_loader=None,
        patience: int = 5,
        ckpt_path: str = "checkpoints/csmf_stage_B.pth",
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

        for epoch in range(epochs):
            total_loss = 0.0
            total_neff = 0.0
            n_batches  = 0

            for x_clean, y_deg in dataloader:
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
                    w    = self._gate_weights(y_deg)
                    neff = self._compute_neff(w).mean().item()

                total_loss += loss.item()
                total_neff += neff
                n_batches  += 1

            if n_batches == 0:
                logger.error(f"Stage B | epoch={epoch} | All batches skipped")
                continue

            avg_loss = total_loss / n_batches
            avg_neff = total_neff / n_batches
            last_loss = avg_loss
            logger.info(
                f"Stage B | Epoch {epoch+1}/{epochs} | "
                f"Loss={avg_loss:.4f} | Neff={avg_neff:.3f}"
            )

            if avg_neff < 1.1:
                logger.warning(
                    f"Stage B | Epoch {epoch+1} | "
                    f"Gate collapse: Neff={avg_neff:.3f} < 1.1"
                )

            if val_loader is not None:
                val_loss = self._eval_hybrid_loss(hybrid_loss, val_loader, epoch)
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
                        break

        logger.info("Stage B complete.")
        if val_loader is None:
            self.save_checkpoint(
                ckpt_path, stage="B", epoch=epochs - 1, loss=last_loss
            )

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
    def _eval_nll_single(self, expert: nn.Module, val_loader) -> float:
        """Evaluate mean NLL of a single expert on val_loader."""
        expert.eval()
        total = 0.0
        n     = 0

        for x_clean, y_deg in val_loader:
            h = self.conditioner(y_deg)
            z, log_det, log_prob = self._expert_forward(expert, x_clean, y_deg, h)
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
    def _eval_hybrid_loss(
        self, hybrid_loss, val_loader, epoch: int
    ) -> float:
        """Evaluate mean hybrid loss on val_loader."""
        self.eval()
        total = 0.0
        n     = 0

        for x_clean, y_deg in val_loader:
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
