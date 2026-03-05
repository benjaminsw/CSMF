"""
ConditionalRealNVP for MNIST Inverse Problems - Multi-Scale Architecture

Version: WP0.3-CondRNVP-v2.1.6
Abbr: COND-RNVP
Last Modified: 2026-02-28
Changelog:
  v2.1.6 (2026-02-28): Rate-limited batch size mismatch warning in inverse() — was flooding
                        logs every epoch during sampling; added _warn_batch counter; warning
                        now logged only once per instance
  v2.1.5 (2026-02-28): Rate-limited recompute warning — added _warn_no_h counter; logged once
  v2.1.3 (2026-02-26): [B] Spec-compliant h caching — forward/inverse accept optional h=
  v2.1.2 (2026-02-26): BUG FIX — removed [F2] z_flat clamp (-50,50) in ScaleBlock inverse;
                        clamp corrupted the coupling unchanged half (mask side) causing inv_err=6.91;
                        replaced with NaN/Inf-only guard (nan_to_num, no value clamping);
                        removed [F3] h clamp (-10,10) in inverse() — same root cause, corrupts
                        s,t recomputation; [F2]/[F3] were treating symptoms not root cause (fixed in COND-COUP v1.4.1)
  v2.1.1 (2026-02-22): [F1] Reduced s_max 10.0→2.0 to prevent exp() explosion across 9 coupling layers;
                        [F2] Added z_flat clamping (-50,50) before inverse coupling loop in ScaleBlock;
                        [F3] Added h clamping (-10,10) after conditioner call in inverse();
                        [A1] Added per-coupling NaN detection in inverse pass with error logging;
                        [A2] Added input normalization range check in forward() with WARNING if x outside [-1,1];
                        [A3] Added h.norm() logging for first 50 batches via _h_log_count counter.
  v2.1.0 (2025-02-04): Added comprehensive debugging for invertibility and log-det issues
  v2.0.1 (2025-02-03): Fixed reshape bug in ScaleBlock inverse
  v2.0 (2025-10-25): Multi-scale with squeeze/unsqueeze and variable factoring
  v1.0 (2025-10-25): Initial flat architecture
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional
import logging

try:
    from configs.mnist_config import MNIST_CONFIG
    from csmf.conditioning.conditioning_networks import MNISTConditioner
    from csmf.flows.coupling_layers import ConditionalAffineCoupling
except ImportError as e:
    logging.error(f"Failed to import dependencies: {e}")
    raise

logger = logging.getLogger(__name__)


class ScaleBlock(nn.Module):
    """Multi-scale block: N coupling layers with optional squeeze operation."""
    
    def __init__(
        self,
        n_layers: int,
        channels: int,
        spatial_dim: int,
        h_dim: int,
        hidden_dims: List[int],
        apply_squeeze: bool = False,
        debug: bool = False
    ):
        super().__init__()
        
        self.n_layers = n_layers
        self.channels = channels
        self.spatial_dim = spatial_dim
        self.h_dim = h_dim
        self.apply_squeeze = apply_squeeze
        self.debug = debug
        
        # Create coupling layers
        dim = channels * spatial_dim * spatial_dim
        self.coupling_layers = nn.ModuleList()
        
        for i in range(n_layers):
            split_dim = dim // 2
            layer = ConditionalAffineCoupling(
                dim=dim,
                split_dim=split_dim,
                h_dim=h_dim,
                hidden_dims=hidden_dims,
                use_batch_norm=False,  # Disable BN for debugging
                s_max=0.5,  # [F1] Reduced from 10.0: exp(10)≈22026 explodes across 9 layers
                debug=self.debug
            )
            self.coupling_layers.append(layer)
            if self.debug:
                logger.debug(f"ScaleBlock: layer {i+1}/{n_layers}, dim={dim}, split={split_dim}")
    
    def forward(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        reverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward through coupling layers with optional squeeze."""
        
        if self.debug:
            logger.debug("=" * 70)
            logger.debug(f"[SCALE DEBUG] Block forward (reverse={reverse})")
            logger.debug(f"  Input: shape={z.shape}, norm={z.norm().item():.6f}")
        
        B, C, H, W = z.shape
        log_det = torch.zeros(B, device=z.device)
        
        # Store input for invertibility check
        if self.debug and not reverse:
            z_input_original = z.clone()
        
        # Flatten for coupling layers
        z_flat = z.reshape(B, -1)
        
        if not reverse:
            # ========== FORWARD: coupling → squeeze ==========
            for i, coupling in enumerate(self.coupling_layers):
                if self.debug:
                    z_before = z_flat.clone()
                
                z_flat, ld = coupling.forward(z_flat, h, reverse=False)
                log_det = log_det + ld
                
                if self.debug:
                    logger.debug(f"  Coupling {i+1}/{self.n_layers}:")
                    logger.debug(f"    Output norm: {z_flat.norm().item():.6f}")
                    logger.debug(f"    Log-det: mean={ld.mean().item():.4f}, sum={log_det.mean().item():.4f}")
                    
                    # Per-layer invertibility check
                    z_test_inv, _ = coupling.forward(z_flat, h, reverse=True)
                    inv_err = (z_before - z_test_inv).abs().max().item()
                    if inv_err > 1e-4:
                        logger.error(f"    ❌ Layer {i+1} invertibility FAILED: error={inv_err:.2e}")
                    else:
                        logger.debug(f"    ✓ Layer {i+1} invertible: error={inv_err:.2e}")
            
            # Reshape back to spatial
            z_out = z_flat.reshape(B, C, H, W)
            
            # Apply squeeze
            if self.apply_squeeze:
                if self.debug:
                    #logger.debug(f"  Applying squeeze: {z_out.shape} → ", end="")
                    shape_before = z_out.shape
                
                z_out = self._squeeze(z_out)
                
                if self.debug:
                    #logger.debug(f"{z_out.shape}")
                    logger.debug(f"  Applying squeeze: {shape_before} → {z_out.shape}")
                    
                    # Squeeze invertibility check
                    z_unsqueeze_test = self._unsqueeze(z_out)
                    squeeze_err = (z_flat.reshape(B, C, H, W) - z_unsqueeze_test).abs().max().item()
                    if squeeze_err > 1e-6:
                        logger.error(f"  ❌ Squeeze NOT invertible: error={squeeze_err:.2e}")
                    else:
                        logger.debug(f"  ✓ Squeeze invertible: error={squeeze_err:.2e}")
        
        else:
            # ========== INVERSE: unsqueeze → coupling (reversed) ==========
            z_out = z_flat.reshape(B, C, H, W)
            
            if self.apply_squeeze:
                if self.debug:
                    #logger.debug(f"  Applying unsqueeze: {z_out.shape} → ", end="")
                    shape_before = z_out.shape
                
                z_out = self._unsqueeze(z_out)
                
                if self.debug:
                    #logger.debug(f"{z_out.shape}")
                    logger.debug(f"  Applying unsqueeze: {shape_before} → {z_out.shape}")
                    
            
            # Re-flatten with correct shape
            z_flat = z_out.reshape(B, -1)
            
            # NaN/Inf guard only — do NOT clamp values (clamping corrupts the
            # coupling's unchanged half and breaks mathematical invertibility)
            if torch.any(torch.isnan(z_flat)) or torch.any(torch.isinf(z_flat)):
                n_bad = (torch.isnan(z_flat) | torch.isinf(z_flat)).sum().item()
                logger.error(
                    f"[ScaleBlock inverse] NaN/Inf in z_flat before coupling ({n_bad} values). "
                    f"Check scale net init and s_max. Replacing with zeros."
                )
                z_flat = torch.nan_to_num(z_flat, nan=0.0, posinf=0.0, neginf=0.0)
            
            if self.debug:
                logger.debug(f"  Flattened for coupling: shape={z_flat.shape}")
            
            # Apply couplings in reverse order
            for i, coupling in enumerate(reversed(self.coupling_layers)):
                z_flat, ld = coupling.forward(z_flat, h, reverse=True)
                log_det = log_det + ld
                
                # [A1] Per-coupling NaN detection in inverse pass
                if torch.any(torch.isnan(z_flat)) or torch.any(torch.isinf(z_flat)):
                    logger.error(
                        f"[A1] NaN/Inf detected after inverse coupling "
                        f"{self.n_layers - i}/{self.n_layers} in ScaleBlock. "
                        f"NaN count: {torch.isnan(z_flat).sum().item()}, "
                        f"Inf count: {torch.isinf(z_flat).sum().item()}"
                    )
                    z_flat = torch.nan_to_num(z_flat, nan=0.0, posinf=1e4, neginf=-1e4)
                
                if self.debug:
                    logger.debug(f"  Coupling {self.n_layers-i}/{self.n_layers} (reversed):")
                    logger.debug(f"    Output norm: {z_flat.norm().item():.6f}")
                    logger.debug(f"    Log-det contribution: {ld.mean().item():.4f}")
            
            # Reshape to match output
            z_out = z_flat.reshape(B, z_out.shape[1], z_out.shape[2], z_out.shape[3])
        
        if self.debug:
            logger.debug(f"  Block output: shape={z_out.shape}, total log_det={log_det.mean().item():.4f}")
            logger.debug("=" * 70)
        
        return z_out, log_det
    
    def _squeeze(self, z: torch.Tensor) -> torch.Tensor:
        """Squeeze: [B, C, H, W] → [B, 4C, H/2, W/2]"""
        B, C, H, W = z.shape
        z = z.reshape(B, C, H // 2, 2, W // 2, 2)
        z = z.permute(0, 1, 3, 5, 2, 4)
        z = z.reshape(B, C * 4, H // 2, W // 2)
        return z
    
    def _unsqueeze(self, z: torch.Tensor) -> torch.Tensor:
        """Unsqueeze: [B, 4C, H, W] → [B, C, 2H, 2W]"""
        B, C4, H, W = z.shape
        C = C4 // 4
        z = z.reshape(B, C, 2, 2, H, W)
        z = z.permute(0, 1, 4, 2, 5, 3)
        z = z.reshape(B, C, H * 2, W * 2)
        return z


class ConditionalRealNVP(nn.Module):
    """Conditional RealNVP flow with multi-scale architecture."""
    
    def __init__(
        self,
        h_dim: int = 64,
        hidden_dims: List[int] = None,
        config: dict = None,
        debug: bool = False
    ):
        """Initialize ConditionalRealNVP with debug mode."""
        super().__init__()
        
        if h_dim <= 0:
            raise ValueError(f"h_dim must be positive, got {h_dim}")
        
        if hidden_dims is None:
            hidden_dims = [256, 256]
        if config is None:
            config = MNIST_CONFIG
        
        self.h_dim = h_dim
        self.hidden_dims = hidden_dims
        self.config = config
        self._cached_h = None
        self.debug = debug
        self._h_log_count  = 0  # [A3] Counter for h.norm() logging during first 50 batches
        self._warn_no_h    = 0  # [v2.1.5] Counter to suppress repeated recompute warnings
        self._warn_batch   = 0  # [v2.1.6] Counter to suppress batch size mismatch warnings
        
        logger.info(f"Initializing ConditionalRealNVP v2.1.0: h_dim={h_dim}, debug={debug}")
        
        # Conditioning network
        self.conditioner = MNISTConditioner(h_dim=h_dim, debug=debug)
        
        # Multi-scale blocks
        self.scale1 = ScaleBlock(3, 1, 28, h_dim, hidden_dims, apply_squeeze=True, debug=debug)
        self.scale2 = ScaleBlock(3, 2, 14, h_dim, hidden_dims, apply_squeeze=True, debug=debug)
        self.scale3 = ScaleBlock(3, 4, 7, h_dim, hidden_dims, apply_squeeze=False, debug=debug)
        
        self.register_buffer('log_2pi', torch.log(torch.tensor(2.0 * torch.pi)))
        
        logger.info("ConditionalRealNVP v2.1.0 initialization complete")
    
    def _factor_out(self, z: torch.Tensor, factor_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Factor out channels."""
        B, C, H, W = z.shape
        n_factor = int(C * factor_ratio)
        n_keep = C - n_factor
        
        z_kept = z[:, :n_keep, :, :]
        z_factored = z[:, n_keep:, :, :]
        
        if self.debug:
            logger.debug(f"[FACTOR] Split [{C},{H},{W}] → kept [{n_keep},{H},{W}] + factored [{n_factor},{H},{W}]")
        
        return z_kept, z_factored
    
    def _unfactor(self, z_kept: torch.Tensor, z_factored: torch.Tensor) -> torch.Tensor:
        """Unfactor: concatenate channels."""
        z = torch.cat([z_kept, z_factored], dim=1)
        
        if self.debug:
            logger.debug(f"[UNFACTOR] Combined {z_kept.shape} + {z_factored.shape} → {z.shape}")
        
        return z
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        compute_h: bool = True,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Forward transform with debugging.
        
        Args:
            h: Pre-computed conditioning vector from external conditioner (e.g. CSMF's
               shared MNISTConditioner). If provided, skips internal self.conditioner(y)
               entirely — satisfies WP0 spec "cache h per mini-batch" requirement.
               If None, falls back to self.conditioner(y) for standalone use.
        """
        
        # Validation
        if x.ndim != 4:
            raise ValueError(f"x must be 4D, got {x.shape}")
        if x.shape != (x.shape[0], 1, 28, 28):
            raise ValueError(f"x must be [B,1,28,28], got {x.shape}")
        
        if self.debug:
            logger.debug("=" * 70)
            logger.debug("[REALNVP FORWARD]")
            logger.debug(f"  Input x: shape={x.shape}, norm={x.norm().item():.6f}")
            x_original = x.clone()  # For final invertibility check
        
        # [A2] Input normalization range check — MNIST should be dequantized+logit, ~[-5, 5]
        x_min, x_max = x.min().item(), x.max().item()
        if x_min < -10.0 or x_max > 10.0:
            logger.warning(
                f"[A2] Input x out of expected range: min={x_min:.3f}, max={x_max:.3f}. "
                f"Expected dequantized+logit-transformed input in ~[-5,5]. "
                f"Check upstream preprocessing (raw [0,1] or [0,255] will cause NaN)."
            )
        
        # Conditioning — use external h if provided (spec-compliant caching path)
        if h is not None:
            # [B] External h from CSMF's shared conditioner — no recomputation
            self._cached_h = h
            if self.debug:
                logger.debug(f"  Conditioning h: source=external, shape={h.shape}, norm={h.norm().item():.6f}")
        elif compute_h:
            h = self.conditioner(y)
            self._cached_h = h
            if self.debug:
                logger.debug(f"  Conditioning h: source=internal, shape={h.shape}, norm={h.norm().item():.6f}")
        else:
            if self._cached_h is None:
                logger.error("forward(): compute_h=False but no cached h and no external h provided")
                raise RuntimeError("No cached h — pass h= or set compute_h=True")
            h = self._cached_h
            if self.debug:
                logger.debug(f"  Conditioning h: source=cached, shape={h.shape}, norm={h.norm().item():.6f}")
        
        B = x.shape[0]
        log_det_total = torch.zeros(B, device=x.device)
        z_factored_list = []
        
        try:
            # Scale 1
            if self.debug:
                logger.debug("\n[SCALE 1] Processing...")
            
            z = x.clone()
            z, log_det = self.scale1.forward(z, h, reverse=False)
            log_det_total = log_det_total + log_det
            
            if self.debug:
                logger.debug(f"[SCALE 1] Output: shape={z.shape}, log_det={log_det.mean().item():.4f}")
            
            # Factor
            z, z_factor1 = self._factor_out(z, factor_ratio=0.5)
            z_factored_list.append(z_factor1)
            
            # Scale 2
            if self.debug:
                logger.debug("\n[SCALE 2] Processing...")
            
            z, log_det = self.scale2.forward(z, h, reverse=False)
            log_det_total = log_det_total + log_det
            
            if self.debug:
                logger.debug(f"[SCALE 2] Output: shape={z.shape}, log_det={log_det.mean().item():.4f}")
            
            # Factor
            z, z_factor2 = self._factor_out(z, factor_ratio=0.5)
            z_factored_list.append(z_factor2)
            
            # Scale 3
            if self.debug:
                logger.debug("\n[SCALE 3] Processing...")
            
            z, log_det = self.scale3.forward(z, h, reverse=False)
            log_det_total = log_det_total + log_det
            
            if self.debug:
                logger.debug(f"[SCALE 3] Output: shape={z.shape}, log_det={log_det.mean().item():.4f}")
            
            z_final = z
            
            # Stability check
            if torch.any(torch.isnan(z_final)) or torch.any(torch.isinf(z_final)):
                logger.warning("NaN/Inf in z_final")
                z_final = torch.clamp(z_final, min=-1e6, max=1e6)
            
        except Exception as e:
            logger.error(f"Forward failed: {e}")
            raise RuntimeError(f"Forward failed: {e}")
        
        # Compute log-probabilities
        log_pz_list = []
        
        # Final latent
        z_final_flat = z_final.reshape(B, -1)
        z_squared = torch.clamp(torch.sum(z_final_flat ** 2, dim=1), max=400.0)
        dim_final = z_final_flat.shape[1]
        log_pz_final = -0.5 * (dim_final * self.log_2pi + z_squared)
        log_pz_list.append(log_pz_final)
        
        if self.debug:
            logger.debug(f"\n[LOG-PROB] Final latent: dim={dim_final}, log_pz={log_pz_final.mean().item():.4f}")
        
        # Factored variables
        for i, z_fact in enumerate(z_factored_list):
            z_fact_flat = z_fact.reshape(B, -1)
            z_squared = torch.clamp(torch.sum(z_fact_flat ** 2, dim=1), max=400.0)
            dim_fact = z_fact_flat.shape[1]
            log_pz_fact = -0.5 * (dim_fact * self.log_2pi + z_squared)
            log_pz_list.append(log_pz_fact)
            
            if self.debug:
                logger.debug(f"[LOG-PROB] Factored {i+1}: dim={dim_fact}, log_pz={log_pz_fact.mean().item():.4f}")
        
        # Total
        log_pz_total = sum(log_pz_list)
        log_prob_total = log_pz_total + log_det_total
        
        if self.debug:
            logger.debug(f"\n[TOTAL]")
            logger.debug(f"  log_pz: {log_pz_total.mean().item():.4f}")
            logger.debug(f"  log_det: {log_det_total.mean().item():.4f}")
            logger.debug(f"  log_prob: {log_prob_total.mean().item():.4f}")
            
            # End-to-end invertibility check
            logger.debug("\n[INVERTIBILITY CHECK]")
            x_recon = self.inverse(z_final, z_factored_list, y)
            inv_err = (x_original - x_recon).abs().max().item()
            inv_mean = (x_original - x_recon).abs().mean().item()
            logger.debug(f"  Max error: {inv_err:.2e} (threshold: 1e-4)")
            logger.debug(f"  Mean error: {inv_mean:.2e}")
            if inv_err > 1e-4:
                logger.error(f"  ❌ END-TO-END INVERTIBILITY FAILED!")
            else:
                logger.debug(f"  ✓ End-to-end invertible")
        
        # NaN check
        nan_mask = torch.isnan(log_prob_total) | torch.isinf(log_prob_total)
        if torch.any(nan_mask):
            logger.warning(f"NaN/Inf in {nan_mask.sum()} samples")
            log_prob_total[nan_mask] = -100.0
        
        if self.debug:
            logger.debug("=" * 70)
        
        return z_final, z_factored_list, log_det_total, log_prob_total
    
    def inverse(
        self,
        z: torch.Tensor,
        z_factored_list: List[torch.Tensor],
        y: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inverse transform with debugging.
        
        Args:
            h: Pre-computed conditioning vector from external conditioner. If provided,
               uses it directly — same h as forward pass → mathematically correct inverse.
               Fallback chain: external h → cached h → recompute (logs WARNING).
        """
        
        if self.debug:
            logger.debug("=" * 70)
            logger.debug("[REALNVP INVERSE]")
            logger.debug(f"  Input z: shape={z.shape}")
        
        # Validation / graceful fallback for callers that provide only z_final.
        if len(z_factored_list) == 0:
            logger.warning(
                "inverse() received empty z_factored_list; sampling missing "
                "factored latents from N(0, I)."
            )
            # EXP-SANITY may pass flattened base samples (e.g. [B, 784]) instead of
            # image-latent z_final ([B, 4, 7, 7]). Coerce to expected shape so
            # fallback sampling can continue.
            if z.dim() != 4:
                B = z.shape[0]
                z_flat = z.reshape(B, -1)
                z_final_dim = 4 * 7 * 7
                if z_flat.shape[1] < z_final_dim:
                    raise ValueError(
                        f"inverse() expected at least {z_final_dim} latent dims for z_final, "
                        f"got {z_flat.shape[1]}"
                    )
                if z_flat.shape[1] != z_final_dim:
                    logger.warning(
                        "inverse() received non-image z with %d dims; using first %d dims "
                        "as z_final and sampling factored latents.",
                        z_flat.shape[1],
                        z_final_dim,
                    )
                z = z_flat[:, :z_final_dim].reshape(B, 4, 7, 7)

            B = z.shape[0]
            z_factor1 = torch.randn(B, 2, 14, 14, device=z.device, dtype=z.dtype)
            z_factor2 = torch.randn(B, 4, 7, 7, device=z.device, dtype=z.dtype)
            z_factored_list = [z_factor1, z_factor2]
        elif len(z_factored_list) == 1:
            logger.warning(
                "inverse() received one factored latent; sampling the second "
                "missing latent from N(0, I)."
            )
            B = z.shape[0]
            z_factor2 = torch.randn(B, 4, 7, 7, device=z.device, dtype=z.dtype)
            z_factored_list = [z_factored_list[0], z_factor2]
        elif len(z_factored_list) != 2:
            raise ValueError(f"Expected 2 factored variables, got {len(z_factored_list)}")
        
        # h resolution: external → cached → recompute (fallback with WARNING)
        if h is not None:
            h_source = "external"
        elif self._cached_h is not None:
            # [v2.1.4] Validate batch size — sampling uses batch=1 but cache may be from
            # training batch=128; stale h causes coupling cat([x_A, x_B_out]) size mismatch
            if self._cached_h.shape[0] != z.shape[0]:
                if not hasattr(self, '_warn_batch') or self._warn_batch < 1:
                    logger.warning(
                        f"inverse(): cached h batch size {self._cached_h.shape[0]} != "
                        f"z batch size {z.shape[0]} — clearing cache, recomputing from y "
                        f"(warning suppressed after this)"
                    )
                    self._warn_batch = getattr(self, '_warn_batch', 0) + 1
                self._cached_h = None
                h = self.conditioner(y)
                h_source = "recomputed"
            else:
                h = self._cached_h
                h_source = "cached"
        else:
            if self._warn_no_h < 1:
                logger.warning(
                    "inverse(): no external h and no cached h — recomputing from self.conditioner(y). "
                    "This may cause forward/inverse h mismatch. Pass h= from CSMF's conditioner. "
                    "(this warning is suppressed after first occurrence)"
                )
                self._warn_no_h += 1
            h = self.conditioner(y)
            h_source = "recomputed"
        
        # [A3] Log h.norm() stats for first 50 calls — includes source for mismatch diagnosis
        if self._h_log_count < 50:
            logger.info(
                f"[A3] inverse() h stats (call {self._h_log_count + 1}/50): "
                f"source={h_source}, norm={h.norm().item():.4f}, mean={h.mean().item():.4f}, "
                f"std={h.std().item():.4f}, max_abs={h.abs().max().item():.4f}"
            )
            self._h_log_count += 1
        
        try:
            x = z.clone()
            
            # Scale 3 inverse
            if self.debug:
                logger.debug("\n[SCALE 3 INVERSE]")
            x, _ = self.scale3.forward(x, h, reverse=True)
            if self.debug:
                logger.debug(f"  Output: {x.shape}")
            
            # Unfactor scale 2
            z_factor2 = z_factored_list[1]
            x = self._unfactor(x, z_factor2)
            
            # Scale 2 inverse
            if self.debug:
                logger.debug("\n[SCALE 2 INVERSE]")
            x, _ = self.scale2.forward(x, h, reverse=True)
            if self.debug:
                logger.debug(f"  Output: {x.shape}")
            
            # Unfactor scale 1
            z_factor1 = z_factored_list[0]
            x = self._unfactor(x, z_factor1)
            
            # Scale 1 inverse
            if self.debug:
                logger.debug("\n[SCALE 1 INVERSE]")
            x, _ = self.scale1.forward(x, h, reverse=True)
            if self.debug:
                logger.debug(f"  Output: {x.shape}")
            
            # Stability
            if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
                logger.warning("NaN/Inf in reconstruction")
                x = torch.clamp(x, min=-1e6, max=1e6)
            
        except Exception as e:
            logger.error(f"Inverse failed: {e}")
            raise RuntimeError(f"Inverse failed: {e}")
        
        assert x.shape == (z.shape[0], 1, 28, 28), f"Shape mismatch: {x.shape}"
        
        if self.debug:
            logger.debug("=" * 70)
        
        return x
    
    def sample(self, n_samples: int, y: torch.Tensor) -> torch.Tensor:
        """Generate samples."""
        if n_samples != y.shape[0]:
            raise ValueError(f"n_samples ({n_samples}) != y.shape[0] ({y.shape[0]})")
        
        device = y.device
        z_final = torch.randn(n_samples, 4, 7, 7, device=device)
        z_factor2 = torch.randn(n_samples, 4, 7, 7, device=device)
        z_factor1 = torch.randn(n_samples, 2, 14, 14, device=device)
        z_factored_list = [z_factor1, z_factor2]
        
        x = self.inverse(z_final, z_factored_list, y)
        return x
    
    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute log-probability."""
        _, _, _, log_prob_total = self.forward(x, y, compute_h=True)
        return log_prob_total
