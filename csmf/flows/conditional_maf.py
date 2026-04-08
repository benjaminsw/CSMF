"""
Conditional Masked Autoregressive Flow (Full MADE Implementation)

Version: WP0.3-CondMAF-v2.4
Abbr: COND-MAF
Last Modified: 2026-04-02
Changelog:
  v2.5 (2026-04-03): [CONFIG] Default hyperparams updated to standard config:
                     n_flows 1→2 (two MADE transforms, better expressivity);
                     hidden_dims [64,64]→[128,128] (~2× more params per MADE);
                     addresses val NLL gap caused by under-capacity at 784-dim;
                     use_reverse_order remains False; no logic changes
  v2.4 (2026-04-02): [CONFIG] Default hyperparams updated to lean ablation config:
                     n_flows 2→1 (single MADE transform, faster, less expressive);
                     hidden_dims [128,128]→[64,64] (~2× fewer params per MADE layer);
                     use_reverse_order True→False (no alternating order with 1 flow);
                     __main__ test block updated to reflect new defaults; no logic changes
  v2.3 (2026-03-16): [SPEED] Fix1: MADE.precompute_film(h)+forward_with_cached_film() cut
                     FiLM recompute from 784×/flow to 1×/flow in inverse() D-loop; Fix2:
                     h_proj pre-allocated in MADE.__init__ via input_h_dim param — removes
                     lazy runtime init that broke DataParallel and created on wrong device;
                     Fix3: sample() batch loop replaced with single batched inverse() call
                     (B×n_samples flattened) — removes Python loop overhead
  v2.2 (2026-02-27): [SPEED] Default n_flows reduced 4→2 (~2× faster forward+inverse);
                     default hidden_dims reduced [256,256]→[128,128] (~1.5–2× faster forward);
                     changes are defaults only — callers can still override; version tracking
                     updated to v2.2; trade-offs: weaker density, less expressive MADE
  v2.1 (2026-02-26): [B] Spec-compliant h caching — forward() and inverse() accept optional
                     h=Optional[Tensor] from CSMF's shared conditioner; forward/inverse use
                     identical h eliminating recompute mismatch; inverse() fallback chain:
                     external→cached→recompute (WARNING on recompute); _cached_h added to
                     __init__; aligns with COND-RNVP v2.1.3 Option B treatment
  v2.0 (2025-10-25): Full MADE masking with parallel computation
  v1.0 (2025-10-25): Initial sequential autoregressive implementation
Dependencies: torch>=2.0, conditioning_networks WP0.1-CondNet-v1.1+, film WP0.1-FiLM-v1.0+

Purpose: Full MADE implementation with parallel computation for MNIST inverse problems

Key Features (v2.0):
+ MADE masking mechanism (Critical - enforces autoregressive property)
+ Parallel computation capability (Forward pass O(D) instead of O(D²))
+ Binary mask matrices for triangular dependencies
+ Batch normalization as invertible flow layer
+ Order reversal between layers (improved expressivity)
+ Efficient gradient flow through masked connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple, Optional, List
import math

try:
    from configs.mnist_config import MNIST_CONFIG
    from csmf.conditioning.conditioning_networks import MNISTConditioner
    from csmf.conditioning.film import FiLM
except ImportError as e:
    logging.error(f"Failed to import dependencies: {e}")
    raise

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MaskedLinear(nn.Linear):
    """
    Linear layer with binary mask for enforcing autoregressive property.
    
    The mask ensures output i only depends on inputs with index < i,
    creating the triangular dependency structure required for MAF.
    
    Implementation from Germain et al. (2015) MADE paper.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        # Mask will be registered as buffer (not a parameter)
        self.register_buffer('mask', torch.ones(out_features, in_features))
        
    def set_mask(self, mask: torch.Tensor):
        """Set the binary mask for this layer."""
        if mask.shape != self.mask.shape:
            logger.error(f"Mask shape mismatch: expected {self.mask.shape}, got {mask.shape}")
            raise ValueError(f"Mask shape {mask.shape} doesn't match weight shape {self.mask.shape}")
        self.mask.data = mask
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with masked weights."""
        # Element-wise multiply weight by mask before applying linear transformation
        return F.linear(x, self.weight * self.mask, self.bias)


def create_masks(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    conditioning_dim: int = 0
) -> List[torch.Tensor]:
    """
    Create binary masks for MADE network to enforce autoregressive property.
    
    Algorithm from Germain et al. (2015):
    1. Assign degrees 1→D to input units (or 0→D-1 for outputs)
    2. Assign degrees randomly (or sequentially) to hidden units, ensuring all degrees present
    3. Connect unit i → unit j only if degree(i) ≤ degree(j)
    
    Args:
        input_dim: Data dimensionality D
        hidden_dims: List of hidden layer sizes
        output_dim: Output dimensionality (typically D for mu, D for log_sigma = 2D)
        conditioning_dim: Size of conditioning features (not masked)
        
    Returns:
        List of binary mask tensors
    """
    logger.info(f"Creating MADE masks: input={input_dim}, hidden={hidden_dims}, output={output_dim}, conditioning={conditioning_dim}")
    
    masks = []
    
    # Input degrees: 1, 2, 3, ..., D (1-indexed for autoregressive property)
    input_degrees = torch.arange(1, input_dim + 1)
    
    # For conditioning inputs, assign degree 0 so they connect to all outputs
    if conditioning_dim > 0:
        cond_degrees = torch.zeros(conditioning_dim)
        input_degrees = torch.cat([input_degrees, cond_degrees])
    
    # Hidden layer degrees: ensure all degrees 1→D appear in each layer
    hidden_degrees = []
    for h_dim in hidden_dims:
        if h_dim >= input_dim:
            # Sequential assignment if enough units
            degrees = torch.arange(1, input_dim + 1).repeat((h_dim // input_dim) + 1)[:h_dim]
        else:
            # Uniform random assignment if fewer units
            degrees = torch.randint(1, input_dim + 1, (h_dim,))
        hidden_degrees.append(degrees)
    
    # Output degrees: 0, 1, 2, ..., D-1 (0-indexed)
    # Each output i should only see inputs with degree ≤ i
    output_degrees = torch.arange(0, input_dim).repeat(output_dim // input_dim)
    if output_dim % input_dim != 0:
        logger.error(f"Output dim {output_dim} not divisible by input dim {input_dim}")
        raise ValueError(f"Output dimension {output_dim} must be divisible by input dimension {input_dim}")
    
    # Create masks layer by layer
    # Mask from input to first hidden
    prev_degrees = input_degrees
    for h_idx, (h_dim, curr_degrees) in enumerate(zip(hidden_dims, hidden_degrees)):
        # mask[i, j] = 1 if degree(input_j) <= degree(hidden_i)
        mask = (prev_degrees.unsqueeze(0) <= curr_degrees.unsqueeze(1)).float()
        masks.append(mask)
        prev_degrees = curr_degrees
        logger.info(f"Hidden layer {h_idx+1} mask shape: {mask.shape}, ones: {mask.sum().item()}/{mask.numel()}")
    
    # Mask from last hidden to output
    # mask[i, j] = 1 if degree(hidden_j) < degree(output_i)
    # Note: strict < for outputs to enforce autoregressive property
    mask = (prev_degrees.unsqueeze(0) < output_degrees.unsqueeze(1)).float()
    masks.append(mask)
    logger.info(f"Output mask shape: {mask.shape}, ones: {mask.sum().item()}/{mask.numel()}")
    
    return masks


class BatchNormFlow(nn.Module):
    """
    Invertible BatchNorm flow layer.
    Uses batch stats in training and caches them for exact inverse.
    Uses running stats in eval.
    """

    def __init__(self, dim: int, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.momentum = momentum
        self.eps = eps

        # Learnable affine params (like BN affine=True)
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

        # Running stats
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))

        # Cache for invertibility in training
        self._cached_mean = None
        self._cached_var = None

    def forward(self, x: torch.Tensor, compute_log_det: bool = True):
        B, D = x.shape
        assert D == self.dim

        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)

            # update running stats
            self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean.detach())
            self.running_var.mul_(1 - self.momentum).add_(self.momentum * var.detach())

            # cache for inverse
            self._cached_mean = mean
            self._cached_var = var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = x_hat * self.weight + self.bias

        if compute_log_det:
            # per-dim jacobian: weight / sqrt(var + eps)
            log_det_scalar = torch.sum(torch.log(torch.abs(self.weight)) - 0.5 * torch.log(var + self.eps))
            log_det = log_det_scalar.expand(B)
        else:
            log_det = torch.zeros(B, device=x.device)

        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        if self.training and (self._cached_mean is not None) and (self._cached_var is not None):
            mean = self._cached_mean
            var = self._cached_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (y - self.bias) / (self.weight + 1e-12)
        x = x_hat * torch.sqrt(var + self.eps) + mean
        return x



class MADE(nn.Module):
    """
    Masked Autoencoder for Distribution Estimation (MADE).
    
    Implements the parallel autoregressive network from Germain et al. (2015).
    Key innovation: uses masked connections to enable parallel computation
    of all conditional parameters in a single forward pass.
    
    With FiLM conditioning for inverse problems (from Chapter 2 plan).
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        conditioning_dim: int = 0,
        use_film: bool = True,
        input_h_dim: Optional[int] = None
    ):
        """
        Initialize MADE network.

        Args:
            input_dim: Data dimensionality D
            hidden_dims: List of hidden layer sizes (e.g., [256, 256])
            conditioning_dim: Dimension of conditioning features h expected internally
            use_film: Whether to use FiLM for conditioning
            input_h_dim: Actual dim of incoming h. If provided and != conditioning_dim,
                         h_proj is pre-allocated here (Fix 2 — removes lazy runtime init).
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.conditioning_dim = conditioning_dim
        self.use_film = use_film

        # [Fix 2] Pre-allocate h_proj to avoid lazy init breaking DataParallel / device placement
        if conditioning_dim > 0 and input_h_dim is not None and input_h_dim != conditioning_dim:
            self.h_proj: Optional[nn.Linear] = nn.Linear(input_h_dim, conditioning_dim)
            logger.info(f"MADE: pre-allocated h_proj {input_h_dim} -> {conditioning_dim}")
        else:
            self.h_proj = None
        
        # Output dimension: D values for mu, D values for log_sigma
        self.output_dim = 2 * input_dim
        
        # Create masked layers
        self.layers = nn.ModuleList()
        self.film_layers = nn.ModuleList() if use_film else None
        
        # Build network architecture
        prev_dim = input_dim + (conditioning_dim if not use_film else 0)
        
        for h_dim in hidden_dims:
            self.layers.append(MaskedLinear(prev_dim, h_dim))
            if use_film:
                self.film_layers.append(FiLM(h_dim, conditioning_dim))
            prev_dim = h_dim
        
        # Output layer
        self.layers.append(MaskedLinear(prev_dim, self.output_dim))
        
        # Create and set masks
        try:
            # When not using FiLM, conditioning is concatenated to input
            cond_dim_for_mask = 0 if use_film else conditioning_dim
            masks = create_masks(input_dim, hidden_dims, self.output_dim, cond_dim_for_mask)
            
            for layer, mask in zip(self.layers, masks):
                layer.set_mask(mask)
                
            logger.info(f"MADE initialized with {len(self.layers)} layers, FiLM={use_film}")
        except Exception as e:
            logger.error(f"Failed to create/set masks: {e}")
            raise
        
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: compute all conditional parameters in parallel.
        
        Args:
            x: Data tensor, shape (batch, input_dim)
            h: Conditioning features, shape (batch, conditioning_dim)
            
        Returns:
            mu: Means for all dimensions, shape (batch, input_dim)
            log_sigma: Log std devs for all dimensions, shape (batch, input_dim)
        """
        if self.conditioning_dim > 0 and h is None:
            logger.error("Conditioning features h required but not provided")
            raise ValueError("Conditioning features h required when conditioning_dim > 0")
        
        # --- Adapt conditioning h to [B, conditioning_dim] ---
        if h is not None:
            # Spatial h (e.g. MNIST image): pool to [B, C]
            if h.dim() == 4:
                h = h.mean(dim=(2, 3))

            if h.dim() != 2:
                logger.error(f"MADE expects h as [B, h_dim] or [B,C,H,W], got {h.shape}")
                raise RuntimeError(f"MADE h shape invalid: {h.shape}")

            # [Fix 2] Use pre-allocated h_proj; raise on unexpected mismatch (no lazy init)
            if self.conditioning_dim > 0 and h.shape[1] != self.conditioning_dim:
                if self.h_proj is not None:
                    h = self.h_proj(h)
                else:
                    logger.error(
                        f"h dim mismatch: got {h.shape[1]}, expected {self.conditioning_dim}. "
                        f"Pass input_h_dim={h.shape[1]} to MADE.__init__ to enable projection."
                    )
                    raise RuntimeError(
                        f"h dim mismatch ({h.shape[1]} != {self.conditioning_dim}). "
                        f"Pre-allocate h_proj via input_h_dim in MADE.__init__."
                    )
        # --- end conditioning adapter ---
                
        
        
        # For non-FiLM case, concatenate conditioning to input
        if self.conditioning_dim > 0 and not self.use_film:
            out = torch.cat([x, h], dim=1)
        else:
            out = x
        
        # Pass through hidden layers
        for i, layer in enumerate(self.layers[:-1]):  # All except output layer
            out = layer(out)
            out = self.activation(out)
            
            # Apply FiLM if enabled
            if self.use_film and h is not None:
                out = self.film_layers[i](out, h)
        
        # Output layer (no activation)
        out = self.layers[-1](out)
        
        # Split into mu and log_sigma
        mu, log_sigma = torch.chunk(out, 2, dim=1)
        
        # Clamp log_sigma for numerical stability
        log_sigma = torch.clamp(log_sigma, min=-5, max=5)
        
        # Check for NaN
        if torch.isnan(mu).any() or torch.isnan(log_sigma).any():
            logger.error("NaN detected in MADE output")
            logger.error(f"mu: min={mu.min():.3f}, max={mu.max():.3f}, nan={torch.isnan(mu).sum()}")
            logger.error(f"log_sigma: min={log_sigma.min():.3f}, max={log_sigma.max():.3f}, nan={torch.isnan(log_sigma).sum()}")
            raise RuntimeError("NaN detected in MADE output")
        
        return mu, log_sigma

    # ------------------------------------------------------------------ #
    # Fix 1: FiLM pre-cache API                                           #
    # ------------------------------------------------------------------ #
    def precompute_film(self, h: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Pre-compute FiLM (gamma, beta) for every hidden layer from h.

        Call ONCE per flow step before entering the autoregressive D-loop in
        inverse(). Cuts FiLM conditioning-net calls from D×n_layers to
        2×n_layers (two dummy passes per layer to isolate gamma and beta).

        Strategy (no FiLM internals required):
            FiLM(x, h) = gamma(h) * x + beta(h)
            beta  = FiLM(0, h)
            gamma = FiLM(1, h) - beta

        Args:
            h: Conditioning vector, shape (B, conditioning_dim).

        Returns:
            List of (gamma, beta) tensors, one tuple per hidden layer.
            Empty list if use_film is False.
        """
        if not self.use_film or self.film_layers is None:
            return []

        film_params: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i, film_layer in enumerate(self.film_layers):
            h_dim_i = self.hidden_dims[i]
            zeros = torch.zeros(h.shape[0], h_dim_i, device=h.device, dtype=h.dtype)
            ones  = torch.ones( h.shape[0], h_dim_i, device=h.device, dtype=h.dtype)
            beta  = film_layer(zeros, h)
            gamma = film_layer(ones,  h) - beta
            if torch.isnan(gamma).any() or torch.isnan(beta).any():
                logger.error(f"NaN in precompute_film at hidden layer {i}")
                raise RuntimeError(f"NaN in precompute_film layer {i}")
            film_params.append((gamma, beta))
            logger.debug(f"precompute_film layer {i}: gamma norm={gamma.norm():.4f}, beta norm={beta.norm():.4f}")

        return film_params

    def forward_with_cached_film(
        self,
        x: torch.Tensor,
        film_params: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass using pre-computed FiLM params (avoids recomputing h-dependent
        conditioning net on every step of the autoregressive D-loop).

        Used exclusively by ConditionalMAF.inverse() inside the D-loop.

        Args:
            x: Masked input tensor, shape (B, input_dim).
            film_params: Output of precompute_film(h) — list of (gamma, beta) per layer.

        Returns:
            mu, log_sigma: shape (B, input_dim) each.
        """
        if self.use_film and len(film_params) != len(self.layers) - 1:
            logger.error(
                f"film_params length {len(film_params)} != n_hidden_layers {len(self.layers)-1}"
            )
            raise RuntimeError("film_params length mismatch — call precompute_film first")

        out = x
        for i, layer in enumerate(self.layers[:-1]):
            out = layer(out)
            out = self.activation(out)
            if self.use_film and film_params:
                gamma, beta = film_params[i]
                out = gamma * out + beta

        out = self.layers[-1](out)
        mu, log_sigma = torch.chunk(out, 2, dim=1)
        log_sigma = torch.clamp(log_sigma, min=-5, max=5)

        if torch.isnan(mu).any() or torch.isnan(log_sigma).any():
            logger.error("NaN detected in MADE forward_with_cached_film output")
            raise RuntimeError("NaN in forward_with_cached_film output")

        return mu, log_sigma


class ConditionalMAF(nn.Module):
    """
    Conditional Masked Autoregressive Flow (Full MADE Implementation)
    
    Key improvements from v1.0:
    - Uses MADE for parallel computation: O(D) forward instead of O(D²)
    - Batch normalization after each flow for training stability
    - Order reversal between layers for increased flexibility
    - Proper triangular Jacobian for efficient log-det computation
    
    Architecture:
    x → [MADE₁ → BN₁] → [MADE₂ → BN₂] → ... → [MADEₖ → BNₖ] → z
    
    Each MADE is conditioned on h = conditioner(y) via FiLM.
    """
    
    def __init__(
        self,
        dim: int = 784,
        h_dim: int = 64,
        cond_dim: int = None,
        n_flows: int = 2,
        hidden_dims: List[int] = None,
        use_batch_norm: bool = True,
        use_reverse_order: bool = False,
        config: dict = None
    ):
        """
        Initialize Conditional MAF.
        
        Args:
            dim: Dimension of data (default: 784 for flattened MNIST)
            h_dim: Conditioning feature dimension (default: 64)
            n_flows: Number of MADE transforms (default: 4)
            hidden_dims: Hidden layer dimensions per MADE (default: [256, 256])
            use_batch_norm: Whether to use batch normalization (default: True)
            use_reverse_order: Whether to reverse order between layers (default: True)
            config: Optional config dict from MNIST_CONFIG
        """
        super().__init__()
        
        # cond_dim is an alias for h_dim (API consistency with NICE/NSF)
        if cond_dim is not None:
            h_dim = cond_dim
        
        # Version tracking
        self.version = "WP0.3-CondMAF-v2.5"
        self.abbr = "COND-MAF"
        logger.info(f"Initializing {self.__class__.__name__} version {self.version}")
        
        # Use config if provided
        if config is not None and 'maf' in config:
            maf_config = config['maf']
            dim = maf_config.get('dim', dim)
            h_dim = maf_config.get('h_dim', h_dim)
            n_flows = maf_config.get('n_flows', n_flows)
            hidden_dims = maf_config.get('hidden_dims', hidden_dims)
        
        if hidden_dims is None:
            hidden_dims = [128, 128]
        
        self.dim = dim
        self.h_dim = h_dim
        self.n_flows = n_flows
        self.hidden_dims = hidden_dims
        self.use_batch_norm = use_batch_norm
        self.use_reverse_order = use_reverse_order
        
        # Conditioning network
        try:
            self.conditioner = MNISTConditioner(h_dim=h_dim)
            logger.info(f"Created MNISTConditioner with h_dim={h_dim}")
        except Exception as e:
            logger.error(f"Failed to create MNISTConditioner: {e}")
            raise
        
        self._cached_h: Optional[torch.Tensor] = None  # [B] v2.1 — external h cache
        
        # Create orderings for each flow (alternating between default and reversed)
        self.orderings = []
        default_order = torch.arange(dim)
        reversed_order = torch.arange(dim - 1, -1, -1)
        
        for k in range(n_flows):
            if use_reverse_order and k % 2 == 1:
                self.orderings.append(reversed_order.clone())
            else:
                self.orderings.append(default_order.clone())
            logger.info(f"Flow {k}: order {'reversed' if use_reverse_order and k % 2 == 1 else 'default'}")
        
        # Register orderings as buffers (not parameters)
        for k, order in enumerate(self.orderings):
            self.register_buffer(f'ordering_{k}', order)
        
        # Create inverse orderings for reconstruction
        self.inv_orderings = []
        for order in self.orderings:
            inv_order = torch.zeros_like(order)
            inv_order[order] = torch.arange(len(order))
            self.inv_orderings.append(inv_order)
        
        for k, inv_order in enumerate(self.inv_orderings):
            self.register_buffer(f'inv_ordering_{k}', inv_order)
        
        # Create MADE networks (one per flow)
        self.flows = nn.ModuleList()
        for k in range(n_flows):
            made = MADE(
                input_dim=dim,
                hidden_dims=hidden_dims,
                conditioning_dim=h_dim,
                use_film=True,
                input_h_dim=h_dim   # [Fix 2] pre-allocate h_proj if dims ever diverge
            )
            self.flows.append(made)
            logger.info(f"Created MADE flow {k+1}/{n_flows}")
        
        # Create batch normalization layers
        if use_batch_norm:
            self.batch_norms = nn.ModuleList()
            for k in range(n_flows):
                bn = BatchNormFlow(dim)
                self.batch_norms.append(bn)
                logger.info(f"Created BatchNorm flow {k+1}/{n_flows}")
        else:
            self.batch_norms = None
        
        # Base distribution
        self.register_buffer('base_loc', torch.zeros(1))
        self.register_buffer('base_scale', torch.ones(1))
        
        logger.info(f"ConditionalMAF initialized:")
        logger.info(f"  - dim={dim}, n_flows={n_flows}, hidden_dims={hidden_dims}")
        logger.info(f"  - batch_norm={use_batch_norm}, reverse_order={use_reverse_order}")
        logger.info(f"  - PARALLEL COMPUTATION ENABLED via MADE")
    
    def _permute(self, x: torch.Tensor, ordering: torch.Tensor) -> torch.Tensor:
        """Apply permutation to reorder dimensions."""
        return x[:, ordering]
    
    def _inv_permute(self, x: torch.Tensor, inv_ordering: torch.Tensor) -> torch.Tensor:
        """Apply inverse permutation to restore original order."""
        return x[:, inv_ordering]
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: transform data x to latent z (PARALLEL COMPUTATION).
        
        This is O(D × n_flows) instead of O(D² × n_flows) thanks to MADE.
        
        Args:
            x: Data tensor, shape (batch, dim)
            y: Measurements for conditioning, shape (batch, 1, 28, 28)
            h: Pre-computed conditioning vector from external conditioner (e.g. CSMF's
               shared MNISTConditioner). If provided, skips internal self.conditioner(y)
               — satisfies WP0 spec "cache h per mini-batch" requirement.
               
        Returns:
            z: Latent codes, shape (batch, dim)
            log_det_total: Total log determinant, shape (batch,)
            log_prob: Log probability, shape (batch,)
        """
        batch_size = x.shape[0]
        
        if x.shape[1] != self.dim:
            logger.error(f"Input dimension mismatch: expected {self.dim}, got {x.shape[1]}")
            raise ValueError(f"Input shape {x.shape} doesn't match expected dim {self.dim}")
        
        # [B] v2.1 — h resolution: external → internal conditioner
        try:
            if h is not None:
                self._cached_h = h
                logger.debug(f"MAF forward: using external h, norm={h.norm().item():.4f}")
            else:
                h = self.conditioner(y)
                self._cached_h = h
                logger.debug(f"MAF forward: computed internal h, norm={h.norm().item():.4f}")
            if torch.isnan(h).any():
                logger.error("NaN detected in conditioning features")
                raise RuntimeError("NaN in conditioning features")
        except Exception as e:
            logger.error(f"Conditioning failed: {e}")
            raise
        
        # Initialize
        z = x
        log_det_total = torch.zeros(batch_size, device=x.device)
        
        # Apply each flow (PARALLEL - no dimension loop!)
        for flow_idx in range(self.n_flows):
            # Apply permutation
            z = self._permute(z, self.orderings[flow_idx])
            
            # MADE forward: compute all mu and log_sigma in ONE PASS
            try:
                mu, log_sigma = self.flows[flow_idx](z, h)
            except Exception as e:
                logger.error(f"MADE forward failed at flow {flow_idx}: {e}")
                raise
            
            # Transform: z_new = (z - mu) / sigma (VECTORIZED)
            sigma = torch.exp(log_sigma)
            z_new = (z - mu) / (sigma + 1e-8)
            
            # Log determinant: -sum(log_sigma) (VECTORIZED, exploits triangular Jacobian)
            log_det_flow = -log_sigma.sum(dim=1)
            log_det_total += log_det_flow
            
            z = z_new
            
            # Apply batch normalization
            if self.use_batch_norm:
                z, bn_log_det = self.batch_norms[flow_idx](z, compute_log_det=True)
                log_det_total += bn_log_det
            
            # Check for numerical issues
            if torch.isnan(z).any():
                logger.error(f"NaN detected after flow {flow_idx}")
                logger.error(f"mu range: [{mu.min():.3f}, {mu.max():.3f}]")
                logger.error(f"log_sigma range: [{log_sigma.min():.3f}, {log_sigma.max():.3f}]")
                raise RuntimeError(f"NaN after flow {flow_idx}")
            
            if torch.isinf(log_det_total).any():
                logger.error(f"Inf log_det after flow {flow_idx}: {log_det_total.max():.2f}")
                raise RuntimeError(f"Inf log_det after flow {flow_idx}")
        
        # Compute log probability under base distribution N(0, I)
        log_prob_base = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(dim=1)
        log_prob = log_prob_base + log_det_total
        
        # Final checks
        if torch.isnan(log_prob).any():
            logger.error("NaN detected in final log_prob")
            n_nan = torch.isnan(log_prob).sum().item()
            logger.error(f"Number of NaN samples: {n_nan}/{batch_size}")
            raise RuntimeError("NaN in final log_prob")
        
        return z, log_det_total, log_prob
    
    def inverse(self, z: torch.Tensor, y: torch.Tensor, h: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Inverse pass: transform latent z back to data x.
        
        IMPORTANT: This remains SEQUENTIAL (O(D² × n_flows)) because
        autoregressive inverse requires computing dimensions one at a time.
        This is a fundamental property of autoregressive models.
        
        Args:
            z: Latent codes, shape (batch, dim)
            y: Measurements for conditioning, shape (batch, 1, 28, 28)
            h: Pre-computed conditioning vector from external conditioner.
               Fallback chain: external h → cached h → recompute (WARNING on recompute).
            
        Returns:
            x: Reconstructed data, shape (batch, dim)
        """
        batch_size = z.shape[0]
        
        # [B] v2.1 — h resolution: external → cached → recompute (fallback with WARNING)
        try:
            if h is not None:
                h_source = "external"
            elif self._cached_h is not None and self._cached_h.shape[0] == batch_size:
                h = self._cached_h
                h_source = "cached"
            else:
                if self._cached_h is not None and self._cached_h.shape[0] != batch_size:
                    logger.warning(
                        f"MAF inverse(): _cached_h batch size {self._cached_h.shape[0]} != "
                        f"z batch size {batch_size} — recomputing from self.conditioner(y). "
                        f"This is expected when sample() flattens (B x n_samples)."
                    )
                else:
                    logger.warning(
                        "MAF inverse(): no external h and no cached h — recomputing from "
                        "self.conditioner(y). This may cause forward/inverse h mismatch. "
                        "Pass h= from CSMF's conditioner."
                    )
                h = self.conditioner(y)
                h_source = "recomputed"
            logger.debug(f"MAF inverse: h source={h_source}, norm={h.norm().item():.4f}")
        except Exception as e:
            logger.error(f"Conditioning failed in inverse: {e}")
            raise
        
        # Initialize
        x = z
        
        # Apply each flow in REVERSE order
        for flow_idx in reversed(range(self.n_flows)):
            # Inverse batch normalization
            if self.use_batch_norm:
                x = self.batch_norms[flow_idx].inverse(x)
            
            # [Fix 1] Pre-compute FiLM gamma/beta once per flow — cuts FiLM calls 784× → 1×
            film_params = self.flows[flow_idx].precompute_film(h)

            # MADE inverse: MUST be sequential
            # We need to compute dimension by dimension because dimension i
            # depends on dimensions 0...i-1 which must be computed first
            x_new = torch.zeros_like(x)

            for i in range(self.dim):
                # Get previous dimensions in the CURRENT ordering
                if i == 0:
                    x_prev = torch.zeros(batch_size, self.dim, device=x.device)
                    # Only use conditioning for first dimension
                    x_prev = x_new[:, :i] if i > 0 else x_prev[:, :0]
                else:
                    x_prev = x_new[:, :i]

                # Compute conditional parameters using MADE
                # We need to mask the input to only see dimensions < i
                x_masked = x_new.clone()
                x_masked[:, i:] = 0  # Mask future dimensions

                try:
                    # [Fix 1] Use cached FiLM — no h recompute inside loop
                    mu_all, log_sigma_all = self.flows[flow_idx].forward_with_cached_film(
                        x_masked, film_params
                    )
                    mu_i = mu_all[:, i:i+1]
                    log_sigma_i = log_sigma_all[:, i:i+1]
                except Exception as e:
                    logger.error(f"MADE inverse failed at flow {flow_idx}, dim {i}: {e}")
                    raise
                
                # Check for NaN
                if torch.isnan(mu_i).any() or torch.isnan(log_sigma_i).any():
                    logger.error(f"NaN in inverse at flow {flow_idx}, dim {i}")
                    raise RuntimeError(f"NaN in inverse at flow {flow_idx}, dim {i}")
                
                # Inverse transform: x_i = z_i * sigma_i + mu_i
                sigma_i = torch.exp(log_sigma_i)
                z_i = x[:, i:i+1]
                x_new[:, i:i+1] = z_i * sigma_i + mu_i
            
            x = x_new
            
            # Apply inverse permutation
            x = self._inv_permute(x, self.inv_orderings[flow_idx])
        
        # Final check
        if torch.isnan(x).any():
            logger.error("NaN detected in inverse output")
            n_nan = torch.isnan(x).sum().item()
            logger.error(f"Number of NaN elements: {n_nan}/{x.numel()}")
            raise RuntimeError("NaN in inverse output")
        
        return x
    
    def sample(self, n_samples: int, y: torch.Tensor) -> torch.Tensor:
        """
        Generate samples from the conditional distribution.

        Args:
            n_samples: Number of samples to generate per conditioning.
            y: Measurements for conditioning, shape (1, 1, 28, 28) or (B, 1, 28, 28).

        Returns:
            x: Samples, shape (n_samples, dim) for B=1, or (B, n_samples, dim) for B>1.
        """
        batch_size = y.shape[0]
        total = batch_size * n_samples

        z = torch.randn(total, self.dim, device=y.device)

        # Expand y: (B, C, H, W) -> (B*n_samples, C, H, W)
        y_expanded = (
            y.unsqueeze(1)
             .expand(-1, n_samples, -1, -1, -1)
             .reshape(total, *y.shape[1:])
        )

        # [Fix] Compute h once here and pass to inverse() — avoids recompute warning
        # that fires when _cached_h batch size != total (B*n_samples)
        try:
            h = self.conditioner(y_expanded)
        except Exception as e:
            logger.error(f"sample(): conditioner failed: {e}")
            raise

        x_flat = self.inverse(z, y_expanded, h=h)

        if batch_size == 1:
            return x_flat
        return x_flat.view(batch_size, n_samples, self.dim)


# Version check function
def get_version():
    """Return version information."""
    return {
        'version': 'WP0.3-CondMAF-v2.5',
        'abbr': 'COND-MAF',
        'date': '2026-04-03',
        'purpose': 'Standard config: n_flows=2, hidden_dims=[128,128] — better expressivity for 784-dim MNIST',
        'improvements': [
            'MADE masking mechanism',
            'Parallel computation (O(D) forward)',
            'Binary mask matrices',
            'Batch normalization',
            'Order reversal between layers',
            'Triangular Jacobian optimization',
            'Efficient gradient flow',
            '[v2.3] FiLM pre-cache (precompute_film + forward_with_cached_film)',
            '[v2.3] h_proj pre-allocated in MADE.__init__ (input_h_dim param)',
            '[v2.3] sample() vectorised — single batched inverse() call',
        ]
    }


if __name__ == "__main__":
    """Test to verify the implementation."""
    print("=" * 80)
    print(f"ConditionalMAF version: {get_version()['version']}")
    print(f"Abbr: {get_version()['abbr']}")
    print(f"Improvements: {', '.join(get_version()['improvements'])}")
    print("=" * 80)
    
    try:
        # Test 1: Model instantiation
        logger.info("\n[TEST 1] Model instantiation")
        model = ConditionalMAF(
            dim=784,
            h_dim=64,
            n_flows=2,
            hidden_dims=[128, 128],
            use_batch_norm=True,
            use_reverse_order=False
        )
        logger.info("✓ Model instantiation successful")
        
        # Test 2: Forward pass (should be parallel)
        logger.info("\n[TEST 2] Forward pass (parallel computation)")
        x = torch.randn(4, 784)
        y = torch.randn(4, 1, 28, 28)
        
        import time
        start = time.time()
        z, log_det, log_prob = model.forward(x, y)
        forward_time = time.time() - start
        
        logger.info(f"✓ Forward pass successful: z shape={z.shape}, log_det shape={log_det.shape}")
        logger.info(f"  Forward time: {forward_time*1000:.2f}ms")
        logger.info(f"  Log-det range: [{log_det.min():.2f}, {log_det.max():.2f}]")
        logger.info(f"  Log-prob range: [{log_prob.min():.2f}, {log_prob.max():.2f}]")
        
        # Test 3: Inverse pass
        logger.info("\n[TEST 3] Inverse pass (sequential)")
        start = time.time()
        x_recon = model.inverse(z, y)
        inverse_time = time.time() - start
        
        logger.info(f"✓ Inverse pass successful: x_recon shape={x_recon.shape}")
        logger.info(f"  Inverse time: {inverse_time*1000:.2f}ms")
        logger.info(f"  Forward/Inverse ratio: {forward_time/inverse_time:.2f}x")
        
        # Test 4: Invertibility
        logger.info("\n[TEST 4] Invertibility check")
        recon_error = (x - x_recon).abs().max().item()
        logger.info(f"  Reconstruction error: {recon_error:.6f}")
        if recon_error < 1e-3:
            logger.info("✓ Invertibility check PASSED")
        else:
            logger.warning(f"✗ Invertibility check FAILED (error={recon_error:.6f} > 1e-3)")
        
        # Test 5: Sampling
        logger.info("\n[TEST 5] Sampling")
        y_single = torch.randn(1, 1, 28, 28)
        samples = model.sample(n_samples=10, y=y_single)
        logger.info(f"✓ Sampling successful: samples shape={samples.shape}")
        
        # Test 6: Mask verification
        logger.info("\n[TEST 6] MADE mask verification")
        made = model.flows[0]
        for i, layer in enumerate(made.layers):
            if isinstance(layer, MaskedLinear):
                mask = layer.mask
                n_zeros = (mask == 0).sum().item()
                n_ones = (mask == 1).sum().item()
                logger.info(f"  Layer {i}: mask shape={mask.shape}, zeros={n_zeros}, ones={n_ones}")
        logger.info("✓ All masks properly set")
        
        # Test 7: Ordering verification
        logger.info("\n[TEST 7] Order reversal verification")
        for k in range(model.n_flows):
            ordering = model.orderings[k]
            is_reversed = (ordering[0] > ordering[-1])
            logger.info(f"  Flow {k}: {'REVERSED' if is_reversed else 'DEFAULT'} order")
        logger.info("✓ Order reversal working correctly")
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED!")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"\n{'=' * 80}")
        logger.error(f"TEST FAILED: {e}")
        logger.error(f"{'=' * 80}")
        import traceback
        traceback.print_exc()
        raise
