"""
================================================================================
FILE:    csmf/physics/forward_models.py
VERSION: WP1.1-FwdMod-v1.0
ABBR:    FWD-MOD
DESC:    Forward operators and adjoints for SR and SAR inverse problems.
         Implements A: x -> y (degradation) and A^T: y -> x (adjoint).
AUTHOR:  CSMF Project
DATE:    2026-02-17
================================================================================

CHANGELOG:
- v1.0 (2026-02-17): Initial implementation. ForwardModel base class,
  SRForwardModel (blur + downsample) with Gaussian kernel caching,
  SARForwardModel (log-domain) with input clamping and shape validation
  on all adjoint inputs. Error logging throughout.
- v1.1 (2026-02-21): Fixed SR adjoint correctness. Replaced bilinear interpolate
  downsample with exact decimation (x[...,::s,::s]) in forward(). Replaced
  nearest upsample in adjoint() with mathematically correct zero-insertion
  (downsample2d_adjoint). Added lazy H,W capture in forward() so adjoint()
  always knows target output size. Fixes test_sr_adjoint_inner_product.
- v1.2 (2026-02-21): Fixed Fourier vs PCG boundary condition mismatch.
  Replaced zero-padding in _blur/_blur_T with circular padding via new
  _circular_conv2d() static method. PCG now uses same circular BCs as
  Fourier solve, fixing test_fourier_vs_pcg_close.
================================================================================
"""

__version__ = "WP1.1-FwdMod-v1.2"

import logging
import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SAR_INPUT_CLAMP_MIN = 1e-8       # min value before log in SAR forward
SAR_OUTPUT_CLAMP_MAX = 1e6       # max value after exp in SAR adjoint
SR_BLUR_SIGMA_MIN   = 0.1        # guard against degenerate kernels
SR_BLUR_SIGMA_MAX   = 10.0


# ─────────────────────────────────────────────────────────────────────────────
# Base Class
# ─────────────────────────────────────────────────────────────────────────────
class ForwardModel(ABC, nn.Module):
    """
    Abstract base class for all forward operators A: x -> y.

    Subclasses must implement:
        forward(x)  : degradation operator
        adjoint(y)  : A^T operator (NOT the inverse, just the adjoint)
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward operator A to clean image x -> degraded y."""
        raise NotImplementedError

    @abstractmethod
    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply adjoint operator A^T to degraded y -> x-space."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared shape-validation helper
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_4d(tensor: torch.Tensor, name: str) -> None:
        """Raise ValueError and log if tensor is not 4-D (N, C, H, W)."""
        if tensor.ndim != 4:
            msg = (
                f"[{__version__}] {name} must be 4-D (N, C, H, W), "
                f"got shape {tuple(tensor.shape)}"
            )
            logger.error(msg)
            raise ValueError(msg)


# ─────────────────────────────────────────────────────────────────────────────
# SR Forward Model
# ─────────────────────────────────────────────────────────────────────────────
class SRForwardModel(ForwardModel):
    """
    Super-resolution forward model: A = D ∘ B
        B  : Gaussian blur  (sigma = blur_sigma)
        D  : Downsample ×s (exact decimation: x[...,::s,::s])

    Adjoint A^T = B^T ∘ D^T
        D^T : Zero-insertion (adjoint of decimation)
        B^T : Transpose conv with flipped kernel

    Additional functionality included:
        - Gaussian kernel caching (computed once, reused)
        - Shape validation on adjoint input
        - Lazy H,W capture on first forward() call for adjoint target size
    """

    def __init__(
        self,
        blur_sigma: float = 1.0,
        downsample_factor: int = 2,
        kernel_size: int = 0,       # 0 = auto-size from sigma
        in_channels: int = 1,
    ):
        super().__init__()

        # ── parameter validation ──────────────────────────────────────
        if not (SR_BLUR_SIGMA_MIN <= blur_sigma <= SR_BLUR_SIGMA_MAX):
            msg = (
                f"[{__version__}] blur_sigma={blur_sigma} out of valid "
                f"range [{SR_BLUR_SIGMA_MIN}, {SR_BLUR_SIGMA_MAX}]"
            )
            logger.error(msg)
            raise ValueError(msg)

        if downsample_factor not in (1, 2, 4):
            msg = f"[{__version__}] downsample_factor must be 1, 2, or 4; got {downsample_factor}"
            logger.error(msg)
            raise ValueError(msg)

        self.blur_sigma       = blur_sigma
        self.downsample_factor = downsample_factor
        self.in_channels      = in_channels

        # ── lazy input size capture (set on first forward call) ───────
        self._input_H: int | None = None
        self._input_W: int | None = None

        # ── auto kernel size (odd, >= 3) ──────────────────────────────
        if kernel_size == 0:
            ks = max(3, 2 * int(math.ceil(3 * blur_sigma)) + 1)
            # ensure odd
            if ks % 2 == 0:
                ks += 1
            kernel_size = ks

        self.kernel_size = kernel_size

        # ── cached Gaussian kernel (registered as buffer) ─────────────
        kernel = self._gaussian_kernel(blur_sigma, kernel_size, in_channels)
        self.register_buffer("kernel", kernel)          # (C, 1, ks, ks)
        self.register_buffer("kernel_T", torch.flip(kernel, dims=[2, 3]))

        logger.info(
            "[%s] SRForwardModel created: sigma=%.2f, downsample=×%d, "
            "kernel_size=%d",
            __version__, blur_sigma, downsample_factor, kernel_size,
        )

    # ------------------------------------------------------------------
    # Kernel construction
    # ------------------------------------------------------------------
    @staticmethod
    def _gaussian_kernel(
        sigma: float,
        kernel_size: int,
        channels: int,
    ) -> torch.Tensor:
        """
        Build a (channels, 1, ks, ks) depthwise Gaussian kernel.
        Cached in __init__ as a buffer.
        """
        half = kernel_size // 2
        coords = torch.arange(kernel_size, dtype=torch.float32) - half
        g1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        g1d = g1d / g1d.sum()
        g2d = torch.outer(g1d, g1d)                    # (ks, ks)
        g2d = g2d.view(1, 1, kernel_size, kernel_size)
        g2d = g2d.repeat(channels, 1, 1, 1)            # (C, 1, ks, ks)
        return g2d

    # ------------------------------------------------------------------
    # Circular convolution helper (matches Fourier solve boundary conditions)
    # ------------------------------------------------------------------
    @staticmethod
    def _circular_conv2d(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """
        2-D depthwise convolution with circular (periodic) padding.
        Matches the boundary conditions assumed by SRProximalFourier FFT solve.
        """
        kh, kw = kernel.shape[-2], kernel.shape[-1]
        pad_h, pad_w = kh // 2, kw // 2
        x_pad = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="circular")
        return F.conv2d(x_pad, kernel, padding=0, groups=x.shape[1])

    # ------------------------------------------------------------------
    # Blur helpers (depthwise conv, circular padding)
    # ------------------------------------------------------------------
    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        return self._circular_conv2d(x, self.kernel)

    def _blur_T(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose blur using flipped kernel with circular padding."""
        return self._circular_conv2d(x, self.kernel_T)

    # ------------------------------------------------------------------
    # Forward:  x (N,C,H,W) → y (N,C,H/d,W/d)
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_4d(x, "SRForwardModel.forward input x")

        if x.shape[1] != self.in_channels:
            msg = (
                f"[{__version__}] Channel mismatch: expected {self.in_channels}, "
                f"got {x.shape[1]}"
            )
            logger.error(msg)
            raise ValueError(msg)

        try:
            # Capture input spatial size for adjoint (lazy)
            if self._input_H is None or self._input_W is None:
                self._input_H, self._input_W = x.shape[2], x.shape[3]
                logger.debug(
                    "[%s] Captured input size H=%d, W=%d",
                    __version__, self._input_H, self._input_W,
                )
            elif (x.shape[2] != self._input_H or x.shape[3] != self._input_W):
                logger.warning(
                    "[%s] Input size changed from (%d,%d) to (%d,%d). "
                    "Updating cached H,W.",
                    __version__, self._input_H, self._input_W,
                    x.shape[2], x.shape[3],
                )
                self._input_H, self._input_W = x.shape[2], x.shape[3]

            # B: Gaussian blur
            x_blur = self._blur(x)

            # D: exact decimation (adjoint = zero-insertion)
            if self.downsample_factor > 1:
                s = self.downsample_factor
                y = x_blur[:, :, ::s, ::s]
            else:
                y = x_blur

            logger.debug(
                "[%s] SR forward: %s -> %s",
                __version__, tuple(x.shape), tuple(y.shape),
            )
            return y

        except Exception as e:
            logger.error("[%s] SR forward failed: %s", __version__, e)
            raise

    # ------------------------------------------------------------------
    # Adjoint: y (N,C,H/d,W/d) → x_hat (N,C,H,W)
    # ------------------------------------------------------------------
    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        self._validate_4d(y, "SRForwardModel.adjoint input y")

        if y.shape[1] != self.in_channels:
            msg = (
                f"[{__version__}] Adjoint channel mismatch: expected "
                f"{self.in_channels}, got {y.shape[1]}"
            )
            logger.error(msg)
            raise ValueError(msg)

        try:
            # D^T: zero-insertion (exact adjoint of decimation)
            if self.downsample_factor > 1:
                if self._input_H is None or self._input_W is None:
                    msg = (
                        f"[{__version__}] adjoint() called before forward(): "
                        "input H,W unknown. Call forward() at least once first."
                    )
                    logger.error(msg)
                    raise RuntimeError(msg)

                s = self.downsample_factor
                y_up = torch.zeros(
                    y.shape[0], y.shape[1],
                    self._input_H, self._input_W,
                    device=y.device, dtype=y.dtype,
                )
                y_up[:, :, ::s, ::s] = y
            else:
                y_up = y

            # B^T: flipped-kernel conv
            x_adj = self._blur_T(y_up)

            logger.debug(
                "[%s] SR adjoint: %s -> %s",
                __version__, tuple(y.shape), tuple(x_adj.shape),
            )
            return x_adj

        except Exception as e:
            logger.error("[%s] SR adjoint failed: %s", __version__, e)
            raise


# ─────────────────────────────────────────────────────────────────────────────
# SAR Forward Model
# ─────────────────────────────────────────────────────────────────────────────
class SARForwardModel(ForwardModel):
    """
    SAR despeckling forward model in log-domain:
        forward(x)  = log(clamp(x, min=eps))   [intensity -> log-intensity]
        adjoint(y)  = clamp(exp(y), max=MAX)    [log-intensity -> intensity]

    Additional functionality included:
        - Input clamping before log (fatal without this)
        - Shape validation on adjoint input
        - Output clamping after exp to prevent overflow
    """

    def __init__(
        self,
        clamp_min: float = SAR_INPUT_CLAMP_MIN,
        clamp_max: float = SAR_OUTPUT_CLAMP_MAX,
    ):
        super().__init__()
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        logger.info(
            "[%s] SARForwardModel created: clamp_min=%.2e, clamp_max=%.2e",
            __version__, clamp_min, clamp_max,
        )

    # ------------------------------------------------------------------
    # Forward: x (intensity) → y (log-intensity)
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_4d(x, "SARForwardModel.forward input x")

        try:
            # Detect and warn about non-positive values before clamping
            n_nonpos = (x <= 0).sum().item()
            if n_nonpos > 0:
                logger.warning(
                    "[%s] SAR forward: %d non-positive values clamped to %.2e",
                    __version__, n_nonpos, self.clamp_min,
                )

            x_safe = x.clamp(min=self.clamp_min)
            y = torch.log(x_safe)

            if torch.isnan(y).any() or torch.isinf(y).any():
                logger.error(
                    "[%s] SAR forward produced NaN/Inf after log. "
                    "Input stats: min=%.4e, max=%.4e",
                    __version__, x.min().item(), x.max().item(),
                )
                raise RuntimeError("SAR forward: NaN/Inf in log output")

            logger.debug(
                "[%s] SAR forward: %s -> %s, y_range=[%.3f, %.3f]",
                __version__, tuple(x.shape), tuple(y.shape),
                y.min().item(), y.max().item(),
            )
            return y

        except RuntimeError:
            raise
        except Exception as e:
            logger.error("[%s] SAR forward failed: %s", __version__, e)
            raise

    # ------------------------------------------------------------------
    # Adjoint: y (log-intensity) → x_hat (intensity)
    # ------------------------------------------------------------------
    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        self._validate_4d(y, "SARForwardModel.adjoint input y")

        try:
            x_hat = torch.exp(y)

            # Clamp to prevent overflow
            n_overflow = (x_hat > self.clamp_max).sum().item()
            if n_overflow > 0:
                logger.warning(
                    "[%s] SAR adjoint: %d values exceeded clamp_max=%.2e",
                    __version__, n_overflow, self.clamp_max,
                )

            x_hat = x_hat.clamp(max=self.clamp_max)

            if torch.isnan(x_hat).any() or torch.isinf(x_hat).any():
                logger.error(
                    "[%s] SAR adjoint produced NaN/Inf after exp. "
                    "y stats: min=%.4e, max=%.4e",
                    __version__, y.min().item(), y.max().item(),
                )
                raise RuntimeError("SAR adjoint: NaN/Inf in exp output")

            logger.debug(
                "[%s] SAR adjoint: %s -> %s",
                __version__, tuple(y.shape), tuple(x_hat.shape),
            )
            return x_hat

        except RuntimeError:
            raise
        except Exception as e:
            logger.error("[%s] SAR adjoint failed: %s", __version__, e)
            raise
