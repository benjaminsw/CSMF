# =============================================================================
# Version: WP1.1-FwdMod-v1.0 | Abbr: FWD-MOD
# File: csmf/physics/forward_models.py
# Description: Forward model operators for imaging inverse problems.
#              BlurDownsampleOperator implements A = D∘B (Gaussian blur +
#              stride-2 avg-pool) and Aᵀ = B∘Dᵀ (bilinear upsample + blur)
#              for MNIST super-resolution / deblurring.
# Dependencies: torch, torch.nn.functional
# Changelog:
#   v1.0 - BlurDownsampleOperator: forward A = avg_pool(blur(x))
#   v1.0 - Adjoint Aᵀ: bilinear_upsample(blur(y)) — Bᵀ=B for symmetric Gaussian
#   v1.0 - Handles flat (B, H*W) and spatial (B, 1, H, W) inputs transparently
#   v1.0 - Gaussian kernel built from sigma/kernel_size at init; stored as buffer
#   v1.0 - NaN guard on forward and adjoint outputs; raises RuntimeError on failure
# =============================================================================

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    """
    Build a normalised 2D Gaussian kernel of shape (1, 1, K, K).

    Args:
        kernel_size: odd integer (e.g. 5)
        sigma:       Gaussian std

    Returns:
        kernel: (1, 1, K, K) float32 tensor summing to 1.0
    """
    if kernel_size % 2 == 0:
        msg = f"FWD-MOD | kernel_size must be odd, got {kernel_size}"
        logger.error(msg)
        raise ValueError(msg)
    if sigma <= 0.0:
        msg = f"FWD-MOD | sigma must be > 0, got {sigma}"
        logger.error(msg)
        raise ValueError(msg)

    half = kernel_size // 2
    xs = torch.arange(-half, half + 1, dtype=torch.float32)
    g1d = torch.exp(-xs ** 2 / (2.0 * sigma ** 2))
    g2d = g1d.unsqueeze(1) * g1d.unsqueeze(0)   # (K, K)
    g2d = g2d / g2d.sum()                        # normalise to sum=1
    return g2d.unsqueeze(0).unsqueeze(0)          # (1, 1, K, K)


# ---------------------------------------------------------------------------
# BlurDownsampleOperator
# ---------------------------------------------------------------------------

class BlurDownsampleOperator(nn.Module):
    """
    Forward model A = D ∘ B for MNIST super-resolution / deblurring.

        B : Gaussian blur  (symmetric kernel ⟹ Bᵀ = B, self-adjoint)
        D : stride-s average-pool downsample
        Dᵀ: bilinear upsample × s² (adjoint of stride-s avg-pool)

    So:  Aᵀ = Bᵀ ∘ Dᵀ = B ∘ (bilinear upsample × s²)

    Args:
        image_size:  (H, W) of the clean input image. Default (28, 28).
        blur_sigma:  Gaussian std. Default 1.0.
        blur_kernel: kernel side length (odd). Default 5.
        downsample:  stride factor s. Default 2.
    """

    def __init__(
        self,
        image_size: tuple = (28, 28),
        blur_sigma: float = 1.0,
        blur_kernel: int = 5,
        downsample: int = 2,
    ) -> None:
        super().__init__()

        self.H, self.W = image_size
        self.downsample = downsample
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma
        self.padding = blur_kernel // 2

        kernel = _make_gaussian_kernel(blur_kernel, blur_sigma)
        self.register_buffer("kernel", kernel)   # (1, 1, K, K)

        self.H_lr = self.H // downsample
        self.W_lr = self.W // downsample

        logger.info(
            "FWD-MOD | BlurDownsampleOperator init | "
            "image=(%d,%d) | sigma=%.2f | kernel=%d | ds=%d | lr=(%d,%d)",
            self.H, self.W, blur_sigma, blur_kernel, downsample,
            self.H_lr, self.W_lr,
        )

    # ------------------------------------------------------------------
    # Internal shape helpers
    # ------------------------------------------------------------------

    def _to_4d(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """(B, H*W) → (B, 1, H, W) or pass (B, 1, H, W) through unchanged."""
        if x.dim() == 2:
            B = x.shape[0]
            if x.shape[1] != H * W:
                msg = (f"FWD-MOD | _to_4d: flat tensor has {x.shape[1]} elements, "
                       f"expected {H}×{W}={H*W}")
                logger.error(msg)
                raise ValueError(msg)
            return x.view(B, 1, H, W)
        elif x.dim() == 4:
            return x
        else:
            msg = f"FWD-MOD | _to_4d: unsupported shape {x.shape}"
            logger.error(msg)
            raise ValueError(msg)

    @staticmethod
    def _maybe_flatten(x4d: torch.Tensor, was_flat: bool) -> torch.Tensor:
        """Return flat (B, H*W) if input was flat, else (B, 1, H, W)."""
        return x4d.flatten(1) if was_flat else x4d

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply A = D ∘ B  (blur then downsample).

        Args:
            x: (B, H*W) flat or (B, 1, H, W) spatial clean image

        Returns:
            y: same format, shape (B, H_lr*W_lr) or (B, 1, H_lr, W_lr)
        """
        was_flat = (x.dim() == 2)
        x4d = self._to_4d(x, self.H, self.W)

        # B: Gaussian blur
        x_blur = F.conv2d(x4d, self.kernel, padding=self.padding, groups=1)

        # D: stride-s average pool
        y4d = F.avg_pool2d(x_blur, kernel_size=self.downsample, stride=self.downsample)

        if torch.any(torch.isnan(y4d)):
            logger.error("FWD-MOD | forward: NaN in output after blur+downsample")
            raise RuntimeError("NaN in BlurDownsampleOperator.forward")

        return self._maybe_flatten(y4d, was_flat)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply Aᵀ = Bᵀ ∘ Dᵀ = B ∘ bilinear_upsample.

        Dᵀ of stride-s avg-pool = bilinear upsample to (H, W), scaled by s².
        Bᵀ = B  (Gaussian kernel is symmetric).

        Args:
            y: (B, H_lr*W_lr) flat or (B, 1, H_lr, W_lr) spatial degraded image

        Returns:
            x: (B, H*W) flat or (B, 1, H, W) spatial (same format as input)
        """
        was_flat = (y.dim() == 2)
        y4d = self._to_4d(y, self.H_lr, self.W_lr)

        # Dᵀ: bilinear upsample + scale by s² (exact adjoint of stride-s avg-pool)
        y_up = F.interpolate(
            y4d,
            size=(self.H, self.W),
            mode="bilinear",
            align_corners=False,
        ) * (self.downsample ** 2)

        # Bᵀ = B: same symmetric Gaussian blur
        x4d = F.conv2d(y_up, self.kernel, padding=self.padding, groups=1)

        if torch.any(torch.isnan(x4d)):
            logger.error("FWD-MOD | adjoint: NaN in output after upsample+blur")
            raise RuntimeError("NaN in BlurDownsampleOperator.adjoint")

        return self._maybe_flatten(x4d, was_flat)


# ---------------------------------------------------------------------------
# Backward compatibility alias
# ---------------------------------------------------------------------------

# train_csmf.py imports SRForwardModel — alias to BlurDownsampleOperator
# backward compat alias — maps downsample_factor -> downsample
class SRForwardModel(BlurDownsampleOperator):
    def __init__(self, blur_sigma=1.0, downsample_factor=2, **kwargs):
        super().__init__(blur_sigma=blur_sigma, downsample=downsample_factor, **kwargs)