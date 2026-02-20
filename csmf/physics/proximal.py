"""
================================================================================
FILE:    csmf/physics/proximal.py
VERSION: WP1.1-Prox-v1.0
ABBR:    PROX
DESC:    Proximal operators for SR and SAR inverse problems.
         Solves: arg min_z [ ||Az-y||²/2σ² + λ||z-x||² ]

         Three solvers:
           ProximalOperator   - general dispatcher (closed-form / PCG)
           SRProximalFourier  - FFT-based solve for SR blur (no downsample)
           SARProximal        - log-domain quadratic approximation

AUTHOR:  CSMF Project
DATE:    2026-02-17
================================================================================

CHANGELOG:
- v1.0 (2026-02-17): Initial implementation. ProximalOperator with closed-form
  dispatch and PCG (diagonal preconditioner + per-iter residual tracking).
  SRProximalFourier using FFT normal equations. SARProximal log-domain
  closed-form with positive clamping. Error logging throughout.
================================================================================
"""

__version__ = "WP1.1-Prox-v1.0"

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
PCG_DEFAULT_ITERS    = 3
PCG_RESIDUAL_TOL     = 1e-6      # early-stop if ||r|| < tol
FOURIER_EPS          = 1e-8      # denominator guard in FFT solve
SAR_OUTPUT_CLAMP_MIN = 1e-8      # SAR intensities must be positive


# ─────────────────────────────────────────────────────────────────────────────
# General Proximal Operator
# ─────────────────────────────────────────────────────────────────────────────
class ProximalOperator:
    """
    General proximal operator for:
        arg min_z  ||Az - y||² / (2σ²)  +  λ||z - x||²

    Closed-form solution via normal equations:
        z = (A⊤A/σ² + λI)⁻¹ (A⊤y/σ² + λx)

    Dispatch logic:
        1. If forward_model exposes .fourier_solve() → SRProximalFourier (fast)
        2. Otherwise                                 → PCG (general)

    Args:
        forward_model : ForwardModel instance with .forward() and .adjoint()
        sigma         : noise standard deviation (measurement noise level)
        lam           : regularisation weight (balance data vs prior)
        pcg_iters     : max PCG iterations (default 3)
    """

    def __init__(
        self,
        forward_model,
        sigma: float = 0.1,
        lam:   float = 0.1,
        pcg_iters: int = PCG_DEFAULT_ITERS,
    ):
        if sigma <= 0:
            msg = f"[{__version__}] sigma must be > 0, got {sigma}"
            logger.error(msg)
            raise ValueError(msg)
        if lam < 0:
            msg = f"[{__version__}] lam must be >= 0, got {lam}"
            logger.error(msg)
            raise ValueError(msg)
        if pcg_iters < 1:
            msg = f"[{__version__}] pcg_iters must be >= 1, got {pcg_iters}"
            logger.error(msg)
            raise ValueError(msg)

        self.A         = forward_model
        self.sigma     = sigma
        self.lam       = lam
        self.pcg_iters = pcg_iters

        logger.info(
            "[%s] ProximalOperator created: sigma=%.4f, lam=%.4f, pcg_iters=%d",
            __version__, sigma, lam, pcg_iters,
        )

    # ------------------------------------------------------------------
    # Public solve entry point
    # ------------------------------------------------------------------
    def solve(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        method: str = "auto",
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Solve proximal problem.

        Args:
            x      : current estimate / prior mean  (N, C, H, W)
            y      : degraded observation            (N, C, H', W')
            method : 'auto' | 'fourier' | 'pcg'
                     'auto' selects Fourier if available, else PCG

        Returns:
            z         : proximal solution, same shape as x
            residuals : list of ||Az_t - y|| per iteration (PCG) or [final] (Fourier)
        """
        self._validate_inputs(x, y)

        if method == "auto":
            if hasattr(self.A, "fourier_solve"):
                method = "fourier"
            else:
                method = "pcg"

        logger.info("[%s] ProximalOperator.solve: method=%s", __version__, method)

        if method == "fourier":
            return self._fourier_dispatch(x, y)
        elif method == "pcg":
            return self._pcg(x, y, num_iters=self.pcg_iters)
        else:
            msg = f"[{__version__}] Unknown method '{method}'. Use 'auto', 'fourier', or 'pcg'."
            logger.error(msg)
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # Fourier dispatch — delegates to SRProximalFourier
    # ------------------------------------------------------------------
    def _fourier_dispatch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[float]]:
        """Delegate to SRProximalFourier via forward_model.fourier_solve()."""
        try:
            z = self.A.fourier_solve(x, y, self.sigma, self.lam)
            residual = torch.norm(self.A.forward(z) - y).item()
            logger.info("[%s] Fourier solve residual: %.6f", __version__, residual)
            return z, [residual]
        except Exception as e:
            logger.error("[%s] Fourier dispatch failed: %s", __version__, e)
            raise

    # ------------------------------------------------------------------
    # PCG solver with diagonal preconditioner + residual tracking
    # ------------------------------------------------------------------
    def _pcg(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        num_iters: int,
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Preconditioned Conjugate Gradient for:
            (A⊤A/σ² + λI) z = b,   b = A⊤y/σ² + λx

        Diagonal preconditioner:
            M⁻¹ = 1 / (diag_AtA/σ² + λ)

        diag(A⊤A) is estimated per-pixel using a single A⊤A pass on ones.

        Returns:
            z         : solution tensor
            residuals : ||Az_t - y|| at each iteration
        """
        sigma2 = self.sigma ** 2

        # ── right-hand side ───────────────────────────────────────────
        try:
            At_y = self.A.adjoint(y) / sigma2        # A⊤y / σ²
        except Exception as e:
            logger.error("[%s] PCG: adjoint(y) failed: %s", __version__, e)
            raise

        b = At_y + self.lam * x                      # (N, C, H, W)

        # ── diagonal preconditioner (M⁻¹) ────────────────────────────
        # Estimate diag(A⊤A) by passing a ones tensor through A⊤A
        try:
            ones = torch.ones_like(x)
            A_ones  = self.A.forward(ones)
            AtA_diag = self.A.adjoint(A_ones)        # approx diag(A⊤A) per pixel
            M_inv = 1.0 / (AtA_diag / sigma2 + self.lam + 1e-8)
        except Exception as e:
            logger.warning(
                "[%s] PCG: diagonal preconditioner failed (%s), "
                "falling back to identity preconditioner.",
                __version__, e,
            )
            M_inv = torch.ones_like(x) / (1.0 / sigma2 + self.lam)

        # ── linear operator: A_op(z) = A⊤Az/σ² + λz ─────────────────
        def A_op(z: torch.Tensor) -> torch.Tensor:
            try:
                Az     = self.A.forward(z)
                At_Az  = self.A.adjoint(Az) / sigma2
                return At_Az + self.lam * z
            except Exception as e:
                logger.error("[%s] PCG A_op failed: %s", __version__, e)
                raise

        # ── initialise ────────────────────────────────────────────────
        z   = x.clone()                              # warm start from prior
        r   = b - A_op(z)                            # initial residual
        d   = M_inv * r                              # preconditioned direction
        rz  = (r * d).sum()                          # r⊤ M⁻¹ r

        residuals: List[float] = []

        # ── iterations ───────────────────────────────────────────────
        for t in range(num_iters):
            Ad    = A_op(d)
            dAd   = (d * Ad).sum()

            if dAd.abs() < 1e-12:
                logger.warning(
                    "[%s] PCG iter %d: dAd=%.2e near zero, stopping early.",
                    __version__, t, dAd.item(),
                )
                break

            alpha = rz / dAd
            z     = z + alpha * d
            r     = r - alpha * Ad

            # residual tracking: ||Az_t - y||
            with torch.no_grad():
                res = torch.norm(self.A.forward(z) - y).item()
            residuals.append(res)

            logger.info(
                "[%s] PCG iter %d/%d: ||Az-y||=%.6f, alpha=%.4e",
                __version__, t + 1, num_iters, res, alpha.item(),
            )

            if res < PCG_RESIDUAL_TOL:
                logger.info(
                    "[%s] PCG converged at iter %d (tol=%.2e)",
                    __version__, t + 1, PCG_RESIDUAL_TOL,
                )
                break

            # preconditioned update
            z_new  = M_inv * r
            rz_new = (r * z_new).sum()
            beta   = rz_new / (rz + 1e-12)
            d      = z_new + beta * d
            rz     = rz_new

        if torch.isnan(z).any() or torch.isinf(z).any():
            logger.error(
                "[%s] PCG produced NaN/Inf. sigma=%.4e, lam=%.4e",
                __version__, self.sigma, self.lam,
            )
            raise RuntimeError("PCG solution contains NaN/Inf")

        return z, residuals

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    def _validate_inputs(self, x: torch.Tensor, y: torch.Tensor) -> None:
        for name, t in [("x", x), ("y", y)]:
            if t.ndim != 4:
                msg = (
                    f"[{__version__}] ProximalOperator: {name} must be 4-D "
                    f"(N, C, H, W), got shape {tuple(t.shape)}"
                )
                logger.error(msg)
                raise ValueError(msg)
        if x.device != y.device:
            msg = (
                f"[{__version__}] ProximalOperator: x ({x.device}) and "
                f"y ({y.device}) must be on same device"
            )
            logger.error(msg)
            raise ValueError(msg)


# ─────────────────────────────────────────────────────────────────────────────
# SR Proximal — FFT-based (fast, O(N log N))
# ─────────────────────────────────────────────────────────────────────────────
class SRProximalFourier:
    """
    FFT-based proximal solver for SR with blur only (no downsampling, d=1).

    Solves in frequency domain:
        Z(ω) = [ conj(H(ω))·Y(ω)/σ²  +  λ·X(ω) ]
               ─────────────────────────────────────
               [ |H(ω)|²/σ²  +  λ  +  ε          ]

    Then z = Re{ IFFT2(Z) }

    This is O(N log N) vs O(N³) for dense pseudo-inverse.

    Note: Only valid when A = B (blur only, no downsample).
          For A = D∘B (blur + downsample), use PCG.

    Args:
        kernel : Gaussian blur kernel (1, 1, ks, ks) or (C, 1, ks, ks)
        sigma  : noise level
        lam    : regularisation weight
    """

    def __init__(
        self,
        kernel: torch.Tensor,
        sigma: float = 0.1,
        lam:   float = 0.1,
    ):
        if sigma <= 0:
            msg = f"[{__version__}] SRProximalFourier: sigma must be > 0, got {sigma}"
            logger.error(msg)
            raise ValueError(msg)
        if lam < 0:
            msg = f"[{__version__}] SRProximalFourier: lam must be >= 0, got {lam}"
            logger.error(msg)
            raise ValueError(msg)
        if kernel.ndim not in (2, 3, 4):
            msg = f"[{__version__}] SRProximalFourier: kernel must be 2-4D, got {kernel.ndim}D"
            logger.error(msg)
            raise ValueError(msg)

        self.kernel = kernel
        self.sigma  = sigma
        self.lam    = lam

        logger.info(
            "[%s] SRProximalFourier created: sigma=%.4f, lam=%.4f, kernel_shape=%s",
            __version__, sigma, lam, tuple(kernel.shape),
        )

    def solve(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sigma: Optional[float] = None,
        lam:   Optional[float] = None,
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        FFT proximal solve.

        Args:
            x     : current estimate  (N, C, H, W)
            y     : blurred observation (N, C, H, W)  — same spatial size as x
            sigma : override instance sigma if provided
            lam   : override instance lam if provided

        Returns:
            z         : proximal solution  (N, C, H, W)
            residuals : [final_residual]
        """
        sigma = sigma if sigma is not None else self.sigma
        lam   = lam   if lam   is not None else self.lam
        sigma2 = sigma ** 2

        for name, t in [("x", x), ("y", y)]:
            if t.ndim != 4:
                msg = (
                    f"[{__version__}] SRProximalFourier: {name} must be 4-D, "
                    f"got {tuple(t.shape)}"
                )
                logger.error(msg)
                raise ValueError(msg)

        if x.shape != y.shape:
            msg = (
                f"[{__version__}] SRProximalFourier: x {tuple(x.shape)} and "
                f"y {tuple(y.shape)} must have same shape (blur-only, no downsample)"
            )
            logger.error(msg)
            raise ValueError(msg)

        try:
            H_size = (x.shape[-2], x.shape[-1])

            # ── FFT of inputs ─────────────────────────────────────────
            X = torch.fft.rfft2(x)                               # (N, C, H, W//2+1)
            Y = torch.fft.rfft2(y)

            # ── FFT of kernel (pad to image size) ─────────────────────
            kernel_2d = self.kernel
            if kernel_2d.ndim == 4:
                kernel_2d = kernel_2d[0, 0]                      # (ks, ks)
            elif kernel_2d.ndim == 3:
                kernel_2d = kernel_2d[0]

            kernel_2d = kernel_2d.to(x.device)
            H = torch.fft.rfft2(kernel_2d, s=H_size)            # (H, W//2+1)

            # ── Wiener-like normal equation in frequency domain ───────
            H_conj  = torch.conj(H)
            H_abs2  = H.abs() ** 2

            numerator   = H_conj * Y / sigma2  +  lam * X       # (N, C, H, W//2+1)
            denominator = H_abs2 / sigma2       +  lam  +  FOURIER_EPS

            Z = numerator / denominator

            # ── IFFT back to spatial domain ───────────────────────────
            z = torch.fft.irfft2(Z, s=H_size)                   # (N, C, H, W)

            if torch.isnan(z).any() or torch.isinf(z).any():
                logger.error(
                    "[%s] SRProximalFourier: NaN/Inf in solution. "
                    "sigma=%.4e, lam=%.4e",
                    __version__, sigma, lam,
                )
                raise RuntimeError("SRProximalFourier solution contains NaN/Inf")

            # ── residual (||Hz - y||) using FFT ──────────────────────
            Hz  = torch.fft.irfft2(H * torch.fft.rfft2(z), s=H_size)
            res = torch.norm(Hz - y).item()

            logger.info(
                "[%s] SRProximalFourier solved: ||Hz-y||=%.6f", __version__, res
            )
            return z, [res]

        except RuntimeError:
            raise
        except Exception as e:
            logger.error("[%s] SRProximalFourier.solve failed: %s", __version__, e)
            raise


# ─────────────────────────────────────────────────────────────────────────────
# SAR Proximal — log-domain closed-form
# ─────────────────────────────────────────────────────────────────────────────
class SARProximal:
    """
    Log-domain proximal operator for SAR despeckling.

    Minimises:
        ||log(z) - log(y)||²  +  λ||z - x||²

    Closed-form approximation (linearised around y):
        z = (λx + y) / (λ + 1)

    Post-solve: clamp z > clamp_min to keep intensities positive
    (fatal if skipped — subsequent SARForwardModel.forward will crash on log).

    Args:
        lam       : regularisation weight
        clamp_min : minimum output value (must be > 0)
    """

    def __init__(
        self,
        lam:       float = 0.1,
        clamp_min: float = SAR_OUTPUT_CLAMP_MIN,
    ):
        if lam < 0:
            msg = f"[{__version__}] SARProximal: lam must be >= 0, got {lam}"
            logger.error(msg)
            raise ValueError(msg)
        if clamp_min <= 0:
            msg = f"[{__version__}] SARProximal: clamp_min must be > 0, got {clamp_min}"
            logger.error(msg)
            raise ValueError(msg)

        self.lam       = lam
        self.clamp_min = clamp_min

        logger.info(
            "[%s] SARProximal created: lam=%.4f, clamp_min=%.2e",
            __version__, lam, clamp_min,
        )

    def solve(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        lam: Optional[float] = None,
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Solve SAR proximal.

        Args:
            x   : current estimate (intensity domain)  (N, C, H, W)
            y   : observed intensity                   (N, C, H, W)
            lam : override instance lam if provided

        Returns:
            z         : proximal solution (intensity domain, > 0)
            residuals : [||log(z) - log(y)||] final residual
        """
        lam = lam if lam is not None else self.lam

        for name, t in [("x", x), ("y", y)]:
            if t.ndim != 4:
                msg = (
                    f"[{__version__}] SARProximal: {name} must be 4-D, "
                    f"got {tuple(t.shape)}"
                )
                logger.error(msg)
                raise ValueError(msg)

        if x.shape != y.shape:
            msg = (
                f"[{__version__}] SARProximal: x {tuple(x.shape)} and "
                f"y {tuple(y.shape)} must have same shape"
            )
            logger.error(msg)
            raise ValueError(msg)

        try:
            # Closed-form: z = (λx + y) / (λ + 1)
            z = (lam * x + y) / (lam + 1.0)

            # Fatal clamp: SAR intensities must remain positive
            n_nonpos = (z <= 0).sum().item()
            if n_nonpos > 0:
                logger.warning(
                    "[%s] SARProximal: %d non-positive values before clamp. "
                    "x_min=%.4e, y_min=%.4e",
                    __version__, n_nonpos, x.min().item(), y.min().item(),
                )
            z = z.clamp(min=self.clamp_min)

            if torch.isnan(z).any() or torch.isinf(z).any():
                logger.error(
                    "[%s] SARProximal: NaN/Inf in solution. "
                    "lam=%.4e, x_range=[%.4e,%.4e], y_range=[%.4e,%.4e]",
                    __version__, lam,
                    x.min().item(), x.max().item(),
                    y.min().item(), y.max().item(),
                )
                raise RuntimeError("SARProximal solution contains NaN/Inf")

            # Residual: ||log(z) - log(y)||
            log_z = torch.log(z.clamp(min=self.clamp_min))
            log_y = torch.log(y.clamp(min=self.clamp_min))
            res   = torch.norm(log_z - log_y).item()

            logger.info(
                "[%s] SARProximal solved: ||log(z)-log(y)||=%.6f", __version__, res
            )
            return z, [res]

        except RuntimeError:
            raise
        except Exception as e:
            logger.error("[%s] SARProximal.solve failed: %s", __version__, e)
            raise
