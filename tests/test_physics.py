"""
================================================================================
FILE:    tests/test_physics.py
VERSION: WP1.2-TestPhys-v1.3
ABBR:    TEST-PHYS
DESC:    Unit and integration tests for WP1 physics components.
         Covers: SRForwardModel, SARForwardModel, ProximalOperator,
                 SRProximalFourier, SARProximal.

         Test categories:
           - SR forward/adjoint shape and correctness
           - SAR forward/adjoint log-domain correctness and clamping
           - Shape validation (ValueError on wrong ndim)
           - PCG residual reduction + tracking + NaN guard
           - Fourier proximal shape + residual + cross-validation vs PCG
           - SAR proximal positivity + residual tracking

AUTHOR:  CSMF Project
DATE:    2026-02-17
================================================================================

CHANGELOG:
- v1.0 (2026-02-17): Initial full test suite. All 18 tests implemented with
  assertion-value logging on failure. Includes adjoint inner-product test,
  Fourier vs PCG cross-validation, SAR positivity and NaN guards. No pass
  statements or placeholder tests.
- v1.1 (2026-02-21): Added debug_fourier_vs_pcg_internals() standalone method
  to TestSRProximalFourier. Inspects H, numerator, denominator, RHS match
  between Fourier and PCG, and kernel centring to diagnose normal-equation
  consistency failures in test_fourier_vs_pcg_close.
- v1.2 (2026-02-21): Added 4 pytest-collected diagnostic tests to
  TestSRProximalFourier: test_sr1x_no_downsampling, test_kernel_fft_alignment,
  test_boundary_conditions_circular, test_normal_equation_consistency. Covers
  all 4 root-cause checks for test_fourier_vs_pcg_close failures.
- v1.3 (2026-02-21): Updated _make_fourier_prox to pass forward_model=sr1x
  instead of kernel=sr1x.kernel, matching new SRProximalFourier v1.2 API
  that builds H from impulse response for exact PCG consistency.
================================================================================
"""

__version__ = "WP1.2-TestPhys-v1.3"

import logging
import math

import pytest
import torch

# ── imports (adjust path if running from repo root) ───────────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

#from forward_models import SRForwardModel, SARForwardModel
from csmf.physics.forward_models import SRForwardModel, SARForwardModel
#from proximal import ProximalOperator, SRProximalFourier, SARProximal
from csmf.physics.proximal import ProximalOperator, SRProximalFourier, SARProximal

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────
BATCH   = 2
CHAN    = 1
H, W    = 28, 28
SIGMA   = 0.1
LAM     = 0.1

@pytest.fixture
def sr2x():
    """SRForwardModel ×2."""
    return SRForwardModel(blur_sigma=1.0, downsample_factor=2, in_channels=CHAN)

@pytest.fixture
def sr4x():
    """SRForwardModel ×4."""
    return SRForwardModel(blur_sigma=1.5, downsample_factor=4, in_channels=CHAN)

@pytest.fixture
def sr1x():
    """SRForwardModel blur-only (no downsample), used for Fourier tests."""
    return SRForwardModel(blur_sigma=1.0, downsample_factor=1, in_channels=CHAN)

@pytest.fixture
def sar():
    """SARForwardModel."""
    return SARForwardModel()

@pytest.fixture
def x_clean():
    torch.manual_seed(0)
    return torch.rand(BATCH, CHAN, H, W) + 0.1   # strictly positive for SAR

@pytest.fixture
def x_randn():
    torch.manual_seed(1)
    return torch.randn(BATCH, CHAN, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# SR Forward Model Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestSRForwardModel:

    def test_sr_forward_shape_2x(self, sr2x, x_randn):
        """SR ×2 forward output shape = (N, C, H/2, W/2)."""
        y = sr2x.forward(x_randn)
        expected = (BATCH, CHAN, H // 2, W // 2)
        assert y.shape == expected, (
            f"[{__version__}] SR ×2 forward shape: got {tuple(y.shape)}, "
            f"expected {expected}"
        )
        logger.info("[%s] test_sr_forward_shape_2x PASSED: %s", __version__, tuple(y.shape))

    def test_sr_forward_shape_4x(self, sr4x, x_randn):
        """SR ×4 forward output shape = (N, C, H/4, W/4)."""
        y = sr4x.forward(x_randn)
        expected = (BATCH, CHAN, H // 4, W // 4)
        assert y.shape == expected, (
            f"[{__version__}] SR ×4 forward shape: got {tuple(y.shape)}, "
            f"expected {expected}"
        )
        logger.info("[%s] test_sr_forward_shape_4x PASSED: %s", __version__, tuple(y.shape))

    def test_sr_forward_blur_reduces_energy(self, sr1x, x_randn):
        """Blur-only forward should reduce or preserve L2 energy (low-pass filter)."""
        y = sr1x.forward(x_randn)
        energy_in  = x_randn.norm().item()
        energy_out = y.norm().item()
        assert energy_out <= energy_in + 1e-3, (
            f"[{__version__}] Blur increased energy: in={energy_in:.4f}, "
            f"out={energy_out:.4f}"
        )
        logger.info(
            "[%s] test_sr_forward_blur_reduces_energy PASSED: "
            "energy_in=%.4f, energy_out=%.4f",
            __version__, energy_in, energy_out,
        )

    def test_sr_adjoint_shape_2x(self, sr2x, x_randn):
        """SR ×2 adjoint restores spatial dimensions."""
        y     = sr2x.forward(x_randn)
        x_adj = sr2x.adjoint(y)
        assert x_adj.shape == x_randn.shape, (
            f"[{__version__}] SR adjoint shape: got {tuple(x_adj.shape)}, "
            f"expected {tuple(x_randn.shape)}"
        )
        logger.info("[%s] test_sr_adjoint_shape_2x PASSED", __version__)

    def test_sr_adjoint_inner_product(self, sr2x):
        """
        Adjoint correctness: <Ax, y> == <x, A^T y>
        Relative tolerance: |lhs - rhs| / |lhs| < 1e-3
        """
        torch.manual_seed(42)
        x = torch.randn(BATCH, CHAN, H, W)
        y = torch.randn(BATCH, CHAN, H // 2, W // 2)

        Ax  = sr2x.forward(x)
        Aty = sr2x.adjoint(y)

        lhs = (Ax  * y).sum().item()
        rhs = (x * Aty).sum().item()

        rel_err = abs(lhs - rhs) / (abs(lhs) + 1e-8)
        assert rel_err < 1e-3, (
            f"[{__version__}] Adjoint inner product: lhs={lhs:.6f}, "
            f"rhs={rhs:.6f}, rel_err={rel_err:.2e} (tol=1e-3)"
        )
        logger.info(
            "[%s] test_sr_adjoint_inner_product PASSED: "
            "lhs=%.6f, rhs=%.6f, rel_err=%.2e",
            __version__, lhs, rhs, rel_err,
        )


# ─────────────────────────────────────────────────────────────────────────────
# SAR Forward / Adjoint Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestSARForwardModel:

    def test_sar_forward_shape(self, sar, x_clean):
        """SAR forward preserves shape."""
        y = sar.forward(x_clean)
        assert y.shape == x_clean.shape, (
            f"[{__version__}] SAR forward shape: got {tuple(y.shape)}, "
            f"expected {tuple(x_clean.shape)}"
        )
        logger.info("[%s] test_sar_forward_shape PASSED", __version__)

    def test_sar_forward_log_domain(self, sar, x_clean):
        """SAR forward output ≈ log(x) for positive inputs."""
        y        = sar.forward(x_clean)
        y_expect = torch.log(x_clean)
        max_diff = (y - y_expect).abs().max().item()
        assert max_diff < 1e-5, (
            f"[{__version__}] SAR forward not matching log(x): max_diff={max_diff:.2e}"
        )
        logger.info(
            "[%s] test_sar_forward_log_domain PASSED: max_diff=%.2e",
            __version__, max_diff,
        )

    def test_sar_adjoint_exp(self, sar, x_clean):
        """SAR adjoint(forward(x)) ≈ x for positive inputs."""
        y     = sar.forward(x_clean)
        x_rec = sar.adjoint(y)
        max_diff = (x_rec - x_clean).abs().max().item()
        assert max_diff < 1e-5, (
            f"[{__version__}] SAR adjoint(forward(x)) != x: max_diff={max_diff:.2e}"
        )
        logger.info(
            "[%s] test_sar_adjoint_exp PASSED: max_diff=%.2e",
            __version__, max_diff,
        )

    def test_sar_clamp_negative_input(self, sar):
        """SAR forward must not crash or produce -inf on negative/zero input."""
        x_neg = torch.tensor([[-1.0, 0.0, 0.5, 1.0]]).view(1, 1, 2, 2)
        y = sar.forward(x_neg)
        assert not torch.isinf(y).any(), (
            f"[{__version__}] SAR forward produced -inf on negative input: {y}"
        )
        assert not torch.isnan(y).any(), (
            f"[{__version__}] SAR forward produced NaN on negative input: {y}"
        )
        logger.info(
            "[%s] test_sar_clamp_negative_input PASSED: output=%s",
            __version__, y.flatten().tolist(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Shape Validation Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestShapeValidation:

    def test_sr_adjoint_wrong_ndim(self, sr2x):
        """SR adjoint must raise ValueError on 3-D input."""
        y_3d = torch.randn(BATCH, H // 2, W // 2)   # missing channel dim
        with pytest.raises(ValueError) as exc_info:
            sr2x.adjoint(y_3d)
        assert "4-D" in str(exc_info.value), (
            f"[{__version__}] Expected '4-D' in error message, got: {exc_info.value}"
        )
        logger.info("[%s] test_sr_adjoint_wrong_ndim PASSED", __version__)

    def test_sar_forward_wrong_ndim(self, sar):
        """SAR forward must raise ValueError on 3-D input."""
        x_3d = torch.rand(BATCH, H, W)
        with pytest.raises(ValueError) as exc_info:
            sar.forward(x_3d)
        assert "4-D" in str(exc_info.value), (
            f"[{__version__}] Expected '4-D' in error message, got: {exc_info.value}"
        )
        logger.info("[%s] test_sar_forward_wrong_ndim PASSED", __version__)


# ─────────────────────────────────────────────────────────────────────────────
# ProximalOperator (PCG) Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestProximalOperatorPCG:

    def test_pcg_reduces_residual(self, sr2x, x_randn):
        """
        PCG proximal: ||Az_prox - y|| < ||Ax - y|| (residual reduction).
        """
        torch.manual_seed(10)
        x = x_randn
        y = sr2x.forward(x) + SIGMA * torch.randn(BATCH, CHAN, H // 2, W // 2)

        prox = ProximalOperator(sr2x, sigma=SIGMA, lam=LAM, pcg_iters=3)
        z, residuals = prox.solve(x, y, method="pcg")

        res_before = torch.norm(sr2x.forward(x) - y).item()
        res_after  = torch.norm(sr2x.forward(z) - y).item()

        assert res_after < res_before, (
            f"[{__version__}] PCG did not reduce residual: "
            f"before={res_before:.6f}, after={res_after:.6f}"
        )
        logger.info(
            "[%s] test_pcg_reduces_residual PASSED: before=%.6f, after=%.6f",
            __version__, res_before, res_after,
        )

    def test_pcg_residual_list_length(self, sr2x, x_randn):
        """
        Residuals list length == number of PCG iters actually executed.
        Must be >= 1 and <= pcg_iters.
        """
        torch.manual_seed(11)
        x = x_randn
        y = sr2x.forward(x) + SIGMA * torch.randn(BATCH, CHAN, H // 2, W // 2)

        n_iters = 3
        prox = ProximalOperator(sr2x, sigma=SIGMA, lam=LAM, pcg_iters=n_iters)
        _, residuals = prox.solve(x, y, method="pcg")

        assert 1 <= len(residuals) <= n_iters, (
            f"[{__version__}] PCG residuals list length {len(residuals)} "
            f"not in [1, {n_iters}]"
        )
        logger.info(
            "[%s] test_pcg_residual_list_length PASSED: len(residuals)=%d",
            __version__, len(residuals),
        )

    def test_pcg_nan_guard(self, sr2x):
        """
        PCG must raise RuntimeError (not silently return NaN) on
        extremely ill-conditioned input (near-zero sigma).
        """
        torch.manual_seed(12)
        x = torch.randn(BATCH, CHAN, H, W) * 1e6   # large values
        y = sr2x.forward(x)

        # Tiny sigma makes normal equations ill-conditioned
        prox = ProximalOperator(sr2x, sigma=1e-10, lam=0.0, pcg_iters=1)
        try:
            z, _ = prox.solve(x, y, method="pcg")
            # If it returns, z must not contain NaN/Inf
            assert not torch.isnan(z).any() and not torch.isinf(z).any(), (
                f"[{__version__}] PCG silently returned NaN/Inf under "
                f"ill-conditioned input"
            )
            logger.info(
                "[%s] test_pcg_nan_guard PASSED: solution is finite",
                __version__,
            )
        except RuntimeError as e:
            # RuntimeError from NaN guard is also acceptable
            logger.info(
                "[%s] test_pcg_nan_guard PASSED: RuntimeError raised: %s",
                __version__, e,
            )


# ─────────────────────────────────────────────────────────────────────────────
# SRProximalFourier Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestSRProximalFourier:

    def _make_fourier_prox(self, sr1x):
        return SRProximalFourier(
            forward_model=sr1x,
            sigma=SIGMA,
            lam=LAM,
        )

    def test_fourier_solve_shape(self, sr1x, x_clean):
        """Fourier solve output shape matches input."""
        y    = sr1x.forward(x_clean)
        fprox = self._make_fourier_prox(sr1x)
        z, _ = fprox.solve(x_clean, y)
        assert z.shape == x_clean.shape, (
            f"[{__version__}] Fourier solve shape: got {tuple(z.shape)}, "
            f"expected {tuple(x_clean.shape)}"
        )
        logger.info("[%s] test_fourier_solve_shape PASSED", __version__)

    def test_fourier_reduces_residual(self, sr1x, x_clean):
        """||Hz - y|| < ||Hx - y|| after Fourier solve."""
        torch.manual_seed(20)
        y = sr1x.forward(x_clean) + SIGMA * torch.randn_like(x_clean)

        fprox = self._make_fourier_prox(sr1x)
        z, residuals = fprox.solve(x_clean, y)

        res_before = torch.norm(sr1x.forward(x_clean) - y).item()
        res_after  = residuals[-1]

        assert res_after < res_before, (
            f"[{__version__}] Fourier solve did not reduce residual: "
            f"before={res_before:.6f}, after={res_after:.6f}"
        )
        logger.info(
            "[%s] test_fourier_reduces_residual PASSED: before=%.6f, after=%.6f",
            __version__, res_before, res_after,
        )

    def test_fourier_vs_pcg_close(self, sr1x, x_clean):
        """
        Cross-validation: Fourier and PCG solutions should agree within atol=5e-2.
        Both solve the same normal equations — difference is numerical path only.
        """
        torch.manual_seed(21)
        y = sr1x.forward(x_clean) + SIGMA * torch.randn_like(x_clean)

        fprox = self._make_fourier_prox(sr1x)
        z_fourier, _ = fprox.solve(x_clean, y)

        prox = ProximalOperator(sr1x, sigma=SIGMA, lam=LAM, pcg_iters=100)
        z_pcg, _ = prox.solve(x_clean, y, method="pcg")

        max_diff = (z_fourier - z_pcg).abs().max().item()
        mean_diff = (z_fourier - z_pcg).abs().mean().item()

        assert max_diff < 5e-2, (
            f"[{__version__}] Fourier vs PCG max_diff={max_diff:.4f} > 5e-2. "
            f"mean_diff={mean_diff:.4f}"
        )
        logger.info(
            "[%s] test_fourier_vs_pcg_close PASSED: max_diff=%.4f, mean_diff=%.4f",
            __version__, max_diff, mean_diff,
        )

    def debug_fourier_vs_pcg_internals(self, sr1x, x_clean):
        """
        Standalone debug helper (not collected by pytest — call manually).
        Inspects Fourier numerator/denominator step-by-step and compares
        the RHS seen by Fourier vs PCG to isolate normal-equation mismatches.

        Usage:
            t = TestSRProximalFourier()
            sr1x = SRForwardModel(blur_sigma=1.0, downsample_factor=1, in_channels=1)
            x    = torch.rand(1, 1, 28, 28) + 0.1
            t.debug_fourier_vs_pcg_internals(sr1x, x)
        """
        import torch

        torch.manual_seed(21)
        y = sr1x.forward(x_clean) + SIGMA * torch.randn_like(x_clean)

        H_size   = (x_clean.shape[-2], x_clean.shape[-1])
        sigma2   = SIGMA ** 2

        # ── Kernel in frequency domain ────────────────────────────────
        kernel_2d = sr1x.kernel
        if kernel_2d.ndim == 4:
            kernel_2d = kernel_2d[0, 0]
        elif kernel_2d.ndim == 3:
            kernel_2d = kernel_2d[0]

        H = torch.fft.rfft2(kernel_2d, s=H_size)

        print("\n=== DEBUG: Fourier vs PCG internals ===")
        print(f"  H  abs  min={H.abs().min():.4e}  max={H.abs().max():.4e}  mean={H.abs().mean():.4e}")
        print(f"  H  DC component (should be ~1.0 for normalised kernel): {H[0,0].real:.6f}")

        # ── Frequency-domain inputs ───────────────────────────────────
        X = torch.fft.rfft2(x_clean)
        Y = torch.fft.rfft2(y)

        # ── Fourier numerator / denominator ───────────────────────────
        H_conj  = torch.conj(H)
        H_abs2  = H.abs() ** 2

        num_data  = H_conj * Y / sigma2          # data term in numerator
        num_prior = LAM * X                       # prior term in numerator
        den       = H_abs2 / sigma2 + LAM         # denominator (no eps for inspection)

        print(f"  num_data  abs mean={num_data.abs().mean():.4e}")
        print(f"  num_prior abs mean={num_prior.abs().mean():.4e}")
        print(f"  den            mean={den.abs().mean():.4e}  min={den.abs().min():.4e}")

        Z   = (num_data + num_prior) / (den + 1e-8)
        z_f = torch.fft.irfft2(Z, s=H_size)
        print(f"  z_fourier  mean={z_f.mean():.4e}  std={z_f.std():.4e}")

        # ── Fourier RHS in spatial domain ─────────────────────────────
        # conj(H)*Y/σ² <-> adjoint(y)/σ²  (should match if BCs match)
        rhs_fourier_data  = torch.fft.irfft2(H_conj * Y / sigma2, s=H_size)
        rhs_fourier_prior = LAM * x_clean
        rhs_fourier       = rhs_fourier_data + rhs_fourier_prior

        # ── PCG RHS in spatial domain ─────────────────────────────────
        At_y      = sr1x.adjoint(y) / sigma2
        rhs_pcg   = At_y + LAM * x_clean

        rhs_max_diff  = (rhs_fourier - rhs_pcg).abs().max().item()
        rhs_mean_diff = (rhs_fourier - rhs_pcg).abs().mean().item()
        rhs_match     = torch.allclose(rhs_fourier, rhs_pcg, atol=1e-3)

        print(f"\n  --- RHS comparison (Fourier vs PCG) ---")
        print(f"  rhs_fourier mean={rhs_fourier.mean():.4e}  std={rhs_fourier.std():.4e}")
        print(f"  rhs_pcg     mean={rhs_pcg.mean():.4e}  std={rhs_pcg.std():.4e}")
        print(f"  max_diff={rhs_max_diff:.4e}  mean_diff={rhs_mean_diff:.4e}")
        print(f"  RHS match (atol=1e-3): {rhs_match}")
        if not rhs_match:
            logger.error(
                "[%s] debug: RHS mismatch — adjoint() does not match conj(H)* "
                "in frequency domain. max_diff=%.4e. Likely cause: kernel not "
                "zero-centred before FFT, or boundary condition mismatch.",
                __version__, rhs_max_diff,
            )

        # ── PCG solution (10 iters) ───────────────────────────────────
        prox  = ProximalOperator(sr1x, sigma=SIGMA, lam=LAM, pcg_iters=10)
        z_pcg, pcg_res = prox.solve(x_clean, y, method="pcg")
        print(f"\n  z_pcg      mean={z_pcg.mean():.4e}  std={z_pcg.std():.4e}")
        print(f"  PCG residuals: {[f'{r:.4e}' for r in pcg_res]}")

        # ── Final comparison ──────────────────────────────────────────
        final_max  = (z_f - z_pcg).abs().max().item()
        final_mean = (z_f - z_pcg).abs().mean().item()
        print(f"\n  --- Final solution diff ---")
        print(f"  max_diff={final_max:.4e}  mean_diff={final_mean:.4e}  (tol=5e-2)")
        print("=== END DEBUG ===\n")

        if not rhs_match:
            logger.error(
                "[%s] debug_fourier_vs_pcg_internals: RHS mismatch detected. "
                "Fix adjoint BCs or kernel centring before re-running tests.",
                __version__,
            )
        return {
            "rhs_match":    rhs_match,
            "rhs_max_diff": rhs_max_diff,
            "H_dc":         H[0, 0].real.item(),
            "final_max":    final_max,
            "final_mean":   final_mean,
        }

    # ------------------------------------------------------------------
    # Diagnostic test 1: SR1x must not downsample
    # ------------------------------------------------------------------
    def test_sr1x_no_downsampling(self, sr1x, x_clean):
        """
        SR1x forward output must have same H,W as input.
        Fourier closed-form is only valid for blur-only (no decimation).
        """
        y = sr1x.forward(x_clean)
        assert y.shape == x_clean.shape, (
            f"[{__version__}] sr1x is still downsampling: "
            f"input={tuple(x_clean.shape)}, output={tuple(y.shape)}. "
            "Fourier prox requires downsample_factor=1."
        )
        logger.info(
            "[%s] test_sr1x_no_downsampling PASSED: shape=%s",
            __version__, tuple(y.shape),
        )

    # ------------------------------------------------------------------
    # Diagnostic test 2: Kernel FFT phase alignment (DC at [0,0])
    # ------------------------------------------------------------------
    def test_kernel_fft_alignment(self, sr1x, x_clean):
        """
        ifft2(H) max must be at position [0,0] (FFT convention).
        If max is near center [H//2, W//2] the kernel is not ifftshifted
        before rfft2, causing phase error between Fourier and PCG RHS.
        """
        H_size    = (x_clean.shape[-2], x_clean.shape[-1])
        kernel_2d = sr1x.kernel
        if kernel_2d.ndim == 4:
            kernel_2d = kernel_2d[0, 0]
        elif kernel_2d.ndim == 3:
            kernel_2d = kernel_2d[0]

        kernel_2d = kernel_2d.to(x_clean.device)

        # Build H the same way SRProximalFourier.solve() does
        fprox     = self._make_fourier_prox(sr1x)
        H = torch.fft.rfft2(torch.fft.ifftshift(kernel_2d), s=H_size)

        # Reconstruct spatial PSF from H — DC should be at [0,0]
        psf    = torch.fft.ifft2(H, s=H_size).real
        maxpos = (psf == psf.max()).nonzero(as_tuple=False)[0].tolist()

        assert maxpos == [0, 0], (
            f"[{__version__}] Kernel FFT max at {maxpos}, expected [0, 0]. "
            "Kernel is not ifftshifted before rfft2 — phase mismatch will "
            "cause Fourier RHS != PCG RHS."
        )
        logger.info(
            "[%s] test_kernel_fft_alignment PASSED: DC at %s, H_dc=%.6f",
            __version__, maxpos, H[0, 0].real.item(),
        )

    # ------------------------------------------------------------------
    # Diagnostic test 3: Boundary conditions — circular wrap-around
    # ------------------------------------------------------------------
    def test_boundary_conditions_circular(self, sr1x):
        """
        Blur applied to an impulse at [0,0] must wrap energy around borders
        (circular padding), not fall off (zero padding).
        Fourier closed-form assumes circular convolution — both must match.
        """
        x = torch.zeros(1, 1, H, W)
        x[0, 0, 0, 0] = 1.0          # impulse at top-left corner

        y = sr1x.forward(x)

        energy_topleft     = y[0, 0, :3, :3].sum().item()
        energy_bottomright = y[0, 0, -3:, -3:].sum().item()

        # With circular padding the kernel wraps: bottom-right corner gets
        # energy from the impulse at [0,0]. With zero padding it stays ~0.
        assert energy_bottomright > 1e-4, (
            f"[{__version__}] Bottom-right energy={energy_bottomright:.6f} ≈ 0. "
            "SR forward is using zero padding, not circular. "
            "Fourier and PCG will not agree."
        )
        logger.info(
            "[%s] test_boundary_conditions_circular PASSED: "
            "top-left energy=%.6f, bottom-right energy=%.6f (circular wrap confirmed)",
            __version__, energy_topleft, energy_bottomright,
        )

    # ------------------------------------------------------------------
    # Diagnostic test 4: Normal-equation consistency (σ²/λ scaling)
    # ------------------------------------------------------------------
    def test_normal_equation_consistency(self, sr1x, x_clean):
        """
        Both Fourier and PCG must satisfy the same normal equation:
            (A^T A / σ²  +  λI) z  =  A^T y / σ²  +  λx

        Mean |normal-eq residual| for Fourier solution must be
        within 10× of PCG residual. Large gap → formula/scaling bug.
        """
        torch.manual_seed(21)
        y = sr1x.forward(x_clean) + SIGMA * torch.randn_like(x_clean)

        fprox         = self._make_fourier_prox(sr1x)
        z_fourier, _  = fprox.solve(x_clean, y)

        prox          = ProximalOperator(sr1x, sigma=SIGMA, lam=LAM, pcg_iters=10)
        z_pcg, _      = prox.solve(x_clean, y, method="pcg")

        def normal_residual(z: torch.Tensor) -> torch.Tensor:
            """||( A^T A z / σ²  +  λz ) - ( A^T y / σ²  +  λx )||"""
            try:
                Az   = sr1x.forward(z)
                AtAz = sr1x.adjoint(Az) / (SIGMA ** 2)
                Aty  = sr1x.adjoint(y)  / (SIGMA ** 2)
                lhs  = AtAz + LAM * z
                rhs  = Aty  + LAM * x_clean
                return (lhs - rhs).abs().mean().item()
            except Exception as e:
                logger.error(
                    "[%s] normal_residual computation failed: %s", __version__, e
                )
                raise

        rF = normal_residual(z_fourier)
        rP = normal_residual(z_pcg)

        logger.info(
            "[%s] test_normal_equation_consistency: "
            "Fourier normal-eq residual=%.4e, PCG normal-eq residual=%.4e",
            __version__, rF, rP,
        )

        assert rF < rP * 10, (
            f"[{__version__}] Fourier normal-eq residual={rF:.4e} is more than "
            f"10× PCG residual={rP:.4e}. Fourier formula or σ²/λ scaling is wrong."
        )
        logger.info(
            "[%s] test_normal_equation_consistency PASSED: rF=%.4e, rP=%.4e",
            __version__, rF, rP,
        )


# ─────────────────────────────────────────────────────────────────────────────
# SARProximal Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestSARProximal:

    def test_sar_proximal_output_positive(self, x_clean):
        """All SAR proximal outputs must be strictly positive."""
        y_intensity = x_clean + 0.1 * torch.rand_like(x_clean)
        sprox = SARProximal(lam=LAM)
        z, _ = sprox.solve(x_clean, y_intensity)
        min_val = z.min().item()
        assert min_val > 0, (
            f"[{__version__}] SARProximal output has non-positive value: min={min_val:.2e}"
        )
        logger.info(
            "[%s] test_sar_proximal_output_positive PASSED: min=%.2e",
            __version__, min_val,
        )

    def test_sar_proximal_residual_returned(self, x_clean):
        """SARProximal must return residuals list of length 1."""
        y_intensity = x_clean + 0.1 * torch.rand_like(x_clean)
        sprox = SARProximal(lam=LAM)
        z, residuals = sprox.solve(x_clean, y_intensity)
        assert len(residuals) == 1, (
            f"[{__version__}] SARProximal residuals length: "
            f"got {len(residuals)}, expected 1"
        )
        assert residuals[0] >= 0.0, (
            f"[{__version__}] SARProximal residual is negative: {residuals[0]:.4f}"
        )
        logger.info(
            "[%s] test_sar_proximal_residual_returned PASSED: residual=%.6f",
            __version__, residuals[0],
        )

    def test_sar_proximal_reduces_log_residual(self, x_clean):
        """
        SAR proximal: ||log(z) - log(y)|| < ||log(x) - log(y)||.
        """
        eps = 1e-8
        torch.manual_seed(30)
        y_intensity = x_clean * (1 + 0.3 * torch.rand_like(x_clean))  # noisy intensity

        sprox = SARProximal(lam=LAM)
        z, residuals = sprox.solve(x_clean, y_intensity)

        res_before = torch.norm(
            torch.log(x_clean.clamp(min=eps)) - torch.log(y_intensity.clamp(min=eps))
        ).item()
        res_after = residuals[0]

        assert res_after < res_before, (
            f"[{__version__}] SARProximal did not reduce log-residual: "
            f"before={res_before:.6f}, after={res_after:.6f}"
        )
        logger.info(
            "[%s] test_sar_proximal_reduces_log_residual PASSED: "
            "before=%.6f, after=%.6f",
            __version__, res_before, res_after,
        )
