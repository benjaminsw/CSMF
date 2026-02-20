"""
================================================================================
FILE:    tests/test_physics.py
VERSION: WP1.2-TestPhys-v1.0
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
================================================================================
"""

__version__ = "WP1.2-TestPhys-v1.0"

import logging
import math

import pytest
import torch

# ── imports (adjust path if running from repo root) ───────────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from forward_models import SRForwardModel, SARForwardModel
from proximal import ProximalOperator, SRProximalFourier, SARProximal

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
            kernel=sr1x.kernel,
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

        prox = ProximalOperator(sr1x, sigma=SIGMA, lam=LAM, pcg_iters=10)
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
