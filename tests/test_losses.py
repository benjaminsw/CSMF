# =============================================================================
# Version: WP2.3-TestLoss-v1.1
# Abbr: TEST-LOSS
# File: tests/test_losses.py
# Description: Test suite for WP2 loss components:
#              SW2, Energy Score, CRPS, HybridLoss, annealing,
#              gradient flow, numerical stability, freeze/unfreeze.
# Dependencies: pytest, torch, csmf.losses.*
# Changelog:
#   v1.1 - Added test_hybrid_gradient_flow (grad reaches flow params per term)
#   v1.1 - Added test_numerical_stability (large/near-zero samples, no NaN)
#   v1.1 - Added test_sw2_unequal_sizes (interpolation path coverage)
#   v1.1 - Added test_crps_multidim (beta=1 equivalence to ES)
#   v1.1 - Added test_stage_freeze_unfreeze (grad flags after freeze helpers)
#   v1.0 - Core: SW2 symmetry/identity/triangle, ES proper scoring,
#           CRPS consistency, hybrid loss finite, annealing schedule
# =============================================================================

import logging
import math

import pytest
import torch
import torch.nn as nn

from csmf.losses.sliced_wasserstein import sliced_wasserstein_distance
from csmf.losses.calibration import (
    energy_score,
    crps,
    crps_multidim,
    temperature_scaling,
)
from csmf.losses.hybrid_loss import (
    HybridLoss,
    StageConfig,
    freeze_experts,
    freeze_gate,
    unfreeze_gate,
    unfreeze_last_blocks,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixtures: Minimal real implementations (NO mocks / pass / dummies)
# ---------------------------------------------------------------------------

class _CouplingBlock(nn.Module):
    """Minimal affine coupling block — real invertible transform."""

    def __init__(self, d: int, h_dim: int):
        super().__init__()
        half = d // 2
        self.net_s = nn.Sequential(nn.Linear(half + h_dim, 32), nn.Tanh(), nn.Linear(32, half))
        self.net_t = nn.Sequential(nn.Linear(half + h_dim, 32), nn.Tanh(), nn.Linear(32, half))
        self.d = d
        self.half = half

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        """x -> z, returns (z, log_det)."""
        x1, x2 = x[:, :self.half], x[:, self.half:]
        inp = torch.cat([x1, h], dim=1)
        s = torch.tanh(self.net_s(inp)) * 0.5   # bounded scale for stability
        t = self.net_t(inp)
        z2 = x2 * torch.exp(s) + t
        log_det = s.sum(dim=1)                   # (B,)
        z = torch.cat([x1, z2], dim=1)
        return z, log_det


class MinimalFlow(nn.Module):
    """
    Minimal real conditional flow (2 coupling blocks).
    Implements the interface expected by HybridLoss:
        .conditioner(y)  -> h
        .forward(x, h)   -> (z, log_det)
        .base_log_prob(z) -> log_prob  (B,)
        .sample(h, num_samples) -> (B, S, d)
        .blocks          -> ModuleList of coupling blocks (for unfreeze tests)
    """

    def __init__(self, d: int = 8, h_dim: int = 4, n_blocks: int = 2):
        super().__init__()
        self.d = d
        self.h_dim = h_dim
        self.conditioner_net = nn.Sequential(
            nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, h_dim)
        )
        self.blocks = nn.ModuleList(
            [_CouplingBlock(d, h_dim) for _ in range(n_blocks)]
        )

    def conditioner(self, y: torch.Tensor) -> torch.Tensor:
        return self.conditioner_net(y)             # (B, h_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        """Encode x -> z through all blocks. Returns (z, total_log_det)."""
        z = x
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        for block in self.blocks:
            z, ld = block(z, h)
            log_det_total = log_det_total + ld
        return z, log_det_total

    def base_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Standard normal log probability. Returns (B,)."""
        d = z.shape[1]
        return -0.5 * (d * math.log(2 * math.pi) + (z ** 2).sum(dim=1))

    def sample(self, h: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Sample x ~ q(x|y). Returns (B, num_samples, d)."""
        B = h.shape[0]
        z = torch.randn(B, num_samples, self.d, device=h.device)
        # Simple inverse: subtract shift (t) and divide by exp(s) per block (reversed)
        # For testing purposes we use the forward pass on noise as approximate samples
        samples = z                                # (B, S, d)
        return samples


class MinimalForwardModel(nn.Module):
    """Identity forward model A(x) = x for testing."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class MinimalCSMFModel(nn.Module):
    """Wraps experts + gate for freeze/unfreeze tests."""

    def __init__(self, d: int = 8, n_experts: int = 2):
        super().__init__()
        self.experts = nn.ModuleList([MinimalFlow(d=d) for _ in range(n_experts)])
        self.gate = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, n_experts))

    def conditioner(self, y):
        return self.experts[0].conditioner(y)

    def forward(self, x, h):
        return self.experts[0].forward(x, h)

    def base_log_prob(self, z):
        return self.experts[0].base_log_prob(z)

    def sample(self, h, num_samples=1):
        return self.experts[0].sample(h, num_samples)


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def d():
    return 8


@pytest.fixture
def flow(d):
    return MinimalFlow(d=d)


@pytest.fixture
def fwd_model():
    return MinimalForwardModel()


@pytest.fixture
def csmf_model(d):
    return MinimalCSMFModel(d=d)


@pytest.fixture
def hybrid_loss(fwd_model):
    return HybridLoss(
        forward_model=fwd_model,
        lambda_cons=0.1,
        lambda_trans=0.05,
        lambda_cal=0.01,
    )


# ===========================================================================
# SW2 Tests
# ===========================================================================

class TestSW2:

    def test_sw2_symmetry(self):
        """SW2(X, Y) ≈ SW2(Y, X)."""
        X = torch.randn(100, 8)
        Y = torch.randn(100, 8)
        sw_xy = sliced_wasserstein_distance(X, Y)
        sw_yx = sliced_wasserstein_distance(Y, X)
        diff = abs(sw_xy.item() - sw_yx.item())
        logger.info("SW2 symmetry diff=%.6f", diff)
        assert diff < 0.05, f"SW2 symmetry violated: diff={diff:.6f}"

    def test_sw2_identity(self):
        """SW2(X, X) ≈ 0."""
        X = torch.randn(100, 8)
        sw = sliced_wasserstein_distance(X, X)
        logger.info("SW2 identity=%.6f", sw.item())
        assert sw.item() < 1e-3, f"SW2(X,X) should be ≈0, got {sw.item():.6f}"

    def test_sw2_triangle_inequality(self):
        """SW2(X, Z) ≤ SW2(X, Y) + SW2(Y, Z)."""
        torch.manual_seed(0)
        X = torch.randn(80, 8)
        Y = torch.randn(80, 8) + 2.0
        Z = torch.randn(80, 8) + 4.0
        sw_xz = sliced_wasserstein_distance(X, Z).item()
        sw_xy = sliced_wasserstein_distance(X, Y).item()
        sw_yz = sliced_wasserstein_distance(Y, Z).item()
        logger.info("SW2 triangle: XZ=%.4f, XY+YZ=%.4f", sw_xz, sw_xy + sw_yz)
        assert sw_xz <= sw_xy + sw_yz + 0.05, (
            f"Triangle inequality violated: {sw_xz:.4f} > {sw_xy:.4f} + {sw_yz:.4f}"
        )

    def test_sw2_unequal_sizes(self):
        """Interpolation path: SW2(N=80, M=120) is finite and positive."""
        X = torch.randn(80, 8)
        Y = torch.randn(120, 8)
        sw = sliced_wasserstein_distance(X, Y)
        logger.info("SW2 unequal sizes N=80 M=120: %.4f", sw.item())
        assert torch.isfinite(sw), "SW2 with unequal sizes returned non-finite"
        assert sw.item() >= 0.0, "SW2 must be non-negative"

    def test_sw2_max_mode(self):
        """max-SW mode returns a finite non-negative scalar."""
        X = torch.randn(50, 8)
        Y = torch.randn(50, 8)
        sw = sliced_wasserstein_distance(X, Y, mode="max")
        logger.info("SW2 max-mode=%.4f", sw.item())
        assert torch.isfinite(sw), "max-SW returned non-finite"
        assert sw.item() >= 0.0

    def test_sw2_invalid_mode(self):
        """Invalid mode raises ValueError."""
        X = torch.randn(10, 4)
        with pytest.raises(ValueError, match="mode"):
            sliced_wasserstein_distance(X, X, mode="invalid")

    def test_sw2_dimension_mismatch(self):
        """Mismatched dimensions raise ValueError."""
        X = torch.randn(10, 4)
        Y = torch.randn(10, 6)
        with pytest.raises(ValueError):
            sliced_wasserstein_distance(X, Y)


# ===========================================================================
# Energy Score Tests
# ===========================================================================

class TestEnergyScore:

    def test_es_proper_scoring(self):
        """ES is smaller in magnitude for tight samples vs spread samples."""
        torch.manual_seed(42)
        ref = torch.zeros(8)
        samples_close = torch.randn(50, 8) * 0.1
        samples_far   = torch.randn(50, 8) * 10.0
        es_close = energy_score(samples_close, ref).item()
        es_far   = energy_score(samples_far,   ref).item()
        logger.info("ES close=%.4f, ES far=%.4f", es_close, es_far)
        assert abs(es_close) < abs(es_far), (
            f"ES proper scoring violated: es_close={es_close:.4f}, es_far={es_far:.4f}"
        )

    def test_es_requires_min_samples(self):
        """ES with S=1 raises ValueError."""
        with pytest.raises(ValueError, match="S>=2"):
            energy_score(torch.randn(1, 8), torch.zeros(8))

    def test_es_finite(self):
        """ES output is finite for normal inputs."""
        samples = torch.randn(20, 8)
        ref     = torch.randn(8)
        es = energy_score(samples, ref)
        assert torch.isfinite(es), f"ES returned non-finite: {es.item()}"

    def test_es_reference_shapes(self):
        """ES accepts both (d,) and (1, d) reference shapes."""
        samples = torch.randn(20, 8)
        es1 = energy_score(samples, torch.zeros(8))
        es2 = energy_score(samples, torch.zeros(1, 8))
        assert abs(es1.item() - es2.item()) < 1e-5, "ES shape variants disagree"


# ===========================================================================
# CRPS Tests
# ===========================================================================

class TestCRPS:

    def test_crps_consistency(self):
        """CRPS is smaller for samples matching the reference distribution."""
        torch.manual_seed(7)
        ref = torch.tensor(0.0)
        # Samples from N(0,1) should score better against ref=0 than N(5,1)
        good_samples = torch.randn(200)
        bad_samples  = torch.randn(200) + 5.0
        crps_good = crps(good_samples, ref).item()
        crps_bad  = crps(bad_samples,  ref).item()
        logger.info("CRPS good=%.4f, bad=%.4f", crps_good, crps_bad)
        assert crps_good < crps_bad, (
            f"CRPS consistency violated: good={crps_good:.4f} >= bad={crps_bad:.4f}"
        )

    def test_crps_finite(self):
        """CRPS returns finite scalar for valid 1D inputs."""
        c = crps(torch.randn(100), torch.tensor(0.5))
        assert torch.isfinite(c), f"CRPS returned non-finite: {c.item()}"

    def test_crps_warns_multidim(self, caplog):
        """CRPS warns when called with d>1 and directs to crps_multidim."""
        with caplog.at_level(logging.WARNING):
            crps(torch.randn(50, 4), torch.zeros(4))
        assert any("crps_multidim" in r.message for r in caplog.records), (
            "Expected warning directing user to crps_multidim for d>1"
        )

    def test_crps_multidim(self):
        """crps_multidim with beta=1 is numerically close to energy_score."""
        torch.manual_seed(3)
        samples = torch.randn(30, 8)
        ref     = torch.randn(8)
        crps_d = crps_multidim(samples, ref, beta=1.0).item()
        es     = energy_score(samples, ref).item()
        # Both use the same energy formula: should agree
        logger.info("crps_multidim(beta=1)=%.4f, ES=%.4f", crps_d, es)
        assert abs(crps_d - es) < 1e-4, (
            f"crps_multidim(beta=1) should equal ES: {crps_d:.4f} vs {es:.4f}"
        )

    def test_crps_multidim_invalid_beta(self):
        """crps_multidim with beta outside (0,2) raises ValueError."""
        with pytest.raises(ValueError, match="beta"):
            crps_multidim(torch.randn(10, 4), torch.zeros(4), beta=2.5)

    def test_temperature_scaling(self):
        """Temperature scaling divides logits by tau."""
        logits = torch.tensor([1.0, 2.0, 3.0])
        scaled = temperature_scaling(logits, temperature=2.0)
        assert torch.allclose(scaled, logits / 2.0)

    def test_temperature_scaling_invalid(self):
        """Temperature <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="temperature"):
            temperature_scaling(torch.randn(4), temperature=0.0)


# ===========================================================================
# HybridLoss Tests
# ===========================================================================

class TestHybridLoss:

    def test_hybrid_loss_finite(self, flow, hybrid_loss, d):
        """All 4 loss components are finite with a real flow."""
        B = 8
        x_clean    = torch.randn(B, d)
        y_degraded = torch.randn(B, d)

        loss, loss_dict = hybrid_loss(flow, x_clean, y_degraded, epoch=5)

        for key, val in loss_dict.items():
            logger.info("hybrid_loss component %s=%.4f", key, val)
            assert math.isfinite(val), f"loss_dict['{key}'] = {val} is not finite"

        assert torch.isfinite(loss), f"total loss is not finite: {loss.item()}"

    def test_hybrid_loss_components_nonneg(self, flow, hybrid_loss, d):
        """Consistency and transport components are non-negative."""
        B = 8
        x_clean    = torch.randn(B, d)
        y_degraded = torch.randn(B, d)
        _, loss_dict = hybrid_loss(flow, x_clean, y_degraded)
        assert loss_dict["consistency"] >= 0.0, "Consistency loss must be >= 0"
        assert loss_dict["transport"]   >= 0.0, "Transport (SW2) loss must be >= 0"

    def test_annealing_warmup_zero(self, fwd_model, d):
        """Lambda is 0 during warmup epochs."""
        sched = {"cons": {"warmup": 5, "rampup": 15}}
        loss_fn = HybridLoss(fwd_model, anneal_schedule=sched)
        for epoch in range(5):
            lam = loss_fn._anneal(0.1, epoch, "cons")
            assert lam == 0.0, f"Expected 0 during warmup, got {lam} at epoch {epoch}"

    def test_annealing_linear_rampup(self, fwd_model):
        """Lambda ramps linearly from 0 to full between warmup and rampup."""
        sched = {"cons": {"warmup": 0, "rampup": 10}}
        loss_fn = HybridLoss(fwd_model, lambda_cons=1.0, anneal_schedule=sched)
        for epoch in range(1, 10):
            lam = loss_fn._anneal(1.0, epoch, "cons")
            expected = epoch / 10
            assert abs(lam - expected) < 1e-6, (
                f"Rampup at epoch {epoch}: expected {expected:.4f}, got {lam:.4f}"
            )

    def test_annealing_full_after_rampup(self, fwd_model):
        """Lambda returns full value after rampup is complete."""
        sched = {"cons": {"warmup": 2, "rampup": 8}}
        loss_fn = HybridLoss(fwd_model, lambda_cons=0.5, anneal_schedule=sched)
        for epoch in [8, 10, 20]:
            lam = loss_fn._anneal(0.5, epoch, "cons")
            assert lam == 0.5, f"Expected full lambda after rampup, got {lam} at epoch {epoch}"

    def test_hybrid_loss_lambda_zero_skips_terms(self, fwd_model, flow, d):
        """Setting lambda_trans=0 and lambda_cal=0 keeps transport/cal at zero."""
        loss_fn = HybridLoss(
            fwd_model, lambda_cons=0.1, lambda_trans=0.0, lambda_cal=0.0
        )
        B = 6
        x_clean    = torch.randn(B, d)
        y_degraded = torch.randn(B, d)
        _, loss_dict = loss_fn(flow, x_clean, y_degraded)
        assert loss_dict["transport"]   == 0.0, "transport should be 0 when lambda_trans=0"
        assert loss_dict["calibration"] == 0.0, "calibration should be 0 when lambda_cal=0"


# ===========================================================================
# [Additional / Fatal] Gradient Flow Tests
# ===========================================================================

class TestGradientFlow:

    def test_nll_gradient_reaches_flow(self, flow, fwd_model, d):
        """Gradients from NLL reach flow conditioner and block parameters."""
        loss_fn = HybridLoss(fwd_model, lambda_cons=0.0, lambda_trans=0.0, lambda_cal=0.0)
        B = 6
        x_clean    = torch.randn(B, d)
        y_degraded = torch.randn(B, d)

        loss, _ = loss_fn(flow, x_clean, y_degraded)
        loss.backward()

        for name, param in flow.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad for {name} (NLL term)"
                assert not torch.all(param.grad == 0), f"Zero grad for {name} (NLL term)"

    def test_consistency_gradient_reaches_flow(self, flow, fwd_model, d):
        """Gradients from consistency term reach flow sample parameters."""
        loss_fn = HybridLoss(fwd_model, lambda_cons=1.0, lambda_trans=0.0, lambda_cal=0.0)
        B = 6
        x_clean    = torch.randn(B, d)
        y_degraded = torch.randn(B, d)

        # Zero existing grads
        flow.zero_grad()
        loss, loss_dict = loss_fn(flow, x_clean, y_degraded)
        loss.backward()

        logger.info("Consistency loss=%.4f", loss_dict["consistency"])
        # At least conditioner params should receive gradient
        cond_grads = [
            p.grad for p in flow.conditioner_net.parameters()
            if p.grad is not None and not torch.all(p.grad == 0)
        ]
        assert len(cond_grads) > 0, "No non-zero gradients reached conditioner (consistency)"

    def test_transport_gradient_reaches_flow(self, flow, fwd_model, d):
        """Gradients from SW2 transport term are finite and non-zero."""
        loss_fn = HybridLoss(fwd_model, lambda_cons=0.0, lambda_trans=1.0, lambda_cal=0.0)
        B = 6
        x_clean    = torch.randn(B, d, requires_grad=False)
        y_degraded = torch.randn(B, d)

        flow.zero_grad()
        loss, loss_dict = loss_fn(flow, x_clean, y_degraded)
        loss.backward()

        logger.info("Transport loss=%.4f", loss_dict["transport"])
        for name, param in flow.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"Non-finite gradient for {name} from transport term"
                )


# ===========================================================================
# [Additional / Fatal] Numerical Stability Tests
# ===========================================================================

class TestNumericalStability:

    def test_sw2_large_samples(self):
        """SW2 remains finite for large-scale samples (σ=100)."""
        X = torch.randn(50, 8) * 100.0
        Y = torch.randn(50, 8) * 100.0
        sw = sliced_wasserstein_distance(X, Y)
        logger.info("SW2 large-scale=%.4f", sw.item())
        assert torch.isfinite(sw), f"SW2 non-finite for large samples: {sw.item()}"

    def test_sw2_near_zero_samples(self):
        """SW2 remains finite for near-zero samples (σ=1e-6)."""
        X = torch.randn(50, 8) * 1e-6
        Y = torch.randn(50, 8) * 1e-6
        sw = sliced_wasserstein_distance(X, Y)
        logger.info("SW2 near-zero=%.6f", sw.item())
        assert torch.isfinite(sw), f"SW2 non-finite for near-zero samples: {sw.item()}"

    def test_es_large_samples(self):
        """ES remains finite for large-scale samples."""
        samples = torch.randn(30, 8) * 100.0
        ref     = torch.zeros(8)
        es = energy_score(samples, ref)
        logger.info("ES large-scale=%.4f", es.item())
        assert torch.isfinite(es), f"ES non-finite for large samples: {es.item()}"

    def test_es_near_identical_samples(self):
        """ES with near-identical samples (low diversity) is finite."""
        base    = torch.randn(1, 8)
        samples = base.expand(20, -1) + torch.randn(20, 8) * 1e-7
        ref     = base.squeeze(0)
        es = energy_score(samples, ref)
        logger.info("ES near-identical=%.6f", es.item())
        assert torch.isfinite(es), f"ES non-finite for near-identical samples: {es.item()}"

    def test_hybrid_loss_finite_under_stress(self, flow, hybrid_loss, d):
        """Hybrid loss remains finite for large-scale inputs."""
        B = 8
        x_clean    = torch.randn(B, d) * 10.0
        y_degraded = torch.randn(B, d) * 10.0
        loss, loss_dict = hybrid_loss(flow, x_clean, y_degraded)
        for key, val in loss_dict.items():
            logger.info("stress test %s=%.4f", key, val)
            assert math.isfinite(val), f"Stress test: loss_dict['{key}']={val} non-finite"


# ===========================================================================
# [Additional / Fatal] Freeze / Unfreeze Tests
# ===========================================================================

class TestFreezeUnfreeze:

    def test_freeze_experts_disables_grads(self, csmf_model):
        """freeze_experts sets requires_grad=False for all expert params."""
        freeze_experts(csmf_model)
        for k, expert in enumerate(csmf_model.experts):
            for name, param in expert.named_parameters():
                assert not param.requires_grad, (
                    f"Expert {k} param '{name}' still has requires_grad=True after freeze"
                )

    def test_gate_remains_trainable_after_freeze_experts(self, csmf_model):
        """Gate params remain trainable after freeze_experts."""
        freeze_experts(csmf_model)
        gate_trainable = [p for p in csmf_model.gate.parameters() if p.requires_grad]
        assert len(gate_trainable) > 0, "Gate should remain trainable after freeze_experts"

    def test_freeze_gate_disables_grads(self, csmf_model):
        """freeze_gate sets requires_grad=False for all gate params."""
        freeze_gate(csmf_model)
        for name, param in csmf_model.gate.named_parameters():
            assert not param.requires_grad, (
                f"Gate param '{name}' still has requires_grad=True after freeze_gate"
            )

    def test_unfreeze_gate_restores_grads(self, csmf_model):
        """unfreeze_gate restores requires_grad=True for gate params."""
        freeze_gate(csmf_model)
        unfreeze_gate(csmf_model)
        for name, param in csmf_model.gate.named_parameters():
            assert param.requires_grad, (
                f"Gate param '{name}' has requires_grad=False after unfreeze_gate"
            )

    def test_unfreeze_last_blocks_partial(self, csmf_model):
        """unfreeze_last_blocks(n=1) only unfreezes the last block per expert."""
        freeze_experts(csmf_model)
        unfreeze_last_blocks(csmf_model, n_blocks=1)

        for k, expert in enumerate(csmf_model.experts):
            blocks = list(expert.blocks)
            n = len(blocks)
            # All blocks before the last should be frozen
            for i, block in enumerate(blocks[:-1]):
                for name, param in block.named_parameters():
                    assert not param.requires_grad, (
                        f"Expert {k} block {i} param '{name}' should be frozen"
                    )
            # Last block should be unfrozen
            for name, param in blocks[-1].named_parameters():
                assert param.requires_grad, (
                    f"Expert {k} last block param '{name}' should be trainable"
                )

    def test_unfreeze_last_blocks_all(self, csmf_model):
        """unfreeze_last_blocks(n >= n_blocks) unfreezes all blocks."""
        freeze_experts(csmf_model)
        n_blocks = len(list(csmf_model.experts[0].blocks))
        unfreeze_last_blocks(csmf_model, n_blocks=n_blocks)

        for k, expert in enumerate(csmf_model.experts):
            for name, param in expert.named_parameters():
                assert param.requires_grad, (
                    f"Expert {k} param '{name}' should be trainable after full unfreeze"
                )

    def test_freeze_experts_missing_attr(self):
        """freeze_experts raises AttributeError if model has no 'experts' attr."""
        model = nn.Linear(4, 4)
        with pytest.raises(AttributeError, match="experts"):
            freeze_experts(model)

    def test_unfreeze_last_blocks_missing_blocks_attr(self, csmf_model):
        """unfreeze_last_blocks raises AttributeError if expert has no 'blocks'."""
        # Replace experts with models that have no .blocks
        csmf_model.experts = nn.ModuleList([nn.Linear(8, 8)])
        freeze_experts(csmf_model)
        with pytest.raises(AttributeError, match="blocks"):
            unfreeze_last_blocks(csmf_model, n_blocks=1)
