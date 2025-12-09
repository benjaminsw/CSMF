# Implementation Plan: test_conditioning.py

**Version:** WP0.4-Test-v1.0  
**Abbr:** TEST-COND  
**Level:** 5 (depends on all WP0 components)  
**Date:** 2025-12-09

---

## Core Functionality

### Test Suite Overview
Comprehensive unit tests for WP0 conditioning components:
1. FiLM layer shape consistency and modulation
2. MNISTConditioner output dimensions
3. ConditionalAffineCoupling invertibility
4. Log-determinant correctness (numerical Jacobian validation)

---

## Additional Functionality (INCLUDED)

### Critical Features (Fatal if missing):
✓ **Numerical Jacobian test** - Compare analytical vs finite-difference Jacobian
✓ **Conditioning determinism** - Different h → different outputs
✓ **Detailed error logging** - Log assertion failures with values
✓ **NO pass statements** - All tests fully implemented

### Recommended Features:
✓ **Multi-scale RealNVP tests** - Test squeeze/unsqueeze operations
✓ **MAF MADE masking tests** - Verify autoregressive property
✓ **Batch normalization tests** - Test BN log-det tracking
✓ **Integration tests** - Test full forward/inverse pipeline
✓ **Stress tests** - Large batch sizes, edge cases

---

## Test Categories

### Category 1: FiLM Layer Tests
**Tests:** 4 tests
- `test_film_shape_consistency()` - Input/output shape matching
- `test_film_modulation()` - γ and β application correctness
- `test_film_spatial_broadcasting()` - 4D tensor handling
- `test_film_conditioning_effect()` - Different h → different output

### Category 2: Conditioning Network Tests
**Tests:** 5 tests
- `test_conditioner_output_shape()` - Output dimension verification
- `test_conditioner_spec_compliant_mode()` - Pooling+flatten path
- `test_conditioner_spatial_mode()` - Stride-2 conv path
- `test_conditioner_batch_processing()` - Batch consistency
- `test_conditioner_determinism()` - Same input → same output

### Category 3: Coupling Layer Tests
**Tests:** 6 tests
- `test_coupling_invertibility()` - x ≈ f⁻¹(f(x))
- `test_coupling_log_det_numerical()` - **CRITICAL: Numerical Jacobian**
- `test_coupling_conditioning()` - Different h → different transform
- `test_coupling_batch_norm()` - BN log-det contribution
- `test_coupling_masking()` - Checkerboard/channel masks
- `test_coupling_stability()` - Scale clamping effectiveness

### Category 4: Flow Architecture Tests
**Tests:** 8 tests
- `test_realnvp_invertibility()` - Multi-scale forward/inverse
- `test_realnvp_squeeze_ops()` - Squeeze/unsqueeze correctness
- `test_realnvp_variable_factoring()` - Dimension splits
- `test_realnvp_log_prob()` - Log-probability computation
- `test_maf_invertibility()` - Autoregressive forward/inverse
- `test_maf_made_masking()` - Autoregressive property verification
- `test_maf_parallel_forward()` - Parallel computation correctness
- `test_maf_sequential_inverse()` - Sequential sampling

### Category 5: Integration Tests
**Tests:** 4 tests
- `test_full_pipeline()` - Dataset → Conditioner → Flow → Loss
- `test_conditioning_determinism()` - **CRITICAL: Different y → different output**
- `test_gradient_flow()` - Backprop through full model
- `test_numerical_stability()` - NaN/Inf detection

---

## Implementation Structure

```python
"""
Unit Tests for WP0 Conditional Flows

Version: WP0.4-Test-v1.0
Abbr: TEST-COND
Last Modified: 2025-12-09
Dependencies: pytest>=6.2.0, torch>=2.0, all WP0 components

Test Categories:
1. FiLM Layer Tests (4 tests)
2. Conditioning Network Tests (5 tests)
3. Coupling Layer Tests (6 tests)
4. Flow Architecture Tests (8 tests)
5. Integration Tests (4 tests)

Total: 27 comprehensive tests
"""

import torch
import torch.nn as nn
import pytest
import logging
from typing import Tuple

# WP0 imports
from csmf.conditioning.film import FiLM
from csmf.conditioning.conditioning_networks import MNISTConditioner
from csmf.flows.coupling_layers import ConditionalAffineCoupling, checkerboard_mask
from csmf.flows.conditional_realnvp import ConditionalRealNVP
from csmf.flows.conditional_maf import ConditionalMAF
from data.mnist_inverse import MNISTInverseDataset

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


#############################################
# CATEGORY 1: FiLM LAYER TESTS (4 tests)
#############################################

def test_film_shape_consistency():
    """Test FiLM preserves input shape"""
    # Test vector input
    # Test spatial input
    # Assert shape matching
    pass

def test_film_modulation():
    """Test γ and β application"""
    # Create FiLM with known γ, β
    # Verify: output = γ * input + β
    # Test with multiple batch sizes
    pass

def test_film_spatial_broadcasting():
    """Test 4D tensor handling [B, C, H, W]"""
    # Test with 4D features
    # Verify correct broadcasting
    # Check no dimension errors
    pass

def test_film_conditioning_effect():
    """Test different h produces different output"""
    # Same f, different h
    # Verify outputs differ
    # Measure difference magnitude
    pass


#############################################
# CATEGORY 2: CONDITIONING NETWORK TESTS (5)
#############################################

def test_conditioner_output_shape():
    """Test output dimension matches h_dim"""
    # Test both modes (spec_compliant, default)
    # Verify output shape
    # Test multiple batch sizes
    pass

def test_conditioner_spec_compliant_mode():
    """Test pooling+flatten architecture"""
    # spec_compliant=True
    # Verify flattened output [B, h_dim]
    # Check architecture matches spec
    pass

def test_conditioner_spatial_mode():
    """Test stride-2 conv architecture"""
    # spec_compliant=False (default)
    # Verify spatial output [B, h_dim, H', W']
    # Check performance mode
    pass

def test_conditioner_batch_processing():
    """Test batch consistency"""
    # Process batch vs individual
    # Verify results match
    # Test batch norm behavior
    pass

def test_conditioner_determinism():
    """Test deterministic output"""
    # Same input twice
    # Verify identical outputs
    # Test in eval mode
    pass


#############################################
# CATEGORY 3: COUPLING LAYER TESTS (6)
#############################################

def test_coupling_invertibility():
    """Test x ≈ f⁻¹(f(x)) within tolerance"""
    # Forward then inverse
    # Compute max absolute error
    # Assert error < 1e-5
    # Log errors if failed
    pass

@pytest.mark.critical
def test_coupling_log_det_numerical():
    """
    CRITICAL: Compare analytical vs numerical Jacobian
    
    Compute:
    1. Analytical: sum of scale parameters
    2. Numerical: finite difference Jacobian determinant
    
    Assert close match (atol=1e-3)
    """
    # Small input for tractable Jacobian
    # Compute analytical log-det
    # Compute numerical Jacobian via finite differences
    # Compare determinants
    # Log detailed error breakdown
    pass

@pytest.mark.critical
def test_coupling_conditioning():
    """
    CRITICAL: Different h → different transformation
    
    Verify conditioning actually affects the transformation
    """
    # Same x, two different h vectors
    # Forward with h1, h2
    # Assert outputs differ significantly
    # Measure conditioning strength
    pass

def test_coupling_batch_norm():
    """Test BN contribution to log-det"""
    # Layer with BN enabled
    # Forward pass
    # Verify log-det includes BN contribution
    # Check BN inverse
    pass

def test_coupling_masking():
    """Test checkerboard and channel masks"""
    # Create masks
    # Test with explicit masks
    # Verify correct partitioning
    # Compare to split-based
    pass

def test_coupling_stability():
    """Test scale clamping prevents overflow"""
    # Create layer with s_max
    # Forward with extreme inputs
    # Verify no NaN/Inf
    # Check clamping effectiveness
    pass


#############################################
# CATEGORY 4: FLOW ARCHITECTURE TESTS (8)
#############################################

def test_realnvp_invertibility():
    """Test multi-scale RealNVP invertibility"""
    # Create model
    # Forward: x → (z, z_factored)
    # Inverse: (z, z_factored) → x_recon
    # Assert x ≈ x_recon
    # Test all scales
    pass

def test_realnvp_squeeze_ops():
    """Test squeeze/unsqueeze correctness"""
    # Test squeeze: [B,C,H,W] → [B,4C,H/2,W/2]
    # Test unsqueeze: [B,4C,H,W] → [B,C,2H,2W]
    # Verify volume preservation
    # Check invertibility
    pass

def test_realnvp_variable_factoring():
    """Test dimension splitting at scales"""
    # Forward pass
    # Check factored variables shapes
    # Verify 50% splits
    # Test reconstruction with factored
    pass

def test_realnvp_log_prob():
    """Test log-probability computation"""
    # Forward pass
    # Check log_prob shape [B]
    # Verify reasonable values
    # Test with known distribution
    pass

def test_maf_invertibility():
    """Test MAF autoregressive invertibility"""
    # Forward (parallel)
    # Inverse (sequential)
    # Assert x ≈ x_recon
    # Test multiple flows
    pass

@pytest.mark.critical
def test_maf_made_masking():
    """
    CRITICAL: Verify autoregressive property
    
    Check that output_i only depends on input_{<i}
    """
    # Create MADE network
    # Inspect mask matrices
    # Verify triangular dependency
    # Test with perturbed inputs
    pass

def test_maf_parallel_forward():
    """Test parallel forward computation"""
    # Time forward pass
    # Verify O(D) complexity
    # Compare to sequential
    # Check correctness
    pass

def test_maf_sequential_inverse():
    """Test sequential sampling"""
    # Sample from base
    # Inverse through flows
    # Verify sequential dependencies
    # Check sample quality
    pass


#############################################
# CATEGORY 5: INTEGRATION TESTS (4)
#############################################

def test_full_pipeline():
    """Test complete forward pass"""
    # Load mini-batch from dataset
    # Extract conditioning: h = conditioner(y)
    # Flow forward: z, log_det = flow(x, h)
    # Compute loss
    # Verify no errors
    pass

@pytest.mark.critical
def test_conditioning_determinism():
    """
    CRITICAL: Different observations → different reconstructions
    
    Verify conditioning mechanism works end-to-end
    """
    # Two different observations y1, y2
    # Extract features h1 = c(y1), h2 = c(y2)
    # Forward same x with h1, h2
    # Assert z1 ≠ z2 (different latents)
    # Sample from both: x1_recon, x2_recon
    # Verify reconstructions differ
    pass

def test_gradient_flow():
    """Test backprop through full model"""
    # Create model
    # Forward pass
    # Compute loss
    # Backward
    # Check all gradients non-None
    # Verify no gradient explosion
    pass

def test_numerical_stability():
    """Test NaN/Inf detection and handling"""
    # Test with edge cases
    # Verify no NaN/Inf in outputs
    # Test clamping mechanisms
    # Check error handling
    pass


#############################################
# HELPER FUNCTIONS
#############################################

def compute_numerical_jacobian(
    func,
    x: torch.Tensor,
    h: torch.Tensor,
    eps: float = 1e-4
) -> torch.Tensor:
    """
    Compute numerical Jacobian via finite differences
    
    Args:
        func: Function to differentiate
        x: Input tensor [B, D]
        h: Conditioning tensor [B, h_dim]
        eps: Finite difference step size
    
    Returns:
        Numerical Jacobian [B, D, D]
    """
    # Implementation here
    pass


def compute_analytical_log_det(
    layer,
    x: torch.Tensor,
    h: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute analytical log-det from layer
    
    Returns:
        z: Transformed tensor
        log_det: Analytical log-determinant
    """
    # Implementation here
    pass


def measure_conditioning_strength(
    layer,
    x: torch.Tensor,
    h1: torch.Tensor,
    h2: torch.Tensor
) -> float:
    """
    Measure how much conditioning affects output
    
    Returns:
        Relative difference between outputs
    """
    # Implementation here
    pass

```

---

## Critical Test Details

### Test 1: Numerical Jacobian Validation

**Purpose:** Verify analytical log-det matches numerical computation

**Method:**
```python
def test_coupling_log_det_numerical():
    layer = ConditionalAffineCoupling(dim=10, split_dim=5, h_dim=8)
    x = torch.randn(2, 10, requires_grad=True)
    h = torch.randn(2, 8)
    
    # Analytical log-det
    z, log_det_analytical = layer.forward(x, h, reverse=False)
    
    # Numerical Jacobian
    jacobian = compute_numerical_jacobian(
        lambda x_: layer.forward(x_, h, reverse=False)[0],
        x, h, eps=1e-4
    )
    log_det_numerical = torch.logdet(jacobian)
    
    # Compare
    error = torch.abs(log_det_analytical - log_det_numerical)
    logger.info(f"Log-det error: {error.max().item():.6f}")
    
    assert torch.allclose(
        log_det_analytical,
        log_det_numerical,
        atol=1e-3
    ), f"Log-det mismatch: analytical={log_det_analytical}, numerical={log_det_numerical}"
```

**Acceptance:** Error < 1e-3

---

### Test 2: Conditioning Determinism

**Purpose:** Verify different observations produce different outputs

**Method:**
```python
@pytest.mark.critical
def test_conditioning_determinism():
    # Setup
    model = ConditionalRealNVP(h_dim=64)
    x = torch.randn(1, 1, 28, 28)
    y1 = torch.randn(1, 1, 28, 28)  # Observation 1
    y2 = torch.randn(1, 1, 28, 28)  # Observation 2
    
    # Forward with different observations
    z1, _, log_det1, _ = model.forward(x, y1)
    z2, _, log_det2, _ = model.forward(x, y2)
    
    # Verify outputs differ
    z_diff = torch.norm(z1 - z2).item()
    logger.info(f"Latent difference: {z_diff:.6f}")
    
    assert z_diff > 0.1, f"Conditioning too weak: z_diff={z_diff}"
    
    # Sample reconstructions
    x1_recon = model.inverse(z1, [], y1)
    x2_recon = model.inverse(z2, [], y2)
    
    recon_diff = torch.norm(x1_recon - x2_recon).item()
    logger.info(f"Reconstruction difference: {recon_diff:.6f}")
    
    assert recon_diff > 0.05, "Reconstructions too similar"
```

**Acceptance:** Latent diff > 0.1, Recon diff > 0.05

---

## Test Execution Strategy

### Phase 1: Component Tests (Run First)
```bash
pytest tests/test_conditioning.py::test_film_shape_consistency -v
pytest tests/test_conditioning.py::test_conditioner_output_shape -v
pytest tests/test_conditioning.py::test_coupling_invertibility -v
```

### Phase 2: Critical Tests (Must Pass)
```bash
pytest tests/test_conditioning.py -m critical -v
```

### Phase 3: Integration Tests (Final Validation)
```bash
pytest tests/test_conditioning.py::test_full_pipeline -v
pytest tests/test_conditioning.py::test_conditioning_determinism -v
```

### Full Suite
```bash
pytest tests/test_conditioning.py -v --tb=short
```

---

## Success Criteria

| Test Category | Tests | Critical | Pass Rate |
|---------------|-------|----------|-----------|
| FiLM | 4 | 1 | 100% |
| Conditioning | 5 | 1 | 100% |
| Coupling | 6 | 3 | 100% |
| Flows | 8 | 2 | 100% |
| Integration | 4 | 2 | 100% |
| **Total** | **27** | **9** | **100%** |

**NO pass statements allowed in final code**

---

## Error Logging Template

```python
try:
    assert condition, "Message"
except AssertionError as e:
    logger.error(f"Test failed: {test_name}")
    logger.error(f"Expected: {expected}")
    logger.error(f"Got: {actual}")
    logger.error(f"Error: {error}")
    logger.error(f"Tolerance: {tol}")
    raise
```

---

## Dependencies

```python
# Test dependencies
pytest>=6.2.0
pytest-cov>=2.12.0

# WP0 components (must be implemented first)
- configs/mnist_config.py (WP0.1-Config-v1.1)
- csmf/conditioning/film.py (WP0.1-FiLM-v1.0)
- csmf/conditioning/conditioning_networks.py (WP0.1-CondNet-v1.1)
- csmf/flows/coupling_layers.py (WP0.2-Coupling-v1.1)
- csmf/flows/conditional_realnvp.py (WP0.3-CondRNVP-v2.0)
- csmf/flows/conditional_maf.py (WP0.3-CondMAF-v2.0)
- data/mnist_inverse.py (WP0.1-MNISTInv-v1.0)
```

---

## Implementation Checklist

- [ ] Implement all 27 tests (NO pass statements)
- [ ] Implement numerical Jacobian computation
- [ ] Implement conditioning strength measurement
- [ ] Add detailed error logging
- [ ] Add pytest markers (@pytest.mark.critical)
- [ ] Add test fixtures for common setups
- [ ] Add parametrized tests for multiple configs
- [ ] Document expected behaviors
- [ ] Create test data generators
- [ ] Add performance benchmarks

---

## Notes

- **NO pass statements** - All tests must be fully implemented
- **Critical tests** marked with @pytest.mark.critical
- **Detailed logging** - Log failures with context
- **Numerical precision** - Use appropriate tolerances (atol=1e-5 for invertibility, 1e-3 for Jacobian)
- **Conditioning verification** - Ensure different h → different output
- **Multi-scale testing** - Test all RealNVP scales
- **MADE masking** - Verify autoregressive property

**This is a COMPREHENSIVE test suite - all WP0 components must pass before proceeding to WP1**
