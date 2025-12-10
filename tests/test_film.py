"""
Test Conditioning Networks and FiLM Layers
Tests CondNet and FiLM functionality with comprehensive validation

Version: WP0.1-TestCondFiLM-v1.0
Last Modified: 2025-12-10
Changelog:
  v1.0 (2025-12-10): Initial implementation with 13 tests (8 core + 5 optional)
Dependencies: torch>=2.0, numpy>=1.20
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json
import traceback

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np

try:
    from csmf.conditioning.conditioning_networks import MNISTConditioner
    from csmf.conditioning.film import FiLM
except ImportError as e:
    print(f"ERROR: Failed to import modules: {e}")
    print(f"Current path: {os.getcwd()}")
    print(f"Sys path: {sys.path}")
    sys.exit(1)


class TestResults:
    """Container for test results"""
    def __init__(self):
        self.tests = {}
        self.tensors = {}
        self.metadata = {
            'version': 'WP0.1-TestCondFiLM-v1.0',
            'timestamp': datetime.now().isoformat(),
            'torch_version': torch.__version__,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
    
    def add_test(self, name, passed, details=None, error=None):
        """Add test result"""
        self.tests[name] = {
            'passed': passed,
            'details': details or {},
            'error': str(error) if error else None
        }
    
    def add_tensor(self, name, tensor):
        """Add tensor for NPZ storage"""
        if isinstance(tensor, torch.Tensor):
            self.tensors[name] = tensor.detach().cpu().numpy()
        else:
            self.tensors[name] = np.array(tensor)
    
    def save(self, output_dir):
        """Save results to NPZ + JSON"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"tcf_v1.0_{timestamp}"
        
        # Save NPZ
        npz_path = os.path.join(output_dir, f"{base_name}.npz")
        np.savez_compressed(npz_path, **self.tensors)
        
        # Save JSON
        json_path = os.path.join(output_dir, f"{base_name}.json")
        with open(json_path, 'w') as f:
            json.dump({
                'metadata': self.metadata,
                'tests': self.tests
            }, f, indent=2)
        
        print(f"\n✓ Results saved:")
        print(f"  NPZ: {npz_path}")
        print(f"  JSON: {json_path}")


def test_condnet_shapes(results, device):
    """Test 1: CondNet shape validation"""
    test_name = "1_condnet_shapes"
    try:
        print(f"\n[{test_name}] Testing CondNet shape validation...")
        
        batch_size = 8
        input_img = torch.randn(batch_size, 1, 28, 28).to(device)
        h_dim = 64
        
        # Test spec-compliant mode
        condnet_spec = MNISTConditioner(h_dim=h_dim, spec_compliant=True).to(device)
        h_spec = condnet_spec(input_img)
        
        # Test performance mode
        condnet_perf = MNISTConditioner(h_dim=h_dim, spec_compliant=False).to(device)
        h_perf = condnet_perf(input_img)
        
        # Validate shapes
        spec_valid = (h_spec.shape == torch.Size([batch_size, h_dim]))
        perf_valid = (h_perf.shape[0] == batch_size and h_perf.shape[1] == h_dim)
        
        passed = spec_valid and perf_valid
        
        results.add_test(test_name, passed, details={
            'spec_shape': list(h_spec.shape),
            'perf_shape': list(h_perf.shape),
            'expected_spec': [batch_size, h_dim],
            'spec_valid': spec_valid,
            'perf_valid': perf_valid
        })
        
        results.add_tensor(f"{test_name}_input", input_img)
        results.add_tensor(f"{test_name}_h_spec", h_spec)
        results.add_tensor(f"{test_name}_h_perf", h_perf)
        
        print(f"  Spec shape: {h_spec.shape} - {'✓' if spec_valid else '✗'}")
        print(f"  Perf shape: {h_perf.shape} - {'✓' if perf_valid else '✗'}")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results.add_test(test_name, False, error=e)


def test_film_modulation(results, device):
    """Test 2: FiLM modulation math verification"""
    test_name = "2_film_modulation"
    try:
        print(f"\n[{test_name}] Testing FiLM modulation math...")
        
        batch_size = 8
        h_dim = 64
        f_dim = 128
        
        film = FiLM(f_dim=f_dim, h_dim=h_dim).to(device)
        
        # Vector features
        h = torch.randn(batch_size, h_dim).to(device)
        f = torch.randn(batch_size, f_dim).to(device)
        
        # Apply FiLM
        f_mod = film(f, h)
        
        # Manual verification
        with torch.no_grad():
            gamma = film.gamma_mlp(h)
            beta = film.beta_mlp(h)
            f_expected = gamma * f + beta
        
        # Check if results match
        diff = torch.abs(f_mod - f_expected).max().item()
        passed = (diff < 1e-5)
        
        results.add_test(test_name, passed, details={
            'input_shapes': {'h': list(h.shape), 'f': list(f.shape)},
            'output_shape': list(f_mod.shape),
            'max_diff': float(diff),
            'formula_verified': passed
        })
        
        results.add_tensor(f"{test_name}_h", h)
        results.add_tensor(f"{test_name}_f", f)
        results.add_tensor(f"{test_name}_f_mod", f_mod)
        results.add_tensor(f"{test_name}_gamma", gamma)
        results.add_tensor(f"{test_name}_beta", beta)
        
        print(f"  Max diff from formula: {diff:.2e} - {'✓' if passed else '✗'}")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results.add_test(test_name, False, error=e)


def test_film_spatial(results, device):
    """Test 3: Spatial conditioning with GAP"""
    test_name = "3_film_spatial"
    try:
        print(f"\n[{test_name}] Testing FiLM spatial→vector GAP...")
        
        batch_size = 8
        h_dim = 64
        f_dim = 128
        
        film = FiLM(f_dim=f_dim, h_dim=h_dim).to(device)
        
        # Spatial conditioning features
        h_spatial = torch.randn(batch_size, h_dim, 4, 4).to(device)
        f_spatial = torch.randn(batch_size, f_dim, 7, 7).to(device)
        
        # Apply FiLM
        f_mod = film(f_spatial, h_spatial)
        
        # Verify GAP applied
        h_pooled = torch.mean(h_spatial, dim=[2, 3])
        with torch.no_grad():
            gamma = film.gamma_mlp(h_pooled)
            beta = film.beta_mlp(h_pooled)
        
        # Check shape preservation
        shape_valid = (f_mod.shape == f_spatial.shape)
        
        # Check modulation applied correctly
        gamma_expanded = gamma.unsqueeze(-1).unsqueeze(-1)
        beta_expanded = beta.unsqueeze(-1).unsqueeze(-1)
        f_expected = gamma_expanded * f_spatial + beta_expanded
        diff = torch.abs(f_mod - f_expected).max().item()
        
        passed = shape_valid and (diff < 1e-5)
        
        results.add_test(test_name, passed, details={
            'h_spatial_shape': list(h_spatial.shape),
            'f_spatial_shape': list(f_spatial.shape),
            'output_shape': list(f_mod.shape),
            'shape_preserved': shape_valid,
            'max_diff': float(diff)
        })
        
        results.add_tensor(f"{test_name}_h_spatial", h_spatial)
        results.add_tensor(f"{test_name}_f_spatial", f_spatial)
        results.add_tensor(f"{test_name}_f_mod", f_mod)
        
        print(f"  Shape preserved: {shape_valid} - {'✓' if shape_valid else '✗'}")
        print(f"  Max diff: {diff:.2e} - {'✓' if diff < 1e-5 else '✗'}")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results.add_test(test_name, False, error=e)


def test_integration(results, device):
    """Test 4: CondNet→FiLM integration"""
    test_name = "4_integration"
    try:
        print(f"\n[{test_name}] Testing CondNet→FiLM pipeline...")
        
        batch_size = 8
        h_dim = 64
        f_dim = 128
        
        # Create pipeline
        condnet = MNISTConditioner(h_dim=h_dim, spec_compliant=False).to(device)
        film = FiLM(f_dim=f_dim, h_dim=h_dim).to(device)
        
        # Input: degraded image
        y = torch.randn(batch_size, 1, 28, 28).to(device)
        
        # Features to modulate (e.g., from flow layer)
        f = torch.randn(batch_size, f_dim, 7, 7).to(device)
        
        # Pipeline: extract features then modulate
        h = condnet(y)  # [B, h_dim, H', W'] or [B, h_dim]
        f_mod = film(f, h)
        
        # Validate
        shape_valid = (f_mod.shape == f.shape)
        no_nans = not torch.isnan(f_mod).any()
        no_infs = not torch.isinf(f_mod).any()
        
        passed = shape_valid and no_nans and no_infs
        
        results.add_test(test_name, passed, details={
            'input_shape': list(y.shape),
            'h_shape': list(h.shape),
            'f_shape': list(f.shape),
            'output_shape': list(f_mod.shape),
            'shape_valid': shape_valid,
            'no_nans': no_nans,
            'no_infs': no_infs
        })
        
        results.add_tensor(f"{test_name}_y", y)
        results.add_tensor(f"{test_name}_h", h)
        results.add_tensor(f"{test_name}_f", f)
        results.add_tensor(f"{test_name}_f_mod", f_mod)
        
        print(f"  Pipeline success: {'✓' if passed else '✗'}")
        print(f"  Shape valid: {shape_valid}, No NaNs: {no_nans}, No Infs: {no_infs}")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results.add_test(test_name, False, error=e)


def test_batch_processing(results, device):
    """Test 5: Batch processing (B=1,8,32)"""
    test_name = "5_batch_processing"
    try:
        print(f"\n[{test_name}] Testing batch processing...")
        
        h_dim = 64
        f_dim = 128
        
        condnet = MNISTConditioner(h_dim=h_dim, spec_compliant=True).to(device)
        film = FiLM(f_dim=f_dim, h_dim=h_dim).to(device)
        
        batch_sizes = [1, 8, 32]
        all_passed = True
        batch_results = {}
        
        for B in batch_sizes:
            y = torch.randn(B, 1, 28, 28).to(device)
            f = torch.randn(B, f_dim).to(device)
            
            h = condnet(y)
            f_mod = film(f, h)
            
            shape_valid = (f_mod.shape[0] == B)
            all_passed = all_passed and shape_valid
            
            batch_results[f"batch_{B}"] = {
                'h_shape': list(h.shape),
                'f_mod_shape': list(f_mod.shape),
                'valid': shape_valid
            }
            
            print(f"  Batch {B}: {h.shape} → {f_mod.shape} - {'✓' if shape_valid else '✗'}")
        
        results.add_test(test_name, all_passed, details=batch_results)
        
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results.add_test(test_name, False, error=e)


def test_different_dims(results, device):
    """Test 6: Different h_dim/f_dim configs"""
    test_name = "6_different_dims"
    try:
        print(f"\n[{test_name}] Testing different h_dim/f_dim configs...")
        
        configs = [
            (32, 64),
            (64, 128),
            (128, 256),
            (256, 512)
        ]
        
        all_passed = True
        dim_results = {}
        
        for h_dim, f_dim in configs:
            condnet = MNISTConditioner(h_dim=h_dim, spec_compliant=True).to(device)
            film = FiLM(f_dim=f_dim, h_dim=h_dim).to(device)
            
            y = torch.randn(4, 1, 28, 28).to(device)
            f = torch.randn(4, f_dim).to(device)
            
            h = condnet(y)
            f_mod = film(f, h)
            
            valid = (h.shape[1] == h_dim and f_mod.shape[1] == f_dim)
            all_passed = all_passed and valid
            
            dim_results[f"h{h_dim}_f{f_dim}"] = {
                'h_shape': list(h.shape),
                'f_mod_shape': list(f_mod.shape),
                'valid': valid
            }
            
            print(f"  h_dim={h_dim}, f_dim={f_dim}: {'✓' if valid else '✗'}")
        
        results.add_test(test_name, all_passed, details=dim_results)
        
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results.add_test(test_name, False, error=e)


def test_gradient_flow(results, device):
    """Test 7: Gradient flow check"""
    test_name = "7_gradient_flow"
    try:
        print(f"\n[{test_name}] Testing gradient flow...")
        
        h_dim = 64
        f_dim = 128
        
        condnet = MNISTConditioner(h_dim=h_dim, spec_compliant=True).to(device)
        film = FiLM(f_dim=f_dim, h_dim=h_dim).to(device)
        
        y = torch.randn(8, 1, 28, 28, requires_grad=True).to(device)
        f = torch.randn(8, f_dim, requires_grad=True).to(device)
        
        # Forward pass
        h = condnet(y)
        f_mod = film(f, h)
        
        # Backward pass
        loss = f_mod.sum()
        loss.backward()
        
        # Check gradients exist
        y_grad_exists = (y.grad is not None) and (not torch.isnan(y.grad).any())
        f_grad_exists = (f.grad is not None) and (not torch.isnan(f.grad).any())
        
        # Check gradients are non-zero
        y_grad_nonzero = (y.grad.abs().max() > 1e-7) if y_grad_exists else False
        f_grad_nonzero = (f.grad.abs().max() > 1e-7) if f_grad_exists else False
        
        passed = y_grad_exists and f_grad_exists and y_grad_nonzero and f_grad_nonzero
        
        results.add_test(test_name, passed, details={
            'y_grad_exists': y_grad_exists,
            'f_grad_exists': f_grad_exists,
            'y_grad_max': float(y.grad.abs().max()) if y_grad_exists else None,
            'f_grad_max': float(f.grad.abs().max()) if f_grad_exists else None,
            'y_grad_nonzero': y_grad_nonzero,
            'f_grad_nonzero': f_grad_nonzero
        })
        
        if y_grad_exists:
            results.add_tensor(f"{test_name}_y_grad", y.grad)
        if f_grad_exists:
            results.add_tensor(f"{test_name}_f_grad", f.grad)
        
        print(f"  Y gradient: {'✓' if y_grad_exists and y_grad_nonzero else '✗'}")
        print(f"  F gradient: {'✓' if f_grad_exists and f_grad_nonzero else '✗'}")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results.add_test(test_name, False, error=e)


def test_device_compatibility(results, device):
    """Test 8: Device compatibility (CPU/GPU)"""
    test_name = "8_device_compatibility"
    try:
        print(f"\n[{test_name}] Testing device compatibility...")
        
        h_dim = 64
        f_dim = 128
        
        devices_to_test = ['cpu']
        if torch.cuda.is_available():
            devices_to_test.append('cuda')
        
        all_passed = True
        device_results = {}
        
        for dev in devices_to_test:
            d = torch.device(dev)
            
            condnet = MNISTConditioner(h_dim=h_dim, spec_compliant=True).to(d)
            film = FiLM(f_dim=f_dim, h_dim=h_dim).to(d)
            
            y = torch.randn(4, 1, 28, 28).to(d)
            f = torch.randn(4, f_dim).to(d)
            
            h = condnet(y)
            f_mod = film(f, h)
            
            valid = (f_mod.device.type == dev)
            all_passed = all_passed and valid
            
            device_results[dev] = {
                'valid': valid,
                'output_device': str(f_mod.device)
            }
            
            print(f"  Device {dev}: {'✓' if valid else '✗'}")
        
        results.add_test(test_name, all_passed, details=device_results)
        
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results.add_test(test_name, False, error=e)


def test_large_batch(results, device):
    """Test 9: Large batch (B=128)"""
    test_name = "9_large_batch"
    try:
        print(f"\n[{test_name}] Testing large batch (B=128)...")
        
        h_dim = 64
        f_dim = 128
        batch_size = 128
        
        condnet = MNISTConditioner(h_dim=h_dim, spec_compliant=True).to(device)
        film = FiLM(f_dim=f_dim, h_dim=h_dim).to(device)
        
        y = torch.randn(batch_size, 1, 28, 28).to(device)
        f = torch.randn(batch_size, f_dim).to(device)
        
        h = condnet(y)
        f_mod = film(f, h)
        
        shape_valid = (f_mod.shape == torch.Size([batch_size, f_dim]))
        no_nans = not torch.isnan(f_mod).any()
        
        passed = shape_valid and no_nans
        
        results.add_test(test_name, passed, details={
            'batch_size': batch_size,
            'output_shape': list(f_mod.shape),
            'shape_valid': shape_valid,
            'no_nans': no_nans
        })
        
        print(f"  Large batch: {'✓' if passed else '✗'}")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results.add_test(test_name, False, error=e)


def test_numerical_stability(results, device):
    """Test 10: Numerical stability (NaN/Inf)"""
    test_name = "10_numerical_stability"
    try:
        print(f"\n[{test_name}] Testing numerical stability...")
        
        h_dim = 64
        f_dim = 128
        
        condnet = MNISTConditioner(h_dim=h_dim, spec_compliant=True).to(device)
        film = FiLM(f_dim=f_dim, h_dim=h_dim).to(device)
        
        # Test with extreme values
        test_cases = {
            'normal': torch.randn(8, 1, 28, 28).to(device),
            'large': torch.randn(8, 1, 28, 28).to(device) * 100,
            'small': torch.randn(8, 1, 28, 28).to(device) * 0.01,
            'zeros': torch.zeros(8, 1, 28, 28).to(device)
        }
        
        all_passed = True
        stability_results = {}
        
        for case_name, y in test_cases.items():
            f = torch.randn(8, f_dim).to(device)
            
            h = condnet(y)
            f_mod = film(f, h)
            
            has_nans = torch.isnan(f_mod).any().item()
            has_infs = torch.isinf(f_mod).any().item()
            valid = not (has_nans or has_infs)
            
            all_passed = all_passed and valid
            
            stability_results[case_name] = {
                'valid': valid,
                'has_nans': has_nans,
                'has_infs': has_infs,
                'output_mean': float(f_mod.mean()),
                'output_std': float(f_mod.std())
            }
            
            print(f"  {case_name}: {'✓' if valid else '✗'} (NaN:{has_nans}, Inf:{has_infs})")
        
        results.add_test(test_name, all_passed, details=stability_results)
        
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results.add_test(test_name, False, error=e)


def test_config_variations(results, device):
    """Test 11: Config variations (channels, kernels)"""
    test_name = "11_config_variations"
    try:
        print(f"\n[{test_name}] Testing config variations...")
        
        configs = [
            {'channels': [32, 64, 128, 64], 'kernel_sizes': [3, 3, 3, 3]},
            {'channels': [16, 32, 64, 32], 'kernel_sizes': [5, 5, 3, 3]},
            {'channels': [64, 128, 256, 128], 'kernel_sizes': [3, 3, 3, 3]}
        ]
        
        all_passed = True
        config_results = {}
        
        for i, config in enumerate(configs):
            condnet = MNISTConditioner(h_dim=64, config=config, spec_compliant=False).to(device)
            
            y = torch.randn(4, 1, 28, 28).to(device)
            h = condnet(y)
            
            valid = (h.shape[0] == 4 and h.shape[1] == 64)
            all_passed = all_passed and valid
            
            config_results[f"config_{i}"] = {
                'config': config,
                'output_shape': list(h.shape),
                'valid': valid
            }
            
            print(f"  Config {i}: {config['channels']} - {'✓' if valid else '✗'}")
        
        results.add_test(test_name, all_passed, details=config_results)
        
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results.add_test(test_name, False, error=e)


def test_output_statistics(results, device):
    """Test 12: Output statistics"""
    test_name = "12_output_statistics"
    try:
        print(f"\n[{test_name}] Computing output statistics...")
        
        h_dim = 64
        f_dim = 128
        
        condnet = MNISTConditioner(h_dim=h_dim, spec_compliant=False).to(device)
        film = FiLM(f_dim=f_dim, h_dim=h_dim).to(device)
        
        # Generate statistics over multiple batches
        n_batches = 10
        h_stats = []
        f_mod_stats = []
        
        for _ in range(n_batches):
            y = torch.randn(16, 1, 28, 28).to(device)
            f = torch.randn(16, f_dim, 7, 7).to(device)
            
            with torch.no_grad():
                h = condnet(y)
                f_mod = film(f, h)
            
            h_stats.append({
                'mean': float(h.mean()),
                'std': float(h.std()),
                'min': float(h.min()),
                'max': float(h.max())
            })
            
            f_mod_stats.append({
                'mean': float(f_mod.mean()),
                'std': float(f_mod.std()),
                'min': float(f_mod.min()),
                'max': float(f_mod.max())
            })
        
        # Aggregate statistics
        stats = {
            'h_features': {
                'mean_avg': np.mean([s['mean'] for s in h_stats]),
                'std_avg': np.mean([s['std'] for s in h_stats]),
                'range': [
                    min([s['min'] for s in h_stats]),
                    max([s['max'] for s in h_stats])
                ]
            },
            'f_modulated': {
                'mean_avg': np.mean([s['mean'] for s in f_mod_stats]),
                'std_avg': np.mean([s['std'] for s in f_mod_stats]),
                'range': [
                    min([s['min'] for s in f_mod_stats]),
                    max([s['max'] for s in f_mod_stats])
                ]
            }
        }
        
        results.add_test(test_name, True, details=stats)
        
        print(f"  H features - mean: {stats['h_features']['mean_avg']:.4f}, std: {stats['h_features']['std_avg']:.4f}")
        print(f"  F modulated - mean: {stats['f_modulated']['mean_avg']:.4f}, std: {stats['f_modulated']['std_avg']:.4f}")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results.add_test(test_name, False, error=e)


def test_timing_benchmarks(results, device):
    """Test 13: Timing benchmarks"""
    test_name = "13_timing_benchmarks"
    try:
        print(f"\n[{test_name}] Running timing benchmarks...")
        
        import time
        
        h_dim = 64
        f_dim = 128
        
        condnet = MNISTConditioner(h_dim=h_dim, spec_compliant=False).to(device)
        film = FiLM(f_dim=f_dim, h_dim=h_dim).to(device)
        
        # Warmup
        for _ in range(5):
            y = torch.randn(16, 1, 28, 28).to(device)
            f = torch.randn(16, f_dim, 7, 7).to(device)
            h = condnet(y)
            _ = film(f, h)
        
        # Benchmark
        n_iters = 100
        
        # CondNet timing
        start = time.time()
        for _ in range(n_iters):
            y = torch.randn(16, 1, 28, 28).to(device)
            with torch.no_grad():
                h = condnet(y)
        condnet_time = (time.time() - start) / n_iters
        
        # FiLM timing
        y = torch.randn(16, 1, 28, 28).to(device)
        h = condnet(y)
        f = torch.randn(16, f_dim, 7, 7).to(device)
        
        start = time.time()
        for _ in range(n_iters):
            with torch.no_grad():
                f_mod = film(f, h)
        film_time = (time.time() - start) / n_iters
        
        timing_results = {
            'condnet_ms': condnet_time * 1000,
            'film_ms': film_time * 1000,
            'total_ms': (condnet_time + film_time) * 1000,
            'n_iterations': n_iters
        }
        
        results.add_test(test_name, True, details=timing_results)
        
        print(f"  CondNet: {condnet_time*1000:.2f} ms/batch")
        print(f"  FiLM: {film_time*1000:.2f} ms/batch")
        print(f"  Total: {(condnet_time+film_time)*1000:.2f} ms/batch")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results.add_test(test_name, False, error=e)


def main():
    """Run all tests and save results"""
    print("="*70)
    print("Test Conditioning Networks and FiLM Layers")
    print("Version: WP0.1-TestCondFiLM-v1.0")
    print("="*70)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Initialize results
    results = TestResults()
    
    # Run all tests
    print("\n" + "="*70)
    print("CORE TESTS (8)")
    print("="*70)
    
    test_condnet_shapes(results, device)
    test_film_modulation(results, device)
    test_film_spatial(results, device)
    test_integration(results, device)
    test_batch_processing(results, device)
    test_different_dims(results, device)
    test_gradient_flow(results, device)
    test_device_compatibility(results, device)
    
    print("\n" + "="*70)
    print("OPTIONAL TESTS (5)")
    print("="*70)
    
    test_large_batch(results, device)
    test_numerical_stability(results, device)
    test_config_variations(results, device)
    test_output_statistics(results, device)
    test_timing_benchmarks(results, device)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    total_tests = len(results.tests)
    passed_tests = sum(1 for t in results.tests.values() if t['passed'])
    
    print(f"\nTotal tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success rate: {100*passed_tests/total_tests:.1f}%")
    
    print("\nDetailed results:")
    for name, test in results.tests.items():
        status = "✓ PASS" if test['passed'] else "✗ FAIL"
        print(f"  {name}: {status}")
        if test['error']:
            print(f"    Error: {test['error']}")
    
    # Save results
    # Calculate relative to script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = str(project_root / "results" / "test")
    #output_dir = "../results/test"
    results.save(output_dir)
    
    print("\n" + "="*70)
    print("All tests completed!")
    print("="*70)


if __name__ == "__main__":
    main()