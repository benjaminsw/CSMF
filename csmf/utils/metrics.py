"""
Enhanced KL Divergence Computation with both KDE and Exact methods.

This module provides functions to compute KL divergence and cross-entropy using:
1. KDE-based approximation (for any two sample sets)
2. Exact computation (when normalizing flows are available)
3. Hybrid approaches combining both methods
"""

import torch
import numpy as np
from scipy.stats import gaussian_kde
from typing import Tuple, Optional, Union, Dict, Any

def compute_exact_cross_entropy(target_samples: torch.Tensor, 
                               flow_model: torch.nn.Module) -> float:
    """
    Compute exact cross-entropy surrogate: -E_p[log q(x)]
    where p is empirical target distribution and q is normalizing flow.
    
    Args:
        target_samples: torch.Tensor [n_samples, d] - samples from target distribution
        flow_model: torch.nn.Module - normalizing flow model
        
    Returns:
        float - exact cross-entropy surrogate
    """
    flow_model.eval()
    with torch.no_grad():
        log_q_x = flow_model.log_prob(target_samples)
        cross_entropy = -log_q_x.mean().item()
    return cross_entropy

def compute_exact_kl_divergence_nf_to_nf(flow_p: torch.nn.Module,
                                        flow_q: torch.nn.Module,
                                        n_samples: int = 10000) -> float:
    """
    Compute exact KL divergence between two normalizing flows: KL(P||Q)
    
    Args:
        flow_p: Source normalizing flow P
        flow_q: Target normalizing flow Q
        n_samples: Number of samples for Monte Carlo estimation
        
    Returns:
        float - exact KL divergence KL(P||Q)
    """
    flow_p.eval()
    flow_q.eval()
    
    with torch.no_grad():
        # Sample from source distribution P
        x_samples = flow_p.sample(n_samples)
        
        # Get exact log probabilities
        log_p_x = flow_p.log_prob(x_samples)
        log_q_x = flow_q.log_prob(x_samples)
        
        # Exact KL divergence
        kl_divergence = (log_p_x - log_q_x).mean().item()
        
    return kl_divergence

def compute_kde_cross_entropy(target_samples: torch.Tensor,
                             generated_samples: torch.Tensor,
                             grid_resolution: int = 100,
                             bandwidth_method: str = 'scott',
                             epsilon: float = 1e-10) -> float:
    """
    Compute cross-entropy using KDE: -E_p[log q(x)]
    where p is estimated from target_samples and q from generated_samples.
    
    Args:
        target_samples: torch.Tensor [n_samples, d] - target distribution samples
        generated_samples: torch.Tensor [n_samples, d] - generated distribution samples
        grid_resolution: int - grid resolution for numerical integration
        bandwidth_method: str - KDE bandwidth method
        epsilon: float - numerical stability constant
        
    Returns:
        float - KDE-based cross-entropy estimate
    """
    # Convert to numpy and transpose for KDE
    target_np = target_samples.detach().cpu().numpy().T
    generated_np = generated_samples.detach().cpu().numpy().T
    
    # Fit KDE to both distributions
    kde_target = gaussian_kde(target_np, bw_method=bandwidth_method)
    kde_generated = gaussian_kde(generated_np, bw_method=bandwidth_method)
    
    # Use same bandwidth for fair comparison
    target_bandwidth = kde_target.factor
    kde_generated.set_bandwidth(target_bandwidth)
    
    # Create evaluation grid
    x_min = min(target_np[0].min(), generated_np[0].min())
    x_max = max(target_np[0].max(), generated_np[0].max())
    y_min = min(target_np[1].min(), generated_np[1].min())
    y_max = max(target_np[1].max(), generated_np[1].max())
    
    # Add margins
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    x_min, x_max = x_min - x_margin, x_max + x_margin
    y_min, y_max = y_min - y_margin, y_max + y_margin
    
    # Create grid
    x_grid = np.linspace(x_min, x_max, grid_resolution)
    y_grid = np.linspace(y_min, y_max, grid_resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = np.vstack([X.ravel(), Y.ravel()])
    
    # Evaluate densities
    p_densities = kde_target(grid_points)
    q_densities = kde_generated(grid_points)
    
    # Add epsilon for stability
    p_densities = np.maximum(p_densities, epsilon)
    q_densities = np.maximum(q_densities, epsilon)
    
    # Cross-entropy: -∫ p(x) log q(x) dx
    cross_entropy_integrand = p_densities * (-np.log(q_densities))
    
    # Numerical integration
    dx = (x_max - x_min) / grid_resolution
    dy = (y_max - y_min) / grid_resolution
    cell_area = dx * dy
    
    cross_entropy = np.sum(cross_entropy_integrand) * cell_area
    
    return float(cross_entropy)

def compute_kde_kl_divergence(target_samples: torch.Tensor, 
                            generated_samples: torch.Tensor,
                            grid_resolution: int = 100,
                            bandwidth_method: str = 'scott',
                            epsilon: float = 1e-10) -> float:
    """
    Compute KL divergence between target and generated samples using KDE.
    
    Args:
        target_samples: torch.Tensor [n_samples, 2] - target distribution samples
        generated_samples: torch.Tensor [n_samples, 2] - generated distribution samples  
        grid_resolution: int - number of grid points per dimension
        bandwidth_method: str - bandwidth selection method ('scott', 'silverman', or float)
        epsilon: float - small value to avoid log(0) issues
        
    Returns:
        float - KL divergence D_KL(P||Q) where P=target, Q=generated
    """
    # Convert to numpy and transpose for KDE
    target_np = target_samples.detach().cpu().numpy().T
    generated_np = generated_samples.detach().cpu().numpy().T
    
    # Fit KDE to both distributions
    if isinstance(bandwidth_method, str):
        kde_target = gaussian_kde(target_np, bw_method=bandwidth_method)
        kde_generated = gaussian_kde(generated_np, bw_method=bandwidth_method)
        
        # Use same bandwidth for fair comparison
        target_bandwidth = kde_target.factor
        kde_generated.set_bandwidth(target_bandwidth)
    else:
        kde_target = gaussian_kde(target_np, bw_method=bandwidth_method)
        kde_generated = gaussian_kde(generated_np, bw_method=bandwidth_method)
    
    # Create evaluation grid
    x_min = min(target_np[0].min(), generated_np[0].min())
    x_max = max(target_np[0].max(), generated_np[0].max())
    y_min = min(target_np[1].min(), generated_np[1].min())
    y_max = max(target_np[1].max(), generated_np[1].max())
    
    # Add margins
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    x_min, x_max = x_min - x_margin, x_max + x_margin
    y_min, y_max = y_min - y_margin, y_max + y_margin
    
    # Create grid
    x_grid = np.linspace(x_min, x_max, grid_resolution)
    y_grid = np.linspace(y_min, y_max, grid_resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = np.vstack([X.ravel(), Y.ravel()])
    
    # Evaluate densities
    p_densities = kde_target(grid_points)
    q_densities = kde_generated(grid_points)
    
    # Add epsilon for stability
    p_densities = np.maximum(p_densities, epsilon)
    q_densities = np.maximum(q_densities, epsilon)
    
    # KL divergence: ∫ p(x) log(p(x)/q(x)) dx
    log_ratio = np.log(p_densities / q_densities)
    kl_integrand = p_densities * log_ratio
    
    # Numerical integration
    dx = (x_max - x_min) / grid_resolution
    dy = (y_max - y_min) / grid_resolution
    cell_area = dx * dy
    
    kl_divergence = np.sum(kl_integrand) * cell_area
    
    return float(kl_divergence)

def comprehensive_divergence_computation(target_samples: torch.Tensor,
                                       generated_samples: torch.Tensor,
                                       flow_model: Optional[torch.nn.Module] = None,
                                       target_flow: Optional[torch.nn.Module] = None,
                                       n_samples_exact: int = 10000,
                                       grid_resolution: int = 100,
                                       bandwidth_method: str = 'scott') -> Dict[str, float]:
    """
    Comprehensive computation of all available divergence metrics.
    
    Args:
        target_samples: Target distribution samples
        generated_samples: Generated distribution samples  
        flow_model: Optional flow model that generated the samples
        target_flow: Optional flow model for target distribution
        n_samples_exact: Number of samples for exact computations
        grid_resolution: Grid resolution for KDE methods
        bandwidth_method: KDE bandwidth method
        
    Returns:
        dict: All computed divergence metrics
    """
    results = {}
    
    # 1. KDE-based methods (always available)
    results['kde_kl_divergence'] = compute_kde_kl_divergence(
        target_samples, generated_samples, grid_resolution, bandwidth_method)
    
    results['kde_cross_entropy'] = compute_kde_cross_entropy(
        target_samples, generated_samples, grid_resolution, bandwidth_method)
    
    # 2. Exact cross-entropy (if flow model available)
    if flow_model is not None:
        results['exact_cross_entropy'] = compute_exact_cross_entropy(
            target_samples, flow_model)
    
    # 3. Exact KL divergence (if both flows available)
    if target_flow is not None and flow_model is not None:
        results['exact_kl_divergence'] = compute_exact_kl_divergence_nf_to_nf(
            target_flow, flow_model, n_samples_exact)
    
    # 4. Additional comparisons
    if flow_model is not None:
        # Compare exact vs KDE cross-entropy
        exact_ce = results.get('exact_cross_entropy')
        kde_ce = results.get('kde_cross_entropy')
        if exact_ce is not None and kde_ce is not None:
            results['cross_entropy_difference'] = abs(exact_ce - kde_ce)
            results['cross_entropy_relative_error'] = abs(exact_ce - kde_ce) / max(abs(exact_ce), 1e-10)
    
    if 'exact_kl_divergence' in results:
        # Compare exact vs KDE KL divergence
        exact_kl = results['exact_kl_divergence']
        kde_kl = results['kde_kl_divergence']
        results['kl_difference'] = abs(exact_kl - kde_kl)
        results['kl_relative_error'] = abs(exact_kl - kde_kl) / max(abs(exact_kl), 1e-10)
    
    return results

def compute_kde_kl_divergence_with_bandwidth_options(target_samples: torch.Tensor,
                                                   generated_samples: torch.Tensor,
                                                   bandwidth_strategy: str = 'same',
                                                   grid_resolution: int = 100) -> float:
    """
    Compute KL divergence with different bandwidth selection strategies.
    """
    target_np = target_samples.detach().cpu().numpy().T
    generated_np = generated_samples.detach().cpu().numpy().T
    
    if bandwidth_strategy == 'same':
        kde_target = gaussian_kde(target_np, bw_method='scott')
        kde_generated = gaussian_kde(generated_np, bw_method='scott')
        target_bandwidth = kde_target.factor
        kde_generated.set_bandwidth(target_bandwidth)
    elif bandwidth_strategy == 'separate':
        kde_target = gaussian_kde(target_np, bw_method='scott')
        kde_generated = gaussian_kde(generated_np, bw_method='scott')
    elif bandwidth_strategy == 'cross_val':
        raise NotImplementedError("Cross-validation bandwidth selection not yet implemented")
    else:
        raise ValueError(f"Unknown bandwidth strategy: {bandwidth_strategy}")
    
    # Create evaluation grid and compute KL divergence
    x_min = min(target_np[0].min(), generated_np[0].min())
    x_max = max(target_np[0].max(), generated_np[0].max())
    y_min = min(target_np[1].min(), generated_np[1].min())
    y_max = max(target_np[1].max(), generated_np[1].max())
    
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    x_min, x_max = x_min - x_margin, x_max + x_margin
    y_min, y_max = y_min - y_margin, y_max + y_margin
    
    x_grid = np.linspace(x_min, x_max, grid_resolution)
    y_grid = np.linspace(y_min, y_max, grid_resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = np.vstack([X.ravel(), Y.ravel()])
    
    p_densities = kde_target(grid_points)
    q_densities = kde_generated(grid_points)
    
    epsilon = 1e-10
    p_densities = np.maximum(p_densities, epsilon)
    q_densities = np.maximum(q_densities, epsilon)
    
    log_ratio = np.log(p_densities / q_densities)
    kl_integrand = p_densities * log_ratio
    
    dx = (x_max - x_min) / grid_resolution
    dy = (y_max - y_min) / grid_resolution
    cell_area = dx * dy
    
    kl_divergence = np.sum(kl_integrand) * cell_area
    return float(kl_divergence)

def adaptive_grid_resolution(target_samples: torch.Tensor,
                           generated_samples: torch.Tensor,
                           max_resolution: int = 200,
                           min_resolution: int = 50) -> int:
    """
    Adaptively choose grid resolution based on data characteristics.
    """
    n_target = target_samples.size(0)
    n_generated = generated_samples.size(0)
    
    total_samples = n_target + n_generated
    
    if total_samples < 500:
        return min_resolution
    elif total_samples < 2000:
        return 100
    else:
        return max_resolution

def get_kde_info(target_samples: torch.Tensor, 
                generated_samples: torch.Tensor) -> dict:
    """
    Get information about KDE fitting for debugging purposes.
    """
    target_np = target_samples.detach().cpu().numpy().T
    generated_np = generated_samples.detach().cpu().numpy().T
    
    kde_target = gaussian_kde(target_np, bw_method='scott')
    kde_generated = gaussian_kde(generated_np, bw_method='scott')
    
    return {
        'target_bandwidth': kde_target.factor,
        'generated_bandwidth': kde_generated.factor,
        'target_n_samples': target_samples.size(0),
        'generated_n_samples': generated_samples.size(0),
        'target_mean': target_samples.mean(dim=0).tolist(),
        'generated_mean': generated_samples.mean(dim=0).tolist(),
        'target_std': target_samples.std(dim=0).tolist(),
        'generated_std': generated_samples.std(dim=0).tolist(),
    }

def get_computation_recommendations(target_samples: torch.Tensor,
                                  generated_samples: torch.Tensor,
                                  flow_model: Optional[torch.nn.Module] = None,
                                  target_flow: Optional[torch.nn.Module] = None) -> Dict[str, str]:
    """
    Provide recommendations on which computation methods to use.
    """
    recommendations = {}
    
    n_target = target_samples.size(0)
    n_generated = generated_samples.size(0)
    total_samples = n_target + n_generated
    
    # Cross-entropy recommendation
    if flow_model is not None:
        recommendations['cross_entropy'] = "Use exact_cross_entropy for highest accuracy"
    else:
        recommendations['cross_entropy'] = "Use kde_cross_entropy (exact method unavailable)"
    
    # KL divergence recommendation  
    if target_flow is not None and flow_model is not None:
        recommendations['kl_divergence'] = "Use exact_kl_divergence for highest accuracy"
    elif total_samples > 2000:
        recommendations['kl_divergence'] = "Use kde_kl_divergence with high grid resolution"
    else:
        recommendations['kl_divergence'] = "Use kde_kl_divergence but expect higher uncertainty"
    
    # Grid resolution recommendation
    if total_samples < 500:
        recommendations['grid_resolution'] = "Use 50-75 (low sample size)"
    elif total_samples < 2000:
        recommendations['grid_resolution'] = "Use 100-150 (medium sample size)"  
    else:
        recommendations['grid_resolution'] = "Use 150-200 (high sample size)"
    
    return recommendations
