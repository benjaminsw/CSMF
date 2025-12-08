import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .base_flow import BaseFlow

class MarginalGaussianization(nn.Module):
    """Marginal Gaussianization using improved CDF estimation and interpolation."""
    
    def __init__(self, dim: int, n_bins: int = 1000):  # Increased default bins
        super().__init__()
        self.dim = dim
        self.n_bins = n_bins
        
        # Store high-resolution CDF data for proper interpolation
        self.register_buffer('x_values', torch.zeros(dim, n_bins))  # Data points
        self.register_buffer('cdf_values', torch.zeros(dim, n_bins))  # CDF values
        self.register_buffer('data_min', torch.zeros(dim))
        self.register_buffer('data_max', torch.zeros(dim))
        self.register_buffer('is_trained', torch.zeros(dim, dtype=torch.bool))
        
    def fit_marginals(self, x: torch.Tensor):
        """Fit marginal CDFs using high-resolution empirical estimation."""
        batch_size = x.size(0)
        
        for d in range(self.dim):
            # Get data for this dimension
            x_d = x[:, d]
            
            # Store data range with more conservative margins
            x_min, x_max = x_d.min(), x_d.max()
            margin = 0.1 * (x_max - x_min)  # Larger margin for better stability
            x_min_extended = x_min - margin
            x_max_extended = x_max + margin
            
            self.data_min[d] = x_min_extended
            self.data_max[d] = x_max_extended
            
            # Create high-resolution grid for CDF estimation
            x_grid = torch.linspace(x_min_extended, x_max_extended, self.n_bins, device=x.device)
            self.x_values[d] = x_grid
            
            # Compute empirical CDF at grid points
            cdf_vals = torch.zeros(self.n_bins, device=x.device)
            for i, xi in enumerate(x_grid):
                cdf_vals[i] = (x_d <= xi).float().mean()
            
            # Ensure monotonicity and proper bounds with better clamping
            cdf_vals = torch.clamp(cdf_vals, min=1e-6, max=1-1e-6)  # More aggressive
            
            # Force strict monotonicity
            for i in range(1, self.n_bins):
                if cdf_vals[i] <= cdf_vals[i-1]:
                    cdf_vals[i] = cdf_vals[i-1] + 1e-8  # Ensure strict increase
            
            # Final normalization
            cdf_vals = torch.clamp(cdf_vals, min=1e-6, max=1-1e-6)
            
            self.cdf_values[d] = cdf_vals
            self.is_trained[d] = True
    
    def marginal_cdf(self, x_d: torch.Tensor, dim_idx: int) -> torch.Tensor:
        """Compute empirical CDF with linear interpolation."""
        if not self.is_trained[dim_idx]:
            # Fallback for untrained dimensions
            x_min, x_max = x_d.min(), x_d.max()
            u = (x_d - x_min) / (x_max - x_min + 1e-8)
            return torch.clamp(u, min=1e-6, max=1-1e-6)  # More aggressive clamping
        
        x_vals = self.x_values[dim_idx]
        cdf_vals = self.cdf_values[dim_idx]
        
        # Linear interpolation for CDF values
        cdf_output = self._interpolate_cdf(x_d, x_vals, cdf_vals)
        return torch.clamp(cdf_output, min=1e-6, max=1-1e-6)  # More aggressive clamping
    
    def _interpolate_cdf(self, x: torch.Tensor, x_vals: torch.Tensor, cdf_vals: torch.Tensor) -> torch.Tensor:
        """Linear interpolation for CDF lookup."""
        # Make input contiguous to avoid warning
        x = x.contiguous()
        x_vals = x_vals.contiguous()
        
        # Find indices for interpolation
        indices = torch.searchsorted(x_vals, x, right=False)
        indices = torch.clamp(indices, 1, len(x_vals) - 1)
        
        # Get surrounding points
        x_left = x_vals[indices - 1]
        x_right = x_vals[indices]
        cdf_left = cdf_vals[indices - 1]
        cdf_right = cdf_vals[indices]
        
        # Linear interpolation
        weight = (x - x_left) / (x_right - x_left + 1e-12)
        weight = torch.clamp(weight, 0, 1)
        
        return cdf_left + weight * (cdf_right - cdf_left)
    
    def inverse_cdf(self, u: torch.Tensor, dim_idx: int) -> torch.Tensor:
        """Compute inverse CDF with linear interpolation."""
        if not self.is_trained[dim_idx]:
            # Fallback for untrained dimensions
            return u  # Identity mapping
        
        x_vals = self.x_values[dim_idx]
        cdf_vals = self.cdf_values[dim_idx]
        
        # Linear interpolation for inverse CDF
        return self._interpolate_inverse_cdf(u, cdf_vals, x_vals)
    
    def _interpolate_inverse_cdf(self, u: torch.Tensor, cdf_vals: torch.Tensor, x_vals: torch.Tensor) -> torch.Tensor:
        """Linear interpolation for inverse CDF lookup."""
        # Make input contiguous to avoid warning
        u = u.contiguous()
        cdf_vals = cdf_vals.contiguous()
        
        # Find indices for interpolation
        indices = torch.searchsorted(cdf_vals, u, right=False)
        indices = torch.clamp(indices, 1, len(cdf_vals) - 1)
        
        # Get surrounding points
        cdf_left = cdf_vals[indices - 1]
        cdf_right = cdf_vals[indices]
        x_left = x_vals[indices - 1]
        x_right = x_vals[indices]
        
        # Linear interpolation
        weight = (u - cdf_left) / (cdf_right - cdf_left + 1e-12)
        weight = torch.clamp(weight, 0, 1)
        
        return x_left + weight * (x_right - x_left)
    
    def inverse_gaussian_cdf(self, u: torch.Tensor) -> torch.Tensor:
        """Apply inverse Gaussian CDF: Φ^(-1)(u) - improved implementation."""
        # More aggressive clamping to avoid extreme values
        u = torch.clamp(u, min=1e-6, max=1-1e-6)
        
        # Protect against extreme erfinv values
        erfinv_input = 2 * u - 1
        erfinv_input = torch.clamp(erfinv_input, min=-0.99999, max=0.99999)
        
        # Use proper inverse normal CDF: Φ^(-1)(u) = √2 * erfinv(2u - 1)
        sqrt_2 = torch.sqrt(torch.tensor(2.0, device=u.device, dtype=u.dtype))
        result = sqrt_2 * torch.erfinv(erfinv_input)
        
        # Final safety clamp to avoid extreme values
        result = torch.clamp(result, min=-10.0, max=10.0)
        return result
    
    def gaussian_cdf(self, z: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian CDF: Φ(z) - proper implementation."""
        # Φ(z) = 0.5 * (1 + erf(z / √2))
        sqrt_2 = torch.sqrt(torch.tensor(2.0, device=z.device, dtype=z.dtype))
        return 0.5 * (1 + torch.erf(z / sqrt_2))
    
    # inside MarginalGaussianization
    def _cdf_and_pdf(self, x, x_vals, cdf_vals, eps=1e-12):
        idx = torch.searchsorted(x_vals, x, right=False).clamp(1, len(x_vals)-1)
        x_l, x_r = x_vals[idx-1], x_vals[idx]
        cdf_l, cdf_r = cdf_vals[idx-1], cdf_vals[idx]
        # Linear interp for CDF
        u = cdf_l + (cdf_r - cdf_l) * (x - x_l) / (x_r - x_l + eps)
        # Slope of the CDF segment ≈ pdf
        p_hat = (cdf_r - cdf_l) / (x_r - x_l + eps)
        return u.clamp(1e-6, 1-1e-6), p_hat.clamp_min(1e-12)

    def forward(self, x):
        z = torch.zeros_like(x)
        log_det = torch.zeros(x.size(0), device=x.device)
        inv_sqrt_2pi = 1.0 / torch.sqrt(torch.tensor(2.0*torch.pi, device=x.device, dtype=x.dtype))
        for d in range(self.dim):
            u_d, p_hat = self._cdf_and_pdf(x[:, d], self.x_values[d], self.cdf_values[d])
            z_d = self.inverse_gaussian_cdf(u_d)
            z[:, d] = z_d
            phi = inv_sqrt_2pi * torch.exp(-0.5 * z_d**2)
            log_det += torch.log(p_hat) - torch.log(phi + 1e-12)
        return z, log_det
    
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Proper inverse transformation using interpolated inverse CDF."""
        x = torch.zeros_like(z)
        
        for d in range(self.dim):
            if self.is_trained[d]:
                # Step 1: Apply Gaussian CDF to get uniform distribution
                u_d = self.gaussian_cdf(z[:, d])
                
                # Step 2: Apply inverse empirical CDF to get original distribution
                x[:, d] = self.inverse_cdf(u_d, d)
            else:
                x[:, d] = z[:, d]  # Identity if not trained
        
        return x


class RandomRotation(nn.Module):
    """Random orthogonal rotation matrix."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # Generate random orthogonal matrix using QR decomposition
        Q, _ = torch.linalg.qr(torch.randn(dim, dim))
        self.register_buffer('rotation_matrix', Q)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotation."""
        z = x @ self.rotation_matrix.T
        # Log determinant is 0 for orthogonal matrices
        log_det = torch.zeros(x.size(0), device=x.device)
        return z, log_det
    
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse rotation (transpose)."""
        return z @ self.rotation_matrix


class RBIGFlow(BaseFlow):
    """
    Rotation-Based Iterative Gaussianization (RBIG) Flow.
    
    Improved version with:
    - High-resolution CDF estimation
    - Proper interpolated inverse transformations
    - Reconstruction quality validation
    """
    
    def __init__(self, dim: int, n_layers: int = 4, n_bins: int = 1000):
        super().__init__(dim)
        self.n_layers = n_layers
        self.n_bins = n_bins
        
        # Create alternating layers: Gaussianization -> Rotation
        self.gaussianization_layers = nn.ModuleList([
            MarginalGaussianization(dim, n_bins) for _ in range(n_layers)
        ])
        
        self.rotation_layers = nn.ModuleList([
            RandomRotation(dim) for _ in range(n_layers)
        ])
        
        self.is_fitted = False
        self._reconstruction_quality = None
    
    def fit_to_data(self, x: torch.Tensor, validate_reconstruction: bool = True):
        """Fit the empirical CDFs to training data with optional validation."""
        current_x = x.clone()
        
        for i in range(self.n_layers):
            # Fit marginal Gaussianization for current layer
            self.gaussianization_layers[i].fit_marginals(current_x)
            
            # Apply transformations to get input for next layer
            with torch.no_grad():
                # Apply Gaussianization
                current_x, _ = self.gaussianization_layers[i](current_x)
                # Apply rotation
                current_x, _ = self.rotation_layers[i](current_x)
        
        self.is_fitted = True
        
        # Validate reconstruction quality
        if validate_reconstruction:
            self._validate_reconstruction_quality(x)
    
    def _validate_reconstruction_quality(self, x: torch.Tensor):
        """Validate the quality of reconstruction."""
        with torch.no_grad():
            # Forward-inverse round trip
            z, _ = self.forward_and_log_det(x)
            x_reconstructed = self.inverse(z)
            
            # Compute reconstruction metrics
            mse = torch.mean((x - x_reconstructed) ** 2).item()
            max_error = torch.max(torch.abs(x - x_reconstructed)).item()
            
            # Store quality metrics
            self._reconstruction_quality = {
                'mse': mse,
                'max_error': max_error,
                'relative_mse': mse / (torch.var(x).item() + 1e-8)
            }
            
            # Print quality information
            print(f"🔍 Reconstruction Quality:")
            print(f"  MSE: {mse:.8f}")
            print(f"  Max Error: {max_error:.8f}")
            print(f"  Relative MSE: {self._reconstruction_quality['relative_mse']:.8f}")
            
            # Warning for poor reconstruction
            if self._reconstruction_quality['relative_mse'] > 0.01:
                print("⚠️  Warning: Poor reconstruction quality detected!")
                print("   Consider increasing n_bins or reducing n_layers.")
    
    def get_reconstruction_quality(self) -> Optional[dict]:
        """Get reconstruction quality metrics."""
        return self._reconstruction_quality
    
    def forward_and_log_det(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: x -> z with log determinant."""
        if not self.is_fitted:
            # Auto-fit on first forward pass
            self.fit_to_data(x)
        
        z = x
        total_log_det = torch.zeros(x.size(0), device=x.device)
        
        for i in range(self.n_layers):
            # Apply marginal Gaussianization
            z, log_det_gauss = self.gaussianization_layers[i](z)
            total_log_det += log_det_gauss
            
            # Apply rotation
            z, log_det_rot = self.rotation_layers[i](z)
            total_log_det += log_det_rot  # Should be 0
        
        return z, total_log_det
    
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse transformation: z -> x."""
        x = z
        
        # Apply inverse transformations in reverse order
        for i in reversed(range(self.n_layers)):
            # Inverse rotation
            x = self.rotation_layers[i].inverse(x)
            # Inverse Gaussianization (now properly implemented)
            x = self.gaussianization_layers[i].inverse(x)
        
        return x
    
    def reset_training(self):
        """Reset training state (useful for retraining on new data)."""
        self.is_fitted = False
        self._reconstruction_quality = None
        for layer in self.gaussianization_layers:
            layer.is_trained.fill_(False)
    
    def sample(self, n_samples: int) -> torch.Tensor:
        """Generate samples from the learned distribution."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before sampling")
        
        # Sample from standard Gaussian
        #device = next(self.parameters()).device
        device = self.gaussianization_layers[0].data_min.device
        z = torch.randn(n_samples, self.dim, device=device)
        
        # Transform through inverse flow
        return self.inverse(z)
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log probability of data points with numerical stability."""
        z, log_det = self.forward_and_log_det(x)
        
        # Check for NaN/inf in transformed data
        if torch.any(torch.isnan(z)) or torch.any(torch.isinf(z)):
            # Return very low but finite log probability
            return torch.full((x.size(0),), -100.0, device=x.device)
        
        # Log probability of standard Gaussian with overflow protection
        log_2pi = torch.log(torch.tensor(2 * torch.pi, device=x.device))
        z_squared = torch.sum(z ** 2, dim=1)
        
        # Clamp to prevent numerical overflow
        z_squared = torch.clamp(z_squared, max=400.0)  # exp(-200) is very small but finite
        
        log_prob_z = -0.5 * (self.dim * log_2pi + z_squared)
        
        # Apply change of variables
        log_prob_total = log_prob_z + log_det
        
        # Replace NaN/inf with very low probability
        nan_mask = torch.isnan(log_prob_total) | torch.isinf(log_prob_total)
        log_prob_total[nan_mask] = -100.0
        
        return log_prob_total