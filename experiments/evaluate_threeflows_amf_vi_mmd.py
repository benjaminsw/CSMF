import torch
import numpy as np
from .threeflows_amf_vi_weights_log import SequentialAMFVI, train_sequential_amf_vi
from data.data_generator import generate_data
import os
import pickle
import csv
from sklearn.metrics import pairwise_distances

# Set seed for reproducible experiments
torch.manual_seed(2025)
np.random.seed(2025)

def gaussian_kernel(x, y, sigma=1.0):
    """Compute Gaussian (RBF) kernel k(x,y) = exp(-||x-y||²/2σ²)"""
    pairwise_sq_dists = torch.cdist(x, y, p=2) ** 2
    return torch.exp(-pairwise_sq_dists / (2 * sigma ** 2))

def polynomial_kernel(x, y, degree=2, coeff=1.0):
    """Compute Polynomial kernel k(x,y) = (x·y + c)^d"""
    dot_product = torch.mm(x, y.t())
    return (dot_product + coeff) ** degree

def compute_mmd_squared_unbiased(X, Y, kernel_func):
    """Compute MMD² using unbiased estimator (excludes diagonal terms)"""
    m, n = X.shape[0], Y.shape[0]
    
    # E[k(x,x')] for X - excluding diagonal terms
    Kxx = kernel_func(X, X)
    term1 = (Kxx.sum() - Kxx.diagonal().sum()) / (m * (m - 1))
    
    # E[k(y,y')] for Y - excluding diagonal terms
    Kyy = kernel_func(Y, Y)
    term2 = (Kyy.sum() - Kyy.diagonal().sum()) / (n * (n - 1))
    
    # E[k(x,y)] between X and Y
    Kxy = kernel_func(X, Y)
    term3 = 2 * Kxy.sum() / (m * n)
    
    return term1 + term2 - term3

def compute_mmd_squared_biased(X, Y, kernel_func):
    """Compute MMD² using biased estimator (includes diagonal terms)"""
    m, n = X.shape[0], Y.shape[0]
    
    # E[k(x,x')] for X - including diagonal terms
    Kxx = kernel_func(X, X)
    term1 = Kxx.sum() / (m * m)
    
    # E[k(y,y')] for Y - including diagonal terms
    Kyy = kernel_func(Y, Y)
    term2 = Kyy.sum() / (n * n)
    
    # E[k(x,y)] between X and Y
    Kxy = kernel_func(X, Y)
    term3 = 2 * Kxy.sum() / (m * n)
    
    return term1 + term2 - term3

def compute_mmd_biased_scikit(target_samples, generated_samples, sigma='auto'):
    """Compute biased MMD with RBF kernel using scikit-learn (similar to sinkhorn file)"""
    target_np = target_samples.detach().cpu().numpy()
    generated_np = generated_samples.detach().cpu().numpy()
    
    if sigma == 'auto':
        # Median heuristic for bandwidth
        combined = np.vstack([target_np[:500], generated_np[:500]])
        distances = pairwise_distances(combined)
        sigma = np.median(distances[distances > 0])
        if sigma == 0:
            sigma = 1.0
    
    def rbf_kernel(X, Y):
        return np.exp(-pairwise_distances(X, Y, squared=True) / (2 * sigma**2))
    
    n_target, n_generated = len(target_np), len(generated_np)
    
    # Compute kernel matrices
    K_tt = rbf_kernel(target_np, target_np)
    K_gg = rbf_kernel(generated_np, generated_np)
    K_tg = rbf_kernel(target_np, generated_np)
    
    # Biased MMD² estimate (includes diagonal)
    mmd_squared = (K_tt.sum() / (n_target * n_target) + 
                   K_gg.sum() / (n_generated * n_generated) - 
                   2 * K_tg.sum() / (n_target * n_generated))
    
    return np.sqrt(max(0, mmd_squared))

def compute_polynomial_mmd_biased_scikit(target_samples, generated_samples, degree=2, coeff=1.0):
    """Compute biased MMD with Polynomial kernel using scikit-learn approach"""
    target_np = target_samples.detach().cpu().numpy()
    generated_np = generated_samples.detach().cpu().numpy()
    
    def polynomial_kernel_scikit(X, Y, degree=2, coeff=1.0):
        return (np.dot(X, Y.T) + coeff) ** degree
    
    n_target, n_generated = len(target_np), len(generated_np)
    
    # Compute kernel matrices
    K_tt = polynomial_kernel_scikit(target_np, target_np, degree, coeff)
    K_gg = polynomial_kernel_scikit(generated_np, generated_np, degree, coeff)
    K_tg = polynomial_kernel_scikit(target_np, generated_np, degree, coeff)
    
    # Biased MMD² estimate (includes diagonal)
    mmd_squared = (K_tt.sum() / (n_target * n_target) + 
                   K_gg.sum() / (n_generated * n_generated) - 
                   2 * K_tg.sum() / (n_target * n_generated))
    
    return np.sqrt(max(0, mmd_squared))

def compute_mmd_comparison(target_samples, generated_samples, sigma=1.0):
    """Compute both biased and unbiased MMD with Gaussian kernel"""
    target_samples = target_samples.detach().cpu()
    generated_samples = generated_samples.detach().cpu()
    
    kernel_func = lambda x, y: gaussian_kernel(x, y, sigma)
    
    # Unbiased MMD²
    mmd_squared_unbiased = compute_mmd_squared_unbiased(target_samples, generated_samples, kernel_func)
    mmd_unbiased = torch.sqrt(torch.clamp(mmd_squared_unbiased, min=0)).item()
    
    # Biased MMD²
    mmd_squared_biased = compute_mmd_squared_biased(target_samples, generated_samples, kernel_func)
    mmd_biased = torch.sqrt(torch.clamp(mmd_squared_biased, min=0)).item()
    
    # Biased MMD using scikit-learn approach (for comparison with sinkhorn file)
    mmd_biased_scikit = compute_mmd_biased_scikit(target_samples, generated_samples, sigma=sigma)
    
    return {
        'mmd_unbiased': mmd_unbiased,
        'mmd_biased': mmd_biased,
        'mmd_biased_scikit': mmd_biased_scikit,
        'sigma': sigma
    }

def compute_polynomial_mmd_comparison(target_samples, generated_samples, degree=2, coeff=1.0):
    """Compute both biased and unbiased MMD with Polynomial kernel"""
    target_cpu = target_samples.detach().cpu()
    generated_cpu = generated_samples.detach().cpu()
    
    poly_kernel_func = lambda x, y: polynomial_kernel(x, y, degree, coeff)
    
    # Unbiased MMD²
    poly_mmd_squared_unbiased = compute_mmd_squared_unbiased(target_cpu, generated_cpu, poly_kernel_func)
    mmd_unbiased = torch.sqrt(torch.clamp(poly_mmd_squared_unbiased, min=0)).item()
    
    # Biased MMD²
    poly_mmd_squared_biased = compute_mmd_squared_biased(target_cpu, generated_cpu, poly_kernel_func)
    mmd_biased = torch.sqrt(torch.clamp(poly_mmd_squared_biased, min=0)).item()
    
    # Biased MMD using scikit-learn approach
    mmd_biased_scikit = compute_polynomial_mmd_biased_scikit(target_samples, generated_samples, degree, coeff)
    
    return {
        'mmd_unbiased': mmd_unbiased,
        'mmd_biased': mmd_biased,
        'mmd_biased_scikit': mmd_biased_scikit,
        'degree': degree,
        'coeff': coeff
    }

def compute_mmd_metrics_single_iteration(target_samples, flow_model):
    """Compute MMD metrics for both kernels with biased/unbiased comparison - single iteration"""
    with torch.no_grad():
        generated_samples = flow_model.sample(10000)
        
        # Gaussian kernel with fixed sigma
        gaussian_results = compute_mmd_comparison(target_samples, generated_samples, sigma=1.0)
        
        # Polynomial kernel with both estimators
        polynomial_results = compute_polynomial_mmd_comparison(target_samples, generated_samples, degree=2, coeff=1.0)
        
        return {
            'gaussian_mmd': gaussian_results,
            'polynomial_mmd': polynomial_results
        }

def compute_mmd_metrics_with_iterations(target_samples, flow_model, n_iterations=10):
    """Compute MMD metrics over multiple iterations and return mean/std"""
    all_gaussian_unbiased = []
    all_gaussian_biased = []
    all_gaussian_biased_scikit = []
    all_polynomial_unbiased = []
    all_polynomial_biased = []
    all_polynomial_biased_scikit = []
    
    for iteration in range(n_iterations):
        metrics = compute_mmd_metrics_single_iteration(target_samples, flow_model)
        
        all_gaussian_unbiased.append(metrics['gaussian_mmd']['mmd_unbiased'])
        all_gaussian_biased.append(metrics['gaussian_mmd']['mmd_biased'])
        all_gaussian_biased_scikit.append(metrics['gaussian_mmd']['mmd_biased_scikit'])
        all_polynomial_unbiased.append(metrics['polynomial_mmd']['mmd_unbiased'])
        all_polynomial_biased.append(metrics['polynomial_mmd']['mmd_biased'])
        all_polynomial_biased_scikit.append(metrics['polynomial_mmd']['mmd_biased_scikit'])
    
    return {
        'gaussian_mmd': {
            'mmd_unbiased_mean': np.mean(all_gaussian_unbiased),
            'mmd_unbiased_std': np.std(all_gaussian_unbiased),
            'mmd_biased_mean': np.mean(all_gaussian_biased),
            'mmd_biased_std': np.std(all_gaussian_biased),
            'mmd_biased_scikit_mean': np.mean(all_gaussian_biased_scikit),
            'mmd_biased_scikit_std': np.std(all_gaussian_biased_scikit),
            'sigma': 1.0
        },
        'polynomial_mmd': {
            'mmd_unbiased_mean': np.mean(all_polynomial_unbiased),
            'mmd_unbiased_std': np.std(all_polynomial_unbiased),
            'mmd_biased_mean': np.mean(all_polynomial_biased),
            'mmd_biased_std': np.std(all_polynomial_biased),
            'mmd_biased_scikit_mean': np.mean(all_polynomial_biased_scikit),
            'mmd_biased_scikit_std': np.std(all_polynomial_biased_scikit),
            'degree': 2,
            'coeff': 1.0
        }
    }

def evaluate_individual_flows_mmd_with_iterations(model, test_data, flow_names, dataset_name, n_iterations=10):
    """Evaluate each individual flow using MMD metrics over multiple iterations"""
    individual_metrics = {}
    
    print(f"  Evaluating individual flows over {n_iterations} iterations...")
    with torch.no_grad():
        for i, (flow, name) in enumerate(zip(model.flows, flow_names)):
            print(f"    Flow {i+1}/{len(flow_names)}: {name}")
            mmd_metrics = compute_mmd_metrics_with_iterations(test_data, flow, n_iterations)
            individual_metrics[name] = mmd_metrics
    
    return individual_metrics

def evaluate_mixture_model_mmd_with_iterations(model, test_data, n_iterations=10):
    """Evaluate mixture model using MMD metrics over multiple iterations"""
    print(f"  Evaluating mixture model over {n_iterations} iterations...")
    mixture_metrics = compute_mmd_metrics_with_iterations(test_data, model, n_iterations)
    return mixture_metrics

def evaluate_single_sequential_dataset_mmd_with_iterations(dataset_name, n_iterations=10):
    """Evaluate a single Sequential model using MMD metrics over multiple iterations"""
    
    print(f"\n{'='*50}")
    print(f"Evaluating MMD for {dataset_name.upper()} dataset ({n_iterations} iterations)")
    print(f"{'='*50}")
    
    # Create test data
    test_data = generate_data(dataset_name, n_samples=10000)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_data = test_data.to(device)
    
    # Load or train model
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, f'trained_model_{dataset_name}.pkl')
    
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
            model = saved_data['model']
    else:
        print(f"Training new model for {dataset_name}")
        model, _, _ = train_sequential_amf_vi(dataset_name, show_plots=False, save_plots=False)
        
        with open(model_path, 'wb') as f:
            pickle.dump({'model': model, 'dataset': dataset_name}, f)
    
    model = model.to(device)
    
    # Get flow names
    flow_type_map = {
        'RealNVPFlow': 'realnvp', 'MAFFlow': 'maf', 'NAFFlowSimplified': 'naf',
        'NICEFlow': 'nice', 'IAFFlow': 'iaf', 'GaussianizationFlow': 'gaussianization',
        'GlowFlow': 'glow', 'TANFlow': 'tan', 'RBIGFlow': 'rbig'
    }
    
    flow_names = [flow_type_map.get(flow.__class__.__name__, flow.__class__.__name__.lower()) 
                  for flow in model.flows]
    
    # Evaluate mixture model over multiple iterations
    mixture_metrics = evaluate_mixture_model_mmd_with_iterations(model, test_data, n_iterations)
    
    # Evaluate individual flows over multiple iterations
    individual_flow_metrics = evaluate_individual_flows_mmd_with_iterations(model, test_data, flow_names, dataset_name, n_iterations)
    
    # Get learned weights
    if hasattr(model, 'weights_trained') and model.weights_trained:
        if hasattr(model, 'log_weights'):
            learned_weights = torch.softmax(model.log_weights, dim=0).detach().cpu().numpy()
        else:
            learned_weights = model.weights.detach().cpu().numpy()
    else:
        learned_weights = np.ones(len(model.flows)) / len(model.flows)
    
    results = {
        'dataset': dataset_name,
        'mixture_metrics': mixture_metrics,
        'individual_flow_metrics': individual_flow_metrics,
        'learned_weights': learned_weights,
        'weights_trained': getattr(model, 'weights_trained', False),
        'flow_names': flow_names,
        'n_iterations': n_iterations
    }
    
    # Print results with mean ± std format
    print(f"\nMMD Results for {dataset_name} ({n_iterations} iterations):")
    print(f"Mixture Model:")
    gauss_mix = mixture_metrics['gaussian_mmd']
    poly_mix = mixture_metrics['polynomial_mmd']
    print(f"   Gaussian MMD (Unbiased): {gauss_mix['mmd_unbiased_mean']:.6f} ± {gauss_mix['mmd_unbiased_std']:.6f}")
    print(f"   Gaussian MMD (Biased):   {gauss_mix['mmd_biased_mean']:.6f} ± {gauss_mix['mmd_biased_std']:.6f}")
    print(f"   Gaussian MMD (Scikit):   {gauss_mix['mmd_biased_scikit_mean']:.6f} ± {gauss_mix['mmd_biased_scikit_std']:.6f}")
    print(f"   Polynomial MMD (Unbiased): {poly_mix['mmd_unbiased_mean']:.6f} ± {poly_mix['mmd_unbiased_std']:.6f}")
    print(f"   Polynomial MMD (Biased):   {poly_mix['mmd_biased_mean']:.6f} ± {poly_mix['mmd_biased_std']:.6f}")
    print(f"   Polynomial MMD (Scikit):   {poly_mix['mmd_biased_scikit_mean']:.6f} ± {poly_mix['mmd_biased_scikit_std']:.6f}")
    print(f"Learned Weights: {learned_weights}")
    
    return results

def comprehensive_mmd_evaluation_with_iterations(n_iterations=10):
    """Comprehensive MMD evaluation of all datasets with multiple iterations"""
    
    datasets = ['banana', 'x_shape', 'bimodal_shared', 'bimodal_different', 'multimodal', 'two_moons', 'rings']
    all_results = {}
    
    # Evaluate each dataset
    for dataset_name in datasets:
        try:
            results = evaluate_single_sequential_dataset_mmd_with_iterations(dataset_name, n_iterations)
            if results is not None:
                all_results[dataset_name] = results
        except Exception as e:
            print(f"Failed to evaluate {dataset_name}: {e}")
            continue
    
    if not all_results:
        print("No models could be evaluated.")
        return None
    
    # Create CSV data with mean ± std
    summary_data = []
    for dataset_name, results in all_results.items():
        weights_status = "Yes" if results['weights_trained'] else "No"
        mixture_metrics = results['mixture_metrics']
        
        # Add mixture model row
        gauss_mix = mixture_metrics['gaussian_mmd']
        poly_mix = mixture_metrics['polynomial_mmd']
        mixture_row = [
            dataset_name,
            'MIXTURE',
            gauss_mix['mmd_unbiased_mean'],
            gauss_mix['mmd_unbiased_std'],
            gauss_mix['mmd_biased_mean'],
            gauss_mix['mmd_biased_std'],
            gauss_mix['mmd_biased_scikit_mean'],
            gauss_mix['mmd_biased_scikit_std'],
            poly_mix['mmd_unbiased_mean'],
            poly_mix['mmd_unbiased_std'],
            poly_mix['mmd_biased_mean'],
            poly_mix['mmd_biased_std'],
            poly_mix['mmd_biased_scikit_mean'],
            poly_mix['mmd_biased_scikit_std'],
            'N/A',  # weight (not applicable for mixture)
            weights_status,
            n_iterations
        ]
        summary_data.append(mixture_row)
        
        # Add individual flow rows
        for i, flow_name in enumerate(results['flow_names']):
            individual_metrics = results['individual_flow_metrics'].get(flow_name, {})
            flow_gaussian = individual_metrics.get('gaussian_mmd', {})
            flow_polynomial = individual_metrics.get('polynomial_mmd', {})
            flow_weight = results['learned_weights'][i]
            
            flow_row = [
                dataset_name,
                flow_name.upper(),
                flow_gaussian.get('mmd_unbiased_mean', 0.0),
                flow_gaussian.get('mmd_unbiased_std', 0.0),
                flow_gaussian.get('mmd_biased_mean', 0.0),
                flow_gaussian.get('mmd_biased_std', 0.0),
                flow_gaussian.get('mmd_biased_scikit_mean', 0.0),
                flow_gaussian.get('mmd_biased_scikit_std', 0.0),
                flow_polynomial.get('mmd_unbiased_mean', 0.0),
                flow_polynomial.get('mmd_unbiased_std', 0.0),
                flow_polynomial.get('mmd_biased_mean', 0.0),
                flow_polynomial.get('mmd_biased_std', 0.0),
                flow_polynomial.get('mmd_biased_scikit_mean', 0.0),
                flow_polynomial.get('mmd_biased_scikit_std', 0.0),
                flow_weight,
                weights_status,
                n_iterations
            ]
            summary_data.append(flow_row)
    
    # Save to CSV
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    csv_filename = f'mmd_metrics_with_iterations_{n_iterations}.csv'
    with open(os.path.join(results_dir, csv_filename), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'dataset', 'model', 
            'gaussian_mmd_unbiased_mean', 'gaussian_mmd_unbiased_std',
            'gaussian_mmd_biased_mean', 'gaussian_mmd_biased_std',
            'gaussian_mmd_biased_scikit_mean', 'gaussian_mmd_biased_scikit_std',
            'polynomial_mmd_unbiased_mean', 'polynomial_mmd_unbiased_std',
            'polynomial_mmd_biased_mean', 'polynomial_mmd_biased_std',
            'polynomial_mmd_biased_scikit_mean', 'polynomial_mmd_biased_scikit_std',
            'weight', 'weights_trained', 'n_iterations'
        ])
        writer.writerows(summary_data)
        print(f'{csv_filename} successfully created')
    
    return all_results

if __name__ == "__main__":
    # Run evaluation with 10 iterations
    results = comprehensive_mmd_evaluation_with_iterations(n_iterations=10)