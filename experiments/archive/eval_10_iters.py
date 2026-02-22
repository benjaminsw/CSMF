import torch
import numpy as np
import os
import pickle
import csv
from statistics import mean, stdev
import traceback

# Import functions from the three evaluation files
# --- robust imports regardless of how the script is launched ---
# Supports: `python -m main.eval_10_iters` (package mode) and `python main/eval_10_iters.py` (script mode)
import os, sys
if __package__ in (None, ''):
    # running as a script; add project root so `main/...` and `amf_vi/...` are importable
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
try:
    # package-relative (works in module mode)
    from .threeflows_amf_vi_weights_log import SequentialAMFVI, train_sequential_amf_vi
except Exception:
    # fallback for script mode or odd runners
    try:
        from main.threeflows_amf_vi_weights_log import SequentialAMFVI, train_sequential_amf_vi
    except Exception:
        from threeflows_amf_vi_weights_log import SequentialAMFVI, train_sequential_amf_vi
# --- end robust import header ---
from data.data_generator import generate_data

# Import specific functions from each evaluation file
from .evaluate_threeflows_amf_vi_weights_log import (
    compute_cross_entropy_surrogate,
    compute_kl_divergence_metric
)
from .evaluate_threeflows_amf_vi_wasserstein import (
    compute_sliced_wasserstein_distance,
    compute_full_wasserstein_distance
)
from .evaluate_threeflows_amf_vi_mmd import (
    compute_mmd_comparison,
    compute_polynomial_mmd_comparison
)

import random
random.seed(2025)
torch.cuda.manual_seed_all(2025)


def compute_single_iteration_metrics(target_samples, flow_model, dataset_name):
    """Compute all metrics for a single iteration"""
    metrics = {}
    
    try:
        # Generate samples for this iteration - ensure matching size
        # nll = 100

        with torch.no_grad():
            target_sample_count = target_samples.shape[0]
            generated_samples = flow_model.sample(target_sample_count)
            print(f"        Target: {target_sample_count} samples, Generated: {generated_samples.shape[0]} samples")

        # 1. NLL (Negative Log-Likelihood) using cross-entropy surrogate
        try:
            # while nll > 10:
            nll = compute_cross_entropy_surrogate(target_samples, flow_model)
            metrics['nll'] = nll
        except Exception as e:
            print(f"Error computing NLL: {e}")
            traceback.print_exc()
            metrics['nll'] = None
        
        # 2. KL Divergence
        try:
            kl_div = compute_kl_divergence_metric(target_samples, flow_model, dataset_name)
            metrics['kl_divergence'] = kl_div
        except Exception as e:
            print(f"Error computing KL divergence: {e}")
            traceback.print_exc()
            metrics['kl_divergence'] = None
        
        # 3. Sliced Wasserstein Distance - REMOVED
        
        # 4. Full Wasserstein Distance
        try:
            full_wd = compute_full_wasserstein_distance(target_samples, generated_samples)
            metrics['full_wasserstein'] = full_wd
        except Exception as e:
            print(f"Error computing Full Wasserstein: {e}")
            traceback.print_exc()
            metrics['full_wasserstein'] = None
        
        # 5. Gaussian MMD (unbiased and biased)
        try:
            gaussian_mmd = compute_mmd_comparison(target_samples, generated_samples, sigma=1.0)
            metrics['gaussian_mmd_unbiased'] = gaussian_mmd['mmd_unbiased']
            metrics['gaussian_mmd_biased'] = gaussian_mmd['mmd_biased']
        except Exception as e:
            print(f"Error computing Gaussian MMD: {e}")
            traceback.print_exc()
            metrics['gaussian_mmd_unbiased'] = None
            metrics['gaussian_mmd_biased'] = None
        
        # 6. Polynomial MMD - REMOVED
        
    except Exception as e:
        print(f"Critical error in compute_single_iteration_metrics: {e}")
        traceback.print_exc()
        return None
    
    return metrics

def compute_metrics_over_iterations(target_samples, flow_model, dataset_name, n_iterations=10):
    """Compute metrics over multiple iterations and return mean/std"""
    all_metrics = {
        'nll': [],
        'kl_divergence': [],
        'full_wasserstein': [],
        'gaussian_mmd_unbiased': [],
        'gaussian_mmd_biased': []
    }
    
    print(f"    Computing metrics over {n_iterations} iterations...")
    
    for iteration in range(n_iterations):
        print(f"      Iteration {iteration + 1}/{n_iterations}")
        
        try:
            metrics = compute_single_iteration_metrics(target_samples, flow_model, dataset_name)
            if metrics is not None:
                for key in all_metrics.keys():
                    if metrics.get(key) is not None:
                        all_metrics[key].append(metrics[key])
        except Exception as e:
            print(f"Error in iteration {iteration + 1}: {e}")
            traceback.print_exc()
            continue
    
    # Calculate mean and std for each metric
    summary_metrics = {}
    for metric_name, values in all_metrics.items():
        if values:  # Only calculate if we have valid values
            summary_metrics[f'{metric_name}_mean'] = mean(values)
            summary_metrics[f'{metric_name}_std'] = stdev(values) if len(values) > 1 else 0.0
            summary_metrics[f'{metric_name}_count'] = len(values)
        else:
            summary_metrics[f'{metric_name}_mean'] = None
            summary_metrics[f'{metric_name}_std'] = None
            summary_metrics[f'{metric_name}_count'] = 0
    
    return summary_metrics

def evaluate_single_dataset_comprehensive(dataset_name, n_iterations=10, n_samples=2000):
    """Evaluate a single dataset with all metrics over multiple iterations"""
    
    print(f"\n{'='*60}")
    print(f"Comprehensive Evaluation: {dataset_name.upper()} ({n_iterations} iterations)")
    print(f"{'='*60}")
    
    try:
        # Create test data and ensure exact sample count
        test_data = generate_data(dataset_name, n_samples=n_samples)
        
        # Verify and fix sample count if needed
        if test_data.shape[0] != n_samples:
            print(f"Warning: generate_data returned {test_data.shape[0]} samples instead of 2000")
            if test_data.shape[0] > n_samples:
                # Truncate to exactly 2000 samples
                test_data = test_data[:n_samples]
                print(f"Truncated to n_samples samples")
            elif test_data.shape[0] >= n_samples-100:  # Allow slight shortage
                print(f"Using {test_data.shape[0]} samples (close enough to n_samples)")
            else:
                print(f"Error: Not enough samples generated ({test_data.shape[0]} < {n_samples})")
                return None
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_data = test_data.to(device)
        print(f"Final test data shape: {test_data.shape}")
        
        # Load or train model
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
        os.makedirs(results_dir, exist_ok=True)
        model_path = os.path.join(results_dir, f'trained_model_{dataset_name}.pkl')
        
        if os.path.exists(model_path):
            print(f"  Loading existing model from {model_path}")
            with open(model_path, 'rb') as f:
                saved_data = pickle.load(f)
                model = saved_data['model']
        else:
            print(f"  Training new model for {dataset_name}")
            model, _, _ = train_sequential_amf_vi(dataset_name, show_plots=False, save_plots=False)
            
            with open(model_path, 'wb') as f:
                pickle.dump({'model': model, 'dataset': dataset_name}, f)
        
        model = model.to(device)
        model.eval()
        
        # Get flow names
        flow_type_map = {
            'RealNVPFlow': 'realnvp', 'MAFFlow': 'maf', 'NAFFlowSimplified': 'naf',
            'NICEFlow': 'nice', 'IAFFlow': 'iaf', 'GaussianizationFlow': 'gaussianization',
            'GlowFlow': 'glow', 'TANFlow': 'tan', 'RBIGFlow': 'rbig'
        }
        
        flow_names = [flow_type_map.get(flow.__class__.__name__, flow.__class__.__name__.lower()) 
                      for flow in model.flows]
        
        # Get learned weights
        if hasattr(model, 'weights_trained') and model.weights_trained:
            if hasattr(model, 'log_weights'):
                learned_weights = torch.softmax(model.log_weights, dim=0).detach().cpu().numpy()
            else:
                learned_weights = model.weights.detach().cpu().numpy()
        else:
            learned_weights = np.ones(len(model.flows)) / len(model.flows)
        
        # Evaluate mixture model
        print(f"  Evaluating mixture model...")
        mixture_metrics = compute_metrics_over_iterations(test_data, model, dataset_name, n_iterations)
        
        # Evaluate individual flows
        print(f"  Evaluating individual flows...")
        individual_metrics = {}
        for i, (flow, name) in enumerate(zip(model.flows, flow_names)):
            print(f"    Flow {i+1}/{len(flow_names)}: {name}")
            try:
                flow_metrics = compute_metrics_over_iterations(test_data, flow, dataset_name, n_iterations)
                individual_metrics[name] = flow_metrics
            except Exception as e:
                print(f"Error evaluating flow {name}: {e}")
                traceback.print_exc()
                individual_metrics[name] = None
        
        results = {
            'dataset': dataset_name,
            'mixture_metrics': mixture_metrics,
            'individual_metrics': individual_metrics,
            'learned_weights': learned_weights,
            'weights_trained': getattr(model, 'weights_trained', False),
            'flow_names': flow_names,
            'n_iterations': n_iterations
        }
        
        # Print summary
        print(f"\n  Results Summary for {dataset_name}:")
        print(f"    Mixture Model Metrics (mean ± std):")
        for metric in ['nll', 'kl_divergence', 'full_wasserstein', 'gaussian_mmd_unbiased']:
            mean_val = mixture_metrics.get(f'{metric}_mean')
            std_val = mixture_metrics.get(f'{metric}_std')
            count_val = mixture_metrics.get(f'{metric}_count', 0)
            if mean_val is not None:
                print(f"      {metric}: {mean_val:.6f} ± {std_val:.6f} (n={count_val})")
            else:
                print(f"      {metric}: FAILED")
        
        print(f"    Learned Weights: {learned_weights}")
        print(f"    Weights Trained: {results['weights_trained']}")
        
        return results
        
    except Exception as e:
        print(f"Critical error evaluating {dataset_name}: {e}")
        traceback.print_exc()
        return None

def comprehensive_evaluation(n_iterations=100):
    """Run comprehensive evaluation on all datasets"""
    
    datasets = [
        'banana',
        'x_shape', 'bimodal_shared',
        'bimodal_different',
        'multimodal',
        'two_moons',
        'rings',
        "BLR", "BPR",
        "Weibull",
        "multimodal-5"
    ]
    # datasets = ['multimodal']
    all_results = {}
    
    print(f"Starting Comprehensive Evaluation ({n_iterations} iterations per metric)")
    print(f"Datasets: {datasets}")
    
    # Evaluate each dataset
    for dataset_name in datasets:
        try:
            results = evaluate_single_dataset_comprehensive(dataset_name, n_iterations)
            if results is not None:
                all_results[dataset_name] = results
        except Exception as e:
            print(f"Failed to evaluate {dataset_name}: {e}")
            traceback.print_exc()
            continue
    
    if not all_results:
        print("No datasets could be evaluated successfully.")
        return None
    
    # Create comprehensive CSV
    print(f"\nCreating comprehensive results CSV...")
    summary_data = []
    
    for dataset_name, results in all_results.items():
        weights_status = "Yes" if results['weights_trained'] else "No"
        mixture_metrics = results['mixture_metrics']
        
        # Add mixture model row
        mixture_row = [
            dataset_name,
            'MIXTURE',
            mixture_metrics.get('nll_mean'),
            mixture_metrics.get('nll_std'),
            mixture_metrics.get('kl_divergence_mean'),
            mixture_metrics.get('kl_divergence_std'),
            mixture_metrics.get('full_wasserstein_mean'),
            mixture_metrics.get('full_wasserstein_std'),
            mixture_metrics.get('gaussian_mmd_unbiased_mean'),
            mixture_metrics.get('gaussian_mmd_unbiased_std'),
            mixture_metrics.get('gaussian_mmd_biased_mean'),
            mixture_metrics.get('gaussian_mmd_biased_std'),
            'N/A',  # weight (not applicable for mixture)
            weights_status,
            n_iterations
        ]
        summary_data.append(mixture_row)
        
        # Add individual flow rows
        for i, flow_name in enumerate(results['flow_names']):
            individual_metrics = results['individual_metrics'].get(flow_name, {})
            flow_weight = results['learned_weights'][i]
            
            if individual_metrics is not None:
                flow_row = [
                    dataset_name,
                    flow_name.upper(),
                    individual_metrics.get('nll_mean'),
                    individual_metrics.get('nll_std'),
                    individual_metrics.get('kl_divergence_mean'),
                    individual_metrics.get('kl_divergence_std'),
                    individual_metrics.get('full_wasserstein_mean'),
                    individual_metrics.get('full_wasserstein_std'),
                    individual_metrics.get('gaussian_mmd_unbiased_mean'),
                    individual_metrics.get('gaussian_mmd_unbiased_std'),
                    individual_metrics.get('gaussian_mmd_biased_mean'),
                    individual_metrics.get('gaussian_mmd_biased_std'),
                    flow_weight,
                    weights_status,
                    n_iterations
                ]
            else:
                # Fill with None values if flow evaluation failed
                flow_row = [dataset_name, flow_name.upper()] + [None] * 11 + [flow_weight, weights_status, n_iterations]
            
            summary_data.append(flow_row)
    
    # Save comprehensive CSV
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    csv_filename = f'comprehensive_evaluation_{n_iterations}_iterations.csv'
    csv_path = os.path.join(results_dir, csv_filename)
    
    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'dataset', 'model', 
                'nll_mean', 'nll_std',
                'kl_divergence_mean', 'kl_divergence_std',
                'full_wasserstein_mean', 'full_wasserstein_std',
                'gaussian_mmd_unbiased_mean', 'gaussian_mmd_unbiased_std',
                'gaussian_mmd_biased_mean', 'gaussian_mmd_biased_std',
                'weight', 'weights_trained', 'n_iterations'
            ])
            writer.writerows(summary_data)
        
        print(f"✅ {csv_filename} successfully created at {csv_path}")
        
    except Exception as e:
        print(f"Error saving CSV: {e}")
        traceback.print_exc()
    
    print(f"\nEvaluation completed! Processed {len(all_results)} datasets successfully.")
    return all_results

if __name__ == "__main__":
    # Run comprehensive evaluation with 10 iterations
    print("Starting Comprehensive Evaluation Script")
    print("=" * 80)
    
    try:
        results = comprehensive_evaluation(n_iterations=10)
        if results:
            print("\n🎉 Comprehensive evaluation completed successfully!")
        else:
            print("\n❌ Comprehensive evaluation failed - no results obtained.")
    except Exception as e:
        print(f"\n💥 Critical error in main execution: {e}")
        traceback.print_exc()
