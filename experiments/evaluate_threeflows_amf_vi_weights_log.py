import torch
import torch.nn.functional as F
import numpy as np
from .threeflows_amf_vi_weights_log import SequentialAMFVI, train_sequential_amf_vi
from data.data_generator import generate_data
from amf_vi.kde_kl_divergence import compute_kde_kl_divergence 
import os
import pickle
import csv

# Set seed for reproducible experiments
torch.manual_seed(2025)
np.random.seed(2025)

def compute_cross_entropy_surrogate(target_samples, flow_model):
    """Compute cross-entropy surrogate for KL divergence: -E_p[log q(x)]"""
    with torch.no_grad():
        log_q = flow_model.log_prob(target_samples)
        return -log_q.mean().item()

def compute_percentage_improvement(target_samples, mixture_model, baseline_flow):
    """Compute percentage improvement of mixture model over a single baseline flow."""
    mixture_cross_entropy = compute_cross_entropy_surrogate(target_samples, mixture_model)
    baseline_cross_entropy = compute_cross_entropy_surrogate(target_samples, baseline_flow)
    
    if baseline_cross_entropy == 0:
        return 0.0
    
    improvement = ((baseline_cross_entropy - mixture_cross_entropy) / baseline_cross_entropy) * 100
    return improvement

def compute_kl_divergence_metric(target_samples, flow_model, dataset_name):
    """
    Compute KL divergence using KDE-based approach as the default method.
    Falls back to histogram method if KDE fails.
    """
    with torch.no_grad():
        try:
            # Generate samples from the flow model
            generated_samples = flow_model.sample(2000)
            
            # Use KDE-based KL divergence as the primary method
            kl_divergence = compute_kde_kl_divergence(
                target_samples=target_samples,
                generated_samples=generated_samples,
                grid_resolution=100,
                bandwidth_method='scott',
                epsilon=1e-10
            )
            
            return kl_divergence
            
        except Exception as e:
            print(f"Warning: KDE-based KL divergence failed for {dataset_name}: {e}")
            print("Falling back to histogram-based method...")
            
            # Fallback to histogram method
            generated_samples = flow_model.sample(2000)
            return compute_kl_divergence_histogram(target_samples, generated_samples)

'''
def compute_kl_divergence_histogram(target_samples, generated_samples):
    """Compute KL divergence between target and generated samples using histogram method."""
    target_np = target_samples.detach().cpu().numpy()
    generated_np = generated_samples.detach().cpu().numpy()
    
    # Simple histogram-based KL divergence estimation
    bins = 50
    
    # Get data range
    x_min = min(target_np[:, 0].min(), generated_np[:, 0].min())
    x_max = max(target_np[:, 0].max(), generated_np[:, 0].max())
    y_min = min(target_np[:, 1].min(), generated_np[:, 1].min())
    y_max = max(target_np[:, 1].max(), generated_np[:, 1].max())
    
    # Create histograms
    hist_target, _, _ = np.histogram2d(target_np[:, 0], target_np[:, 1], 
                                       bins=bins, range=[[x_min, x_max], [y_min, y_max]])
    hist_generated, _, _ = np.histogram2d(generated_np[:, 0], generated_np[:, 1], 
                                          bins=bins, range=[[x_min, x_max], [y_min, y_max]])
    
    # Normalize to probabilities
    hist_target = hist_target / hist_target.sum()
    hist_generated = hist_generated / hist_generated.sum()
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    hist_target = hist_target + epsilon
    hist_generated = hist_generated + epsilon
    
    # Compute KL divergence
    kl_div = np.sum(hist_target * np.log(hist_target / hist_generated))
    
    return kl_div
'''

def evaluate_individual_flows(model, test_data, flow_names, dataset_name):
    """Evaluate each individual flow against test data."""
    individual_metrics = {}
    
    with torch.no_grad():
        for i, (flow, name) in enumerate(zip(model.flows, flow_names)):
            # Compute metrics: KL divergence and cross-entropy surrogate only
            kl_divergence = compute_kl_divergence_metric(test_data, flow, dataset_name)
            cross_entropy = compute_cross_entropy_surrogate(test_data, flow)
            
            # Store metrics
            individual_metrics[name] = {
                'kl_divergence': kl_divergence,
                'cross_entropy_surrogate': cross_entropy,
            }
    
    return individual_metrics

def evaluate_single_sequential_dataset(dataset_name):
    """Evaluate or train+evaluate a single Sequential model."""
    
    print(f"\n{'='*50}")
    print(f"Evaluating Sequential {dataset_name.upper()} dataset")
    print(f"{'='*50}")
    
    # Create test data
    test_data = generate_data(dataset_name, n_samples=2000)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_data = test_data.to(device)
    
    # Check if model exists, if not train it
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, f'trained_model_{dataset_name}.pkl')
    
    if os.path.exists(model_path):
        print(f"Loading existing Sequential model from {model_path}")
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
            model = saved_data['model']
    else:
        print(f"Training new Sequential model for {dataset_name}")
        model, _, _ = train_sequential_amf_vi(dataset_name, show_plots=False, save_plots=False)
        
        # Save the trained model
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'dataset': dataset_name
            }, f)
    
    model = model.to(device)
    
    # Dynamically determine flow names based on actual flows in the model
    flow_type_map = {
        'RealNVPFlow': 'realnvp',
        'MAFFlow': 'maf', 
        'NAFFlowSimplified': 'naf',
        'NICEFlow': 'nice',
        'IAFFlow': 'iaf',
        'GaussianizationFlow': 'gaussianization',
        'GlowFlow': 'glow',
        'TANFlow': 'tan',
        'RBIGFlow': 'rbig'
    }
    
    flow_names = []
    for flow in model.flows:
        flow_class_name = flow.__class__.__name__
        flow_name = flow_type_map.get(flow_class_name, flow_class_name.lower())
        flow_names.append(flow_name)
    
    # Compute metrics: KL divergence, cross-entropy surrogate, and percentage improvements
    model.eval()
    kl_divergence = compute_kl_divergence_metric(test_data, model, dataset_name)
    cross_entropy = compute_cross_entropy_surrogate(test_data, model)
    
    # Compute percentage improvements for all flows dynamically
    percentage_improvements = {}
    for i, name in enumerate(flow_names):
        improvement = compute_percentage_improvement(test_data, model, model.flows[i])
        percentage_improvements[f'vs_{name}'] = improvement
    
    # Evaluate individual flows
    individual_flow_metrics = evaluate_individual_flows(model, test_data, flow_names, dataset_name)
    
    # Get learned weights (handle both log_weights and weights attributes)
    if model.weights_trained:
        print('*** learned weights is extracted ***')
        if hasattr(model, 'log_weights'):
            learned_weights = F.softmax(model.log_weights, dim=0).detach().cpu().numpy()
        else:
            learned_weights = model.weights.detach().cpu().numpy()
    else:
        learned_weights = np.ones(len(model.flows)) / len(model.flows)
    
    results = {
        'dataset': dataset_name,
        'kl_divergence': kl_divergence,
        'cross_entropy_surrogate': cross_entropy,
        'percentage_improvements': percentage_improvements,
        'individual_flow_metrics': individual_flow_metrics,
        'learned_weights': learned_weights,
        'weights_trained': model.weights_trained,
        'flow_names': flow_names
    }
    
    print(f"📊 Overall Sequential Mixture Results for {dataset_name}:")
    print(f"   KL Divergence: {kl_divergence:.3f}")
    print(f"   Cross-Entropy Surrogate: {cross_entropy:.3f}")
    for name, improvement in percentage_improvements.items():
        print(f"   % Improvement {name}: {improvement:.1f}%")
    print(f"   Learned Weights: {learned_weights}")
    print(f"   Weights Trained: {model.weights_trained}")
    
    return results

def comprehensive_sequential_evaluation():
    """Comprehensive evaluation of all Sequential AMF-VI models."""
    
    # Define datasets to evaluate
    datasets = ['banana', 'x_shape', 'bimodal_shared', 'bimodal_different', 'multimodal', 'two_moons', 'rings']
    
    all_results = {}
    
    # Evaluate each dataset
    for dataset_name in datasets:
        try:
            results = evaluate_single_sequential_dataset(dataset_name)
            if results is not None:
                all_results[dataset_name] = results
        except Exception as e:
            print(f"⚠ Failed to evaluate {dataset_name}: {e}")
            continue
    
    if not all_results:
        print("⚠ No Sequential models could be trained/evaluated.")
        return None
    
    # Create summary data and save to CSV
    summary_data = []
    for dataset_name, results in all_results.items():
        weights_status = "Yes" if results['weights_trained'] else "No"
        
        # Add rows for all flows dynamically with individual flow metrics
        for i, flow_name in enumerate(results['flow_names']):
            improvement_key = f'vs_{flow_name}'
            improvement = results['percentage_improvements'].get(improvement_key, 0.0)
            
            # Get individual flow metrics
            individual_metrics = results['individual_flow_metrics'].get(flow_name, {})
            flow_kl = individual_metrics.get('kl_divergence', 0.0)
            flow_ce = individual_metrics.get('cross_entropy_surrogate', 0.0)
            flow_weight = results['learned_weights'][i] if results['weights_trained'] else 1/len(results['learned_weights'])
            
            summary_data.append([
                dataset_name,
                results['kl_divergence'],
                results['cross_entropy_surrogate'],
                flow_name.upper(),
                flow_kl,
                flow_ce,
                flow_weight,
                improvement,
                weights_status
            ])
    
    # Save mixture metrics to CSV
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'sequential_comprehensive_metrics.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'mixture_kl_divergence', 'mixture_cross_entropy_surrogate', 'flow', 'flow_kl_divergence', 'flow_cross_entropy_surrogate', 'flow_weight', 'percentage_improvement', 'weights_trained'])
        writer.writerows(summary_data)
        print('sequential_comprehensive_metrics.csv is successfully created')
    
    return all_results

if __name__ == "__main__":
    results = comprehensive_sequential_evaluation()