import torch
import numpy as np
import ot
from .threeflows_amf_vi_weights_log import SequentialAMFVI, train_sequential_amf_vi
from data.data_generator import generate_data
import os
import pickle
import csv

# Set seed for reproducible experiments
torch.manual_seed(2025)
np.random.seed(2025)

def compute_sliced_wasserstein_distance(target_samples, generated_samples, n_projections=100):
    """Compute Sliced Wasserstein Distance using random projections."""
    target_np = target_samples.detach().cpu().numpy()
    generated_np = generated_samples.detach().cpu().numpy()
    
    # Generate random unit vectors for projections
    directions = np.random.randn(n_projections, target_np.shape[1])
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    
    wasserstein_distances = []
    for direction in directions:
        # Project samples onto direction
        proj_target = target_np @ direction
        proj_generated = generated_np @ direction
        
        # Sort projections
        proj_target_sorted = np.sort(proj_target)
        proj_generated_sorted = np.sort(proj_generated)
        
        # Compute 1D Wasserstein (L1 distance between sorted arrays)
        wd_1d = np.mean(np.abs(proj_target_sorted - proj_generated_sorted))
        wasserstein_distances.append(wd_1d)
    
    return np.mean(wasserstein_distances)

def compute_full_wasserstein_distance(target_samples, generated_samples):
    """Compute full 2-Wasserstein distance using optimal transport."""
    target_np = target_samples.detach().cpu().numpy()
    generated_np = generated_samples.detach().cpu().numpy()
    
    # Create uniform distributions
    a = np.ones(len(target_np)) / len(target_np)
    b = np.ones(len(generated_np)) / len(generated_np)
    
    # Compute cost matrix (squared L2 distances)
    cost_matrix = ot.dist(target_np, generated_np, metric='sqeuclidean')
    
    # Compute optimal transport cost
    wasserstein_dist = ot.emd2(a, b, cost_matrix)
    
    return np.sqrt(wasserstein_dist)  # Return 2-Wasserstein distance

def compute_wasserstein_distance(target_samples, flow_model, metric_type='sliced'):
    """Compute Wasserstein distance between target and flow-generated samples."""
    with torch.no_grad():
        generated_samples = flow_model.sample(2000)
        
        if metric_type == 'sliced':
            return compute_sliced_wasserstein_distance(target_samples, generated_samples)
        elif metric_type == 'full':
            return compute_full_wasserstein_distance(target_samples, generated_samples)
        else:
            raise ValueError("metric_type must be 'sliced' or 'full'")

def evaluate_individual_flows_wasserstein(model, test_data, flow_names, dataset_name):
    """Evaluate each individual flow using both Wasserstein distances."""
    individual_metrics = {}
    
    with torch.no_grad():
        for i, (flow, name) in enumerate(zip(model.flows, flow_names)):
            # Compute both sliced and full Wasserstein distances
            sliced_wd = compute_wasserstein_distance(test_data, flow, 'sliced')
            full_wd = compute_wasserstein_distance(test_data, flow, 'full')
            
            individual_metrics[name] = {
                'sliced_wasserstein': sliced_wd,
                'full_wasserstein': full_wd,
            }
    
    return individual_metrics

def evaluate_single_sequential_dataset_wasserstein(dataset_name):
    """Evaluate a single Sequential model using Wasserstein distances."""
    
    print(f"\n{'='*50}")
    print(f"Evaluating Wasserstein for {dataset_name.upper()} dataset")
    print(f"{'='*50}")
    
    # Create test data
    test_data = generate_data(dataset_name, n_samples=2000)
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
    
    # Compute mixture model Wasserstein distances
    model.eval()
    mixture_sliced_wd = compute_wasserstein_distance(test_data, model, 'sliced')
    mixture_full_wd = compute_wasserstein_distance(test_data, model, 'full')
    
    # Evaluate individual flows
    individual_flow_metrics = evaluate_individual_flows_wasserstein(model, test_data, flow_names, dataset_name)
    
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
        'mixture_sliced_wasserstein': mixture_sliced_wd,
        'mixture_full_wasserstein': mixture_full_wd,
        'individual_flow_metrics': individual_flow_metrics,
        'learned_weights': learned_weights,
        'weights_trained': getattr(model, 'weights_trained', False),
        'flow_names': flow_names
    }
    
    print(f"📊 Wasserstein Results for {dataset_name}:")
    print(f"   Mixture Sliced Wasserstein: {mixture_sliced_wd:.4f}")
    print(f"   Mixture Full Wasserstein: {mixture_full_wd:.4f}")
    print(f"   Learned Weights: {learned_weights}")
    
    return results

def comprehensive_wasserstein_evaluation():
    """Comprehensive Wasserstein evaluation of all datasets."""
    
    datasets = ['banana', 'x_shape', 'bimodal_shared', 'bimodal_different', 'multimodal', 'two_moons', 'rings']
    all_results = {}
    
    # Evaluate each dataset
    for dataset_name in datasets:
        try:
            results = evaluate_single_sequential_dataset_wasserstein(dataset_name)
            if results is not None:
                all_results[dataset_name] = results
        except Exception as e:
            print(f"❌ Failed to evaluate {dataset_name}: {e}")
            continue
    
    if not all_results:
        print("❌ No models could be evaluated.")
        return None
    
    # Create CSV data
    summary_data = []
    for dataset_name, results in all_results.items():
        weights_status = "Yes" if results['weights_trained'] else "No"
        
        for i, flow_name in enumerate(results['flow_names']):
            individual_metrics = results['individual_flow_metrics'].get(flow_name, {})
            flow_sliced_wd = individual_metrics.get('sliced_wasserstein', 0.0)
            flow_full_wd = individual_metrics.get('full_wasserstein', 0.0)
            flow_weight = results['learned_weights'][i]
            
            summary_data.append([
                dataset_name,
                results['mixture_sliced_wasserstein'],
                results['mixture_full_wasserstein'],
                flow_name.upper(),
                flow_sliced_wd,
                flow_full_wd,
                flow_weight,
                weights_status
            ])
    
    # Save to CSV
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'wasserstein_comprehensive_metrics.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'mixture_sliced_wasserstein', 'mixture_full_wasserstein', 
                        'flow', 'flow_sliced_wasserstein', 'flow_full_wasserstein', 
                        'flow_weight', 'weights_trained'])
        writer.writerows(summary_data)
        print('✅ wasserstein_comprehensive_metrics.csv successfully created')
    
    return all_results

if __name__ == "__main__":
    results = comprehensive_wasserstein_evaluation()