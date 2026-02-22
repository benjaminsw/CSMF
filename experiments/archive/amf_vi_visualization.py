#!/usr/bin/env python3
"""
AMF-VI Posterior Visualization Script

This script creates comprehensive visualizations for individual datasets 
(banana, x_shape, bimodal_shared, bimodal_different, multimodal, two_moons, rings)
showing the true data distribution and samples from individual flows (RealNVP, MAF, RBIG) 
as well as the combined AMF-VI mixture model.

Each dataset gets its own separate PNG file.

Usage:
    python visualize_amf_vi_posteriors.py

Requirements:
    - Saved AMF-VI model files in the results/ directory
    - All the flow implementations and data generation functions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from pathlib import Path
from data.data_generator import generate_data

# Import the SequentialAMFVI class so pickle can find it
try:
    from threeflows_amf_vi_weights_log import SequentialAMFVI
    print("✅ Successfully imported SequentialAMFVI class")
except ImportError as e:
    print(f"⚠️ Could not import SequentialAMFVI: {e}")
    print("Make sure threeflows_amf_vi_weights_log.py is in your Python path")
    # You might need to add the path:
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)  # Go up one level if in main/ folder
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    try:
        from threeflows_amf_vi_weights_log import SequentialAMFVI
        print("✅ Successfully imported SequentialAMFVI after adding path")
    except ImportError:
        print("❌ Still cannot import SequentialAMFVI. Please check file paths.")

# Set seed for reproducible visualizations
torch.manual_seed(2025)
np.random.seed(2025)

def load_trained_model(dataset_name, results_dir='results'):
    """Load a trained AMF-VI model from pickle file."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try current directory first, then parent directory
    possible_paths = [
        os.path.join(script_dir, results_dir, f'trained_model_{dataset_name}.pkl'),  # ./results/
        os.path.join(script_dir, '..', results_dir, f'trained_model_{dataset_name}.pkl'),  # ../results/
    ]
    
    for model_path in possible_paths:
        if os.path.exists(model_path):
            break
    else:
        # If neither path exists, use the first one for error reporting
        model_path = possible_paths[0]
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
        return saved_data['model'], saved_data.get('flow_losses', []), saved_data.get('weight_losses', [])
    except Exception as e:
        print(f"Error loading model: {e}")
        print("This usually means the SequentialAMFVI class is not available in the current environment.")
        print("Try running the script from the same directory where you trained the models.")
        
        # Try to provide a more helpful error message
        if "SequentialAMFVI" in str(e):
            print("\n💡 Quick fix:")
            print("1. Make sure threeflows_amf_vi_weights_log.py is in the same directory")
            print("2. Or run: 'from threeflows_amf_vi_weights_log import SequentialAMFVI' before loading")
        
        raise e

def generate_samples_from_model(model, n_samples=1000):
    """Generate samples from trained model and individual flows."""
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # Generate samples from the mixture model
        mixture_samples = model.sample(n_samples)
        
        # Generate samples from individual flows
        individual_samples = {}
        flow_types = []
        
        for i, flow in enumerate(model.flows):
            # Determine flow type from class name
            flow_name = flow.__class__.__name__.lower()
            if 'realnvp' in flow_name:
                flow_type = 'realnvp'
            elif 'maf' in flow_name:
                flow_type = 'maf'
            elif 'rbig' in flow_name:
                flow_type = 'rbig'
            elif 'iaf' in flow_name:
                flow_type = 'iaf'
            elif 'gaussianization' in flow_name:
                flow_type = 'gaussianization'
            elif 'naf' in flow_name:
                flow_type = 'naf'
            elif 'glow' in flow_name:
                flow_type = 'glow'
            elif 'nice' in flow_name:
                flow_type = 'nice'
            elif 'tan' in flow_name:
                flow_type = 'tan'
            else:
                flow_type = f'flow_{i}'
            
            flow_types.append(flow_type)
            
            try:
                flow.eval()
                samples = flow.sample(n_samples)
                individual_samples[flow_type] = samples
            except Exception as e:
                print(f"Warning: Could not sample from {flow_type}: {e}")
                # Create dummy samples as fallback
                individual_samples[flow_type] = torch.randn(n_samples, 2, device=device)
    
    return mixture_samples, individual_samples, flow_types

def create_single_dataset_visualization(dataset_name, 
                                      n_samples=2000, 
                                      figsize=(20, 4),
                                      save_path=None,
                                      show_plot=True):
    """
    Create a visualization for a single dataset showing all flow comparisons.
    
    Args:
        dataset_name: Name of the dataset to visualize
        n_samples: Number of samples to generate for visualization
        figsize: Figure size tuple (width, height)
        save_path: Path to save the figure (optional)
        show_plot: Whether to display the plot
    """
    
    n_cols = 5  # True, RealNVP, MAF, RBIG, AMF-VI
    
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    
    # Define colors for consistency
    colors = {
        'true': '#1f77b4',      # Blue
        'realnvp': '#ff7f0e',   # Orange  
        'maf': '#2ca02c',       # Green
        'rbig': '#d62728',      # Red
        'amf_vi': '#9467bd',    # Purple
    }
    
    # Column titles
    col_titles = ['True Data', 'RealNVP', 'MAF', 'RBIG', 'AMF-VI']
    
    print(f"Processing {dataset_name}...")
    
    try:
        # Load trained model
        model, flow_losses, weight_losses = load_trained_model(dataset_name)
        device = next(model.parameters()).device
        
        # Generate true data
        true_data = generate_data(dataset_name, n_samples=n_samples)
        true_data = true_data.to(device)
        
        # Generate model samples
        mixture_samples, individual_samples, flow_types = generate_samples_from_model(model, n_samples)
        
        # Plot true data
        true_np = true_data.cpu().numpy()
        axes[0].scatter(true_np[:, 0], true_np[:, 1], 
                       alpha=0.6, c=colors['true'], s=15, edgecolors='none')
        axes[0].set_title(col_titles[0], fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Plot individual flows (columns 1-3: RealNVP, MAF, RBIG)
        expected_flows = ['realnvp', 'maf', 'rbig']
        for col, flow_name in enumerate(expected_flows, 1):
            if flow_name in individual_samples:
                samples_np = individual_samples[flow_name].cpu().numpy()
                axes[col].scatter(samples_np[:, 0], samples_np[:, 1],
                                 alpha=0.6, c=colors[flow_name], s=15, edgecolors='none')
            else:
                # Flow not available, show empty plot with message
                axes[col].text(0.5, 0.5, f'{flow_name.upper()}\nNot Available', 
                               transform=axes[col].transAxes, 
                               ha='center', va='center', fontsize=10, 
                               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
            
            axes[col].set_title(col_titles[col], fontsize=14, fontweight='bold')
            axes[col].grid(True, alpha=0.3)
        
        # Plot AMF-VI mixture (column 4)
        mixture_np = mixture_samples.cpu().numpy()
        axes[4].scatter(mixture_np[:, 0], mixture_np[:, 1],
                       alpha=0.6, c=colors['amf_vi'], s=15, edgecolors='none')
        axes[4].set_title(col_titles[4], fontsize=14, fontweight='bold')
        axes[4].grid(True, alpha=0.3)
        
        # Set consistent axis limits for all plots
        all_data = [true_np]
        for samples in individual_samples.values():
            all_data.append(samples.cpu().numpy())
        all_data.append(mixture_np)
        
        all_data_combined = np.vstack(all_data)
        x_min, x_max = np.percentile(all_data_combined[:, 0], [2, 98])
        y_min, y_max = np.percentile(all_data_combined[:, 1], [2, 98])
        
        # Add some padding
        x_pad = (x_max - x_min) * 0.1
        y_pad = (y_max - y_min) * 0.1
        
        for col in range(n_cols):
            axes[col].set_xlim(x_min - x_pad, x_max + x_pad)
            axes[col].set_ylim(y_min - y_pad, y_max + y_pad)
            axes[col].set_aspect('equal', adjustable='box')
            
            # Remove tick labels for cleaner look (except leftmost for y-axis and bottom for x-axis)
            if col > 0:
                axes[col].set_yticklabels([])
        
        # Print learned weights information
        if hasattr(model, 'weights') and model.weights_trained:
            weights = model.weights.detach().cpu().numpy()
            print(f"  {dataset_name} learned weights: {dict(zip(flow_types, weights))}")
        
        # Add dataset title as suptitle
        dataset_title = dataset_name.replace('_', ' ').title()
        #fig.suptitle(f'{dataset_title} - AMF-VI Posterior Comparison', 
        #             fontsize=16, fontweight='bold', y=1.02)
        
    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")
        # Fill all plots with error message
        for col in range(n_cols):
            axes[col].text(0.5, 0.5, f'Error loading\n{dataset_name}', 
                           transform=axes[col].transAxes, 
                           ha='center', va='center', fontsize=10,
                           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
            axes[col].set_title(col_titles[col], fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    # plt.subplots_adjust(top=0.85, wspace=0.05)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1, wspace=0.05)
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5, facecolor='white')
        print(f"✅ Figure saved to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig

def create_detailed_analysis_plot(datasets=['banana', 'x_shape', 'bimodal_shared', 'bimodal_different', 'multimodal', 'two_moons', 'rings'],
                                save_path=None,
                                show_plot=True):
    """Create a detailed analysis plot with statistical comparisons."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Statistical comparison data
    comparison_data = {'datasets': [], 'realnvp_kl': [], 'maf_kl': [], 'rbig_kl': [], 'amfvi_kl': []}
    weight_data = {'datasets': [], 'realnvp_weight': [], 'maf_weight': [], 'rbig_weight': []}
    
    for dataset_name in datasets:
        try:
            model, flow_losses, weight_losses = load_trained_model(dataset_name)
            device = next(model.parameters()).device
            
            # Generate true data for KL divergence estimation
            true_data = generate_data(dataset_name, n_samples=2000).to(device)
            
            # Get learned weights
            if hasattr(model, 'weights') and model.weights_trained:
                weights = model.weights.detach().cpu().numpy()
                comparison_data['datasets'].append(dataset_name.replace('_', '-').title())
                
                # Simple KL divergence approximation (placeholder - would need proper implementation)
                # For now, use negative log likelihood as proxy
                model.eval()
                with torch.no_grad():
                    mixture_log_prob = model.log_prob(true_data).mean().item()
                    
                    # Individual flow log probs
                    flow_log_probs = []
                    for flow in model.flows:
                        try:
                            log_prob = flow.log_prob(true_data).mean().item()
                            flow_log_probs.append(-log_prob)  # Convert to "divergence" proxy
                        except:
                            flow_log_probs.append(float('inf'))
                    
                    comparison_data['realnvp_kl'].append(flow_log_probs[0] if len(flow_log_probs) > 0 else 0)
                    comparison_data['maf_kl'].append(flow_log_probs[1] if len(flow_log_probs) > 1 else 0)
                    comparison_data['rbig_kl'].append(flow_log_probs[2] if len(flow_log_probs) > 2 else 0)
                    comparison_data['amfvi_kl'].append(-mixture_log_prob)
                
                # Store weights
                weight_data['datasets'].append(dataset_name.replace('_', '-').title())
                weight_data['realnvp_weight'].append(weights[0] if len(weights) > 0 else 0)
                weight_data['maf_weight'].append(weights[1] if len(weights) > 1 else 0)
                weight_data['rbig_weight'].append(weights[2] if len(weights) > 2 else 0)
        
        except Exception as e:
            print(f"Error in detailed analysis for {dataset_name}: {e}")
            continue
    
    # Plot 1: Performance comparison (negative log likelihood)
    if comparison_data['datasets']:
        x_pos = np.arange(len(comparison_data['datasets']))
        width = 0.2
        
        axes[0, 0].bar(x_pos - 1.5*width, comparison_data['realnvp_kl'], width, label='RealNVP', color='#ff7f0e', alpha=0.8)
        axes[0, 0].bar(x_pos - 0.5*width, comparison_data['maf_kl'], width, label='MAF', color='#2ca02c', alpha=0.8)
        axes[0, 0].bar(x_pos + 0.5*width, comparison_data['rbig_kl'], width, label='RBIG', color='#d62728', alpha=0.8)
        axes[0, 0].bar(x_pos + 1.5*width, comparison_data['amfvi_kl'], width, label='AMF-VI', color='#9467bd', alpha=0.8)
        
        axes[0, 0].set_xlabel('Datasets')
        axes[0, 0].set_ylabel('Negative Log Likelihood')
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(comparison_data['datasets'])
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Learned weights
    if weight_data['datasets']:
        x_pos = np.arange(len(weight_data['datasets']))
        width = 0.25
        
        axes[0, 1].bar(x_pos - width, weight_data['realnvp_weight'], width, label='RealNVP', color='#ff7f0e', alpha=0.8)
        axes[0, 1].bar(x_pos, weight_data['maf_weight'], width, label='MAF', color='#2ca02c', alpha=0.8)
        axes[0, 1].bar(x_pos + width, weight_data['rbig_weight'], width, label='RBIG', color='#d62728', alpha=0.8)
        
        axes[0, 1].set_xlabel('Datasets')
        axes[0, 1].set_ylabel('Learned Weight')
        axes[0, 1].set_title('Learned Flow Weights')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(weight_data['datasets'])
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Training loss curves (example from first dataset)
    try:
        model, flow_losses, weight_losses = load_trained_model(datasets[0])
        if flow_losses and weight_losses:
            colors = ['#ff7f0e', '#2ca02c', '#d62728']  # RealNVP, MAF, RBIG
            flow_names = ['RealNVP', 'MAF', 'RBIG']
            
            for i, (losses, name, color) in enumerate(zip(flow_losses, flow_names, colors)):
                if losses and not all(np.isnan(losses)):
                    axes[1, 0].plot(losses, label=f'{name} Flow', color=color, alpha=0.7)
            
            if weight_losses:
                axes[1, 0].plot(weight_losses, label='Weight Learning', color='#9467bd', linewidth=2)
            
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title(f'Training Curves ({datasets[0].replace("_", "-").title()})')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')
    except:
        axes[1, 0].text(0.5, 0.5, 'Training curves\nnot available', 
                       transform=axes[1, 0].transAxes, ha='center', va='center')
    
    # Plot 4: Model complexity comparison
    complexity_data = {'Flow Type': [], 'Parameters': []}
    try:
        model, _, _ = load_trained_model(datasets[0])  # Use first dataset as example
        flow_names = ['RealNVP', 'MAF', 'RBIG']
        
        for i, (flow, name) in enumerate(zip(model.flows, flow_names)):
            n_params = sum(p.numel() for p in flow.parameters())
            complexity_data['Flow Type'].append(name)
            complexity_data['Parameters'].append(n_params)
        
        if complexity_data['Flow Type']:
            bars = axes[1, 1].bar(complexity_data['Flow Type'], complexity_data['Parameters'], 
                                 color=['#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
            axes[1, 1].set_xlabel('Flow Type')
            axes[1, 1].set_ylabel('Number of Parameters')
            axes[1, 1].set_title('Model Complexity Comparison')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, complexity_data['Parameters']):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + max(complexity_data['Parameters'])*0.01,
                               f'{value:,}', ha='center', va='bottom', fontsize=9)
    
    except Exception as e:
        print(f"Error in complexity analysis: {e}")
        axes[1, 1].text(0.5, 0.5, 'Complexity data\nnot available', 
                       transform=axes[1, 1].transAxes, ha='center', va='center')
    
    plt.tight_layout()
    
    if save_path:
        analysis_path = save_path.replace('.png', '_analysis.png')
        plt.savefig(analysis_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Analysis figure saved to {analysis_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig

def main():
    """Main execution function."""
    print("🎨 AMF-VI Posterior Visualization Script")
    print("=" * 50)
    
    # Define all 7 datasets to visualize
    datasets_to_viz = ['banana', 'x_shape', 'bimodal_shared', 'bimodal_different', 'multimodal', 'two_moons', 'rings']
    
    # Create results directory path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try to find results directory in current dir or parent dir
    possible_results_dirs = [
        os.path.join(script_dir, 'results'),      # ./results/
        os.path.join(script_dir, '..', 'results')  # ../results/
    ]
    
    results_dir = None
    for dir_path in possible_results_dirs:
        if os.path.exists(dir_path):
            results_dir = dir_path
            break
    
    if results_dir is None:
        results_dir = possible_results_dirs[0]  # Default to first option
    
    print(f"📁 Looking for models in: {results_dir}")
    
    # Check if model files exist
    missing_files = []
    for dataset in datasets_to_viz:
        model_file = os.path.join(results_dir, f'trained_model_{dataset}.pkl')
        if not os.path.exists(model_file):
            missing_files.append(model_file)
    
    if missing_files:
        print("❌ Missing trained model files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease run the AMF-VI training script first to generate the model files.")
        return
    
    print(f"📊 Visualizing datasets: {datasets_to_viz}")
    print(f"📁 Looking for models in: {results_dir}")
    
    # Create individual plots for each dataset
    print("\n🎯 Creating individual posterior comparison plots...")
    
    # Consistent figure size for all datasets
    figsize = (20, 4)  # Wide but not too tall for single row
    
    created_files = []
    
    for dataset_name in datasets_to_viz:
        # Create save path for this dataset
        save_path = os.path.join(results_dir, f'{dataset_name}_posterior_comparison.png')
        
        try:
            # Create visualization for this dataset
            print(f"Creating plot for {dataset_name}...")
            fig = create_single_dataset_visualization(
                dataset_name=dataset_name,
                n_samples=2000,
                figsize=figsize,
                save_path=save_path,
                show_plot=False  # Set to True if you want to display plots
            )
            created_files.append(save_path)
            
        except Exception as e:
            print(f"❌ Error creating plot for {dataset_name}: {e}")
            continue
    
    # Create detailed analysis plot
    print("\n📈 Creating detailed analysis plot...")
    analysis_save_path = os.path.join(results_dir, 'amf_vi_detailed_analysis.png')
    try:
        fig_analysis = create_detailed_analysis_plot(
            datasets=datasets_to_viz,
            save_path=analysis_save_path,
            show_plot=False  # Set to True if you want to display plots
        )
        created_files.append(analysis_save_path)
    except Exception as e:
        print(f"❌ Error creating analysis plot: {e}")
    
    print("\n✅ Visualization complete!")
    print(f"📸 Created {len(created_files)} files:")
    for file_path in created_files:
        print(f"   - {os.path.basename(file_path)}")
    
    # Print summary statistics
    print("\n📋 Summary:")
    for dataset in datasets_to_viz:
        try:
            model, _, _ = load_trained_model(dataset)
            if hasattr(model, 'weights') and model.weights_trained:
                weights = model.weights.detach().cpu().numpy()
                flow_types = ['RealNVP', 'MAF', 'RBIG'][:len(weights)]
                print(f"  {dataset.replace('_', '-').title()}:")
                for flow_type, weight in zip(flow_types, weights):
                    print(f"    - {flow_type}: {weight:.3f}")
        except Exception as e:
            print(f"  {dataset}: Error loading model - {e}")

if __name__ == "__main__":
    main()