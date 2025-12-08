"""
Visualize synthetic datasets from data_generator.
Usage: python visualize_data.py [dataset_names...]
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
from data_generator import generate_data, get_available_datasets


def plot_dataset(data, title, ax=None, color='steelblue'):
    """Plot single dataset with statistics."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    data_np = data.detach().cpu().numpy()
    ax.scatter(data_np[:, 0], data_np[:, 1], alpha=0.6, s=20, c=color, 
               edgecolors='black', linewidth=0.1)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Add statistics
    mean = data_np.mean(axis=0)
    std = data_np.std(axis=0)
    stats_text = f'μ=({mean[0]:.2f}, {mean[1]:.2f})\nσ=({std[0]:.2f}, {std[1]:.2f})\nn={len(data)}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    return ax


def plot_all_datasets(datasets=None, n_samples=1000, save_path=None):
    """Plot all datasets in a grid."""
    if datasets is None:
        datasets = get_available_datasets()
    
    n_datasets = len(datasets)
    n_cols = 3
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1) if n_datasets > 1 else [axes]
    
    colors = ['steelblue', 'crimson', 'forestgreen', 'darkorange', 
              'mediumpurple', 'chocolate', 'teal']
    
    for i, dataset_name in enumerate(datasets):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        try:
            data = generate_data(dataset_name, n_samples=n_samples)
            color = colors[i % len(colors)]
            plot_dataset(data, dataset_name.replace('_', ' ').title(), ax, color)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading\n{dataset_name}', 
                   ha='center', va='center', transform=ax.transAxes)
            print(f"Error loading {dataset_name}: {e}")
    
    # Hide empty subplots
    for i in range(n_datasets, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.suptitle('AMF-VI Synthetic Datasets', fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize synthetic datasets')
    parser.add_argument('datasets', nargs='*', 
                       help='Dataset names (default: all)', 
                       default=None)
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of samples per dataset')
    parser.add_argument('--save', type=str, default=None,
                       help='Save plot to file')
    parser.add_argument('--list', action='store_true',
                       help='List available datasets')
    
    args = parser.parse_args()
    
    if args.list:
        print("Available datasets:")
        for name in get_available_datasets():
            print(f"  - {name}")
        return
    
    # Use specified datasets or all
    datasets = args.datasets if args.datasets else get_available_datasets()
    
    # Validate dataset names
    available = get_available_datasets()
    invalid = [d for d in datasets if d not in available]
    if invalid:
        print(f"Invalid datasets: {invalid}")
        print(f"Available: {available}")
        return
    
    print(f"Plotting datasets: {datasets}")
    
    # Create plot
    fig = plot_all_datasets(datasets, args.n_samples, args.save)
    plt.show()


if __name__ == "__main__":
    main()
