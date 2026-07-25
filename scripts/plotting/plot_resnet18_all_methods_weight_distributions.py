import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_all_methods_weight_distributions(checkpoint_paths, output_path="plots/resnet18_all_methods_weight_distributions.png"):
    """
    Loads model checkpoints for Baseline and all 4 pruning methods, extracts the active weights
    (non-zero values) across all Conv/Linear layers, and plots their distributions
    in a 2x3 grid of subplots for comparison.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, dpi=150)
    axes = axes.flatten()
    
    methods = ['Baseline (Dense)', 'DADP (Hebbian)', 'Magnitude', 'SNIP', 'RigL']
    
    for idx, method in enumerate(methods):
        path = checkpoint_paths.get(method)
        ax = axes[idx]
        
        if not path or not os.path.exists(path):
            print(f"Warning: Checkpoint path for {method} not found at '{path}'. Skipping subplot.")
            ax.text(0.5, 0.5, f"Checkpoint Not Found\n({path})", 
                    ha='center', va='center', fontsize=11, color='gray')
            ax.set_title(method, fontsize=12, fontweight='bold')
            continue
            
        print(f"Loading checkpoint for {method} from {path}...")
        state_dict = torch.load(path, map_location='cpu')
        model_dict = state_dict.get('model_state_dict', state_dict)
        
        active_weights = []
        
        for key in model_dict.keys():
            # Extract weights of Conv and Linear layers (avoiding batch norm params)
            if key.endswith('.weight') and 'bn' not in key and 'downsample' not in key:
                w_tensor = model_dict[key]
                if w_tensor.is_sparse:
                    w_tensor = w_tensor.to_dense()
                    
                mask_key = key.replace('.weight', '.mask')
                
                # Apply mask if it exists in the state dict
                if mask_key in model_dict:
                    mask_tensor = model_dict[mask_key]
                    if mask_tensor.is_sparse:
                        mask_tensor = mask_tensor.to_dense()
                    w_tensor = w_tensor * mask_tensor
                
                w_flat = w_tensor.cpu().numpy().flatten()
                
                # Filter out zero connections to only analyze active parameter values
                w_active = w_flat[w_flat != 0.0]
                active_weights.append(w_active)
        
        if len(active_weights) == 0:
            print(f"Warning: No active weights found for {method}!")
            continue
            
        all_w = np.concatenate(active_weights)
        print(f"Method: {method} | Active Parameters analyzed: {all_w.size:,}")
        
        # Plot density histogram (PDF) using the same 'darkblue' color theme
        ax.hist(all_w, bins=250, color='darkblue', alpha=0.95, density=True, edgecolor='black', linewidth=0.1)
        ax.set_title(f"{method} Weight Distribution", fontsize=12, fontweight='bold', pad=8)
        ax.set_ylabel("Probability Density", fontsize=10)
        ax.set_xlabel("Weight Value", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Highlight zero line
        ax.axvline(0.0, color='black', linestyle=':', alpha=0.5, linewidth=1)
        
        # Add quick summary metrics text on subplot
        sparsity_str = "0%" if method == 'Baseline (Dense)' else "~99%"
        stats_text = (f"Sparsity: {sparsity_str}\n"
                      f"Mean: {np.mean(all_w):.4f}\n"
                      f"Std: {np.std(all_w):.4f}\n"
                      f"Count: {all_w.size:,}")
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))

    # Format the 6th subplot as a clean info panel
    ax_info = axes[5]
    ax_info.axis('off')
    info_text = (
        "📊 Weight Distribution Notes:\n\n"
        "• Baseline (Dense): Unpruned Gaussian weight distribution.\n"
        "• DADP (Hebbian): Unimodal bell curve centered at zero.\n"
        "• Magnitude: Bimodal distribution with a hard exclusion\n"
        "  zone centered at zero (|w| < threshold).\n"
        "• SNIP: Smooth distribution showing zero-drift.\n"
        "• RigL: Dense-like distribution via active regrowth."
    )
    ax_info.text(0.05, 0.5, info_text, va='center', ha='left', fontsize=11,
                 bbox=dict(boxstyle='round,pad=1.0', facecolor='#f9f9f9', alpha=0.9, edgecolor='lightgray'))

    plt.suptitle("Model-Wide Active Weight Distribution Profiles (ResNet-18 at ~99% Sparsity)", y=0.98, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Weight distribution comparison plot successfully saved to {output_path}")

if __name__ == '__main__':
    # Local paths (default) - customize as needed
    CHECKPOINTS = {
        'Baseline (Dense)': 'results/resnet18_cifar10_experiments/models/baseline_resnet18_CIFAR10.pth',
        'DADP (Hebbian)': 'results/resnet18_cifar10_experiments/models/hebbian_resnet18_CIFAR10_thr0.0005_dt500_best.pth',
        'Magnitude': 'results/resnet18_cifar10_experiments/models/magnitude_resnet18_CIFAR10_sp0.99.pth',
        'SNIP': 'results/resnet18_cifar10_experiments/models/snip_resnet18_CIFAR10_sp0.99.pth',
        'RigL': 'results/resnet18_cifar10_experiments/models/rigl_resnet18_CIFAR10_sp0.99.pth'
    }
    plot_all_methods_weight_distributions(CHECKPOINTS)
