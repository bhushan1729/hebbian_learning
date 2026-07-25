import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_bert_weight_distributions(checkpoint_paths, output_path="plots/bert_tiny_all_methods_weight_distributions.png"):
    """
    Loads model checkpoints for all 4 pruning methods on BERT-Tiny, extracts active weights
    (non-zero values in 2D linear matrices, avoiding 1D LayerNorm and biases),
    and plots their distributions in a clean 2x2 grid with dynamic percentile x-limits.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=200)
    axes = axes.flatten()
    
    methods = ['DADP (Hebbian)', 'Magnitude', 'SNIP', 'RigL']
    
    # Store all weights to determine common zoomed x-limits
    all_methods_weights = {}
    valid_weights = []
    
    for method in methods:
        path = checkpoint_paths.get(method)
        if not path or not os.path.exists(path):
            continue
            
        print(f"Loading checkpoint for {method} from {path}...")
        state_dict = torch.load(path, map_location='cpu')
        model_dict = state_dict.get('model_state_dict', state_dict)
        
        active_weights = []
        for key in model_dict.keys():
            # Extract 2D Linear weight matrices (excluding unpruned embedding tables and 1D biases/LayerNorm)
            if key.endswith('.weight') and 'embeddings' not in key:
                w_tensor = model_dict[key]
                if hasattr(w_tensor, 'is_sparse') and w_tensor.is_sparse:
                    w_tensor = w_tensor.to_dense()
                    
                if w_tensor.dim() > 1:
                    mask_key = key.replace('.weight', '.mask')
                    if mask_key in model_dict:
                        mask_tensor = model_dict[mask_key]
                        if hasattr(mask_tensor, 'is_sparse') and mask_tensor.is_sparse:
                            mask_tensor = mask_tensor.to_dense()
                        w_tensor = w_tensor * mask_tensor
                    
                    w_flat = w_tensor.cpu().numpy().flatten()
                    w_active = w_flat[w_flat != 0.0]
                    active_weights.append(w_active)
                    
        if active_weights:
            concatenated = np.concatenate(active_weights)
            all_methods_weights[method] = concatenated
            valid_weights.append(concatenated)
            print(f"Method: {method} | Active Parameters analyzed: {concatenated.size:,}")

    # Determine optimal zoomed x-limits based on percentiles across all methods to avoid outlier stretching
    if valid_weights:
        global_concat = np.concatenate(valid_weights)
        x_min = np.percentile(global_concat, 0.05)
        x_max = np.percentile(global_concat, 99.95)
        padding = (x_max - x_min) * 0.1
        xlim = (x_min - padding, x_max + padding)
    else:
        xlim = (-0.2, 0.2)

    for idx, method in enumerate(methods):
        ax = axes[idx]
        if method not in all_methods_weights:
            ax.text(0.5, 0.5, f"Checkpoint Not Found", ha='center', va='center', fontsize=11, color='gray')
            ax.set_title(method, fontsize=12, fontweight='bold')
            continue
            
        all_w = all_methods_weights[method]
        
        # Plot smooth density histogram without black line clutter
        ax.hist(all_w, bins=150, range=xlim, color='#1f77b4', alpha=0.85, density=True, edgecolor='none')
        
        ax.set_title(f"{method} Weight Distribution", fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel("Probability Density", fontsize=10)
        ax.set_xlabel("Weight Value", fontsize=10)
        ax.set_xlim(xlim)
        ax.grid(True, linestyle='--', alpha=0.4)
        
        # Highlight zero line
        ax.axvline(0.0, color='crimson', linestyle='--', alpha=0.7, linewidth=1.2)
        
        # Add summary stats box
        stats_text = (f"Sparsity: ~95%\n"
                      f"Mean: {np.mean(all_w):.4f}\n"
                      f"Std: {np.std(all_w):.4f}\n"
                      f"Active: {all_w.size:,}")
        ax.text(0.96, 0.94, stats_text, transform=ax.transAxes, fontsize=9.5,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#cccccc'))

    plt.suptitle("Model-Wide Active Weight Distribution Profiles (BERT-Tiny on SST-2)", y=0.98, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"BERT weight distribution comparison plot successfully saved to {output_path}")

if __name__ == '__main__':
    # Local paths (default) - customize as needed
    CHECKPOINTS = {
        'DADP (Hebbian)': 'results/bert_experiments/results/models/hebbian_bert_SST2_thr3e-06_dt500_best.pth',
        'Magnitude': 'results/bert_experiments/results/models/magnitude_bert_SST2_sp0.95.pth',
        'SNIP': 'results/bert_experiments/results/models/snip_bert_SST2_sp0.95.pth',
        'RigL': 'results/bert_experiments/results/models/rigl_bert_SST2_sp0.95.pth'
    }
    plot_bert_weight_distributions(CHECKPOINTS)
