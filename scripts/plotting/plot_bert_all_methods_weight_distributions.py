import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_bert_weight_distributions(checkpoint_paths, output_path="plots/bert_tiny_all_methods_weight_distributions.png"):
    """
    Loads model checkpoints for all 4 pruning methods on BERT-Tiny, extracts active weights
    (non-zero values in 2D linear matrices, avoiding 1D LayerNorm and biases),
    and plots their distributions in a 2x2 grid of subplots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, dpi=150)
    axes = axes.flatten()
    
    methods = ['DADP (Hebbian)', 'Magnitude', 'SNIP', 'RigL']
    colors = {
        'DADP (Hebbian)': '#d62728',   # Red
        'Magnitude': '#1f77b4',        # Blue
        'SNIP': '#2ca02c',             # Green
        'RigL': '#9467bd'              # Purple
    }
    
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
            # Extract 2D Linear weight matrices, filtering out 1D biases and LayerNorm parameters
            if key.endswith('.weight'):
                w_tensor = model_dict[key]
                if w_tensor.dim() > 1: # Isolates actual attention and projection matrices
                    mask_key = key.replace('.weight', '.mask')
                    
                    # Apply mask if present
                    if mask_key in model_dict:
                        w_tensor = w_tensor * model_dict[mask_key]
                    
                    w_flat = w_tensor.cpu().numpy().flatten()
                    w_active = w_flat[w_flat != 0.0]
                    active_weights.append(w_active)
        
        if len(active_weights) == 0:
            print(f"Warning: No active weights found for {method}!")
            continue
            
        all_w = np.concatenate(active_weights)
        print(f"Method: {method} | Active Parameters analyzed: {all_w.size:,}")
        
        # Plot density histogram
        ax.hist(all_w, bins=250, color=colors[method], alpha=0.85, density=True, edgecolor='black', linewidth=0.3)
        ax.set_title(f"{method} Weight Distribution", fontsize=12, fontweight='bold', pad=8)
        ax.set_ylabel("Probability Density", fontsize=10)
        ax.set_xlabel("Weight Value", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Highlight zero line
        ax.axvline(0.0, color='black', linestyle=':', alpha=0.5, linewidth=1)
        
        # Add summary stats box
        stats_text = (f"Sparsity: ~95%\n"
                      f"Mean: {np.mean(all_w):.4f}\n"
                      f"Std: {np.std(all_w):.4f}\n"
                      f"Count: {all_w.size:,}")
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.suptitle("Model-Wide Active Weight Distribution Profiles (BERT-Tiny at ~95% Sparsity)", y=0.98, fontsize=14, fontweight='bold')
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
