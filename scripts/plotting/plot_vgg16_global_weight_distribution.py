import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_global_weight_distribution(dense_checkpoint_path, sparse_checkpoint_path):
    """
    Plots global weight distribution before and after pruning with customized y-axis scales.
    Panel 1: Ticks from 1e5 to 7e5
    Panel 2: Ticks from 1e4 to 7e4
    """
    if not os.path.exists(dense_checkpoint_path):
        print(f"Error: Dense checkpoint not found at {dense_checkpoint_path}")
        return
    if not os.path.exists(sparse_checkpoint_path):
        print(f"Error: Sparse checkpoint not found at {sparse_checkpoint_path}")
        return
        
    print("Loading checkpoints...")
    dense_state = torch.load(dense_checkpoint_path, map_location='cpu')
    sparse_state = torch.load(sparse_checkpoint_path, map_location='cpu')
    
    dense_dict = dense_state.get('model_state_dict', dense_state)
    sparse_dict = sparse_state.get('model_state_dict', sparse_state)
    
    dense_all_weights = []
    sparse_all_weights = []
    
    print("Extracting and masking all weight tensors...")
    for key in dense_dict.keys():
        if key.endswith('.weight') and 'bn' not in key and 'downsample' not in key:
            mask_key = key.replace('.weight', '.mask')
            
            # Apply mask to dense model if present
            if mask_key in dense_dict:
                dense_w = (dense_dict[key] * dense_dict[mask_key]).cpu().numpy().flatten()
            else:
                dense_w = dense_dict[key].cpu().numpy().flatten()
            
            dense_w = dense_w[dense_w != 0.0]
            dense_all_weights.append(dense_w)
            
            # Apply mask to sparse model
            if key in sparse_dict:
                if mask_key in sparse_dict:
                    sparse_w = (sparse_dict[key] * sparse_dict[mask_key]).cpu().numpy().flatten()
                else:
                    sparse_w = sparse_dict[key].cpu().numpy().flatten()
                
                active_sparse_w = sparse_w[sparse_w != 0.0]
                sparse_all_weights.append(active_sparse_w)
    
    dense_global = np.concatenate(dense_all_weights)
    sparse_global = np.concatenate(sparse_all_weights)
    
    print(f"Total active connections in dense model: {dense_global.size:,}")
    print(f"Total active connections in sparse model: {sparse_global.size:,}")
    
    # 2. Plotting (sharex=True to keep weight range same, sharey=False for independent scales)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=False, dpi=150)
    
    # Panel 1: Global Dense Distribution
    ax1.hist(dense_global, bins=250, color='darkblue', alpha=0.95, density=False)
    ax1.set_title("Global weight distribution before pruning", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Weight Value", fontsize=10)
    ax1.set_ylabel("Count", fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Configure Y-axis 1: Limits [0, 7.5e5] and ticks [1e5, 2e5, ..., 7e5]
    ax1.set_ylim(0, 7.5e5)
    ax1.set_yticks([1e5, 2e5, 3e5, 4e5, 5e5, 6e5, 7e5])
    ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    
    # Panel 2: Global Sparse Distribution (active only)
    ax2.hist(sparse_global, bins=250, color='darkblue', alpha=0.95, density=False)
    ax2.set_title("Global weight distribution after pruning and retraining", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Weight Value", fontsize=10)
    ax2.set_ylabel("Count", fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # Configure Y-axis 2: Limits [0, 7.5e4] and ticks [1e4, 2e4, ..., 7e4]
    ax2.set_ylim(0, 7.5e4)
    ax2.set_yticks([1e4, 2e4, 3e4, 4e4, 5e4, 6e4, 7e4])
    ax2.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    
    plt.suptitle("Model-Wide (Global) Weight Distribution Analysis", y=1.02, fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    output_img = "plots/vgg16_global_weight_distribution_custom_scale.png"
    os.makedirs('plots', exist_ok=True)
    plt.savefig(output_img, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"Global distribution plot successfully saved to {output_img}")

if __name__ == '__main__':
    DENSE_CHECKPOINT = "results/vgg16_cifar10_experiments/models/baseline_vgg16_CIFAR10.pth"
    SPARSE_CHECKPOINT = "results/vgg16_cifar10_experiments/models/hebbian_vgg16_CIFAR10_thr5e-06_dt500.pth"
    plot_global_weight_distribution(DENSE_CHECKPOINT, SPARSE_CHECKPOINT)
