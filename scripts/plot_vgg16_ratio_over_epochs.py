import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Define file paths to compare (aiming for ~90% target sparsity)
comparison_files = {
    'DADP (Hebbian)': 'results/vgg16_cifar10_experiments/history_hebbian_vgg16_CIFAR10_thr5e-06_dt500.json',
    'Magnitude': 'results/vgg16_cifar10_experiments/history_magnitude_vgg16_CIFAR10_sp0.9.json',
    'SNIP': 'results/vgg16_cifar10_experiments/history_snip_vgg16_CIFAR10_sp0.9.json',
    'RigL': 'results/vgg16_cifar10_experiments/history_rigl_vgg16_CIFAR10_sp0.9.json'
}

styles = {
    'DADP (Hebbian)': {'color': '#d62728', 'marker': 'o', 'linestyle': '-'},
    'Magnitude': {'color': '#1f77b4', 'marker': 's', 'linestyle': '--'},
    'SNIP': {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.'},
    'RigL': {'color': '#9467bd', 'marker': 'd', 'linestyle': ':'}
}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, dpi=150)

for ax in [ax1, ax2]:
    ax.grid(True, linestyle='--', alpha=0.5)

for method, path in comparison_files.items():
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Skipping.")
        continue
        
    with open(path, 'r') as f:
        data = json.load(f)
        
    active_w = data.get('active_connections', [])
    active_n = data.get('active_neurons', [])
    layer_sparsity_dict = data.get('layer_sparsity', {})
    
    if not active_w or not active_n or not layer_sparsity_dict:
        print(f"Warning: Missing data in {path}. Skipping.")
        continue
        
    # Get total weights and total neurons from epoch_1
    epoch_keys = sorted(list(layer_sparsity_dict.keys()), key=lambda x: int(x.split('_')[1]))
    first_epoch_data = layer_sparsity_dict[epoch_keys[0]]
    
    total_weights = sum(metrics.get('total_weights', 0) for metrics in first_epoch_data.values())
    total_neurons = sum(metrics.get('total_neurons', 0) for metrics in first_epoch_data.values())
    
    if total_weights == 0 or total_neurons == 0:
        print(f"Warning: Total weights/neurons is 0 for {path}. Skipping.")
        continue
        
    epochs = np.arange(1, len(active_w) + 1)
    
    # Compute ratios
    weights_ratio = [w / total_weights for w in active_w]
    neurons_ratio = [n / total_neurons for n in active_n]
    
    # Plot 1: Active Weights / Total Weights
    ax1.plot(epochs, weights_ratio, 
             color=styles[method]['color'], marker=styles[method]['marker'], 
             linestyle=styles[method]['linestyle'], label=method, linewidth=2, markersize=6)
             
    # Plot 2: Active Neurons / Total Neurons
    ax2.plot(epochs, neurons_ratio, 
             color=styles[method]['color'], marker=styles[method]['marker'], 
             linestyle=styles[method]['linestyle'], label=method, linewidth=2, markersize=6)

# Labels & Styling
ax1.set_title('Global Active Capacity Over Epochs\nActive Weights to Total Weights Ratio (~90% Sparsity)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Active Weights Ratio', fontsize=10)
ax1.legend(loc='best', fontsize=9, frameon=True)

ax2.set_title('Global Neuron Survival Over Epochs\nActive Neurons to Total Neurons Ratio (~90% Sparsity)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Active Neurons Ratio', fontsize=10)
ax2.set_xlabel('Epoch', fontsize=10)
plt.xticks(np.arange(1, 21))

plt.tight_layout()

# Save image
output_path = 'results/vgg16_cifar10_experiments/vgg16_cifar10_ratio_over_epochs.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"Global weights and neurons ratio over epochs plot successfully saved to {output_path}")
