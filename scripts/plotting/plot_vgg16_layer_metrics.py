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

# Collect layer data
layer_data = {}

for method, path in comparison_files.items():
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Skipping.")
        continue
        
    with open(path, 'r') as f:
        data = json.load(f)
        
    # Get last epoch layer sparsity dict
    layer_sparsity_dict = data.get('layer_sparsity', {})
    if not layer_sparsity_dict:
        continue
        
    # Find the last epoch key dynamically
    last_epoch_key = sorted(list(layer_sparsity_dict.keys()), key=lambda x: int(x.split('_')[1]))[-1]
    last_epoch_data = layer_sparsity_dict[last_epoch_key]
    
    layer_data[method] = {
        'layers': [],
        'weights_ratio': [],
        'neurons_ratio': []
    }
    
    # Sort layers by their sequential order in VGG16
    sorted_layers = sorted(list(last_epoch_data.keys()), key=lambda name: (
        0 if 'features' in name else 1,
        int(name.split('.')[1])
    ))
    
    for layer_name in sorted_layers:
        metrics = last_epoch_data[layer_name]
        active_w = metrics.get('active_weights', 0)
        total_w = metrics.get('total_weights', 1)
        active_n = metrics.get('active_neurons', 0)
        total_n = metrics.get('total_neurons', 1)
        
        layer_data[method]['layers'].append(layer_name)
        layer_data[method]['weights_ratio'].append(active_w / total_w)
        layer_data[method]['neurons_ratio'].append(active_n / total_n)

# Plotting Subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=150)

# Set common grid & formatting
for ax in [ax1, ax2]:
    ax.grid(True, linestyle='--', alpha=0.5)

# Plot 1: Active Weights Ratio
ax1.set_title('Layer-wise Capacity Distribution\nActive Weights to Total Weights Ratio (~90% Global Sparsity)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Active Weights Ratio', fontsize=10)
ax1.set_xlabel('Network Layer Name', fontsize=10)

for method, data in layer_data.items():
    ax1.plot(data['layers'], data['weights_ratio'], 
             color=styles[method]['color'], marker=styles[method]['marker'], 
             linestyle=styles[method]['linestyle'], label=method, linewidth=2, markersize=7)

ax1.legend(loc='upper right', frameon=True)
ax1.tick_params(axis='x', labelrotation=90, labelsize=8)

# Plot 2: Active Neurons Ratio
ax2.set_title('Layer-wise Neuron Retention\nActive Neurons to Total Neurons Ratio (~90% Global Sparsity)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Active Neurons Ratio', fontsize=10)
ax2.set_xlabel('Network Layer Name', fontsize=10)

for method, data in layer_data.items():
    ax2.plot(data['layers'], data['neurons_ratio'], 
             color=styles[method]['color'], marker=styles[method]['marker'], 
             linestyle=styles[method]['linestyle'], label=method, linewidth=2, markersize=7)

ax2.tick_params(axis='x', labelrotation=90, labelsize=8)

plt.tight_layout()

# Save image
output_path = 'results/vgg16_cifar10_experiments/vgg16_cifar10_layer_wise_comparison.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"Layer-wise active weights and neurons comparison plots successfully saved to {output_path}")
