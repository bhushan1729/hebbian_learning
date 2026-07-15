import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Define file paths to compare (aiming for ~99% target sparsity)
comparison_files = {
    'DADP (Hebbian)': 'results/resnet18_cifar10_experiments/history_hebbian_resnet18_CIFAR10_thr0.0005_dt500.json',
    'Magnitude': 'results/resnet18_cifar10_experiments/history_magnitude_resnet18_CIFAR10_sp0.99.json',
    'SNIP': 'results/resnet18_cifar10_experiments/history_snip_resnet18_CIFAR10_sp0.99.json',
    'RigL': 'results/resnet18_cifar10_experiments/history_rigl_resnet18_CIFAR10_sp0.99.json'
}

styles = {
    'DADP (Hebbian)': {'color': '#d62728', 'marker': 'o', 'linestyle': '-'},
    'Magnitude': {'color': '#1f77b4', 'marker': 's', 'linestyle': '--'},
    'SNIP': {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.'},
    'RigL': {'color': '#9467bd', 'marker': 'd', 'linestyle': ':'}
}

# Custom sorting order key function for ResNet-18 layers
def get_layer_order(name):
    if name == 'conv1':
        return (0, 0, 0, 0)
    if name.startswith('layer'):
        parts = name.split('.')
        layer_num = int(parts[0].replace('layer', ''))
        block_num = int(parts[1])
        if parts[2] == 'downsample':
            return (1, layer_num, block_num, 3) # Put downsample after conv2
        elif parts[2].startswith('conv'):
            conv_num = int(parts[2].replace('conv', ''))
            return (1, layer_num, block_num, conv_num)
    if name == 'fc':
        return (2, 0, 0, 0)
    return (3, 0, 0, 0)

# Collect layer data
layer_data = {}

for method, path in comparison_files.items():
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Skipping.")
        continue
        
    with open(path, 'r') as f:
        data = json.load(f)
        
    layer_sparsity_dict = data.get('layer_sparsity', {})
    if not layer_sparsity_dict:
        continue
        
    # Find last epoch key dynamically
    last_epoch_key = sorted(list(layer_sparsity_dict.keys()), key=lambda x: int(x.split('_')[1]))[-1]
    last_epoch_data = layer_sparsity_dict[last_epoch_key]
    
    layer_data[method] = {
        'layers': [],
        'weights_ratio': [],
        'neurons_ratio': []
    }
    
    # Sort layers sequentially
    sorted_layers = sorted(list(last_epoch_data.keys()), key=get_layer_order)
    
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
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, dpi=150)

for ax in [ax1, ax2]:
    ax.grid(True, linestyle='--', alpha=0.5)

# Plot 1: Active Weights Ratio
ax1.set_title('Layer-wise Capacity Distribution\nActive Weights to Total Weights Ratio (~99% Global Sparsity)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Active Weights Ratio', fontsize=11)

for method, data in layer_data.items():
    ax1.plot(data['layers'], data['weights_ratio'], 
             color=styles[method]['color'], marker=styles[method]['marker'], 
             linestyle=styles[method]['linestyle'], label=method, linewidth=2, markersize=7)

ax1.legend(loc='upper right', frameon=True)

# Plot 2: Active Neurons Ratio
ax2.set_title('Layer-wise Neuron Retention\nActive Neurons to Total Neurons Ratio (~99% Global Sparsity)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Active Neurons Ratio', fontsize=11)
ax2.set_xlabel('Network Layer Name', fontsize=11)

for method, data in layer_data.items():
    ax2.plot(data['layers'], data['neurons_ratio'], 
             color=styles[method]['color'], marker=styles[method]['marker'], 
             linestyle=styles[method]['linestyle'], label=method, linewidth=2, markersize=7)

# Format X-axis tick labels
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.tight_layout()

# Save image
output_path = 'results/resnet18_cifar10_experiments/resnet18_cifar10_layer_wise_comparison_99.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"Layer-wise active weights and neurons comparison plots at 99% successfully saved to {output_path}")
