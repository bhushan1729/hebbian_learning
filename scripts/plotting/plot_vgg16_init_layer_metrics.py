import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Define file paths to compare (VGG-16 across the 6 successful initialization runs)
comparison_files = {
    'Kaiming Normal': 'results/init_ablation_vgg16/results/history_init_run_vgg16_kaiming_normal.json',
    'Kaiming Uniform': 'results/init_ablation_vgg16/results/history_init_run_vgg16_kaiming_uniform.json',
    'Xavier Normal': 'results/init_ablation_vgg16/results/history_init_run_vgg16_xavier_normal.json',
    'Xavier Uniform': 'results/init_ablation_vgg16/results/history_init_run_vgg16_xavier_uniform.json',
    'Orthogonal': 'results/init_ablation_vgg16/results/history_init_run_vgg16_orthogonal.json',
    'Normal (sigma=0.1)': 'results/init_ablation_vgg16/results/history_init_run_vgg16_normal_0.1.json'
}

styles = {
    'Kaiming Normal': {'color': '#d62728', 'marker': 'o', 'linestyle': '-'},
    'Kaiming Uniform': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--'},
    'Xavier Normal': {'color': '#1f77b4', 'marker': '^', 'linestyle': '-.'},
    'Xavier Uniform': {'color': '#17becf', 'marker': 'v', 'linestyle': ':'},
    'Orthogonal': {'color': '#2ca02c', 'marker': 'd', 'linestyle': '-'},
    'Normal (sigma=0.1)': {'color': '#9467bd', 'marker': 'x', 'linestyle': '--'}
}

# Custom sorting order key function for VGG-16 layers
def get_vgg_layer_order(name):
    if name.startswith('features.'):
        idx = int(name.split('.')[1])
        return (0, idx)
    elif name.startswith('classifier.'):
        idx = int(name.split('.')[1])
        return (1, idx)
    return (2, 0)

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
    sorted_layers = sorted(list(last_epoch_data.keys()), key=get_vgg_layer_order)
    
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
ax1.set_title('Layer-wise Capacity Distribution\nActive Weights to Total Weights Ratio (VGG-16 Initialization Sensitivity Sweep)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Active Weights Ratio', fontsize=11)

for method, data in layer_data.items():
    ax1.plot(data['layers'], data['weights_ratio'], 
             color=styles[method]['color'], marker=styles[method]['marker'], 
             linestyle=styles[method]['linestyle'], label=method, linewidth=2, markersize=7)

ax1.legend(loc='upper right', frameon=True)

# Plot 2: Active Neurons Ratio
ax2.set_title('Layer-wise Neuron Retention\nActive Neurons to Total Neurons Ratio (VGG-16 Initialization Sensitivity Sweep)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Active Neurons Ratio', fontsize=11)
ax2.set_xlabel('Network Layer Name', fontsize=11)

for method, data in layer_data.items():
    ax2.plot(data['layers'], data['neurons_ratio'], 
             color=styles[method]['color'], marker=styles[method]['marker'], 
             linestyle=styles[method]['linestyle'], label=method, linewidth=2, markersize=7)

# Format X-axis tick labels
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.tight_layout()

# Save image to both results subfolder and primary plots directory
output_paths = [
    'results/init_ablation_vgg16/plots/vgg16_init_layer_wise_comparison.png',
    'plots/vgg16_init_layer_wise_comparison.png'
]

for out_path in output_paths:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', dpi=300)

plt.close()

print(f"Layer-wise active weights and neurons comparison plots successfully saved to: {output_paths}")
