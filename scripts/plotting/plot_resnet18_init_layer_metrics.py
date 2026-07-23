import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Define file paths to compare (ResNet-18 across the 7 initialization runs)
comparison_files = {
    'Kaiming Normal': 'results/init_ablation_resnet18/results/history_init_run_resnet18_kaiming_normal.json',
    'Kaiming Uniform': 'results/init_ablation_resnet18/results/history_init_run_resnet18_kaiming_uniform.json',
    'Xavier Normal': 'results/init_ablation_resnet18/results/history_init_run_resnet18_xavier_normal.json',
    'Xavier Uniform': 'results/init_ablation_resnet18/results/history_init_run_resnet18_xavier_uniform.json',
    'Orthogonal': 'results/init_ablation_resnet18/results/history_init_run_resnet18_orthogonal.json',
    'Normal (sigma=0.02)': 'results/init_ablation_resnet18/results/history_init_run_resnet18_normal_0.02.json',
    'Normal (sigma=0.1)': 'results/init_ablation_resnet18/results/history_init_run_resnet18_normal_0.1.json'
}

styles = {
    'Kaiming Normal': {'color': '#d62728', 'marker': 'o', 'linestyle': '-'},
    'Kaiming Uniform': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--'},
    'Xavier Normal': {'color': '#1f77b4', 'marker': '^', 'linestyle': '-.'},
    'Xavier Uniform': {'color': '#17becf', 'marker': 'v', 'linestyle': ':'},
    'Orthogonal': {'color': '#2ca02c', 'marker': 'd', 'linestyle': '-'},
    'Normal (sigma=0.02)': {'color': '#bcbd22', 'marker': 'p', 'linestyle': '-.'},
    'Normal (sigma=0.1)': {'color': '#9467bd', 'marker': 'x', 'linestyle': '--'}
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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=150)

for ax in [ax1, ax2]:
    ax.grid(True, linestyle='--', alpha=0.5)

# Plot 1: Active Weights Ratio
ax1.set_title('Layer-wise Capacity Distribution\nActive Weights to Total Weights Ratio (ResNet-18 Initialization Sweep)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Active Weights Ratio', fontsize=10)
ax1.set_xlabel('Network Layer Name', fontsize=10)

for method, data in layer_data.items():
    lw = 2.0 if styles[method]['linestyle'] == '-' else 1.3
    ax1.plot(data['layers'], data['weights_ratio'], 
             color=styles[method]['color'], marker=styles[method]['marker'], 
             linestyle=styles[method]['linestyle'], label=method, linewidth=lw, markersize=7)

ax1.legend(loc='upper right', frameon=True)
ax1.tick_params(axis='x', labelrotation=90, labelsize=8)

# Plot 2: Active Neurons Ratio
ax2.set_title('Layer-wise Neuron Retention\nActive Neurons to Total Neurons Ratio (ResNet-18 Initialization Sweep)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Active Neurons Ratio', fontsize=10)
ax2.set_xlabel('Network Layer Name', fontsize=10)

for method, data in layer_data.items():
    lw = 2.0 if styles[method]['linestyle'] == '-' else 1.3
    ax2.plot(data['layers'], data['neurons_ratio'], 
             color=styles[method]['color'], marker=styles[method]['marker'], 
             linestyle=styles[method]['linestyle'], label=method, linewidth=lw, markersize=7)

# Remove legend from ax2 to prevent overlap/clutter as ax1 already displays it
ax2.tick_params(axis='x', labelrotation=90, labelsize=8)

plt.tight_layout()

# Save image to both results subfolder and primary plots directory
output_paths = [
    'results/init_ablation_resnet18/plots/resnet18_init_layer_wise_comparison.png',
    'plots/resnet18_init_layer_wise_comparison.png'
]

for out_path in output_paths:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', dpi=300)

plt.close()

print(f"Layer-wise active weights and neurons comparison plots successfully saved to: {output_paths}")
