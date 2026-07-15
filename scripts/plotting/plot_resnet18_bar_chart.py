import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Define file paths to compare (aiming for ~99% target sparsity)
comparison_files = {
    'DADP (Hebbian, thr=0.0005)': 'results/resnet18_cifar10_experiments/history_hebbian_resnet18_CIFAR10_thr0.0005_dt500.json',
    'SNIP (sp=0.99)': 'results/resnet18_cifar10_experiments/history_snip_resnet18_CIFAR10_sp0.99.json',
    'Magnitude (sp=0.99)': 'results/resnet18_cifar10_experiments/history_magnitude_resnet18_CIFAR10_sp0.99.json',
    'RigL (sp=0.99)': 'results/resnet18_cifar10_experiments/history_rigl_resnet18_CIFAR10_sp0.99.json'
}

# Collect layer-wise sparsity
methods = ['DADP (Hebbian, thr=0.0005)', 'SNIP (sp=0.99)', 'Magnitude (sp=0.99)', 'RigL (sp=0.99)']
colors = {
    'DADP (Hebbian, thr=0.0005)': '#d62728',
    'SNIP (sp=0.99)': '#2ca02c',
    'Magnitude (sp=0.99)': '#1f77b4',
    'RigL (sp=0.99)': '#9467bd'
}

layer_sparsities = {m: [] for m in methods}
layers = []

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

# Fetch the data
for method in methods:
    path = comparison_files[method]
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        continue
    with open(path, 'r') as f:
        data = json.load(f)
    
    layer_dict = data.get('layer_sparsity', {})
    last_epoch_key = sorted(list(layer_dict.keys()), key=lambda x: int(x.split('_')[1]))[-1]
    last_epoch_data = layer_dict[last_epoch_key]
    
    # Sort layers sequentially
    sorted_layers = sorted(list(last_epoch_data.keys()), key=get_layer_order)
    
    if not layers:
        layers = sorted_layers
        
    for layer in layers:
        # Sparsity is stored as ratio, convert to percentage
        sp_percentage = last_epoch_data[layer]['sparsity'] * 100
        layer_sparsities[method].append(sp_percentage)

# Set up the plot
x = np.arange(len(layers))
width = 0.2

plt.figure(figsize=(18, 8), dpi=150)
plt.grid(True, linestyle='--', alpha=0.3, zorder=0)

# Plot bars
for i, method in enumerate(methods):
    plt.bar(x + (i - 1.5) * width, layer_sparsities[method], width, 
            label=method, color=colors[method], zorder=3)

# Formatting
plt.title('ResNet-18 Layer-wise Network Sparsity Distribution\nComparing DADP vs. SNIP, Magnitude, and RigL at ~99% Global Sparsity', fontsize=13, fontweight='bold')
plt.xlabel('Layer Name', fontsize=11)
plt.ylabel('Sparsity (%)', fontsize=11)
plt.xticks(x, layers, rotation=45, ha='right', fontsize=9)
plt.ylim(0, 110)
plt.legend(loc='lower left', fontsize=10, frameon=True)

# Save image
output_path = 'plots/resnet18_layer_sparsity_bar_chart.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"ResNet-18 layer-wise sparsity bar chart successfully saved to {output_path}")
