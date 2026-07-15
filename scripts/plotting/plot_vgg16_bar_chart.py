import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Define file paths to compare (aiming for ~90% target sparsity)
comparison_files = {
    'DADP (Hebbian, thr=5e-6)': 'results/vgg16_cifar10_experiments/history_hebbian_vgg16_CIFAR10_thr5e-06_dt500.json',
    'SNIP (sp=0.90)': 'results/vgg16_cifar10_experiments/history_snip_vgg16_CIFAR10_sp0.9.json',
    'Magnitude (sp=0.90)': 'results/vgg16_cifar10_experiments/history_magnitude_vgg16_CIFAR10_sp0.9.json',
    'RigL (sp=0.90)': 'results/vgg16_cifar10_experiments/history_rigl_vgg16_CIFAR10_sp0.9.json'
}

# Collect layer-wise sparsity
methods = ['DADP (Hebbian, thr=5e-6)', 'SNIP (sp=0.90)', 'Magnitude (sp=0.90)', 'RigL (sp=0.90)']
colors = {
    'DADP (Hebbian, thr=5e-6)': '#d62728',
    'SNIP (sp=0.90)': '#2ca02c',
    'Magnitude (sp=0.90)': '#1f77b4',
    'RigL (sp=0.90)': '#9467bd'
}

layer_sparsities = {m: [] for m in methods}
layers = []

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
    sorted_layers = sorted(list(last_epoch_data.keys()), key=lambda name: (
        0 if 'features' in name else 1,
        int(name.split('.')[1])
    ))
    
    if not layers:
        layers = sorted_layers
        
    for layer in layers:
        # Sparsity is stored as ratio, convert to percentage
        sp_percentage = last_epoch_data[layer]['sparsity'] * 100
        layer_sparsities[method].append(sp_percentage)

# Set up the plot
x = np.arange(len(layers))
width = 0.2

plt.figure(figsize=(15, 8), dpi=150)
plt.grid(True, linestyle='--', alpha=0.3, zorder=0)

# Plot bars
for i, method in enumerate(methods):
    plt.bar(x + (i - 1.5) * width, layer_sparsities[method], width, 
            label=method, color=colors[method], zorder=3)

# Formatting
plt.title('VGG16 Layer-wise Network Sparsity Distribution\nComparing DADP vs. SNIP, Magnitude, and RigL at ~90% Global Sparsity', fontsize=13, fontweight='bold')
plt.xlabel('Layer Name', fontsize=11)
plt.ylabel('Sparsity (%)', fontsize=11)
plt.xticks(x, layers, rotation=45, ha='right', fontsize=9)
plt.ylim(0, 110)
plt.legend(loc='lower left', fontsize=10, frameon=True)

# Save image
output_path = 'plots/vgg16_layer_sparsity_bar_chart.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"VGG16 layer-wise sparsity bar chart successfully saved to {output_path}")
