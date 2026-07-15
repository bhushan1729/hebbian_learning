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

plt.figure(figsize=(10, 6), dpi=150)
plt.grid(True, linestyle='--', alpha=0.5)

for method, path in comparison_files.items():
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Skipping.")
        continue
        
    with open(path, 'r') as f:
        data = json.load(f)
        
    active_w = data.get('active_connections', [])
    active_n = data.get('active_neurons', [])
    
    if not active_w or not active_n:
        print(f"Warning: Missing data in {path}. Skipping.")
        continue
        
    epochs = np.arange(1, len(active_w) + 1)
    
    # Compute active weights per active neuron ratio
    ratios = [w / n for w, n in zip(active_w, active_n)]
    
    plt.plot(epochs, ratios, 
             color=styles[method]['color'], marker=styles[method]['marker'], 
             linestyle=styles[method]['linestyle'], label=method, linewidth=2, markersize=7)

plt.title('Connection Density Over Epochs\nTotal Active Weights to Total Active Neurons Ratio (~90% Sparsity)', fontsize=12, fontweight='bold', pad=10)
plt.xlabel('Epoch', fontsize=11)
plt.ylabel('Active Weights / Active Neurons Ratio', fontsize=11)
plt.xticks(np.arange(1, 21))
plt.legend(loc='best', fontsize=10, frameon=True)

# Save image
output_path = 'results/vgg16_cifar10_experiments/vgg16_cifar10_ratio_over_epochs.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"Ratio over epochs plot successfully saved to {output_path}")
