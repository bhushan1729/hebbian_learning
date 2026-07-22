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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

for ax in [ax1, ax2]:
    ax.grid(True, linestyle='--', alpha=0.5)

max_epochs = 0
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
        
    max_epochs = max(max_epochs, len(active_w))
    epochs = np.arange(1, len(active_w) + 1)
    
    # Plot 1: Actual Active Weights Count
    ax1.plot(epochs, active_w, 
             color=styles[method]['color'], marker=styles[method]['marker'], 
             linestyle=styles[method]['linestyle'], label=method, linewidth=2, markersize=6)
             
    # Plot 2: Actual Active Neurons Count
    ax2.plot(epochs, active_n, 
             color=styles[method]['color'], marker=styles[method]['marker'], 
             linestyle=styles[method]['linestyle'], label=method, linewidth=2, markersize=6)

# Labels & Styling
ax1.set_title('Global Active Capacity Over Epochs\nActual Active Weights Count (~99% Sparsity)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Active Weights Count', fontsize=10)
ax1.set_xlabel('Epoch', fontsize=10)
ax1.legend(loc='best', fontsize=9, frameon=True)
ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

ax2.set_title('Global Neuron Survival Over Epochs\nActual Active Neurons Count (~99% Sparsity)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Active Neurons Count', fontsize=10)
ax2.set_xlabel('Epoch', fontsize=10)
ax2.legend(loc='best', fontsize=9, frameon=True)
ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

# Set alternate epoch ticks (even numbers)
epoch_ticks = np.arange(2, max_epochs + 1, 2)
for ax in [ax1, ax2]:
    ax.set_xticks(epoch_ticks)

plt.tight_layout()

# Save image
output_path = 'results/resnet18_cifar10_experiments/resnet18_cifar10_ratio_over_epochs_99.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"Global active weights and neurons count over epochs plot at 99% successfully saved to {output_path}")
