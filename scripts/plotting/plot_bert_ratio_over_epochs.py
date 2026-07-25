import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Define file paths to compare (aiming for ~95% target sparsity)
comparison_files = {
    'DADP (Hebbian)': 'results/bert_experiments/results/history_hebbian_bert_SST2_thr3e-06_dt500.json',
    'Magnitude': 'results/bert_experiments/results/history_magnitude_bert_SST2_sp0.95.json',
    'SNIP': 'results/bert_experiments/results/history_snip_bert_SST2_sp0.95.json',
    'RigL': 'results/bert_experiments/results/history_rigl_bert_SST2_sp0.95.json'
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
    # Fallback to key 'active' if 'active_connections' doesn't exist
    if not active_w:
        active_w = data.get('active', [])
        
    active_n = data.get('active_neurons', [])
    
    if not active_w or not active_n:
        print(f"Warning: Missing data in {path}. Skipping.")
        continue
        
    max_epochs = max(max_epochs, len(active_w))
    epochs = np.arange(1, len(active_w) + 1)
    
    # Plot 1: Actual Active Weights Count
    lw = 2.0 if styles[method]['linestyle'] == '-' else 1.3
    ax1.plot(epochs, active_w, 
             color=styles[method]['color'], marker=styles[method]['marker'], 
             linestyle=styles[method]['linestyle'], label=method, linewidth=lw, markersize=6)
             
    # Plot 2: Actual Active Neurons Count
    ax2.plot(epochs, active_n, 
             color=styles[method]['color'], marker=styles[method]['marker'], 
             linestyle=styles[method]['linestyle'], label=method, linewidth=lw, markersize=6)

# Labels & Styling
ax1.set_title('Global Active Capacity Over Epochs\nActual Active Weights Count (~95% Sparsity)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Active Weights Count', fontsize=10)
ax1.set_xlabel('Epoch', fontsize=10)
ax1.legend(loc='best', fontsize=9, frameon=True)
ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

ax2.set_title('Global Neuron Survival Over Epochs\nActual Active Neurons Count (~95% Sparsity)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Active Neurons Count', fontsize=10)
ax2.set_xlabel('Epoch', fontsize=10)
ax2.legend(loc='best', fontsize=9, frameon=True)
ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

# Set alternate epoch ticks (even numbers or consecutive numbers)
epoch_ticks = np.arange(1, max_epochs + 1)
for ax in [ax1, ax2]:
    ax.set_xticks(epoch_ticks)

plt.tight_layout()

# Save image
output_path = 'plots/bert_tiny_ratio_over_epochs_95.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"Global active weights and neurons count over epochs plot at 95% successfully saved to {output_path}")
