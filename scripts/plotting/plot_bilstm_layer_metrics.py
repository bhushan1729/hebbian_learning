import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Define file paths to compare (aiming for ~80% target sparsity)
comparison_files = {
    'DADP (Hebbian)': 'results/bilstm_crf_experiments/history_hebbian_bilstm_crf_CoNLL2003_thr5e-05_dt500.json',
    'Magnitude': 'results/bilstm_crf_experiments/history_magnitude_bilstm_crf_CoNLL2003_sp0.8.json',
    'SNIP': 'results/bilstm_crf_experiments/history_snip_bilstm_crf_CoNLL2003_sp0.8.json',
    'RigL': 'results/bilstm_crf_experiments/history_rigl_bilstm_crf_CoNLL2003_sp0.8.json'
}

styles = {
    'DADP (Hebbian)': {'color': '#d62728', 'marker': 'o', 'linestyle': '-'},
    'Magnitude': {'color': '#1f77b4', 'marker': 's', 'linestyle': '--'},
    'SNIP': {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.'},
    'RigL': {'color': '#9467bd', 'marker': 'd', 'linestyle': ':'}
}

# Custom sorting order key function for BiLSTM-CRF layers
def get_layer_order(name):
    if 'forward_cells' in name:
        if 'fc_ih' in name:
            return 0
        return 1
    elif 'backward_cells' in name:
        if 'fc_ih' in name:
            return 2
        return 3
    elif 'hidden2tag' in name:
        return 4
    return 5

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
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, dpi=150)

for ax in [ax1, ax2]:
    ax.grid(True, linestyle='--', alpha=0.5)

# Plot 1: Active Weights Ratio
ax1.set_title('Layer-wise Capacity Distribution\nActive Weights to Total Weights Ratio (~80% Global Sparsity)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Active Weights Ratio', fontsize=11)

for method, data in layer_data.items():
    ax1.plot(data['layers'], data['weights_ratio'], 
             color=styles[method]['color'], marker=styles[method]['marker'], 
             linestyle=styles[method]['linestyle'], label=method, linewidth=2, markersize=7)

ax1.legend(loc='upper right', frameon=True)

# Plot 2: Active Neurons Ratio
ax2.set_title('Layer-wise Neuron Retention\nActive Neurons to Total Neurons Ratio (~80% Global Sparsity)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Active Neurons Ratio', fontsize=11)
ax2.set_xlabel('Network Layer Name', fontsize=11)

for method, data in layer_data.items():
    ax2.plot(data['layers'], data['neurons_ratio'], 
             color=styles[method]['color'], marker=styles[method]['marker'], 
             linestyle=styles[method]['linestyle'], label=method, linewidth=2, markersize=7)

# Format X-axis tick labels
plt.xticks(rotation=15, ha='right', fontsize=9)
plt.tight_layout()

# Save image
output_path = 'results/bilstm_crf_experiments/bilstm_crf_layer_wise_comparison.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"Layer-wise active weights and neurons comparison plots successfully saved to {output_path}")
