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

def get_layer_order(name):
    if 'layer.0' in name:
        base = 0
    elif 'layer.1' in name:
        base = 10
    elif 'pooler' in name:
        base = 20
    elif 'classifier' in name:
        base = 30
    else:
        base = 40
        
    if 'query' in name:
        offset = 1
    elif 'key' in name:
        offset = 2
    elif 'value' in name:
        offset = 3
    elif 'attention.output.dense' in name:
        offset = 4
    elif 'intermediate.dense' in name:
        offset = 5
    elif 'output.dense' in name:
        offset = 6
    else:
        offset = 0
    return base + offset

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
        
        # Shorten layer name for display
        short_name = layer_name.replace("bert.encoder.", "").replace("attention.self.", "attn.")
        
        layer_data[method]['layers'].append(short_name)
        layer_data[method]['weights_ratio'].append(active_w / total_w)
        layer_data[method]['neurons_ratio'].append(active_n / total_n)

# Plotting Subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2), dpi=300)

for ax in [ax1, ax2]:
    ax.grid(True, linestyle='--', alpha=0.5)

# Generate numeric layer indices for x-axis ticks
sample_method = list(layer_data.keys())[0] if layer_data else None
num_layers = len(layer_data[sample_method]['layers']) if sample_method and layer_data[sample_method]['layers'] else 13
layer_indices = [str(i + 1) for i in range(num_layers)]
x_coords = np.arange(num_layers)

# Plot 1: Active Weights Ratio
ax1.set_title('(a) Active Weights Ratio (MiniBERT @ 95% Sparsity)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Active Weights Ratio', fontsize=10, fontweight='bold')
ax1.set_xlabel(f'Layer Index (1 to {num_layers})', fontsize=10, fontweight='bold')

for method, data in layer_data.items():
    if not data['layers']:
        continue
    lw = 2.0 if styles[method]['linestyle'] == '-' else 1.3
    ax1.plot(x_coords, data['weights_ratio'], 
             color=styles[method]['color'], marker=styles[method]['marker'], 
             linestyle=styles[method]['linestyle'], label=method, linewidth=lw, markersize=6)

ax1.set_xticks(x_coords)
ax1.set_xticklabels(layer_indices, rotation=0, fontsize=9.5, ha='center')
ax1.legend(loc='upper left', frameon=True, fontsize=9)

# Plot 2: Active Neurons Ratio
ax2.set_title('(b) Active Neurons Ratio (MiniBERT @ 95% Sparsity)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Active Neurons Ratio', fontsize=10, fontweight='bold')
ax2.set_xlabel(f'Layer Index (1 to {num_layers})', fontsize=10, fontweight='bold')

for method, data in layer_data.items():
    if not data['layers']:
        continue
    lw = 2.0 if styles[method]['linestyle'] == '-' else 1.3
    ax2.plot(x_coords, data['neurons_ratio'], 
             color=styles[method]['color'], marker=styles[method]['marker'], 
             linestyle=styles[method]['linestyle'], label=method, linewidth=lw, markersize=6)

ax2.set_xticks(x_coords)
ax2.set_xticklabels(layer_indices, rotation=0, fontsize=9.5, ha='center')
# Legend on right plot is removed to avoid redundancy

plt.tight_layout()

# Save image
output_path = 'plots/bert_tiny_layer_wise_comparison_95.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"Layer-wise active weights and neurons comparison plots successfully saved to {output_path}")
