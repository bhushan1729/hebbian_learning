import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Define file paths to compare (aiming for ~95% target sparsity)
comparison_files = {
    'DADP (Hebbian, thr=3e-6)': 'results/bert_experiments/results/history_hebbian_bert_SST2_thr3e-06_dt500.json',
    'SNIP (sp=0.95)': 'results/bert_experiments/results/history_snip_bert_SST2_sp0.95.json',
    'Magnitude (sp=0.95)': 'results/bert_experiments/results/history_magnitude_bert_SST2_sp0.95.json',
    'RigL (sp=0.95)': 'results/bert_experiments/results/history_rigl_bert_SST2_sp0.95.json'
}

methods = ['DADP (Hebbian, thr=3e-6)', 'SNIP (sp=0.95)', 'Magnitude (sp=0.95)', 'RigL (sp=0.95)']
colors = {
    'DADP (Hebbian, thr=3e-6)': '#d62728',
    'SNIP (sp=0.95)': '#2ca02c',
    'Magnitude (sp=0.95)': '#1f77b4',
    'RigL (sp=0.95)': '#9467bd'
}

layer_sparsities = {m: [] for m in methods}
layers = []

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

# Fetch data
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
    
    # Sort layers
    sorted_layers = sorted(list(last_epoch_data.keys()), key=get_layer_order)
    
    if not layers:
        layers = sorted_layers
        
    for layer in layers:
        if layer in last_epoch_data:
            # We want sparsity in percentage
            sparsity_val = last_epoch_data[layer]['sparsity'] * 100
            layer_sparsities[method].append(sparsity_val)
        else:
            layer_sparsities[method].append(0.0)

# Shorten layer names for visualization labels
short_layer_names = []
for name in layers:
    # Remove "bert." or other prefixes
    short_name = name.replace("bert.encoder.", "").replace("attention.self.", "attn.")
    short_layer_names.append(short_name)

# Plotting
x = np.arange(len(layers))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

for i, method in enumerate(methods):
    offset = (i - 1.5) * width
    ax.bar(x + offset, layer_sparsities[method], width, 
           label=method, color=colors[method], alpha=0.9, edgecolor='black', linewidth=0.5)

ax.set_ylabel('Layer-wise Sparsity (%)', fontsize=11)
ax.set_title('BERT-Tiny Layer-wise Sparsity Comparison\nComparison of Parameter Allocations at ~95% Target Sparsity', fontsize=12, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(short_layer_names, rotation=45, ha='right', fontsize=9)
ax.set_ylim(0, 105)
ax.grid(True, linestyle='--', alpha=0.4, axis='y')
ax.legend(loc='lower right', fontsize=10)

# Save the plot
output_path = 'plots/bert_tiny_layer_sparsity_bar_chart.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"BERT-Tiny layer sparsity bar chart successfully saved to {output_path}")
