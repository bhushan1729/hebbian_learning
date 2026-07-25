import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np

# Find all JSON history files
files = glob.glob('results/bert_experiments/results/*.json')

# Initialize data structures to collect (sparsity, accuracy) for each method
data_points = {
    'hebbian': [],
    'magnitude': [],
    'snip': [],
    'rigl': []
}

dense_baseline_acc = 82.68  # Default fallback if baseline file not found

for f in files:
    with open(f, 'r') as file:
        data = json.load(file)
        sparsity = data.get('sparsity', [])
        test_acc = data.get('test_acc', [])
        
        final_sparsity = sparsity[-1] * 100 if sparsity else 0.0
        final_acc = max(test_acc) if test_acc else 0.0
        
        config = data.get('config', {})
        mode = config.get('mode', 'unknown')
        
        if mode == 'baseline':
            dense_baseline_acc = final_acc
        elif mode in data_points:
            # Avoid duplicates
            data_points[mode].append((final_sparsity, final_acc))

# Sort points by sparsity for each method
for mode in data_points:
    data_points[mode] = sorted(data_points[mode], key=lambda x: x[0])

# Plotting
plt.figure(figsize=(10, 7), dpi=150)

styles = {
    'hebbian': {'color': '#d62728', 'marker': 'o', 'label': 'DADP (Hebbian)', 'linewidth': 2},
    'magnitude': {'color': '#1f77b4', 'marker': 's', 'label': 'Magnitude', 'linewidth': 2},
    'snip': {'color': '#2ca02c', 'marker': '^', 'label': 'SNIP', 'linewidth': 2},
    'rigl': {'color': '#9467bd', 'marker': 'd', 'label': 'RigL', 'linewidth': 2}
}

# 1. Plot Dense Baseline
plt.axhline(y=dense_baseline_acc, color='darkgray', linestyle='--', linewidth=1.5,
            label=f'Dense Baseline ({dense_baseline_acc:.2f}%)')

# 2. Plot lines for each pruning method
for mode in ['hebbian', 'magnitude', 'snip', 'rigl']:
    pts = data_points[mode]
    if not pts:
        continue
    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    plt.plot(x, y, color=styles[mode]['color'], marker=styles[mode]['marker'],
             linewidth=styles[mode]['linewidth'], label=styles[mode]['label'], markersize=8)

# 3. Add text labels on all points dynamically avoiding overlaps
points_by_sp = {}
for mode in ['hebbian', 'magnitude', 'snip', 'rigl']:
    for sp, acc in data_points[mode]:
        sp_round = round(sp, 1)
        if sp_round not in points_by_sp:
            points_by_sp[sp_round] = []
        points_by_sp[sp_round].append((acc, mode, sp))

for sp_round, pts in points_by_sp.items():
    pts = sorted(pts, key=lambda val: val[0], reverse=True)
    for rank, (acc, mode, sp_orig) in enumerate(pts):
        offset_y = 0.6 if rank % 2 == 0 else -1.4
        if mode == 'hebbian':
            offset_y = 0.8
        elif mode == 'rigl':
            offset_y = -1.6
        plt.text(sp_orig, acc + offset_y, f'{acc:.1f}%', 
                 color=styles[mode]['color'], fontsize=8, ha='center', fontweight='bold')

# Styling and Labels
plt.title('BERT-Tiny SST-2 Sentiment Classification Pruning Benchmark\nTest Accuracy vs. Network Sparsity (10 Epochs)', fontsize=13, fontweight='bold', pad=10)
plt.xlabel('Sparsity (%)', fontsize=11)
plt.ylabel('Test Accuracy (%)', fontsize=11)
plt.xlim(50, 101)
plt.ylim(45, 87)

# Add grid lines
plt.grid(True, linestyle='--', alpha=0.5)

# Place legend
plt.legend(loc='lower left', fontsize=10)

# Save plot
os.makedirs('plots', exist_ok=True)
plot_path = 'plots/bert_tiny_sst2_sparsity_vs_accuracy.png'
plt.savefig(plot_path, bbox_inches='tight')
print(f"Comparison plot saved successfully to {plot_path}!")
plt.close()
