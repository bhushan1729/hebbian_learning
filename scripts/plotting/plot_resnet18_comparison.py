import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np

# Find all JSON history files
files = glob.glob('results/resnet18_cifar10_experiments/*.json')

# Initialize data structures to collect (sparsity, accuracy) for each method
data_points = {
    'hebbian': [],
    'magnitude': [],
    'snip': [],
    'rigl': []
}

dense_baseline_acc = 76.06  # Default fallback if baseline file not found

for f in files:
    with open(f, 'r') as file:
        data = json.load(file)
        sparsity = data.get('sparsity', [])
        test_acc = data.get('test_acc', [])
        
        final_sparsity = sparsity[-1] * 100 if sparsity else 0.0
        final_acc = test_acc[-1] if test_acc else 0.0
        
        config = data.get('config', {})
        mode = config.get('mode', 'unknown')
        
        if mode == 'baseline':
            dense_baseline_acc = final_acc
        elif mode in data_points:
            # Avoid duplicate 100% or extreme sparsity points if any
            if mode == 'hebbian' and final_sparsity == 100.0 and len([p for p in data_points['hebbian'] if p[0] == 100.0]) > 0:
                continue
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
        if acc < 20.0:
            continue
        offset_y = 0.6 if rank % 2 == 0 else -1.4
        if mode == 'hebbian':
            offset_y = 0.8
        elif mode == 'rigl':
            offset_y = -1.6
        plt.text(sp_orig, acc + offset_y, f'{acc:.1f}%', 
                 color=styles[mode]['color'], fontsize=8, ha='center', fontweight='bold')

# Styling and Labels
plt.title('ResNet-18 CIFAR-10 Pruning Benchmark\nTest Accuracy vs. Network Sparsity', fontsize=13, fontweight='bold', pad=10)
plt.xlabel('Sparsity (%)', fontsize=11)
plt.ylabel('Test Accuracy (%)', fontsize=11)
plt.xlim(67, 101)
plt.ylim(55, 80)

# Add grid lines
plt.grid(True, linestyle='--', alpha=0.5)

# Place legend
plt.legend(loc='lower left', fontsize=10, frameon=True)

# Save the plot
output_path = 'results/resnet18_cifar10_experiments/resnet18_cifar10_sparsity_vs_accuracy.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"Benchmark plot successfully generated and saved to {output_path}")
