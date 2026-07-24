import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np

# Find all ResNet-18 history JSON files
files = glob.glob('results/resnet18_cifar10_experiments/*.json')

data_points = {
    'hebbian': [],
    'magnitude': [],
    'snip': [],
    'rigl': []
}

dense_baseline_acc = 76.06

# Parse JSONs
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
            # Avoid duplicated 100% collapsed points for clean curves
            if mode == 'hebbian' and final_sparsity == 100.0 and len([p for p in data_points['hebbian'] if p[0] == 100.0]) > 0:
                continue
            data_points[mode].append((final_sparsity, final_acc))

# Sort points by sparsity for plotting
for mode in data_points:
    data_points[mode] = sorted(data_points[mode], key=lambda x: x[0])

# Compute Accuracy Change (%) relative to baseline
accuracy_changes = {}
for mode in data_points:
    accuracy_changes[mode] = []
    for sp, acc in data_points[mode]:
        acc_change = acc - dense_baseline_acc
        accuracy_changes[mode].append((sp, acc_change))

# Plotting
plt.figure(figsize=(10, 6), dpi=150)
plt.grid(True, linestyle='--', alpha=0.5)

colors = {
    'hebbian': '#d62728',   # Red
    'magnitude': '#1f77b4',  # Blue
    'snip': '#2ca02c',       # Green
    'rigl': '#9467bd'        # Purple
}

labels = {
    'hebbian': 'DADP (Hebbian)',
    'magnitude': 'Magnitude (One-shot)',
    'snip': 'SNIP (One-shot)',
    'rigl': 'RigL (Dynamic)'
}

markers = {
    'hebbian': 'o',
    'magnitude': 's',
    'snip': '^',
    'rigl': 'd'
}

# Plot baseline reference line at 0.0% accuracy change
plt.axhline(y=0.0, color='black', linestyle='-', linewidth=1.2, label='Dense Baseline')

# Plot curves
for mode in ['hebbian', 'magnitude', 'snip', 'rigl']:
    pts = accuracy_changes[mode]
    if not pts:
        continue
    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    plt.plot(x, y, color=colors[mode], marker=markers[mode], markersize=7, 
             linewidth=2.2, label=labels[mode])

# 3. Add text labels on all points dynamically avoiding overlaps
points_by_sp = {}
for mode in ['hebbian', 'magnitude', 'snip', 'rigl']:
    for sp, acc_change in accuracy_changes[mode]:
        sp_round = round(sp, 1)
        if sp_round not in points_by_sp:
            points_by_sp[sp_round] = []
        points_by_sp[sp_round].append((acc_change, mode, sp))

for sp_round, pts in points_by_sp.items():
    pts = sorted(pts, key=lambda val: val[0], reverse=True)
    for rank, (acc_change, mode, sp_orig) in enumerate(pts):
        if acc_change < -20.0:
            continue
        offset_y = 0.3 if rank % 2 == 0 else -0.7
        if mode == 'hebbian':
            offset_y = 0.4
        elif mode == 'rigl':
            offset_y = -0.8
        plt.text(sp_orig, acc_change + offset_y, f'{acc_change:+.1f}%', 
                 color=colors[mode], fontsize=8, ha='center', fontweight='bold')

# Styling to match Song Han's paper layout
plt.title('ResNet-18 CIFAR-10 Accuracy Trade-off vs. Sparsity\nRelative Accuracy Change from Dense Baseline', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Parameters Pruned Away (%)', fontsize=11)
plt.ylabel('Accuracy Change (%)', fontsize=11)

# Format X-axis and Y-axis to focus on the interesting high-sparsity trade-off region
plt.xlim(65, 101)
plt.ylim(-20.0, 2.0)  # Focus on the area showing recovery/collapse boundaries

# Format tick labels to show percentage symbols
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{int(x)}%'))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, pos: f'{y:+.1f}%' if y != 0 else '0.0%'))

# Add legend
plt.legend(loc='lower left', fontsize=10, frameon=True)

# Save the plot
output_path = 'plots/resnet18_accuracy_loss_comparison.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"ResNet-18 accuracy loss trade-off plot successfully saved to {output_path}")
