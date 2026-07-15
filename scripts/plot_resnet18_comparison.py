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

# 3. Add text labels on key points
# Annotate key DADP points
hebbian_pts = data_points['hebbian']
for sp, acc in hebbian_pts:
    if sp > 88.0:
        if abs(sp - 89.62) < 0.5:
            plt.text(sp - 2.8, acc + 0.6, f'{acc:.2f}%', color='#d62728', fontweight='bold', fontsize=9)
        elif abs(sp - 95.58) < 0.5:
            plt.text(sp - 2.8, acc + 0.8, f'{acc:.2f}%', color='#d62728', fontweight='bold', fontsize=9)
        elif abs(sp - 99.23) < 0.5:
            plt.text(sp - 2.8, acc + 0.8, f'{acc:.2f}%', color='#d62728', fontweight='bold', fontsize=9)

# Annotate collapsed / high-sparsity points for other methods
snip_pts = data_points['snip']
for sp, acc in snip_pts:
    if abs(sp - 99.00) < 0.1:
        plt.text(sp - 3.2, acc - 1.5, f'{acc:.2f}%', color='#2ca02c', fontweight='bold', fontsize=9)

mag_pts = data_points['magnitude']
for sp, acc in mag_pts:
    if abs(sp - 99.00) < 0.1:
        plt.text(sp - 3.2, acc - 1.5, f'{acc:.2f}%', color='#1f77b4', fontweight='bold', fontsize=9)

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
