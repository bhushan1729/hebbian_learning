import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np

# Find all VGG-16 history JSON files
files = glob.glob('results/vgg16_cifar10_experiments/*.json')

data_points = {
    'hebbian': [],
    'magnitude': [],
    'snip': [],
    'rigl': []
}

dense_baseline_acc = 84.46

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

# Styling to match Song Han's paper layout
plt.title('VGG-16 CIFAR-10 Accuracy Trade-off vs. Sparsity\nRelative Accuracy Change from Dense Baseline', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Parameters Pruned Away (%)', fontsize=11)
plt.ylabel('Accuracy Change (%)', fontsize=11)

# Format X-axis and Y-axis to focus on the interesting high-sparsity trade-off region
plt.xlim(65, 101)
plt.ylim(-15.0, 1.5)  # Focus on the area showing recovery/collapse boundaries

# Format tick labels to show percentage symbols
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{int(x)}%'))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, pos: f'{y:+.1f}%' if y != 0 else '0.0%'))

# Add legend
plt.legend(loc='lower left', fontsize=10, frameon=True)

# Save the plot
output_path = 'plots/vgg16_accuracy_loss_comparison.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"Accuracy loss trade-off plot successfully saved to {output_path}")
