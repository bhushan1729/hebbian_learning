import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np

# Find all JSON history files
files = glob.glob('results/bilstm_crf_experiments/*.json')

# Initialize data structures to collect (sparsity, accuracy) for each method
data_points = {
    'hebbian': [],
    'magnitude': [],
    'snip': [],
    'rigl': []
}

dense_baseline_acc = 93.69  # Default fallback if baseline file not found

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
            # We want to filter out duplicated 100% sparsity points for clean plotting
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
    if sp > 25.0:
        if abs(sp - 29.09) < 0.5:
            plt.text(sp - 2.8, acc + 0.6, f'{acc:.2f}%', color='#d62728', fontweight='bold', fontsize=9)
        elif abs(sp - 77.96) < 0.5:
            plt.text(sp - 2.8, acc - 1.2, f'{acc:.2f}%', color='#d62728', fontweight='bold', fontsize=9)
        elif abs(sp - 82.47) < 0.5:
            plt.text(sp - 2.8, acc + 0.6, f'{acc:.2f}%', color='#d62728', fontweight='bold', fontsize=9)
        elif abs(sp - 100.0) < 0.5:
            plt.text(sp - 3.8, acc + 0.8, f'{acc:.2f}%', color='#d62728', fontweight='bold', fontsize=9)

# Annotate key points of other methods at 95%
for mode in ['snip', 'magnitude', 'rigl']:
    pts = data_points[mode]
    for sp, acc in pts:
        if abs(sp - 95.00) < 0.1:
            plt.text(sp - 2.8, acc - 1.2, f'{acc:.2f}%', color=styles[mode]['color'], fontweight='bold', fontsize=9)

# Styling and Labels
plt.title('BiLSTM-CRF CoNLL2003 Pruning Benchmark\nTest Accuracy vs. Network Sparsity', fontsize=13, fontweight='bold', pad=10)
plt.xlabel('Sparsity (%)', fontsize=11)
plt.ylabel('Test Accuracy (%)', fontsize=11)
plt.xlim(-2, 103)
plt.ylim(80, 96)

# Add grid lines
plt.grid(True, linestyle='--', alpha=0.5)

# Place legend
plt.legend(loc='lower left', fontsize=10, frameon=True)

# Save the plot
output_path = 'results/bilstm_crf_experiments/bilstm_crf_sparsity_vs_accuracy.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"Benchmark plot successfully generated and saved to {output_path}")
