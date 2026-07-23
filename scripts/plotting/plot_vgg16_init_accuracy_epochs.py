import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Define file paths to compare (VGG-16 across all 7 initialization runs)
comparison_files = {
    'Kaiming Normal': 'results/init_ablation_vgg16/results/history_init_run_vgg16_kaiming_normal.json',
    'Kaiming Uniform': 'results/init_ablation_vgg16/results/history_init_run_vgg16_kaiming_uniform.json',
    'Xavier Normal': 'results/init_ablation_vgg16/results/history_init_run_vgg16_xavier_normal.json',
    'Xavier Uniform': 'results/init_ablation_vgg16/results/history_init_run_vgg16_xavier_uniform.json',
    'Orthogonal': 'results/init_ablation_vgg16/results/history_init_run_vgg16_orthogonal.json',
    'Normal (sigma=0.02) [Collapsed]': 'results/init_ablation_vgg16/results/history_init_run_vgg16_normal_0.02.json',
    'Normal (sigma=0.1)': 'results/init_ablation_vgg16/results/history_init_run_vgg16_normal_0.1.json'
}

styles = {
    'Kaiming Normal': {'color': '#d62728', 'marker': 'o', 'linestyle': '-'},
    'Kaiming Uniform': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--'},
    'Xavier Normal': {'color': '#1f77b4', 'marker': '^', 'linestyle': '-.'},
    'Xavier Uniform': {'color': '#17becf', 'marker': 'v', 'linestyle': ':'},
    'Orthogonal': {'color': '#2ca02c', 'marker': 'd', 'linestyle': '-'},
    'Normal (sigma=0.02) [Collapsed]': {'color': '#7f7f7f', 'marker': 'x', 'linestyle': ':'},
    'Normal (sigma=0.1)': {'color': '#9467bd', 'marker': '*', 'linestyle': '--'}
}

plt.figure(figsize=(10, 6), dpi=150)
plt.grid(True, linestyle='--', alpha=0.5)

for method, path in comparison_files.items():
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Skipping.")
        continue
        
    with open(path, 'r') as f:
        data = json.load(f)
        
    test_accs = data.get('test_acc', [])
    if not test_accs:
        continue
        
    epochs = np.arange(1, len(test_accs) + 1)
    lw = 2.0 if styles[method]['linestyle'] == '-' else 1.3
    plt.plot(epochs, test_accs, 
             color=styles[method]['color'], marker=styles[method]['marker'], 
             linestyle=styles[method]['linestyle'], label=method, linewidth=lw, markersize=5)

plt.title('VGG-16 Initialization Sensitivity Study\nTest Accuracy vs. Epochs on CIFAR-10', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Epoch', fontsize=11, fontweight='bold')
plt.ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
plt.xlim(1, 20)
plt.ylim(0, 100)
plt.legend(loc='lower right', frameon=True, fontsize=9)
plt.tight_layout()

# Save image to both results subfolder and primary plots directory
output_paths = [
    'results/init_ablation_vgg16/plots/vgg16_init_accuracy_over_epochs.png',
    'plots/vgg16_init_accuracy_over_epochs.png'
]

for out_path in output_paths:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', dpi=300)

plt.close()

print(f"VGG-16 accuracy over epochs plots successfully saved to: {output_paths}")
