import os
import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    workspace_path = "c:\\Users\\Admin\\OneDrive\\Desktop\\hebbian_learning"
    output_dirs = [
        os.path.join(workspace_path, "plots"),
        os.path.join(workspace_path, "results/mlp_mnist_experiments"),
        "C:\\Users\\Admin\\.gemini\\antigravity-ide\\brain\\12bb9f2c-386e-4c81-8bc8-b4faa5833f1f\\plots"
    ]
    
    # Ensure all directories exist
    for d in output_dirs:
        os.makedirs(d, exist_ok=True)
        
    files = {
        'DADP (Hebbian, thr=1e-5)': 'results/mlp_mnist_experiments/history_hebbian_mlp_MNIST_thr1e-05_dt500.json',
        'SNIP (sp=0.95)': 'results/mlp_mnist_experiments/history_snip_mlp_MNIST_sp0.95.json',
        'Magnitude (sp=0.95)': 'results/mlp_mnist_experiments/history_magnitude_mlp_MNIST_sp0.95.json',
        'RigL (sp=0.95)': 'results/mlp_mnist_experiments/history_rigl_mlp_MNIST_sp0.95.json'
    }
    
    # We want to extract final epoch's sparsity for fc1, fc2, fc3
    layers_ordered = ['fc1', 'fc2', 'fc3']
    methods_data = {}
    
    for method_name, rel_path in files.items():
        f = os.path.join(workspace_path, rel_path)
        if not os.path.exists(f):
            print(f"Warning: {f} not found.")
            continue
            
        with open(f, 'r') as fh:
            data = json.load(fh)
            
        if 'layer_sparsity' in data and data['layer_sparsity']:
            epochs = list(data['layer_sparsity'].keys())
            epochs.sort(key=lambda x: int(x.split('_')[1]))
            last_epoch = epochs[-1]
            
            methods_data[method_name] = {}
            for layer in layers_ordered:
                val = data['layer_sparsity'][last_epoch].get(layer, {}).get('sparsity', 0.0)
                methods_data[method_name][layer] = val * 100.0
                
    if not methods_data:
        print("Error: No data loaded.")
        return
        
    # Plotting grouped bar chart
    x = np.arange(len(layers_ordered)) # label locations
    width = 0.2 # width of bars
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    colors = {
        'DADP (Hebbian, thr=1e-5)': '#d62728', # Crimson
        'SNIP (sp=0.95)': '#2ca02c',          # Green
        'Magnitude (sp=0.95)': '#1f77b4',     # Blue
        'RigL (sp=0.95)': '#9467bd'           # Purple
    }
    
    # Position offsets for the 4 bars
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    
    for idx, (method_name, layer_data) in enumerate(methods_data.items()):
        sparsities = [layer_data[l] for l in layers_ordered]
        rects = ax.bar(x + offsets[idx], sparsities, width, label=method_name, color=colors[method_name])
        
        # Add labels on top of bars
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7, fontweight='bold')
            
    # Add title, labels, ticks
    ax.set_title('Layer-wise Network Sparsity Distribution\nComparing DADP vs. SNIP, Magnitude, and RigL at ~95% Global Sparsity', fontsize=12, fontweight='bold')
    ax.set_xlabel('Layer Name', fontsize=10)
    ax.set_ylabel('Sparsity (%)', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(['fc1 (Input Layer)', 'fc2 (Hidden Layer)', 'fc3 (Output Layer)'], fontsize=9, fontweight='bold')
    ax.set_ylim(0, 110) # extra space for labels
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='lower left', fontsize=9)
    
    plt.tight_layout()
    
    for d in output_dirs:
        save_path = os.path.join(d, "mlp_mnist_layer_sparsity_comparison.png")
        plt.savefig(save_path, bbox_inches='tight')
        
    print("Layer sparsity plots saved successfully!")

if __name__ == '__main__':
    main()
