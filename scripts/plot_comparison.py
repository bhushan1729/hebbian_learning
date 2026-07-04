import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np

def main():
    workspace_path = "c:\\Users\\Admin\\OneDrive\\Desktop\\hebbian_learning"
    output_dir = os.path.join(workspace_path, "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(workspace_path, "results/mlp_mnist_experiments/history_*.json"))
    print(f"Found {len(files)} history files.")
    
    # Organize data by method
    data_by_method = {
        'DADP (Hebbian)': [],
        'Magnitude': [],
        'SNIP': [],
        'RigL': []
    }
    dense_acc = None
    
    for f in files:
        name = os.path.basename(f).replace("history_", "").replace(".json", "")
        try:
            with open(f, "r") as fh:
                data = json.load(fh)
            
            final_te_acc = data["test_acc"][-1] if data.get("test_acc") else 0.0
            
            if data.get("sparsity") and isinstance(data["sparsity"], list):
                final_sp = data["sparsity"][-1]
            else:
                final_sp = data.get("sparsity", 0.0)
            
            # Convert to percentage
            sparsity_pct = final_sp * 100.0 if final_sp <= 1.0 else final_sp
            
            if "baseline" in name:
                dense_acc = final_te_acc
            elif "hebbian" in name:
                data_by_method['DADP (Hebbian)'].append((sparsity_pct, final_te_acc))
            elif "magnitude" in name:
                data_by_method['Magnitude'].append((sparsity_pct, final_te_acc))
            elif "snip" in name:
                data_by_method['SNIP'].append((sparsity_pct, final_te_acc))
            elif "rigl" in name:
                data_by_method['RigL'].append((sparsity_pct, final_te_acc))
        except Exception as e:
            print(f"Error parsing {f}: {e}")
            
    # Sort points by sparsity for correct line plotting
    for method in data_by_method:
        data_by_method[method].sort(key=lambda x: x[0])
        
    # --- PLOT 1: FULL VIEW (70% - 100% Sparsity) ---
    plt.figure(figsize=(9, 6), dpi=150)
    
    # Reference line for Dense Baseline
    if dense_acc is not None:
        plt.axhline(y=dense_acc, color='gray', linestyle='--', alpha=0.7, label=f'Dense Baseline ({dense_acc:.2f}%)')
        
    colors = {
        'DADP (Hebbian)': '#d62728', # Red (our method)
        'Magnitude': '#1f77b4',     # Blue
        'SNIP': '#2ca02c',          # Green
        'RigL': '#9467bd'           # Purple
    }
    
    markers = {
        'DADP (Hebbian)': 'o',
        'Magnitude': 's',
        'SNIP': '^',
        'RigL': 'd'
    }
    
    for method, points in data_by_method.items():
        if not points:
            continue
        sparsities, accuracies = zip(*points)
        plt.plot(sparsities, accuracies, label=method, color=colors[method], 
                 marker=markers[method], markersize=6, linewidth=2)
        
        # Annotate DADP points specifically
        if method == 'DADP (Hebbian)':
            for s, a in points:
                if s > 95.0: # Only annotate high sparsities to avoid overlap
                    plt.annotate(f"{a:.2f}%", (s, a), textcoords="offset points", 
                                 xytext=(-15, 10), ha='center', fontsize=8, fontweight='bold', color=colors[method])
                    
    plt.title("MLP MNIST Pruning Benchmark\nTest Accuracy vs. Network Sparsity", fontsize=12, fontweight='bold')
    plt.xlabel("Sparsity (%)", fontsize=10)
    plt.ylabel("Test Accuracy (%)", fontsize=10)
    plt.xlim(68, 100.5)
    plt.ylim(75, 99.5)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower left', fontsize=9)
    
    plot_path_full = os.path.join(output_dir, "mlp_mnist_sparsity_vs_accuracy_full.png")
    plt.savefig(plot_path_full, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {plot_path_full}")

    # --- PLOT 2: ZOOMED VIEW (70% - 98.5% Sparsity) ---
    plt.figure(figsize=(9, 6), dpi=150)
    
    # Reference line for Dense Baseline
    if dense_acc is not None:
        plt.axhline(y=dense_acc, color='gray', linestyle='--', alpha=0.7, label=f'Dense Baseline ({dense_acc:.2f}%)')
        
    for method, points in data_by_method.items():
        if not points:
            continue
        # Filter points to only show sparsity <= 98.5% for zoomed clarity
        zoomed_points = [p for p in points if p[0] <= 98.5]
        if not zoomed_points:
            continue
        sparsities, accuracies = zip(*zoomed_points)
        plt.plot(sparsities, accuracies, label=method, color=colors[method], 
                 marker=markers[method], markersize=6, linewidth=2)
        
    plt.title("MLP MNIST Pruning Benchmark (Zoomed View)\nTest Accuracy vs. Network Sparsity (Up to 98% Sparsity)", fontsize=12, fontweight='bold')
    plt.xlabel("Sparsity (%)", fontsize=10)
    plt.ylabel("Test Accuracy (%)", fontsize=10)
    plt.xlim(68, 98.5)
    plt.ylim(96.0, 99.0)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower left', fontsize=9)
    
    plot_path_zoom = os.path.join(output_dir, "mlp_mnist_sparsity_vs_accuracy_zoom.png")
    plt.savefig(plot_path_zoom, bbox_inches='tight')
    plt.close()
    print(f"Zoomed comparison plot saved to {plot_path_zoom}")

if __name__ == "__main__":
    main()
