import json
import matplotlib.pyplot as plt
import argparse
import os

def plot_history(json_files, output_dir=None, output_name=None):
    plt.figure(figsize=(18, 10))
    
    # Define metrics to plot
    metrics = [
        ('train_loss', 'Training Loss', 'Loss'),
        ('train_acc', 'Training Accuracy', 'Accuracy (%)'),
        ('test_acc', 'Test Accuracy', 'Accuracy (%)'),
        ('sparsity', 'Network Sparsity', 'Sparsity (0.0 - 1.0)'),
        ('active_connections', 'Active Connections', 'Count'),
        ('active_neurons', 'Active Neurons', 'Count')
    ]
    
    num_metrics = len(metrics)
    cols = 3
    rows = (num_metrics + cols - 1) // cols
    
    plt.figure(figsize=(18, 5 * rows))
    
    for i, (key, title, ylabel) in enumerate(metrics, 1):
        plt.subplot(rows, cols, i)
        for json_file in json_files:
            if not os.path.exists(json_file):
                print(f"Warning: {json_file} not found.")
                continue
                
            with open(json_file, 'r') as f:
                history = json.load(f)
            
            if key in history and len(history[key]) > 0:
                label = os.path.basename(json_file).replace('history_', '').replace('.json', '')
                data = history[key]
                color = plt.gca().get_lines()[-1].get_color() if plt.gca().get_lines() else None
                line, = plt.plot(data, label=label, marker='o', markersize=4)
                
                # Annotate EACH point
                for x, y in enumerate(data):
                    if 'loss' in key:
                        val_str = f"{y:.3f}"
                    elif 'acc' in key or 'sparsity' in key:
                        val_str = f"{y:.2f}"
                    else:
                        val_str = f"{int(y):,}" if y > 1000 else f"{int(y)}"
                    
                    plt.text(x, y, val_str, fontsize=6, color=line.get_color(), 
                             ha='center', va='bottom', fontweight='light')

                # Annotate threshold if available in config
                if 'config' in history and 'prune_threshold' in history['config']:
                    thresh = history['config']['prune_threshold']
                    if key in ['sparsity', 'active_connections', 'active_neurons']:
                        idx = 0
                        if 'sparsity' in history:
                            for s_idx, s_val in enumerate(history['sparsity']):
                                if s_val > 0:
                                    idx = s_idx
                                    break
                        plt.annotate(f"T={thresh}", (idx, data[idx]), textcoords="offset points", 
                                     xytext=(0,-15), ha='center', fontsize=7, fontweight='bold', 
                                     color='red', arrowprops=dict(arrowstyle="->", color='red', alpha=0.5))
        
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    
    if not output_name:
        # Generate name from first file if not provided
        first_file = str(json_files[0])
        first_label = os.path.basename(first_file).replace('history_', '').replace('.json', '')
        output_name = f"plot_{first_label}.png"
        
    if output_dir:
        output_dir = str(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, output_name)
    else:
        output_path = output_name
        
    plt.savefig(output_path, dpi=300)
    print(f"Visualization saved to {output_path}")
    if not output_dir: # Only show if not saving to a specific remote dir to avoid blocking in non-GUI envs
        try:
            plt.show()
        except:
            print("Could not display plot (no GUI).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training history from JSON files')
    parser.add_argument('files', nargs='+', help='Path to one or more history_*.json files')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save the plot')
    parser.add_argument('--output_name', type=str, default=None, help='Filename for the plot (e.g. experiment.png)')
    args = parser.parse_args()
    
    if not args.files:
        print("Error: No history files provided.")
    else:
        plot_history(args.files, args.output_dir, args.output_name)
