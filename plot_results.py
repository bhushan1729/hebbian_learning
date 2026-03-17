import json
import matplotlib.pyplot as plt
import argparse
import os

def plot_history(json_files):
    plt.figure(figsize=(15, 10))
    
    # Define metrics to plot
    # Each tuple: (json_key, plot_title, ylabel)
    metrics = [
        ('train_loss', 'Training Loss', 'Loss'),
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
                plt.plot(history[key], label=label, marker='o', markersize=4)
        
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    output_file = 'experiment_results.png'
    plt.savefig(output_file, dpi=300)
    print(f"Visualization saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training history from JSON files')
    parser.add_argument('files', nargs='+', help='Path to one or more history_*.json files')
    args = parser.parse_args()
    
    plot_history(args.files)
