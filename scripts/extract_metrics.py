import json
import os
import glob

def process_dir(dir_path, experiment_name):
    print(f"## {experiment_name}\n")
    
    files = glob.glob(os.path.join(dir_path, "*.json"))
    
    # We want a specific sorting order
    # Baseline, Hebbian, SNIP, Magnitude, RigL
    def sort_key(f):
        base = os.path.basename(f).lower()
        if 'baseline' in base: return 0, base
        if 'hebbian' in base: return 1, base
        if 'snip' in base: return 2, base
        if 'magnitude' in base: return 3, base
        if 'rigl' in base: return 4, base
        return 5, base

    files.sort(key=sort_key)
    
    table_header = "| Method | Final Sparsity | Active Connections | Final Train Acc | Final Test Acc | Peak Test Acc |"
    print(table_header)
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            
        base = os.path.basename(file)
        # Parse method name
        if 'baseline' in base:
            method = "Baseline"
        elif 'hebbian' in base:
            if '1e-6' in base: method = "Hebbian (1e-6)"
            elif '1e-5' in base: method = "Hebbian (1e-5)"
            elif '5e-5' in base: method = "Hebbian (5e-5)"
            elif '1e-4' in base: method = "Hebbian (1e-4)"
            else: method = "Hebbian"
        elif 'snip' in base:
            if '0.8' in base: method = "SNIP (80%)"
            elif '0.9' in base: method = "SNIP (90%)"
            elif '0.7' in base: method = "SNIP (70%)"
            else: method = "SNIP"
        elif 'magnitude' in base:
            if '0.8' in base: method = "Magnitude (80%)"
            elif '0.9' in base: method = "Magnitude (90%)"
            elif '0.7' in base: method = "Magnitude (70%)"
            else: method = "Magnitude"
        elif 'rigl' in base:
            if '0.8' in base: method = "RigL (80%)"
            elif '0.9' in base: method = "RigL (90%)"
            elif '0.7' in base: method = "RigL (70%)"
            else: method = "RigL"
        else:
            method = base
            
        final_sparsity = data["sparsity"][-1] * 100
        active_conn = data["active_connections"][-1]
        final_tr_acc = data["train_acc"][-1]
        final_te_acc = data["test_acc"][-1]
        peak_te_acc = max(data["test_acc"])
        
        row = f"| {method} | {final_sparsity:.2f}% | {active_conn:,} | {final_tr_acc:.2f}% | {final_te_acc:.2f}% | {peak_te_acc:.2f}% |"
        print(row)
    print("\n")

print("# Pruning Methods Comparison\n")

print("Detailed comparison of various pruning methods across different architectures and datasets.\n")

process_dir(r"C:\Users\Admin\OneDrive\Desktop\hebbian_learning\results\mlp_mnist_10epoch_all", "MLP MNIST (10 Epochs)")
process_dir(r"C:\Users\Admin\OneDrive\Desktop\hebbian_learning\results\cnn_cifar_10epoch_all", "CNN CIFAR-10 (10 Epochs)")
process_dir(r"C:\Users\Admin\OneDrive\Desktop\hebbian_learning\results\vgg16_cifar_epoch20_all", "VGG16 CIFAR-10 (20 Epochs)")
