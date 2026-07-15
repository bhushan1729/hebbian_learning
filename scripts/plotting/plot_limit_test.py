import os
import re
import matplotlib.pyplot as plt
import numpy as np

def main():
    log_path = "logs/hebbian_mlp_MNIST_thr1e-05_dt500_epoch100.log"
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    if not os.path.exists(log_path):
        print(f"Error: Log file not found at {log_path}")
        return
        
    epochs = []
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    sparsities = []
    active_conns = []
    
    # Regex to match the epoch metrics line in the log file
    # Example: "   1    |  0.2044  |  93.85 % |  0.1175  |  96.29 % |  0.2441  |   505436  "
    pattern = re.compile(
        r"^\s*(\d+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*%\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*%\s*\|\s*([\d\.]+)\s*\|\s*(\d+)"
    )
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.match(line)
            if match:
                epochs.append(int(match.group(1)))
                train_losses.append(float(match.group(2)))
                train_accs.append(float(match.group(3)))
                test_losses.append(float(match.group(4)))
                test_accs.append(float(match.group(5)))
                sparsities.append(float(match.group(6)))
                active_conns.append(int(match.group(7)))
                
    if not epochs:
        print("Error: Could not parse any epoch metrics from the log file.")
        return
        
    print(f"Successfully parsed {len(epochs)} epochs from log file.")
    
    # Create a 2x2 grid of plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150)
    fig.suptitle("DADP MLP MNIST 100-Epoch Ablation Limit Test (Threshold: 1e-5)", fontsize=14, fontweight='bold', y=0.98)
    
    # Plot 1: Loss
    ax_loss = axes[0, 0]
    ax_loss.plot(epochs, train_losses, label="Train Loss", color="#1f77b4", linewidth=1.5)
    ax_loss.plot(epochs, test_losses, label="Test Loss", color="#ff7f0e", linewidth=1.5)
    ax_loss.set_title("Training & Test Loss", fontsize=11, fontweight='bold')
    ax_loss.set_xlabel("Epoch", fontsize=9)
    ax_loss.set_ylabel("Loss", fontsize=9)
    ax_loss.grid(True, linestyle='--', alpha=0.5)
    ax_loss.legend(fontsize=8)
    
    # Plot 2: Accuracy
    ax_acc = axes[0, 1]
    ax_acc.plot(epochs, train_accs, label="Train Acc", color="#2ca02c", linewidth=1.5)
    ax_acc.plot(epochs, test_accs, label="Test Acc", color="#d62728", linewidth=1.5)
    ax_acc.set_title("Training & Test Accuracy", fontsize=11, fontweight='bold')
    ax_acc.set_xlabel("Epoch", fontsize=9)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=9)
    ax_acc.grid(True, linestyle='--', alpha=0.5)
    ax_acc.legend(fontsize=8)
    
    # Annotate peak test accuracy
    peak_idx = np.argmax(test_accs)
    peak_epoch = epochs[peak_idx]
    peak_acc = test_accs[peak_idx]
    ax_acc.annotate(
        f"Peak: {peak_acc:.2f}% (Epoch {peak_epoch})",
        xy=(peak_epoch, peak_acc),
        xytext=(peak_epoch - 25, peak_acc - 5),
        arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.8),
        fontsize=8,
        fontweight='bold'
    )
    
    # Plot 3: Sparsity
    ax_sp = axes[1, 0]
    ax_sp.plot(epochs, sparsities, color="#9467bd", linewidth=1.8, label="Sparsity")
    ax_sp.set_title("Network Sparsity over Epochs", fontsize=11, fontweight='bold')
    ax_sp.set_xlabel("Epoch", fontsize=9)
    ax_sp.set_ylabel("Sparsity (0.0 - 1.0)", fontsize=9)
    ax_sp.grid(True, linestyle='--', alpha=0.5)
    ax_sp.legend(fontsize=8)
    
    # Plot 4: Active Connections
    ax_conn = axes[1, 1]
    ax_conn.plot(epochs, active_conns, color="#bcbd22", linewidth=1.8, label="Active Connections")
    ax_conn.set_title("Active Weight Connections Count", fontsize=11, fontweight='bold')
    ax_conn.set_xlabel("Epoch", fontsize=9)
    ax_conn.set_ylabel("Count", fontsize=9)
    ax_conn.grid(True, linestyle='--', alpha=0.5)
    ax_conn.legend(fontsize=8)
    
    plt.tight_layout()
    plot_save_path = os.path.join(plots_dir, "mnist_mlp_limit_test_100epochs.png")
    plt.savefig(plot_save_path, bbox_inches='tight')
    print(f"Plot saved successfully to {plot_save_path}")
    
if __name__ == "__main__":
    main()
