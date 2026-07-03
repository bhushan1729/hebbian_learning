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
    losses = []
    pruned_counts = []
    
    # Regex to match the epoch table row:
    # Example: "   1    |  0.2044  |  93.85 % |  0.1175  |  96.29 % |  0.2441  |   505436  "
    pattern = re.compile(
        r"^\s*(\d+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*%\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*%\s*\|\s*([\d\.]+)\s*\|\s*(\d+)"
    )
    
    current_pruned = 0
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if "[Pruning]" in line:
                # Find all integers prefixed with +
                nums = [int(n) for n in re.findall(r"\+(\d+)", line)]
                current_pruned += sum(nums)
            else:
                match = pattern.match(line)
                if match:
                    epoch = int(match.group(1))
                    train_loss = float(match.group(2))
                    
                    epochs.append(epoch)
                    losses.append(train_loss)
                    pruned_counts.append(current_pruned)
                    
                    # Reset pruned accumulator for next epoch
                    current_pruned = 0
                    
    if not epochs:
        print("Error: Could not parse any metrics from the log file.")
        return
        
    print(f"Successfully compiled {len(epochs)} epochs.")
    
    # Create the dual-axis plot
    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=150)
    
    # Title and Grid
    plt.title("DADP Self-Regulating Negative Feedback Loop\nTraining Loss vs. Pruned Weights per Epoch (MNIST + MLP)", fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Plot 1: Training Loss (Smooth Line on Left Axis)
    color_loss = '#1f77b4' # Blue
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Training Loss', color=color_loss, fontsize=10, fontweight='bold')
    line_loss = ax1.plot(epochs, losses, color=color_loss, linewidth=2, label='Training Loss', marker='o', markersize=3)
    ax1.tick_params(axis='y', labelcolor=color_loss)
    
    # Plot 2: Pruned Weights (Bar Chart on Right Axis)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color_prune = '#d62728' # Red
    ax2.set_ylabel('Weights Pruned During Epoch', color=color_prune, fontsize=10, fontweight='bold')
    
    # Using alpha to make bars semi-transparent so the line is visible behind/in front of them
    bars_prune = ax2.bar(epochs, pruned_counts, color=color_prune, alpha=0.3, width=0.6, label='Weights Pruned')
    ax2.tick_params(axis='y', labelcolor=color_prune)
    
    # Add a horizontal dashed line showing when pruning slows down (Epoch 30+)
    ax1.axvline(x=30, color='gray', linestyle=':', alpha=0.7, label='Pruning Equilibrium (Epoch 30)')
    
    # Annotate specific examples of the bounce cycle
    # E.g. Epoch 86 (loss drops, prune spikes, epoch 87 loss bounces)
    if len(epochs) >= 97:
        # Cycle A (Epoch 86/87)
        ax1.annotate(
            "Loss Drop (0.0126)",
            xy=(86, losses[85]),
            xytext=(70, losses[85] + 0.05),
            arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.8),
            fontsize=8
        )
        ax1.annotate(
            "Loss Bounce (0.0184)",
            xy=(87, losses[86]),
            xytext=(88, losses[86] + 0.05),
            arrowprops=dict(facecolor='red', arrowstyle='->', lw=0.8),
            fontsize=8,
            color='red'
        )
        
        # Cycle B (Epoch 96/97)
        ax1.annotate(
            "Loss Drop (0.0122)",
            xy=(96, losses[95]),
            xytext=(80, losses[95] + 0.02),
            arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.8),
            fontsize=8
        )
        ax1.annotate(
            "Loss Bounce (0.0200)",
            xy=(97, losses[96]),
            xytext=(98, losses[96] + 0.03),
            arrowprops=dict(facecolor='red', arrowstyle='->', lw=0.8),
            fontsize=8,
            color='red'
        )

    # Combine legends
    lines = line_loss + [plt.Rectangle((0,0),1,1,fc=color_prune,alpha=0.3)]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plot_save_path = os.path.join(plots_dir, "mnist_mlp_loss_vs_pruning_cycles.png")
    plt.savefig(plot_save_path, bbox_inches='tight')
    print(f"Plot saved successfully to {plot_save_path}")

if __name__ == "__main__":
    main()
