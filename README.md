# Hebbian-Inspired Structural Pruning on MNIST

This repository implements an activity-dependent structural sparsification algorithm for neural networks, inspired by Hebbian learning principles ("neurons that fire together, wire together").

The core idea is to track the "importance" of each connection during training and permanently prune connections that contribute little to the learning process.

## 🧠 Pruning Logic: The Hebbian Proxy

We use **Gradient × Activation** as a proxy for connection importance. A connection $w_{ij}$ between neuron $i$ in the previous layer and neuron $j$ in the current layer is considered important if:

$$importance_{ij} = E[ |a_i \cdot \frac{\partial L}{\partial y_j}| ]$$

Where:
- $a_i$ is the activation of the previous neuron.
- $\frac{\partial L}{\partial y_j}$ is the gradient of the loss with respect to the pre-activation output of the current neuron.

### Implementation:
1. **Binary Mask ($M$):** Each weight layer $W$ is associated with a binary mask $M$.
2. **Effective Weights:** During the forward pass, we use $W_{eff} = W \odot M$.
3. **Periodic Updates:** Every $K$ steps, we update the mask:
   $$M = (importance > threshold).float()$$
4. **Permanent Pruning:** Once a connection is masked (set to 0), it remains 0 for the rest of the training, following the biological principle of synaptic pruning.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- Torchvision

### Installation
```bash
pip install torch torchvision
```

## 🏗️ Baseline Architecture

The project uses a 3-layer MLP (Multi-Layer Perceptron) for MNIST classification:

| Layer | Type | Input Size | Output Size | Parameters (Weights) |
| :--- | :--- | :--- | :--- | :--- |
| **fc1** | Linear | 784 (Input) | 512 | 401,408 |
| **fc2** | Linear | 512 | 512 | 262,144 |
| **fc3** | Linear | 512 | 10 (Output) | 5,120 |
| **Total** | | | **1,034 Neurons** | **668,672 Connections** |

*Note: Neurons refer to hidden and output units. Connections refer to unique synaptic weights.*

### 🧪 How to Run

> [!NOTE]
> If running in Google Colab, please **manually mount your Google Drive** before running the script. The script will look for a folder named `hebbian_learning` in your `MyDrive`.

#### 1️⃣ Baseline Training
```bash
python main.py --mode baseline --epochs 10 --exp_name my_baseline_run
```

#### 2️⃣ Hebbian Pruning Experiment
```bash
python main.py --mode hebbian --epochs 10 --prune_threshold 0.000001 --exp_name hebbian_gentle_prune
```

### 🛠️ CLI Arguments
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--mode` | `hebbian` | `baseline` or `hebbian` |
| `--dataset` | `MNIST` | `MNIST` or `CIFAR10` |
| `--exp_name` | `None` | Custom name for .pth and .json files |
| `--prune_interval` | `500` | Steps between pruning updates |
| `--prune_threshold`| `0.0001`| Importance cutoff for pruning |
| `--colab` | `False` | Enable Google Colab pathing |
This will save checkpoints and history results to `/content/drive/MyDrive/hebbian_learning/`.

## 📈 Monitoring Results

The training loop outputs a clean, professional table:

```text
===========================================================================
 Epoch  | Tr Loss  | Tr Acc  | Te Loss  | Te Acc  | Sparsity |   Active   
---------------------------------------------------------------------------
   1    |  0.1542  |  95.42% |  0.1204  |  96.30% |  0.0000  |   668672   
   2    |  0.0821  |  97.51% |  0.0911  |  97.12% |  0.1524  |   566782   
===========================================================================
```

- **Sparsity:** The fraction of total connections that have been pruned.
- **Active:** Number of connections currently active ($W_{eff} \neq 0$).

### 🚀 Optimizing Your Pruning
If your sparsity reaches **1.0000** (total brain death), your threshold is too high. Try these settings:
- **Baseline Accuracy**: `--prune_threshold 0.000001` (very gentle)
- **Aggressive Pruning**: `--prune_threshold 0.0001` (cuts more, but riskier)

Results are saved as `.pth` checkpoints and `.json` history files in the `./results` directory.

## 📊 Visualizing Results

The `plot_results.py` script allows you to compare multiple experiments visually. It reads the `.json` history files and generates an `experiment_results.png` image.

```bash
# Compare baseline and hebbian runs
python plot_results.py results/history_baseline_MNIST.json results/history_hebbian_MNIST.json

# Save to a specific directory (e.g., Google Drive) with a custom name
python plot_results.py results/history_hebbian_MNIST.json --output_dir /content/drive/MyDrive/hebbian_learning/plots --output_name my_hebbian_plot.png
```

It plots:
- Loss and Accuracy curves.
- Structural Sparsity over time.
- Active Connections and Neurons.

---
## 🛠️ Hyperparameters
- `--lr`: Learning rate (default: 0.001)
- `--prune_interval`: Number of steps between pruning checks (default: 500)
- `--prune_threshold`: Importance threshold for pruning (default: 0.0001)
- `--batch_size`: Training batch size (default: 64)
- `--exp_name`: Custom name for experiment outputs.
