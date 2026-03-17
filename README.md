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

## 📊 Running Experiments

You can run two types of experiments: `baseline` (standard MLP) and `hebbian` (masked MLP with pruning).

### 1. Baseline Experiment
To train a standard dense network on MNIST:
```bash
python main.py --mode baseline --epochs 10
```

### 2. Hebbian Pruning Experiment
To train with activity-dependent pruning:
```bash
python main.py --mode hebbian --epochs 10 --prune_interval 500 --prune_threshold 0.0001
```

### ☁️ Google Colab Usage
If you are running in Google Colab with GPU support, use the `--colab` flag to automatically mount Google Drive and save results there.

```python
# In a Colab cell:
!python main.py --colab --mode hebbian --epochs 20
```
This will save checkpoints and history results to `/content/drive/MyDrive/hebbian_learning/`.

## 📈 Monitoring Results

The training loop outputs:
- **Loss and Accuracy** for both training and validation sets.
- **Sparsity:** The fraction of total connections that have been pruned.
- **Pruned Count:** Total number of connections cut.

Results are saved as `.pth` checkpoints and `.json` history files in the `./results` directory.

## 🛠️ Hyperparameters
- `--lr`: Learning rate (default: 0.001)
- `--prune_interval`: Number of steps between pruning checks (default: 500)
- `--prune_threshold`: Importance threshold for pruning (default: 0.0001)
- `--batch_size`: Training batch size (default: 64)
