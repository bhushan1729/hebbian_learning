# Hebbian-Inspired Structural Pruning: MNIST & CIFAR-10

This repository implements an activity-dependent structural sparsification algorithm for neural networks, inspired by Hebbian learning principles ("neurons that fire together, wire together").

The core idea is to track the "importance" of each connection during training and permanently prune connections that contribute little to the learning process.

## 🧠 Pruning Logic: The Hebbian Proxy

We use **Gradient × Activation** as a proxy for connection importance. A connection $w_{ij}$ between neuron $i$ in the previous layer and neuron $j$ in the current layer is considered important if:

$$importance_{ij} = E[ |a_i \cdot \frac{\partial L}{\partial y_j}| ]$$

Where:
- $a_i$ is the activation of the previous neuron.
- $\frac{\partial L}{\partial y_j}$ is the gradient of the loss with respect to the pre-activation output of the current neuron.

---

## 🏗️ Supported Architectures

We support three primary architectures, each with a **Baseline** and **Hebbian** (Masked) variant:

| Architecture | Description | Target Datasets |
| :--- | :--- | :--- |
| **MLP** | 3-Layer Multi-Layer Perceptron (784-512-512-10) | MNIST |
| **CNN** | 2 Conv Layers + 2 FC Layers | MNIST, CIFAR-10 |
| **VGG16** | 16-Layer Deep CNN (with Batch Norm & Adaptive Pooling) | CIFAR-10, MNIST |

### 🚀 VGG16 Highlights
- **Numerical Stability**: Includes `nn.BatchNorm2d` to handle the depth of 16 layers.
- **Robustness**: Uses `nn.AdaptiveAvgPool2d((1, 1))` to handle varying input sizes (e.g., 28x28 for MNIST and 32x32 for CIFAR-10) without architectural changes.
- **Inplace ReLU Fix**: All activations use `inplace=False` to ensure compatibility with backward hooks during pruning.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch & Torchvision

### How to Run

> [!NOTE]
> For Google Colab, use the `--colab` flag to automatically handle pathing to `/content/drive/MyDrive/hebbian_learning/`.

#### 1️⃣ Basic MLP Run (MNIST)
```bash
python scripts/main.py --arch mlp --dataset MNIST --mode hebbian --prune_threshold 0.0001
```

#### 2️⃣ Deep VGG16 Run (CIFAR-10)
```bash
python scripts/main.py --arch vgg16 --dataset CIFAR10 --mode hebbian --prune_threshold 1e-6 --prune_interval 1000
```

### 🛠️ CLI Arguments
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--arch` | `mlp` | `mlp`, `cnn`, or `vgg16` |
| `--dataset`| `MNIST` | `MNIST` or `CIFAR10` |
| `--mode` | `hebbian` | `baseline` or `hebbian` |
| `--epochs` | `10` | Number of training epochs |
| `--prune_interval` | `500` | Steps between pruning updates |
| `--prune_threshold`| `1e-4` | Importance cutoff for pruning |
| `--colab` | `False` | Enable Google Colab pathing |

---

## 📈 Key Findings: VGG16 on CIFAR-10

In our benchmarks on the CIFAR-10 dataset (20 Epochs):

- **Baseline Test Acc**: 84.46% (15.2M Connections)
- **Hebbian (1e-6) Test Acc**: **85.16%** (**4.3M Connections**)
- **Sparsity**: **71.60%**

> [!IMPORTANT]
> **Observation**: Pruning over **71%** of the model connections actually **increased** the final test accuracy by **0.70%** compared to the baseline. This suggests the Hebbian pruning algorithm acts as a powerful regularizer, removing redundant weights and improving generalization in deep architectures.

---

## 📊 Monitoring & Visualizing

Results (checkpoints and JSON logs) are saved to `/results`. You can visualize comparisons using:

```bash
python scripts/plot_results.py results/history_baseline_MNIST.json results/history_hebbian_MNIST.json --output_dir plots/
```
