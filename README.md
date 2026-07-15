# Hebbian-Inspired Structural Pruning: DADP Pipeline

This repository implements an activity-dependent structural sparsification algorithm for neural networks, inspired by Hebbian learning principles ("neurons that fire together, wire together"), alongside multiple state-of-the-art pruning baselines.

The core idea is to track the "importance" of each connection during training and permanently prune connections that contribute little to the learning process.

---

## 🧠 Pruning Logic: The Hebbian Proxy

We use **Gradient × Activation** as a proxy for connection importance. A connection $w_{ij}$ between neuron $i$ in the previous layer and neuron $j$ in the current layer is considered important if:

$$importance_{ij} = E[ |a_i \cdot \frac{\partial L}{\partial y_j}| ]$$

Where:
- $a_i$ is the activation of the previous neuron.
- $\frac{\partial L}{\partial y_j}$ is the gradient of the loss with respect to the pre-activation output of the current neuron.

---

## 🏗️ Supported Architectures & Datasets

We support five primary architectures, each with a **Baseline** and a **Pruned** (Masked) variant, across diverse datasets:

| Architecture | Description | Target Datasets |
| :--- | :--- | :--- |
| **MLP** | 3-Layer Multi-Layer Perceptron (784-512-512-10) | MNIST |
| **CNN** | 2 Conv Layers + 2 FC Layers | MNIST |
| **VGG16** | 16-Layer Deep CNN (with Batch Norm & Adaptive Pooling) | CIFAR-10 |
| **ResNet-18** | Branching Residual CNN Architecture | CIFAR-10 |
| **BiLSTM-CRF** | Bidirectional LSTM with Conditional Random Fields | CoNLL-2003 |
| **Transformer** | Mini-Transformer model for sequence classification | SST-2, IMDB |

---

## ✂️ Supported Pruning Modes

1. **baseline**: Standard dense, unpruned training.
2. **hebbian**: Progressive Hebbian-inspired pruning (DADP) during training.
3. **snip**: One-shot Single-path Network Importance Pruning at initialization.
4. **magnitude**: One-shot standard weight magnitude-based pruning.
5. **rigl**: Dynamic gradient-based regrowth and pruning framework.

---

## 📂 Project Directory Structure

```
├── docs/               # Outlines, suggestions, and research notes
├── notebooks/          # Colab-ready experimental runner notebooks
├── scripts/            # Core training and pruning engine code
│   ├── plotting/       # Visualization and figure generator scripts
│   ├── data_loader.py  # Dataset loading & preprocessing pipelines
│   ├── engine.py       # Trainer class and evaluation methods
│   ├── main.py         # Primary training CLI entry point
│   ├── model.py        # Masked module and network architectures
│   └── structured_pruning.py # Physical model compression logic
└── results/            # Saved checkpoints, metrics, and JSON logs (git-ignored)
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch & Torchvision

### How to Run

> [!NOTE]
> For Google Colab, use the `--colab` flag to automatically handle pathing to `/content/drive/MyDrive/hebbian_learning/`.

#### 1️⃣ Basic Hebbian MLP Run (MNIST)
```bash
python scripts/main.py --arch mlp --dataset MNIST --mode hebbian --prune_threshold 0.0001
```

#### 2️⃣ Deep ResNet-18 Run (CIFAR-10)
```bash
python scripts/main.py --arch resnet18 --dataset CIFAR10 --mode hebbian --prune_threshold 5e-6 --prune_interval 500
```

#### 3️⃣ Sequence Labeling SNIP Run (CoNLL-2003)
```bash
python scripts/main.py --arch bilstm_crf --dataset CoNLL2003 --mode snip --sparsity 0.90
```

### 🛠️ CLI Arguments
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--arch` | `mlp` | `mlp`, `cnn`, `vgg16`, `resnet18`, `bilstm_crf`, `transformer` |
| `--dataset`| `MNIST` | `MNIST`, `CIFAR10`, `SST2`, `IMDB`, `CoNLL2003` |
| `--mode` | `hebbian` | `baseline`, `hebbian`, `snip`, `magnitude`, `rigl` |
| `--epochs` | `10` | Number of training epochs |
| `--prune_interval` | `500` | Steps between dynamic pruning updates |
| `--prune_threshold`| `1e-4` | Importance cutoff for DADP (Hebbian) |
| `--sparsity` | `0.5` | Target sparsity fraction for SNIP, Magnitude, and RigL |
| `--colab` | `False` | Enable Google Colab pathing |

---

## 📊 Visualizing Results

The plotting scripts have been grouped into [scripts/plotting/](file:///c:/Users/Admin/OneDrive/Desktop/hebbian_learning/scripts/plotting/) to quickly generate benchmark curves, layer metrics, and training metrics over epochs. 

Run the comparison scripts for each architecture:
```bash
# VGG16 Benchmarks
python scripts/plotting/plot_vgg16_comparison.py
python scripts/plotting/plot_vgg16_layer_metrics.py
python scripts/plotting/plot_vgg16_ratio_over_epochs.py

# ResNet-18 Benchmarks
python scripts/plotting/plot_resnet18_comparison.py
python scripts/plotting/plot_resnet18_layer_metrics.py
python scripts/plotting/plot_resnet18_ratio_over_epochs.py

# BiLSTM-CRF Benchmarks
python scripts/plotting/plot_bilstm_comparison.py
python scripts/plotting/plot_bilstm_layer_metrics.py
python scripts/plotting/plot_bilstm_ratio_over_epochs.py
```
Outputs are automatically written to their respective experimental results folder (e.g., `results/vgg16_cifar10_experiments/`).
