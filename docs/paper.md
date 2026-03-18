# Dynamic Activity-Dependent Pruning: A Hebbian Approach to Structural Sparsity

**Author**: [Your Name/Bhushan]
**Date**: March 2026

---

## 1. Abstract

Modern deep neural networks are notoriously over-parameterized, leading to significant wasted computation during both training and inference. While various pruning techniques exist, most rely on post-hoc weight magnitude analysis or static pre-training sensitivity measures. In this paper, we propose **Dynamic Activity-Dependent Pruning (DADP)**, a biologically-inspired structural plasticity mechanism. DADP utilizes an importance signal derived from the expectation of the product of pre-synaptic activations and post-synaptic error gradients: $importance_{ij} = E[ |a_i \cdot \frac{\partial L}{\partial y_j}| ]$. Our experiments on the MNIST dataset demonstrate that DADP can eliminate over **93% of network connections** during a single training pass while maintaining **99.7% of baseline accuracy**. Unlike the Lottery Ticket Hypothesis, which requires iterative training and resetting, DADP identifies efficient sub-networks dynamically, matching the developmental "synapse elimination" observed in biological nervous systems.

---

## 2. Introduction

### 2.1 The Problem of Over-parameterization
Standard artificial neural networks (ANNs) utilize dense connectivity between layers. However, biological brains are sparsely connected and highly efficient. 

### 2.2 Biological Motivation
In neurobiology, the principle of **Hebbian Learning** ("neurons that fire together, wire together") suggests that synaptic strength is a function of correlated activity. Furthermore, during early development, the brain undergoes a period of massive **synaptic pruning**, where redundant connections are removed based on their lack of contribution to functional circuits.

### 2.3 Our Contribution: DADP
We bridge the gap between backpropagation and Hebbian structural plasticity.
- **Dynamic**: Pruning happens *while* the model learns.
- **Activity-Aware**: It doesn't just look at weight size; it looks at whether the connection is actually "firing" and "helping" the loss.
- **Single-Pass**: No need for expensive iterative pruning/re-training cycles.

---

## 3. Methodology

### 3.1 The Importance Signal
We define the importance of a connection $w_{ij}$ as the moving average of its gradient magnitude:
$$I_{ij} = \frac{1}{N} \sum_{n=1}^{N} | a_i^{(n)} \cdot \delta_j^{(n)} |$$
where $a_i$ is the input from the previous layer and $\delta_j$ is the error signal (gradient) from the current layer.

### 3.2 Permanent Masking
We maintain a binary mask $M \in \{0, 1\}$. Every $K$ steps, we update the mask:
$$M_{ij} = 
\begin{cases} 
1 & \text{if } I_{ij} > \tau \\
0 & \text{otherwise}
\end{cases}$$
Once $M_{ij}$ is set to 0, the connection is effectively removed ($w_{ij} = 0$) and is no longer updated by the optimizer.

---

## 4. Experimental Results (MNIST)

| Metric | Baseline | DADP (Optimal) |
| :--- | :--- | :--- |
| **Accuracy** | 97.82% | **97.58%** |
| **Sparsity** | 0.00% | **93.28%** |
| **Active Connections** | 668,672 | **44,888** |
| **Active Neurons** | 1,034 | **383** |

*(Refer to generated plots for loss convergence and structural decay curves.)*

---

## 5. Conclusion & Future Work
DADP proves that the vast majority of connections in a standard MLP are non-essential for basic digit recognition and can be identified and removed using simple activity-dependent rules during training. Future work will explore "regrowth" mechanisms (synaptogenesis) and applications to Transformer architectures.
