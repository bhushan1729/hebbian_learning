# Sparse Network Adaptation: Retraining DADP Subnetworks Without Forgetting
**A Formal Proposal for Sparse Transfer Learning**

This document formalizes the idea of taking a highly sparse neural network trained on one dataset via **Dynamic Activity-Dependent Pruning (DADP)** and retraining/adapting the **same** sparse structure to a new dataset with minimal effort and without losing performance on the previous dataset.

---

## 1. Concept Motivation & Research Question

When DADP trains a network on a dataset (like MNIST), it prunes away 95%–98% of the connections, leaving a highly optimized, sparse subnetwork (e.g., 2% active connections). 

The core research question is:
> *Can we repurpose/retrain this specific sparse subnetwork (the active 2%) on a new task (e.g., Fashion-MNIST or CIFAR-10) with minimal training cost, while preventing catastrophic forgetting of the original task?*

This explores the **transferability of sparse topologies** and parameter-efficient adaptation of biologically inspired sparse networks.

---

## 2. Proposed Methodologies

To retrain the *same* active sparse parameters without losing performance on the original dataset, we propose three distinct pathways:

### Approach A: DADP-Guided Elastic Weight Consolidation (DADP-EWC)
Traditional Elastic Weight Consolidation (EWC) calculates the Fisher Information Matrix (FIM) post-training to regularize parameter updates on subsequent tasks. 
Since DADP already tracks the expected Gradient $\times$ Activation metric over time, we can reuse this tracked importance directly as a regularization coefficient without any post-hoc computation.

*   **Mathematical Formulation:**
    During retraining on Task 2, we optimize the following regularized loss function restricted to the active mask $M$:
    $$\mathcal{L}_{\text{total}} = \mathcal{L}_2(W \odot M) + \frac{\lambda}{2} \sum_{i \in \text{active}} \hat{I}_i^{(1)} \left( w_i - w_i^{(1)} \right)^2$$
    where:
    *   $\mathcal{L}_2$ is the classification loss on the new dataset (Task 2).
    *   $\hat{I}_i^{(1)} = E\left[ \left| a_{\text{pre}} \cdot \frac{\partial \mathcal{L}_1}{\partial y_{\text{post}}} \right| \right]$ is the DADP importance score of connection $i$ after Task 1.
    *   $w_i^{(1)}$ is the frozen optimal weight value of connection $i$ after Task 1.
    *   $\lambda$ is a hyperparameter balancing consolidation vs. plastic adaptation.
*   **Why it works:** Weights that were critical for Task 1 (high $\hat{I}^{(1)}$) are heavily restricted from changing, whereas less important weights in the 2% subnetwork can adapt freely to learn Task 2.

---

### Approach B: Parameter-Efficient Sparse Adapters (PE-Sparse-Adapters)
To ensure **100% lossless performance** on Task 1, we can freeze the active parameters $\theta^{(1)} = W \odot M$ completely and introduce task-specific adapters.

*   **Mathematical Formulation:**
    For Task 2, we introduce a task-specific scale vector $s^{(2)}$ and bias vector $b^{(2)}$ applied directly to the active connections of the sparse network. During the forward pass of Task 2:
    $$y = f\left(x; \left(W^{(1)} \odot s^{(2)}\right) + b^{(2)}\right)$$
    During backward pass, we **only update** $s^{(2)}$ and $b^{(2)}$. The base weights $W^{(1)}$ remain strictly frozen.
*   **Minimal Effort:** Since the network is 98% sparse, the number of active weights is tiny. Learning a scaling factor per active connection requires negligible parameter overhead (e.g., 2% of the original network size) and converges extremely fast.
*   **Zero Forgetting:** When evaluating Task 1, we remove the Task 2 scaling factors, recovering the original Task 1 subnetwork perfectly.

---

### Approach C: Active Subnetwork Rehearsal (Sparse Replay)
Rather than freezing or heavily regularizing, we train the sparse subnetwork on Task 2 using standard optimization while interleaving a tiny percentage of Task 1 samples (experience replay / rehearsal).

*   **Why it works:** Because the capacity is restricted to a small sparse subnetwork (e.g., 2% of the parameters), the network is naturally regularized against massive representation drifts. A very small rehearsal buffer (e.g., 100 images from Task 1) is sufficient to stabilize the weights and maintain high performance on both tasks simultaneously.

---

## 3. Comparison of Approaches

| Metric | DADP-EWC (Approach A) | PE-Sparse-Adapters (Approach B) | Sparse Replay (Approach C) |
| :--- | :--- | :--- | :--- |
| **Task 1 Accuracy Loss** | Extremely low (depends on $\lambda$) | **Absolute Zero (Lossless)** | Low (stabilizes with buffer) |
| **Task 2 Adaptation Capacity** | Medium-High | Medium (restricted by scaling) | **High** |
| **Inference Overhead** | None (uses same weights) | Minor (applies scaling factors) | None (uses same weights) |
| **Additional Parameter Cost** | Zero (saves old weight vector) | Tiny (scaling vectors per task) | Zero |
| **Task-ID needed at test time?** | No (single joint weight set) | Yes (to select the correct adapter) | No (single joint weight set) |

---

## 4. Experimental Validation Plan

To validate this proposal:
1.  **Stage 1: Pre-training**
    *   Train the base CNN on MNIST with DADP ($\tau = 1e-5$).
    *   Extract the sparse subnetwork (e.g., 97% sparsity) and record the final weights $W^{(1)}$ and DADP importance values $\hat{I}^{(1)}$.
2.  **Stage 2: Retraining on Task 2 (e.g., Fashion-MNIST)**
    *   **Baseline:** Fine-tune the subnetwork on Fashion-MNIST without any protection (expect catastrophic forgetting of MNIST).
    *   **Test Approach A (DADP-EWC):** Fine-tune with the DADP-based quadratic weight penalty.
    *   **Test Approach B (PE-Sparse-Adapters):** Freeze weights and train only scaling multipliers.
3.  **Stage 3: Evaluation**
    *   Plot the trade-off curves: Accuracy on MNIST vs. Accuracy on Fashion-MNIST.
    *   Verify if the DADP-guided constraint or sparse adapters outperform the baseline fine-tuning.
