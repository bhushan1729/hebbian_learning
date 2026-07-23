# Proposed Integration of Advanced Pruning Methods
## Technical Analysis and Blueprint for `pruning_utils.py`

This document outlines the mathematical blueprints and Python design signatures for integrating **Fine-Pruning**, **SparseGPT**, and **DeepGraph/Graph-Centrality Pruning** into `scripts/pruning_utils.py` when requested.

---

### 1. Fine-Pruning (Iterative / Defended)

Fine-Pruning represents two distinct paradigms in the literature:
1. **Iterative Fine-Pruning (Standard)**: Periodically deleting a fraction of weights followed by short training/fine-tuning phases.
2. **Fine-Pruning for Backdoor Defense (Security)**: Deactivating neurons that are quiet on clean validation data to eliminate backdoor triggers, followed by fine-tuning.

#### Mathematical Blueprint (Security Defense)
Let $a_i^l(x)$ represent the activation of the $i$-th neuron in layer $l$ for clean input $x \in \mathcal{D}_{val}$. 
We compute the average activation profile:
$$\bar{a}_i^l = \frac{1}{|\mathcal{D}_{val}|} \sum_{x \in \mathcal{D}_{val}} a_i^l(x)$$
Neurons are sorted by their average activation $\bar{a}_i^l$, and the lowest-activated neurons are deactivated by zeroing out their corresponding rows in the weight matrix:
$$W^l_{i, :} \leftarrow 0$$
Followed by standard SGD fine-tuning to recover clean accuracy.

#### Code Outline
```python
def fine_prune_step(model, val_loader, device, prune_neuron_fraction=0.1):
    """
    Accumulates neuron activations on clean validation data, 
    prunes the least active neurons (row-wise deactivation), 
    and returns a neuron mask.
    """
    model.eval()
    activation_sums = {}
    sample_count = 0
    
    # 1. Forward pass to accumulate activations
    # (Using forward hooks to catch intermediate activations)
    ...
    
    # 2. Sort and deactivate the lowest activated neurons
    # Row-wise zeroing of weights
    ...
```

---

### 2. SparseGPT (Hessian-based Optimal Brain Surgeon)

SparseGPT is a state-of-the-art post-training pruning method designed to compress very large models (e.g. LLMs) layer-by-layer without requiring retraining. It solves the local reconstruction problem using approximate second-order information.

#### Mathematical Blueprint
For a layer with weights $W$, input activations $X$, and output $Y = WX$, the local reconstruction error under a column-pruning mask $M$ is optimized via the Hessian matrix $H = 2 X X^\top$.
SparseGPT computes the inverse Hessian $H^{-1}$ using a Cholesky decomposition with a ridge penalty:
$$H^{-1} = \left( X X^\top + \lambda I \right)^{-1}$$
To prune the $i$-th column of $W$, SparseGPT calculates the Optimal Brain Surgeon (OBS) weight update for the remaining active weights in that row to compensate for the pruned parameter:
$$\delta W = - \frac{W_{:, i}}{[H^{-1}]_{ii}} \cdot H^{-1}_{:, i}$$
This update is performed column-by-column across the layer.

#### Code Outline
```python
def sparsegpt_layer_prune(layer, inputs, target_sparsity=0.5, lambda_reg=1e-4):
    """
    Applies column-wise Hessian-based reconstruction pruning (Optimal Brain Surgeon)
    to a specific weight layer (Linear / Conv) using input activations.
    """
    W = layer.weight.data  # Shape: (d_out, d_in)
    d_out, d_in = W.shape
    
    # 1. Compute covariance matrix H = X X^T
    # X shape: (d_in, num_samples)
    X = inputs.reshape(-1, d_in).T
    H = torch.matmul(X, X.T)
    
    # 2. Add regularization and compute inverse Hessian using Cholesky
    H_inv = torch.inverse(H + lambda_reg * torch.eye(d_in, device=W.device))
    
    # 3. Iterative column-wise OBS updates
    # (Updates the remaining active weights dynamically as columns are pruned)
    ...
```

---

### 3. DeepGraph / Graph-Centrality Pruning

DeepGraph pruning models the neural network as a directed bipartite computational graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where nodes $\mathcal{V}$ represent neurons/channels and edges $\mathcal{E}$ represent weights. Connections are pruned based on graph topology metrics (e.g. PageRank, Degree Centrality, or Betweenness Centrality) rather than local weight values.

#### Mathematical Blueprint
Let $A \in \mathbb{R}^{V \times V}$ represent the weighted adjacency matrix of the network, where $A_{ij} = |W_{ij}|$ is the weight magnitude between node $i$ and node $j$.
The **Degree Centrality** $C_D(i)$ of neuron $i$ is calculated as the sum of its incoming and outgoing weight strengths:
$$C_D(i) = \sum_j A_{ji} + \sum_k A_{ik}$$
Connections associated with nodes having the lowest centrality scores are pruned. Alternatively, PageRank centrality vectors $v$ can be computed using the power iteration method:
$$v^{(t+1)} = \alpha A D^{-1} v^{(t)} + (1 - \alpha) \frac{\mathbf{1}}{V}$$
where $D$ is the diagonal out-degree matrix.

#### Code Outline
```python
def graph_centrality_pruning(model, prune_fraction=0.2):
    """
    Constructs the model's global weighted adjacency matrix,
    calculates PageRank or Degree Centrality for all neurons,
    and prunes weights associated with low-centrality nodes.
    """
    # 1. Build weighted adjacency matrix across layers
    ...
    
    # 2. Run power iteration to compute PageRank centrality
    ...
    
    # 3. Apply masks to the lowest centrality paths
    ...
```
