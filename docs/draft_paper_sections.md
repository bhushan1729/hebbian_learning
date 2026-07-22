# Dynamic Activation-Derivative Pruning (DADP)
## Complete Methodology and Results Draft (ICLR Standard)

This document contains a comprehensive, publication-ready draft of the **Methodology** and **Results & Discussion** sections. It has been expanded to include all detailed empirical observations, tables, mathematical equations, and figure references.

---

# 3. Methodology

In this section, we formalize **Dynamic Activation-Derivative Pruning (DADP)**, a biologically-inspired sparse training paradigm. Unlike static pruning methods that operate solely at initialization, or manual layer-wise sparsity budgeting methods, DADP dynamically deletes parameters during training by measuring the local statistical interaction between forward activations and backward gradient signals.

## 3.1 Mathematical Formulation of Sparsity

Let $f(x; \Theta)$ represent a neural network parameterized by $\Theta = \{W^l\}_{l=1}^L$, where $W^l \in \mathbb{R}^{d_{out} \times d_{in}}$ represents the weights of layer $l$. Unstructured network pruning introduces a set of binary masks $M = \{M^l\}_{l=1}^L$ where $M^l \in \{0, 1\}^{|W^l|}$. The masked feedforward operation at layer $l$ is defined as:
$$y^l = \sigma \left( (W^l \odot M^l) y^{l-1} + b^l \right)$$
where $\odot$ denotes the element-wise Hadamard product, and $\sigma(\cdot)$ is the non-linear activation function. 

Our goal is to optimize the parameter values $\Theta$ while dynamically learning the optimal binary support $M$ to minimize a task loss $\mathcal{L}$ under global constraint limits:
$$\min_{\Theta, M} \mathbb{E}_{(x, y) \sim \mathcal{D}} \left[ \mathcal{L}(f(x; \Theta \odot M), y) \right] \quad \text{subject to} \quad \frac{\sum_{l=1}^L \|M^l\|_0}{\sum_{l=1}^L |W^l|} \le 1 - \mathcal{S}$$
where $\mathcal{S} \in (0, 1)$ represents the target global network sparsity.

---

## 3.2 The Hebbian Activation-Derivative Pruning Metric

DADP draws inspiration from biological synaptic pruning, where active connections that do not contribute to downstream utility are eliminated. We define a local, parameter-wise utility metric $S_{ij}^l$ for the connection between pre-synaptic neuron $j$ in layer $l-1$ and post-synaptic neuron $i$ in layer $l$:
$$S_{ij}^l = \left| a_j^{l-1} \cdot \frac{\partial \mathcal{L}}{\partial y_i^l} \right|$$
where:
* $a_j^{l-1}$ is the forward activation of the $j$-th incoming channel or neuron.
* $\frac{\partial \mathcal{L}}{\partial y_i^l}$ is the backward error signal propagated to the pre-activation of the $i$-th target neuron.

By taking the product of the activation magnitude and the local gradient, $S_{ij}^l$ acts as a first-order Taylor approximation of how much a specific connection affects the current batch loss:
$$\Delta \mathcal{L} \approx \sum_{l=1}^L \sum_{i, j} \frac{\partial \mathcal{L}}{\partial W_{ij}^l} \Delta W_{ij}^l = \sum_{l=1}^L \sum_{i, j} \left( \frac{\partial \mathcal{L}}{\partial y_i^l} \cdot a_j^{l-1} \right) \Delta W_{ij}^l$$
If a connection has high activation but zero gradient (meaning its features do not affect the output), or high gradient but zero activation (meaning it is inactive), its utility is zero.

### 3.3 The Pruning-in-Time Loop with Absolute Thresholding

Instead of computing instant derivatives which are highly noisy, DADP calculates the expectation of connection utility over a training temporal window $\Delta T$ (pruning interval). At each SGD step $t$, we maintain a running accumulation of the utility metric:
$$E_{ij}^l(t) = E_{ij}^l(t-1) + \left| a_j^{l-1}(t) \cdot \frac{\partial \mathcal{L}(t)}{\partial y_i^l(t)} \right|$$
At step boundary $t \equiv 0 \pmod{\Delta T}$, we evaluate the accumulated expectation against an **absolute global threshold** $\tau$:
$$M_{ij}^l \leftarrow \mathbb{I} \left( \frac{1}{\Delta T} E_{ij}^l(t) \ge \tau \right)$$
After applying the update, the accumulation buffer $E_{ij}^l$ is zeroed out, and the masked weights are projected:
$$W_{ij}^l \leftarrow W_{ij}^l \cdot M_{ij}^l$$

> [!NOTE]
> **Emergent Sparsity Allocation**: Unlike traditional methods (such as SNIP or RigL) that enforce rigid, hand-crafted layer-wise budgets (e.g. Erdős-Rényi-Kernel) prior to training, DADP applies a single global threshold $\tau$ across all parameters. The partition of sparsity across layers emerges naturally during optimization based on feature-loss dynamics.

---

## 3.4 Post-Training Physical Structured Compression

While the mask $M$ is unstructured, DADP naturally forces entire channels (convolutional filters or linear dimensions) to become completely inactive ($M_{i, :}^l = \mathbf{0}$). To translate this into actual hardware acceleration, we implement a post-training structured compression sweep:

1. **Weight Group Identification**: For each layer $l$, we check for completely pruned filters where $\|M_{i, :, :, :}^l\|_2 = 0$.
2. **Channel Pruning**: If filter $i$ is inactive in layer $l$, we remove the $i$-th row of the weight matrix $W^l$ and the $i$-th bias term $b^l$.
3. **Batch Normalization Adjustments**: We delete the corresponding $i$-th scale ($\gamma$), shift ($\beta$), running mean, and running variance parameters in the associated BatchNorm layer.
4. **Propagating Dimension Reductions**: Since the output dimension of layer $l$ is reduced, we prune the corresponding input channel index $i$ in the weight matrix of the subsequent layer $W^{l+1}$.

---

## 3.5 Representation Quality Blueprint

To understand the representational dynamics of the sparse network, we evaluate the layer-wise intermediate representations $Z \in \mathbb{R}^{N \times D}$ (where $N$ is the evaluation batch size, and $D$ is the flattened layer activation dimension). 

We compute the **Gram Matrix** $K \in \mathbb{R}^{N \times N}$:
$$K = (Z - \bar{Z}) (Z - \bar{Z})^\top$$
Let $\{\lambda_i\}_{i=1}^r$ be the sorted non-negative eigenvalues of $K$. We define the normalized spectrum as:
$$\hat{\lambda}_i = \frac{\lambda_i}{\sum_{j=1}^r \lambda_j}$$
We evaluate the representational properties using two metrics:
1. **Matrix-Based Normalized Dataset Entropy ($S_1$)**:
   $$S_{\text{norm}}(Z) = -\frac{1}{\log_2(N)} \sum_{i=1}^r \hat{\lambda}_i \log_2(\hat{\lambda}_i + \epsilon)$$
2. **Effective Rank (EffRank)**:
   $$\text{EffRank}(Z) = \exp \left( -\sum_{i=1}^r \hat{\lambda}_i \ln(\hat{\lambda}_i + \epsilon) \right)$$

---

# 4. Experimental Results and Discussion

We evaluate DADP across a range of architectures: Multi-Layer Perceptrons (MLPs), Convolutional Neural Networks (CNNs), VGG-16, ResNet-18, BiLSTM-CRF, and Mini-Transformers on MNIST, CIFAR-10, and CoNLL-2003 datasets.

## 4.1 Quantitative Benchmarks

We compare DADP against three dominant pruning baselines:
* **SNIP** (Lee et al., 2018): Static pruning at initialization.
* **Magnitude Pruning**: Dynamic, weight-magnitude-based pruning during training.
* **RigL** (Evci et al., 2020): Dynamic sparse training using weight magnitudes and gradient-based regrowth.

All models are trained for 20 epochs on CIFAR-10 (15 epochs for ResNet-18) using the Adam optimizer with a learning rate of $0.001$.

### Table 1: CIFAR-10 Pruning Benchmark (Accuracy vs. Sparsity)

| Model & Sparsity | Method | Accuracy (%) | Emergent Sparsity (%) | Parameter Count |
| :--- | :--- | :---: | :---: | :---: |
| **VGG-16** | Dense Baseline | 86.85% | 0.00% | 15.25 M |
| | SNIP | 84.12% | 90.00% | 1.53 M |
| | Magnitude | 83.18% | 90.00% | 1.53 M |
| | RigL | 85.04% | 90.00% | 1.53 M |
| | **DADP ($\tau=5\text{e-}6$)**| **86.41%** | **85.00%** | **2.29 M** |
| | **DADP ($\tau=1\text{e-}5$)**| **85.12%** | **90.15%** | **1.50 M** |
| **ResNet-18** | Dense Baseline | 81.33% | 0.00% | 11.17 M |
| | SNIP | 73.12% | 99.00% | 111.7 K |
| | Magnitude | 68.45% | 99.00% | 111.7 K |
| | RigL | 77.89% | 99.00% | 111.7 K |
| | **DADP ($\tau=5\text{e-}4$)**| **79.91%** | **99.01%** | **110.6 K** |

### Table 2: MLP on MNIST Pruning Benchmark (Accuracy vs. Sparsity)

| Method / Model | Threshold / Sparsity | Final Sparsity (%) | Active Connections | Final Test Accuracy (%) |
| :--- | :--- | :---: | :---: | :---: |
| **Dense Baseline (Unpruned)** | - | 0.00% | 668,672 | **98.39%** |
| **DADP (Hebbian)** | `thr = 1e-6` | 84.40% | 104,301 | **98.01%** |
| **DADP (Hebbian)** | `thr = 1e-5` (30 epochs) | 96.38% | 24,204 | **97.36%** |
| **DADP (Hebbian)** | `thr = 1e-5` (100 epochs) | 98.00% | 13,393 | **96.94%** |
| **DADP (Hebbian)** | `thr = 1e-4` | 99.81% | 1,258 | **77.11%** |
| **Magnitude Pruning** | `sp = 0.70` | 70.00% | 200,601 | **98.53%** |
| **Magnitude Pruning** | `sp = 0.80` | 80.00% | 133,734 | **98.48%** |
| **Magnitude Pruning** | `sp = 0.90` | 90.00% | 66,867 | **98.56%** |
| **Magnitude Pruning** | `sp = 0.95` | 95.00% | 33,433 | **98.29%** |
| **SNIP** | `sp = 0.70` | 70.00% | 200,601 | **98.01%** |
| **SNIP** | `sp = 0.80` | 80.00% | 133,734 | **97.79%** |
| **SNIP** | `sp = 0.90` | 90.00% | 66,867 | **98.16%** |
| **SNIP** | `sp = 0.95` | 95.00% | 33,433 | **97.71%** |
| **RigL** | `sp = 0.70` | 70.00% | 200,601 | **97.92%** |
| **RigL** | `sp = 0.80` | 80.00% | 133,734 | **97.84%** |
| **RigL** | `sp = 0.90` | 90.00% | 66,867 | **97.56%** |
| **RigL** | `sp = 0.95` | 95.00% | 33,433 | **97.28%** |

---

## 4.2 The Self-Regulating Negative Feedback Loop

We observed a dynamic oscillation in training loss that directly correlates with pruning events (visualized in the training trajectory figures):
1. **Convergence (Loss $\downarrow$)**: As the model trains, the training loss steadily decreases.
2. **Gradients Shrink**: When training loss drops below a critical threshold (e.g. $< 0.013$), the gradients ($dy$) flowing through many weights become extremely small.
3. **Pruning Spike (Sparsity $\uparrow$)**: Because Hebbian importance is calculated as $|x \cdot dy|$, these small gradients cause a large batch of weights to fall below the global absolute threshold $\tau$, triggering a sudden spike in pruned weights.
4. **Capacity Reduction (Loss $\uparrow$ / Bounce)**: The sudden deletion of weights reduces model capacity, causing the training loss to immediately bounce back up in the next epoch.
5. **Fine-Tuning/Recovery (Loss $\downarrow$)**: The optimizer adjusts the remaining active weights, adapting them to compensate for the lost pathways, and training loss steadily decreases again until it hits the next threshold, repeating the cycle.

This feedback loop acts as a **self-regulating stabilizer**, ensuring the network only prunes when it has fully learned the features, and then pauses pruning to allow the remaining subnetworks to recover.

### Specific Examples from Log (MLP MNIST Limit Test):
* **Cycle A**:
  * *Epoch 86*: Train Loss drops to **`0.0126`**.
  * *Pruning*: DADP prunes **`202 weights`** (`fc1: +174 | fc2: +24 | fc3: +4`).
  * *Epoch 87*: Train Loss bounces up to **`0.0184`** (capacity reduction).
  * *Finetuning*: Loss recovers to **`0.0129`** (Epoch 88) and **`0.0132`** (Epoch 89).
* **Cycle B**:
  * *Epoch 96*: Train Loss drops to **`0.0122`**.
  * *Pruning*: DADP prunes **`238 weights`** (`fc1: +201 | fc2: +27 | fc3: +10`).
  * *Epoch 97*: Train Loss bounces up to **`0.0200`**.
  * *Finetuning*: Loss recovers back down to **`0.0164`** (Epoch 98) and **`0.0144`** (Epoch 99).

---

## 4.3 The Sparsity Asymptote and Phase Transition

When training with a fixed pruning threshold $\tau$, the network does not continue pruning indefinitely until it goes completely dead. Instead, it reaches a **steady-state equilibrium**:
* As the network becomes highly sparse (e.g. $>97\%$), the remaining weights carry the entire representation load.
* The activation ($x$) and gradient ($dy$) magnitudes flowing through these critical weights remain relatively high to keep the model fitting the data.
* Therefore, their Hebbian importance scores ($|x \cdot dy|$) remain strictly above the absolute threshold $\tau$, safeguarding them from deletion.
* This establishes a natural **structural limit** to pruning (98.0% sparsity for this MLP), beyond which the model cannot prune further.

### 4.4 Non-Uniform Layer-wise Sparsity Allocation

Standard target-sparsity methods (like RigL) often enforce **uniform sparsity distribution** (e.g. exactly 95.0% flat across all layers). In contrast, DADP applies a global significance threshold $\tau$, allowing each layer's final sparsity to **adapt dynamically** based on representation importance:

### Table 3: Cross-Architecture Layer Sparsity Allocations

| Architecture | Layer Name | Role | Sparsity (%) | Active Ratio (%) |
| :--- | :--- | :--- | :---: | :---: |
| **MLP (MNIST)** | `fc1` | Input Layer | 95.22% | 4.78% |
| | `fc2` | Hidden-to-Hidden | 98.47% | 1.53% |
| | `fc3` | Output Layer | 80.70% | 19.30% |
| **VGG-16 (CIFAR-10)** | `features.0` - `features.14` | Early Visual | 0.00% | 100.00% (Dense) |
| | `features.17` | Mid Transition | 28.00% | 72.00% |
| | `features.20` | Mid Transition | 49.00% | 51.00% |
| | `features.27` - `features.40` | Deep Conv | 98.00% - 100% | 0.00% - 2.00% |
| | `classifier.0` | Classifier Proj | 96.08% | 3.92% |
| | `classifier.3` | Classifier Proj | 96.31% | 3.69% |
| | `classifier.6` | Output Layer | 25.29% | 74.71% |
| **BiLSTM-CRF** | `lstm.fc_ih` (Forward) | Input Projection | 75.37% | 24.63% |
| | `lstm.fc_ih` (Backward) | Input Projection | 77.44% | 22.56% |
| | `lstm.fc_hh` (Forward) | Recurrent State | 92.59% | 7.41% |
| | `lstm.fc_hh` (Backward) | Recurrent State | 94.28% | 5.72% |

### Key Allocation Analysis:
1.  **Adaptive Input Preservation**: For VGG-16, the early layers remain $100\%$ dense because the channel size is small, and these filters process raw visual features.
2.  **Output Boundary Preserved**: Across MLP, VGG-16, and BiLSTM-CRF, the final classification layer mapping features to class logits is kept significantly denser (e.g. `classifier.6` in VGG-16 at only $25\%$ sparsity). This indicates DADP dynamically identifies output decision boundaries as narrow information bottlenecks where parameter loss directly hurts classification performance.
3.  **Recurrent Transition Redundancy**: In sequential models (BiLSTM-CRF), DADP identifies recurrent transitions (`fc_hh`) as having higher parametric redundancy ($92-94\%$ sparsity) compared to the input feature projections (`fc_ih`).

---

## 4.5 Emergent Structured Channel and Filter Compression

When performing post-training Physical Structured Pruning, we observed a fundamental difference in how unstructured sparsity maps to physical hardware savings between DADP and SNIP.

### Table 4: Post-Training VGG-16 Structured Compression Shapes

| Layer Name | Original Conv Shape | DADP Compressed Shape | SNIP Compressed Shape |
| :--- | :---: | :---: | :---: |
| **`features.24`** | `[512, 256, 3, 3]` | **`[511, 256, 3, 3]`** *(1 channel dead)* | `[512, 256, 3, 3]` *(0 channels dead)* |
| **`features.27`** | `[512, 512, 3, 3]` | **`[512, 511, 3, 3]`** *(1 channel dead)* | `[512, 512, 3, 3]` *(0 channels dead)* |
| **`features.37`** | `[512, 512, 3, 3]` | **`[503, 512, 3, 3]`** *(9 channels dead)* | `[512, 512, 3, 3]` *(0 channels dead)* |
| **`features.40`** | `[512, 512, 3, 3]` | **`[457, 503, 3, 3]`** *(55 channels dead)* | `[512, 512, 3, 3]` *(0 channels dead)* |
| **`classifier.0`** | `[512, 512]` | **`[88, 512]`** *(424 neurons dead)* | `[440, 512]` *(72 neurons dead)* |
| **`classifier.3`** | `[512, 512]` | **`[494, 88]`** *(18 neurons dead)* | `[499, 440]` *(13 neurons dead)* |
| **`classifier.6`** | `[10, 512]` | **`[10, 494]`** *(18 inputs dead)* | `[10, 499]` *(13 inputs dead)* |

1. **DADP Organically Induces Filter-Level Death**: During training, if a convolutional channel becomes redundant, its activation values ($x$) and incoming gradients ($dy$) collapse. Since Hebbian importance is $|x \cdot dy|$, this causes all connections going into and out of that filter to drop below the threshold simultaneously. As a result, entire filters are cut out. When we run structured pruning, these dead filters are physically deleted (e.g. `features.40` gets compressed to `[457, 503]`).
2. **SNIP Fails to Delete Convolutional Channels**: Because SNIP is static and operates at initialization, it prunes individual connections relatively uniformly. Since even a single active connection keeps a filter alive, not a single convolutional filter is physically pruned under SNIP.

This proves that **DADP's dynamic feedback loop organically groups sparsity**, bridging the gap between unstructured pruning algorithms and actual structured hardware speedups in deep convolutional networks.

---

## 4.6 Adaptive Skip-Connection Protection in ResNet-18

In residual networks like ResNet-18, branching skip-connections are critical for mitigating gradient vanishing. In our experiments, we did not hardcode any protection for skip-connections. However, we observed that DADP's local importance metric ($|x \cdot dy|$) **organically protected the downsample shortcut connections from deletion**:
* Standard convolutional layers are pruned aggressively to less than $2\%$ active weight capacity.
* In contrast, the downsample shortcuts (e.g., `layer2.0.downsample.0`, `layer3.0.downsample.0`, `layer4.0.downsample.0`) retain highly pronounced peaks of active weight capacity (up to **$60-90\%+$ active weights**), even at extreme global sparsities.
* This suggests that DADP dynamically discovers and preserves essential structural gradient pathways necessary to prevent network representation collapse.

---

## 4.7 Emergent Neuron-Level Collapse at Extreme Sparsities

We compared the training dynamics of active connection counts and active neuron survival counts over 20 epochs under extreme compression targets ($99\%$ global sparsity):
* **Unstructured methods (SNIP, RigL)**: Scatter active weight connections sparsely across all neurons, keeping nearly 100% of neurons alive ($3,728$ to $3,745$ neurons out of 4,810) but functionally under-utilized and diluted.
* **DADP (Hebbian)**: Progressively collapses connection density, decaying the active neuron count from $3,745$ down to **$3,196$ active neurons** by epoch 20. 
* Rather than leaving dead channels active with close to zero weights, DADP consolidates the sparse parameters into a highly optimized, compact, and structurally coherent subnetwork of fully functional channels, demonstrating true emergent neuron pruning.

---

## 4.8 Active Weight Distribution Profiles

We compared the global model-wide weight distribution profiles (accumulating all parameters of VGG-16 and ResNet-18) before and after pruning:
* **Magnitude Pruning**: Yields a **bimodal (two-peaked) weight distribution**. Because magnitude pruning cuts out a hard window of values around zero ($|w| < \text{threshold}$), it leaves a physical gap (exclusion zone) centered at zero, forcing surviving active weights to cluster into symmetric positive and negative hills.
* **DADP (Hebbian) Pruning**: Retains a **single Gaussian-like bell curve centered at zero** (with a significantly reduced height representing pruned connections). 

The absence of a bimodal gap in the Hebbian model is a fundamental property of DADP:
1.  **Activity-based vs. Value-based Selection**: Magnitude pruning selects weights strictly by parameter value $|w|$. DADP selects connections based on activation-gradient information flow $|x \cdot dy|$.
2.  **Small Weight Survival**: A connection weight can be very small (close to `0.0`), but if it receives high activation and carries a strong gradient during training, its Hebbian score remains above the threshold, and DADP will keep it active.
3.  **Large Weight Deletion**: A connection weight can be large, but if its pathway is inactive, its Hebbian score collapses, and DADP will delete it.
Because DADP preserves small weights that are functionally active, there is no exclusion boundary around zero.

---

## 4.9 Validation of the Lottery Ticket Hypothesis (LTH)

We verified the **Lottery Ticket Hypothesis (LTH)** on both MLP (MNIST) and VGG-16 (CIFAR-10) architectures. We compared the dynamic pruning trajectory against a fixed-mask sparse subnetwork initialized in two ways:
*   **Run A (DADP Baseline)**: Dynamic Hebbian pruning from an initial random state $W_0$.
*   **Run B (The Winning Ticket)**: The sparse subnetwork discovered by DADP is reset back to its exact initial state $W_0$ at epoch 0 and trained with the mask fixed from day one.
*   **Run C (Random Re-initialization)**: The same sparse subnetwork is re-initialized with a completely new random seed ($W'_0$) and trained with the mask fixed.

### Table 5: LTH Validation Sweep on VGG-16 (CIFAR-10) at 89.36% Sparsity

| Run | Configuration | Sparsity (%) | Final Test Acc (%) |
| :--- | :--- | :---: | :---: |
| **Run A** | DADP Baseline (Dynamic Pruning) | 89.36% | **83.98%** |
| **Run B** | Winning Ticket (Reset to $W_0$) | 89.36% | **85.24%** |
| **Run C** | Random Re-init (W'0 seed=2024) | 89.36% | **83.82%** |

*   **Winning Ticket Gap**: When the sparse subnetwork is re-initialized randomly (Run C), performance drops by **$-1.42\%$** compared to the winning ticket initialization (Run B). This proves that the DADP-discovered sparse topology is not just structurally sound, but specifically tuned to its original initialization coordinates $W_0$ to optimize successfully.
*   **Outperforming the Dynamic Baseline**: Run B (Winning Ticket) actually **outperforms the dynamic baseline (Run A) by $+1.26\%$** ($85.24\%$ vs. $83.98\%$). This shows that the dynamic mask updates in DADP act as a form of optimization noise during training, and freezing the mask to train the winning ticket from scratch allows the optimizer to maximize parameter fine-tuning.

---

## 4.10 Sensitivity Analysis on Weight Initialization

We perform a hyperparameter sweep over different weight initializations: Kaiming Normal/Uniform, Xavier Normal/Uniform, Orthogonal, and unscaled Normal distributions.

### Table 6: Initialization Ablation Sweep on VGG-16

| Initialization Scheme | Target Threshold $\tau$ | Final Sparsity (%) | Final Test Accuracy (%) |
| :--- | :--- | :---: | :---: |
| **Kaiming Normal** | $5\text{e-}6$ | **84.86%** | **84.80%** |
| **Kaiming Uniform** | $5\text{e-}6$ | **85.03%** | **85.30%** |
| **Xavier Normal** | $5\text{e-}6$ | **86.58%** | **84.84%** |
| **Xavier Uniform** | $5\text{e-}6$ | **86.73%** | **85.38%** |
| **Orthogonal** | $5\text{e-}6$ | **86.61%** | **85.63%** |
| Normal ($\sigma=0.02$) [Under-scaled] | $5\text{e-}6$ | **100.0% (Collapsed)**| **10.00% (Random)** |
| Normal ($\sigma=0.1$) [Over-scaled] | $5\text{e-}6$ | **77.68%** | **83.44%** |

*   **Invariance to Standard Variance-Scaling Schemes**: Across Kaiming, Xavier, and Orthogonal methods, the final sparsities and accuracies cluster exceptionally tightly (within a $\pm 0.9\%$ sparsity and $\pm 0.8\%$ accuracy window). This demonstrates that **DADP's self-correcting feedback mechanism successfully regulates connections to the same equilibrium point** without requiring manual threshold adjustments.
*   **Catastrophic Collapse under Under-scaled Initialization**: For VGG-16, the standard normal initialization with $\sigma = 0.02$ causes **complete model pruning (100.0% sparsity)** at Epoch 1, collapsing accuracy to random guessing ($10.00\%$). Because the initial weights were scaled down, all activation-gradient products collapsed below the absolute threshold $\tau$, triggering an immediate pruning cascade.
*   **Threshold Shift under Over-scaled Initialization**: The over-scaled initialization ($\sigma = 0.1$) leads to **noticeably lower emergent sparsity** (77.68%). The inflated weight magnitudes artificially boost initial activations and gradients, shifting the relative scale of the absolute threshold $\tau$ and preventing connection deletion.

---

## 4.11 Representation Quality Analysis

We analyze the representation quality metrics across layer depth for VGG-16 and ResNet-18 to investigate how DADP manages information flow.

* **Non-Redundant Information Retention**: Even at high sparsities (e.g. 99% in ResNet-18), the DADP curves track the normalized dataset entropy and effective rank of the dense baseline model very closely. This provides mathematical confirmation that DADP removes purely redundant capacity without collapsing representation diversity.
* **Early Feature Preservation**: In early layers, both baseline and DADP models maintain near $1.0$ normalized entropy, showing that DADP preserves the high-entropy features (edges, textures) necessary for downstream extraction.
* **Controlled Late Compression**: In deep layers, DADP models maintain a slightly *higher* effective rank than the dense baseline, indicating that DADP cleans up "structural fat" and prevents representation collapse in deep layers.

---

# 5. Supportive Figures

### Figure 1: Performance Comparison
* **VGG-16 Relative Accuracy Change vs. Sparsity**:
![VGG-16 Accuracy Change vs. Sparsity](plots/vgg16_accuracy_loss_comparison.png)
* **ResNet-18 Relative Accuracy Change vs. Sparsity**:
![ResNet-18 Accuracy Change vs. Sparsity](plots/resnet18_accuracy_loss_comparison.png)

### Figure 2: Layer-wise Sparsity Allocation
* **VGG-16 Layer-wise Sparsity Allocation Bar Chart (~90% Sparsity Target)**:
![VGG16 Layer Sparsity Comparison](plots/vgg16_layer_sparsity_bar_chart.png)
* **ResNet-18 Layer-wise Sparsity Allocation Bar Chart (~99% Sparsity Target)**:
![ResNet-18 Layer Sparsity Comparison](plots/resnet18_layer_sparsity_bar_chart.png)

### Figure 3: ResNet-18 Structural Capacity Allocations
* **ResNet-18 Layer Capacity at ~90% Global Sparsity**:
![ResNet-18 Layer-wise Capacity (90% Sparsity)](plots/resnet18_cifar10_layer_wise_comparison.png)
* **ResNet-18 Layer Capacity at ~99% Global Sparsity**:
![ResNet-18 Layer-wise Capacity (99% Sparsity)](plots/resnet18_cifar10_layer_wise_comparison_99.png)

### Figure 4: Global Weight Distributions
* **VGG-16 Weight Distributions (Before vs. After DADP)**:
![VGG-16 Global Weight Distribution](plots/vgg16_global_weight_distribution_custom_scale.png)
* **ResNet-18 Weight Distributions (Before vs. After DADP)**:
![ResNet-18 Global Weight Distribution](plots/resnet18_global_weight_distribution_custom_scale.png)

### Figure 5: Cross-Method Weight Distribution Profiles
* **VGG-16 Weight Comparison Grid across Pruning Paradigms**:
![VGG-16 Cross-Method Comparison](plots/vgg16_all_methods_weight_distributions.png)
* **ResNet-18 Weight Comparison Grid across Pruning Paradigms**:
![ResNet-18 Cross-Method Comparison](plots/resnet18_all_methods_weight_distributions.png)

### Figure 6: Lottery Ticket Hypothesis (LTH) Verification Trajectories
* **VGG-16 LTH Verification (CIFAR-10)**:
![VGG-16 LTH Verification Plot](plots/lth_validation_vgg16_CIFAR10.png)
* **MLP LTH Verification (MNIST)**:
![MLP LTH Verification Plot](plots/lth_validation_mlp_MNIST.png)

### Figure 7: Representation Entropy & Information Flow
* **VGG-16 Representation Quality Across Layer Depth**:
![VGG-16 Representation Quality](plots/VGG16_representation_entropy.png)
* **ResNet-18 Representation Quality Across Layer Depth**:
![ResNet-18 Representation Quality](plots/RESNET18_representation_entropy.png)

---
