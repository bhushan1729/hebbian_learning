# DADP Benchmarking - Intermediate Research Observations

This file accumulates structural and dynamic observations of DADP (Hebbian pruning) behavior gathered during experiments to support our ICLR paper submission.

---

## 🔍 Observation 1: The Self-Regulating Negative Feedback Loop (Pruning Cycles)

### 📈 Behavior Description
We observed a dynamic oscillation in training loss that directly correlates with pruning events:
1. **Convergence (Loss $\downarrow$)**: As the model trains, the training loss steadily decreases.
2. **Gradients Shrink**: When training loss drops below a critical threshold (e.g. $< 0.013$), the gradients ($dy$) flowing through many weights become extremely small.
3. **Pruning Spike (Sparsity $\uparrow$)**: Because Hebbian importance is calculated as $|x \cdot dy|$, these small gradients cause a large batch of weights to fall below the `1e-5` threshold, triggering a sudden spike in pruned weights (e.g., $+235$ weights pruned).
4. **Capacity Reduction (Loss $\uparrow$ / Bounce)**: The sudden deletion of weights reduces model capacity, causing the training loss to immediately bounce back up in the next epoch (e.g. rising from $0.012$ to $0.018$).
5. **Fine-Tuning/Recovery (Loss $\downarrow$)**: The optimizer adjusts the remaining active weights, adapting them to compensate for the lost pathways, and training loss steadily decreases again until it hits the next threshold, repeating the cycle.

This feedback loop acts as a **self-regulating stabilizer**, ensuring the network only prunes when it has fully learned the features, and then pauses pruning to allow the remaining subnetworks to recover.

### 🧪 Supporting Evidence (100-Epoch Limit Test)
*   **Result Log File**: [`logs/hebbian_mlp_MNIST_thr1e-05_dt500_epoch100.log`](file:///c:/Users/Admin/OneDrive/Desktop/hebbian_learning/logs/hebbian_mlp_MNIST_thr1e-05_dt500_epoch100.log)
*   **Result JSON File**: [`results/mlp_mnist_experiments/results/history_hebbian_mlp_MNIST_thr1e-05_dt500_epoch100.json`](file:///c:/Users/Admin/OneDrive/Desktop/hebbian_learning/results/mlp_mnist_experiments/results/history_hebbian_mlp_MNIST_thr1e-05_dt500_epoch100.json)

#### Specific Examples from Log:
*   **Cycle A**:
    *   *Epoch 86*: Train Loss drops to **`0.0126`**.
    *   *Pruning*: DADP prunes **`202 weights`** (`fc1: +174 | fc2: +24 | fc3: +4`).
    *   *Epoch 87*: Train Loss bounces up to **`0.0184`** (capacity reduction).
    *   *Finetuning*: Loss recovers to **`0.0129`** (Epoch 88) and **`0.0132`** (Epoch 89).
*   **Cycle B**:
    *   *Epoch 96*: Train Loss drops to **`0.0122`**.
    *   *Pruning*: DADP prunes **`238 weights`** (`fc1: +201 | fc2: +27 | fc3: +10`).
    *   *Epoch 97*: Train Loss bounces up to **`0.0200`**.
    *   *Finetuning*: Loss recovers back down to **`0.0164`** (Epoch 98) and **`0.0144`** (Epoch 99).

---

## 🔍 Observation 2: The Sparsity Asymptote & Phase Transition

### 📈 Behavior Description
When training with a fixed pruning threshold (e.g. `1e-5`), the network does not continue pruning indefinitely until it goes completely dead. Instead, it reaches a **steady-state equilibrium**:
*   As the network becomes highly sparse (e.g. $>97\%$), the remaining weights carry the entire representation load.
*   The activation ($x$) and gradient ($dy$) magnitudes flowing through these critical weights remain relatively high to keep the model fitting the data.
*   Therefore, their Hebbian importance scores ($|x \cdot dy|$) remain strictly above the `1e-5` threshold, safeguarding them from deletion.
*   This establishes a natural **structural limit** to pruning (98.0% sparsity for this MLP), beyond which the model cannot prune further.

### 🧪 Supporting Evidence (100-Epoch Limit Test)
*   Between **Epoch 70 (97.76% sparsity)** and **Epoch 100 (98.00% sparsity)**, the network only pruned a total of **0.24%** of its weights.
*   **Overfitting Boundary**: Training accuracy remained high ($99.53\%$), but test accuracy decayed slightly from a peak of **$97.69\%$ (Epoch 11)** to **$96.94\%$ (Epoch 100)**, while test loss rose from **$0.0945 \rightarrow 0.2277$**, indicating that at $98\%$ sparsity, the model's capacity limit has been exceeded, leading to slight overfitting/memorization.
