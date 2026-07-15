# Layer-wise Sparsity Analysis

This document provides a detailed breakdown of how Hebbian Pruning selectively removes connections across the different layers of convolutional neural networks trained on CIFAR-10. Because Hebbian Pruning relies on localized activity metrics instead of global parameter targets, it organically carves out an implicit sparsity gradient.

---

## 1. CNN CIFAR-10 (Hebbian 1e-5)

### Initial Parameter Counts vs Pruned Connections
The standard CNN for CIFAR-10 started with **544,864** connection weights. After the training loop, **342,940** connections were organically pruned, leaving 201,924 active connections.

| Layer | Initial Connections | Connections Pruned | Final Active | **Layer Sparsity** |
| :--- | :--- | :--- | :--- | :--- |
| **conv1** | 864 | 0 | 864 | **0.00%** |
| **conv2** | 18,432 | 1,330 | 17,102 | **7.22%** |
| **fc1** | 524,288 | 341,020 | 183,268 | **65.04%** |
| **fc2** | 1,280 | 590 | 690 | **46.09%** |
| **Total** | 544,864 | 342,940 | 201,924 | **62.94%** |

> [!NOTE]
> ### Key Observations
> **1. Feature Extraction Preservation**  
> The `conv1` layer represents core fundamental edge/shape detection. The Hebbian process recognized it was 100% required, actively refusing to prune a single connection.
> 
> **2. Dense Over-Parameterization Targeting**  
> Over **99.4% of the pruned weight budget fell directly on `fc1`**. This implies the vast majority of network "fat" lies in the jump from flattened convolutional spaces to the fully connected dense layer. The algorithm perfectly self-targeted the structural bottleneck.

---

## 2. VGG16 CIFAR-10 (Hebbian 1e-6)

### Initial Parameter Counts vs Pruned Connections
VGG16 features 16 separate learnable layers (13 convolutional and 3 dense layers), totaling an initial **15,239,872 active connections**. Across the 20 epochs, **11,091,944** connections were pruned dynamically.

| Layer | Initial Connections | Connections Pruned | Final Active | **Layer Sparsity** |
| :--- | :--- | :--- | :--- | :--- |
| **features.0** (conv1_1) | 1,728 | 0 | 1,728 | **0.00%** |
| **features.3** (conv1_2) | 36,864 | 0 | 36,864 | **0.00%** |
| **features.7** (conv2_1) | 73,728 | 0 | 73,728 | **0.00%** |
| **features.10** (conv2_2) | 147,456 | 0 | 147,456 | **0.00%** |
| **features.14** (conv3_1) | 294,912 | 0 | 294,912 | **0.00%** |
| **features.17** (conv3_2) | 589,824 | 1,191 | 588,633 | **0.20%** |
| **features.20** (conv3_3) | 589,824 | 50,627 | 539,197 | **8.58%** |
| **features.24** (conv4_1) | 1,179,648 | 372,194 | 807,454 | **31.55%** |
| **features.27** (conv4_2) | 2,359,296 | 1,789,152 | 570,144 | **75.83%** |
| **features.30** (conv4_3) | 2,359,296 | 2,015,927 | 343,369 | **85.45%** |
| **features.34** (conv5_1) | 2,359,296 | 2,037,642 | 321,654 | **86.37%** |
| **features.37** (conv5_2) | 2,359,296 | 2,156,648 | 202,648 | **91.41%** |
| **features.40** (conv5_3) | 2,359,296 | 2,187,541 | 171,755 | **92.72%** |
| **classifier.0** (fc1) | 262,144 | 239,794 | 22,350 | **91.47%** |
| **classifier.3** (fc2) | 262,144 | 240,848 | 21,296 | **91.87%** |
| **classifier.6** (fc3) | 5,120 | 380 | 4,740 | **7.42%** |
| **Total** | **15,239,872** | **11,091,944** | **4,146,928** | **~72.79%** |

> [!NOTE]
> ### Key Observations
> **1. Foundational Representation Lock**  
> The algorithm organically leaves the first 5 entire convolutional layers completely untouched at **0% sparsity**. This mimics biological neural preservation—early visual cortex pathways remain dense and hard-wired because mutating fundamental edge/shape detectors permanently damages all downstream abstraction.
> 
> **2. Deep Redundancy Extraction**  
> The deep convolutional layers (`conv4_2` to `conv5_3`) theoretically map millions of dense combinations of highly abstract feature shapes. Hebbian selectively wiped out over **75% to 92%** of these layers seamlessly. It implies that complex 10-class categorization relies on sparse, highly-specialized abstract routes rather than mathematically dense matrices.
> 
> **3. Logit Output Protection**  
> While the dense fully-connected layers (`fc1` and `fc2`) form the classic parameter-heavy bottlenecks and are gutted at ~91% sparsity, the final output layer `classifier.6` is nearly entirely spared (**only 7.42% parameter loss**). The algorithm intrinsically recognized that the final combinations deciding the direct logits are hyper-critical for holding accuracy, shielding them from the aggressive pruning applied immediately prior.
