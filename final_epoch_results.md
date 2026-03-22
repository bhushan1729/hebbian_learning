# MNIST Experiment Results Summary

This document summarizes the final results for MNIST pruning experiments. **$\Delta$** indicates the change relative to the **Baseline**.

## CNN MNIST Experiments

### CNN MNIST (10 Epochs)
| Metric | Baseline | Hebbian (1e-6) | $\Delta$ | Hebbian (1e-5) | $\Delta$ | Hebbian (1e-4) | $\Delta$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Final Sparsity** | 0.0% | 85.59% | +85.59% | 95.75% | +95.75% | **99.02%** | +99.02% |
| **Active Connections** | 421,408 | 60,704 | -85.6% | 17,905 | -95.8% | **4,113** | -99.0% |
| **Active Neurons** | 234 | 179 | -23.5% | 149 | -36.3% | **99** | -57.7% |
| **Final Train Acc** | **99.73%** | 99.70% | -0.03% | 99.45% | -0.28% | 97.84% | -1.89% |
| **Final Test Acc** | **99.24%** | 99.09% | -0.15% | 99.10% | -0.14% | 97.71% | -1.53% |
| **Peak Test Acc** | **99.24%** | 99.10% | -0.14% | 99.16% | -0.08% | 98.25% | -0.99% |

### CNN MNIST (20 Epochs)
| Metric | Baseline | Hebbian (1e-6) | $\Delta$ | Hebbian (1e-5) | $\Delta$ | Hebbian (1e-4) | $\Delta$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Final Sparsity** | 0.0% | 91.61% | +91.61% | 97.03% | +97.03% | **99.04%** | +99.04% |
| **Active Connections** | 421,408 | 35,355 | -91.6% | 12,534 | -97.0% | **4,040** | -99.0% |
| **Active Neurons** | 234 | 169 | -27.8% | 121 | -48.3% | **89** | -62.0% |
| **Final Train Acc** | **99.92%** | 99.78% | -0.14% | 99.62% | -0.30% | 98.73% | -1.19% |
| **Final Test Acc** | 98.30% | **99.15%** | **+0.85%** | 98.78% | +0.48% | 98.29% | -0.01% |
| **Peak Test Acc** | **99.23%** | 99.15% | -0.08% | 99.04% | -0.19% | 98.54% | -0.69% |

---

## MLP MNIST Experiments

### MLP MNIST (10 Epochs)
| Metric | Baseline | Hebbian (1e-6) | $\Delta$ | Hebbian (1e-5) | $\Delta$ | Hebbian (1e-4) | $\Delta$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Final Sparsity** | 0.0% | 68.57% | +68.57% | 93.29% | +93.29% | **99.90%** | +99.90% |
| **Active Connections** | 668,672 | 210,174 | -68.6% | 44,888 | -93.3% | **665** | -99.9% |
| **Active Neurons** | 1,034 | 794 | -23.2% | 383 | -63.0% | **14** | -98.6% |
| **Final Train Acc** | **99.33%** | 99.30% | -0.03% | 98.96% | -0.37% | 35.66% | -63.67% |
| **Final Test Acc** | 97.82% | **98.11%** | **+0.29%** | 97.58% | -0.24% | 35.85% | -61.97% |
| **Peak Test Acc** | 98.01% | **98.14%** | **+0.13%** | 97.85% | -0.16% | 96.80% | -1.21% |

### MLP MNIST (20 Epochs)
| Metric | Baseline | Hebbian (1e-6) | $\Delta$ | Hebbian (1e-5) | $\Delta$ | Hebbian (1e-4) | $\Delta$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Final Sparsity** | 0.0% | 82.46% | +82.46% | 95.83% | +95.83% | **99.95%** | +99.95% |
| **Active Connections** | 668,672 | 117,271 | -82.5% | 27,912 | -95.8% | **311** | -99.9% |
| **Active Neurons** | 1,034 | 559 | -45.9% | 271 | -73.8% | **14** | -98.6% |
| **Final Train Acc** | 99.53% | **99.70%** | **+0.17%** | 99.40% | -0.13% | 33.06% | -66.47% |
| **Final Test Acc** | 97.63% | **98.05%** | **+0.42%** | 97.62% | -0.01% | 33.62% | -64.01% |
| **Peak Test Acc** | 98.19% | **98.31%** | **+0.12%** | 97.94% | -0.25% | 96.66% | -1.53% |

---



# CIFAR-10 Experiment Results Summary

## CNN CIFAR-10 Experiments

### CNN CIFAR-10 (10 Epochs)
| Metric | Baseline | Hebbian (1e-6) | $\Delta$ | Hebbian (1e-5) | $\Delta$ | Hebbian (1e-4) | $\Delta$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Final Sparsity** | 0.0% | 53.57% | +53.57% | 56.59% | +56.59% | **92.37%** | +92.37% |
| **Active Connections** | 544,864 | 252,992 | -53.6% | 236,510 | -56.6% | **41,557** | -92.4% |
| **Active Neurons** | 234 | 166 | -29.1% | 165 | -29.5% | **141** | -39.7% |
| **Final Train Acc** | **94.54%** | 93.86% | -0.68% | 92.64% | -1.90% | 77.57% | -16.97% |
| **Final Test Acc** | 70.82% | 71.37% | +0.55% | 70.71% | -0.11% | **71.69%** | **+0.87%** |
| **Peak Test Acc** | 72.58% | **73.06%** | **+0.48%** | 72.64% | +0.06% | 71.69% | -0.89% |

### CNN CIFAR-10 (20 Epochs)
| Metric | Baseline | Hebbian (1e-6) | $\Delta$ | Hebbian (1e-5) | $\Delta$ | Hebbian (1e-4) | $\Delta$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Final Sparsity** | 0.0% | 61.96% | +61.96% | 63.59% | +63.59% | **92.73%** | +92.73% |
| **Active Connections** | 544,864 | 207,293 | -62.0% | 198,361 | -63.6% | **39,629** | -92.7% |
| **Active Neurons** | 234 | 154 | -34.2% | 163 | -30.3% | **140** | -40.2% |
| **Final Train Acc** | **97.55%** | 97.41% | -0.14% | 96.06% | -1.49% | 83.37% | -14.18% |
| **Final Test Acc** | 68.81% | 70.08% | +1.27% | **70.25%** | **+1.44%** | 69.93% | +1.12% |
| **Peak Test Acc** | 71.27% | **72.59%** | **+1.32%** | 72.26% | +0.99% | 71.06% | -0.21% |

---

## MLP CIFAR-10 Experiments

### MLP CIFAR-10 (10 Epochs)
| Metric | Baseline | Hebbian (1e-6) | $\Delta$ | Hebbian (1e-5) | $\Delta$ | Hebbian (1e-4) | $\Delta$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Final Sparsity** | 0.0% | 10.11% | +10.11% | 20.59% | +20.59% | **99.99%** | +99.99% |
| **Active Connections** | 1,840,128 | 1,654,163 | -10.1% | 1,461,252 | -20.6% | **140** | -99.99% |
| **Active Neurons** | 1,034 | 767 | -25.8% | 736 | -28.8% | **10** | -99.0% |
| **Final Train Acc** | 67.33% | **67.52%** | **+0.19%** | 67.28% | -0.05% | 9.77% | -57.56% |
| **Final Test Acc** | 51.85% | 52.00% | +0.15% | **52.54%** | **+0.69%** | 10.00% | -41.85% |
| **Peak Test Acc** | **52.90%** | 52.34% | -0.56% | 52.54% | -0.36% | 45.73% | -7.17% |

### MLP CIFAR-10 (20 Epochs)
| Metric | Baseline | Hebbian (1e-6) | $\Delta$ | Hebbian (1e-5) | $\Delta$ | Hebbian (1e-4) | $\Delta$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Final Sparsity** | 0.0% | 10.85% | +10.85% | 20.02% | +20.02% | **99.99%** | +99.99% |
| **Active Connections** | 1,840,128 | 1,640,390 | -10.9% | 1,471,813 | -20.0% | **4** | -99.99% |
| **Active Neurons** | 1,034 | 730 | -29.4% | 703 | -32.0% | **4** | -99.6% |
| **Final Train Acc** | 79.95% | 80.10% | +0.15% | **81.03%** | **+1.08%** | 10.03% | -69.92% |
| **Final Test Acc** | 50.52% | 51.97% | +1.45% | **52.72%** | **+2.20%** | 10.00% | -40.52% |
| **Peak Test Acc** | 52.42% | 52.93% | +0.51% | **53.69%** | **+1.27%** | 44.17% | -8.25% |

---

## VGG16 CIFAR-10 Experiments

### VGG16 CIFAR-10 (20 Epochs)
| Metric | Baseline | Hebbian (1e-6) | $\Delta$ | Hebbian (1e-5) | $\Delta$ | Hebbian (1e-4) | $\Delta$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Final Sparsity** | 0.0% | 71.60% | +71.60% | **100.0%** | +100.0% | TBD | - |
| **Active Connections** | 15,239,872 | 4,328,264 | -71.6% | **0** | -100% | TBD | - |
| **Active Neurons** | **5,258** | 4,742 | -9.8% | **0** | -100% | TBD | - |
| **Final Train Acc** | **95.66%** | 95.50% | -0.16% | 9.63% | -86.03% | TBD | - |
| **Final Test Acc** | 84.46% | **85.16%** | **+0.70%** | 10.00% | -74.46% | TBD | - |
| **Peak Test Acc** | **85.31%** | 85.16% | -0.15% | 30.44% | -54.87% | TBD | - |

> [!CAUTION]
> **Brain Death Observed**: The 1e-5 threshold is **too aggressive** for VGG16 on CIFAR-10. The model reached **100.0% sparsity** by epoch 18, resulting in 0 active connections and random-guess (10.0%) accuracy. 

> [!NOTE]
> **Observation**: The Hebbian (1e-6) pruning stage for VGG16 on CIFAR-10 successfully reduced the model size by over **71%** while actually **increasing** the final test accuracy by **0.70%**. This suggests that Hebbian pruning is acting as an effective regularizer for the deep VGG16 architecture.
