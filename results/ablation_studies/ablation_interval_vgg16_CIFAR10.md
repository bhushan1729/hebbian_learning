# Ablation Study: Effect of Pruning Interval (Δt) on VGG16 (CIFAR10)

Fixed threshold $\tau = 5e-06$ | Trained for 20 epochs over seeds [42, 512, 1729].

| Prune Interval (Δt) | Final Sparsity (%) | Final Test Acc (%) | Peak Test Acc (%) |
| :--- | :---: | :---: | :---: |
| `Δt = 100` (7.82 prunes/epoch) | 93.30 ± 0.18% | 84.82 ± 0.20% | 84.96 ± 0.28% |
| `Δt = 250` (3.13 prunes/epoch) | 91.58 ± 0.46% | 83.36 ± 0.63% | 84.98 ± 0.52% |
| `Δt = 500` (1.56 prunes/epoch) | 90.42 ± 0.93% | 84.20 ± 0.73% | 84.71 ± 1.11% |
| `Δt = 1000` (0.78 prunes/epoch) | 89.62 ± 0.72% | 84.02 ± 2.11% | 85.14 ± 0.58% |
| `Δt = 1500` (0.52 prunes/epoch) | 89.37 ± 0.80% | 84.53 ± 0.35% | 85.06 ± 0.26% |
| `Δt = 2000` (0.39 prunes/epoch) | 88.84 ± 0.75% | 84.50 ± 0.87% | 85.21 ± 0.03% |
| `Δt = 5000` (0.16 prunes/epoch) | 88.35 ± 0.64% | 83.62 ± 0.64% | 84.83 ± 0.40% |
