# Ablation Study: Effect of Pruning Interval (Δt) on RESNET18 (CIFAR10)

Fixed threshold $\tau = 1e-05$ | Trained for 20 epochs over seeds [42, 512, 1729].

| Prune Interval (Δt) | Final Sparsity (%) | Final Test Acc (%) | Peak Test Acc (%) |
| :--- | :---: | :---: | :---: |
| `Δt = 100` (7.82 prunes/epoch) | 92.11 ± 0.15% | 76.12 ± 0.40% | 76.90 ± 0.13% |
| `Δt = 250` (3.13 prunes/epoch) | 91.64 ± 0.03% | 76.92 ± 0.52% | 77.25 ± 0.31% |
| `Δt = 500` (1.56 prunes/epoch) | 91.41 ± 0.08% | 76.76 ± 0.16% | 77.25 ± 0.24% |
| `Δt = 1000` (0.78 prunes/epoch) | 91.13 ± 0.04% | 76.21 ± 0.40% | 77.37 ± 0.39% |
| `Δt = 1500` (0.52 prunes/epoch) | 90.97 ± 0.05% | 76.84 ± 0.29% | 77.35 ± 0.10% |
| `Δt = 2000` (0.39 prunes/epoch) | 90.71 ± 0.01% | 76.80 ± 0.12% | 77.13 ± 0.27% |
| `Δt = 5000` (0.16 prunes/epoch) | 90.87 ± 0.03% | 76.86 ± 0.40% | 77.37 ± 0.42% |
