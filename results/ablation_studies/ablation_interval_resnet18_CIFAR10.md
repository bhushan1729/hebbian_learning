# Ablation Study: Effect of Pruning Interval (Δt) on RESNET18 (CIFAR10)

Fixed threshold $\tau = 1e-05$ | Trained for 2 epochs over seeds [42, 512, 1729].

| Prune Interval (Δt) | Final Sparsity (%) | Final Test Acc (%) | Peak Test Acc (%) |
| :--- | :---: | :---: | :---: |
| `Δt = 100` (7.82 prunes/epoch) | 84.49 ± 0.13% | 66.19 ± 0.62% | 66.19 ± 0.62% |
