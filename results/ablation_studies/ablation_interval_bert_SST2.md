# Ablation Study: Effect of Pruning Interval (Δt) on BERT (SST2)

Fixed threshold $\tau = 2e-06$ | Trained for 10 epochs over seeds [42, 512, 1729].

| Prune Interval (Δt) | Final Sparsity (%) | Final Test Acc (%) | Peak Test Acc (%) |
| :--- | :---: | :---: | :---: |
| `Δt = 100` (10.53 prunes/epoch) | 97.11 ± 0.12% | 51.91 ± 1.41% | 51.91 ± 1.41% |
| `Δt = 250` (4.21 prunes/epoch) | 96.58 ± 0.50% | 72.06 ± 13.41% | 72.06 ± 13.41% |
| `Δt = 500` (2.11 prunes/epoch) | 90.07 ± 2.99% | 80.35 ± 0.69% | 81.65 ± 0.28% |
| `Δt = 1000` (1.05 prunes/epoch) | 85.67 ± 1.01% | 80.70 ± 0.19% | 81.96 ± 0.14% |
| `Δt = 1500` (0.70 prunes/epoch) | 84.65 ± 0.73% | 80.85 ± 0.41% | 82.38 ± 0.29% |
| `Δt = 2000` (0.53 prunes/epoch) | 80.77 ± 3.70% | 80.73 ± 0.28% | 82.34 ± 0.47% |
| `Δt = 5000` (0.21 prunes/epoch) | 74.91 ± 4.41% | 81.19 ± 0.09% | 82.72 ± 0.22% |
