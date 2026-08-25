# Statistical Sweep Summary: MLP on MNIST

Executed across 3 seeds: `[42, 512, 1729]` over `20` epochs.

| Threshold (τ) | Final Sparsity (%) | Final Test Acc (%) | Peak Test Acc (%) |
| :--- | :---: | :---: | :---: |
| `τ = 1e-6` | 84.90 ± 0.56% | 97.91 ± 0.12% | 98.18 ± 0.02% |
| `τ = 5e-6` | 94.60 ± 0.14% | 97.55 ± 0.09% | 98.02 ± 0.03% |
| `τ = 1e-5` | 96.19 ± 0.11% | 97.38 ± 0.15% | 97.91 ± 0.08% |
| `τ = 5e-5` | 98.35 ± 0.04% | 96.69 ± 0.04% | 96.92 ± 0.05% |
