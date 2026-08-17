# Statistical Sweep Summary: BERT on SST2

Executed across 3 seeds: `[42, 512, 1729]` over `20` epochs.

| Threshold (τ) | Final Sparsity (%) | Final Test Acc (%) | Peak Test Acc (%) |
| :--- | :---: | :---: | :---: |
| `τ = 1e-6` | 71.52 ± 4.93% | 79.01 ± 0.57% | 82.19 ± 0.39% |
| `τ = 2e-6` | 91.69 ± 2.03% | 78.67 ± 0.66% | 81.57 ± 0.30% |
| `τ = 3e-6` | 96.03 ± 0.79% | 80.85 ± 1.06% | 82.11 ± 0.37% |
| `τ = 4e-6` | 96.87 ± 0.16% | 64.18 ± 12.70% | 64.22 ± 12.76% |
