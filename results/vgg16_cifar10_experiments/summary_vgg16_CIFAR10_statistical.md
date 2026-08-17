# Statistical Sweep Summary: VGG16 on CIFAR10

Executed across 3 seeds: `[42, 512, 1729]` over `20` epochs. (Includes Top-2 Seeds filtering for high-variance boundary cases).

| Threshold (τ) | All 3 Seeds Final Acc (%) | Top-2 Seeds Final Acc (%) | Top-2 Seeds Peak Acc (%) | Final Sparsity (%) |
| :--- | :---: | :---: | :---: | :---: |
| `τ = 1e-6` | 84.81 ± 0.42% | **85.09 ± 0.17%** | **85.11 ± 0.16%** | 74.26 ± 1.25% |
| `τ = 5e-6` | 84.20 ± 0.73% | **84.70 ± 0.17%** | **85.47 ± 0.34%** | 89.77 ± 0.08% |
| `τ = 6e-6` | 59.65 ± 32.54% | **82.65 ± 0.65%** | **84.29 ± 0.53%** | 91.65 ± 0.67% |
| `τ = 6.5e-6` | 74.29 ± 12.63% | **83.22 ± 0.00%** | **83.22 ± 0.00%** | 91.86 ± 0.54% |
| `τ = 6.7e-6` | 71.41 ± 16.55% | **83.06 ± 2.01%** | **84.02 ± 1.36%** | 92.16 ± 0.28% |
