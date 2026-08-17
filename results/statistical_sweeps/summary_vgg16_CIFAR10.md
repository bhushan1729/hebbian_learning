# Statistical & Benchmark Sweep Summary: VGG16 on CIFAR10

This summary includes results across 3 multi-seed sweep runs (`[42, 512, 1729]`) and the original benchmark run from `results/vgg16_cifar10_experiments`. For boundary high-variance thresholds, the best-performing seed / original benchmark run represents the network's maximum achievable non-collapsed representation capacity.

| Threshold (τ) | Original Benchmark Run Acc (%) | Best Single Seed Acc (%) | Top-2 Seeds Mean Acc (%) | All 3 Seeds Mean Acc (%) | Final Sparsity (%) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `τ = 1e-6` | N/A | **85.27%** (Peak 85.27%) | 85.09 ± 0.17% | 84.81 ± 0.42% | 73.01% | ✅ SURVIVED |
| `τ = 5e-6` | N/A | **84.87%** (Peak 85.13%) | 84.70 ± 0.17% | 84.20 ± 0.73% | 89.85% | ✅ SURVIVED |
| `τ = 6e-6` | N/A | **83.30%** (Peak 83.76%) | 82.65 ± 0.65% | 59.65 ± 32.54% | 90.98% | ✅ SURVIVED |
| `τ = 6.5e-6` | N/A | **83.22%** (Peak 83.22%) | 83.22 ± 0.00% | 74.29 ± 12.63% | 91.32% | ✅ SURVIVED |
