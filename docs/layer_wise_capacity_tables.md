# Layer-wise Network Capacity and Sparsity Tables

Here are the detailed parameter counts and active capacity allocations extracted from the best-performing dynamic Hebbian pruning (DADP) configurations across four distinct neural network architectures (MLP/ANN, VGG-16, ResNet-18, and BiLSTM-CRF):


### 📊 MLP/ANN (DADP thr=1e-5, 97.36% Acc) Layer-wise Capacity Table
| Layer Name | Total Param Count | Active Param Count | Active Weights (%) | Sparsity (%) |
| :--- | :---: | :---: | :---: | :---: |
| **`fc1`** | 401.4K | 19.2K | 4.78% | 95.22% |
| **`fc2`** | 262.1K | 4.0K | 1.53% | 98.47% |
| **`fc3`** | 5.1K | 988 | 19.30% | 80.70% |
| **TOTAL** | **668.7K** | **24.2K** | **3.62%** | **96.38%** |


### 📊 VGG-16 (DADP thr=5e-6, 85.02% Acc) Layer-wise Capacity Table
| Layer Name | Total Param Count | Active Param Count | Active Weights (%) | Sparsity (%) |
| :--- | :---: | :---: | :---: | :---: |
| **`features.0`** | 1.7K | 1.7K | 100.00% | 0.00% |
| **`features.3`** | 36.9K | 36.6K | 99.41% | 0.59% |
| **`features.7`** | 73.7K | 73.7K | 100.00% | 0.00% |
| **`features.10`** | 147.5K | 147.5K | 100.00% | 0.00% |
| **`features.14`** | 294.9K | 294.6K | 99.89% | 0.11% |
| **`features.17`** | 589.8K | 424.8K | 72.02% | 27.98% |
| **`features.20`** | 589.8K | 299.4K | 50.77% | 49.23% |
| **`features.24`** | 1.18M | 190.8K | 16.18% | 83.82% |
| **`features.27`** | 2.36M | 55.6K | 2.35% | 97.65% |
| **`features.30`** | 2.36M | 30.9K | 1.31% | 98.69% |
| **`features.34`** | 2.36M | 19.5K | 0.83% | 99.17% |
| **`features.37`** | 2.36M | 7.4K | 0.32% | 99.68% |
| **`features.40`** | 2.36M | 14.4K | 0.61% | 99.39% |
| **`classifier.0`** | 262.1K | 10.3K | 3.92% | 96.08% |
| **`classifier.3`** | 262.1K | 9.7K | 3.69% | 96.31% |
| **`classifier.6`** | 5.1K | 3.8K | 74.71% | 25.29% |
| **TOTAL** | **15.24M** | **1.62M** | **10.64%** | **89.36%** |


### 📊 ResNet-18 (DADP thr=0.0005, 73.67% Acc) Layer-wise Capacity Table
| Layer Name | Total Param Count | Active Param Count | Active Weights (%) | Sparsity (%) |
| :--- | :---: | :---: | :---: | :---: |
| **`conv1`** | 9.4K | 8.9K | 94.98% | 5.02% |
| **`layer1.0.conv1`** | 36.9K | 26.5K | 71.77% | 28.23% |
| **`layer1.0.conv2`** | 36.9K | 19.5K | 52.96% | 47.04% |
| **`layer1.1.conv1`** | 36.9K | 19.1K | 51.87% | 48.13% |
| **`layer1.1.conv2`** | 36.9K | 6.9K | 18.84% | 81.16% |
| **`layer2.0.conv1`** | 73.7K | 3.5K | 4.72% | 95.28% |
| **`layer2.0.conv2`** | 147.5K | 196 | 0.13% | 99.87% |
| **`layer2.0.downsample.0`** | 8.2K | 469 | 5.73% | 94.27% |
| **`layer2.1.conv1`** | 147.5K | 40 | 0.03% | 99.97% |
| **`layer2.1.conv2`** | 147.5K | 51 | 0.03% | 99.97% |
| **`layer3.0.conv1`** | 294.9K | 0 | 0.00% | 100.00% |
| **`layer3.0.conv2`** | 589.8K | 0 | 0.00% | 100.00% |
| **`layer3.0.downsample.0`** | 32.8K | 150 | 0.46% | 99.54% |
| **`layer3.1.conv1`** | 589.8K | 0 | 0.00% | 100.00% |
| **`layer3.1.conv2`** | 589.8K | 0 | 0.00% | 100.00% |
| **`layer4.0.conv1`** | 1.18M | 0 | 0.00% | 100.00% |
| **`layer4.0.conv2`** | 2.36M | 0 | 0.00% | 100.00% |
| **`layer4.0.downsample.0`** | 131.1K | 159 | 0.12% | 99.88% |
| **`layer4.1.conv1`** | 2.36M | 0 | 0.00% | 100.00% |
| **`layer4.1.conv2`** | 2.36M | 0 | 0.00% | 100.00% |
| **`fc`** | 5.1K | 251 | 4.90% | 95.10% |
| **TOTAL** | **11.17M** | **85.8K** | **0.77%** | **99.23%** |


### 📊 BiLSTM-CRF (DADP thr=5e-5, 92.75% Acc) Layer-wise Capacity Table
| Layer Name | Total Param Count | Active Param Count | Active Weights (%) | Sparsity (%) |
| :--- | :---: | :---: | :---: | :---: |
| **`lstm.forward_cells.0.fc_ih`** | 32.8K | 0 | 0.00% | 100.00% |
| **`lstm.forward_cells.0.fc_hh`** | 16.4K | 0 | 0.00% | 100.00% |
| **`lstm.backward_cells.0.fc_ih`** | 32.8K | 14.1K | 43.17% | 56.83% |
| **`lstm.backward_cells.0.fc_hh`** | 16.4K | 3.2K | 19.68% | 80.32% |
| **`hidden2tag`** | 1.0K | 43 | 4.20% | 95.80% |
| **TOTAL** | **99.3K** | **17.4K** | **17.53%** | **82.47%** |