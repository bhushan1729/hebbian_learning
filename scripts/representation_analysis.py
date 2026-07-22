import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Add scripts directory to path for clean imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from data_loader import get_data_loaders
from model import BaselineVGG16, get_resnet18, convert_to_masked_model
from structured_pruning import load_sparse_checkpoint

# =====================================================================
# 1. MATRIX-BASED ENTROPY & EFFECTIVE RANK COMPUTATION
# =====================================================================

def compute_matrix_entropy(Z: torch.Tensor, eps: float = 1e-12) -> Tuple[float, float, float]:
    """
    Computes Matrix-Based Entropy (S_1) and Effective Rank for activation matrix Z.
    Z shape: (N, D) where N is number of samples/tokens, D is feature dimension.
    """
    N, D = Z.shape
    if N < 2:
        return 0.0, 0.0, 1.0

    # Center representations across batch
    Z = Z - Z.mean(dim=0, keepdim=True)
    
    # Construct Gram Matrix K = Z Z^T in R^(N x N)
    K = torch.matmul(Z, Z.T)
    
    # Eigenvalues of symmetric Gram matrix
    eigenvalues = torch.linalg.eigvalsh(K)
    eigenvalues = torch.clamp(eigenvalues, min=0.0) # Ensure non-negative
    
    trace_K = torch.sum(eigenvalues)
    if trace_K <= eps:
        return 0.0, 0.0, 1.0
    
    # Normalized eigenvalues (p_i)
    p = eigenvalues / trace_K
    p = p[p > eps] # Filter numerical zero eigenvalues
    
    # Von Neumann Matrix Entropy S_1(Z) in bits (base 2 log)
    entropy = -torch.sum(p * torch.log2(p)).item()
    
    # Max possible entropy is log2(N)
    max_entropy = np.log2(N)
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    
    # Effective Rank exp(S_1 in nats)
    entropy_nats = -torch.sum(p * torch.log(p)).item()
    eff_rank = np.exp(entropy_nats)
    
    return entropy, norm_entropy, eff_rank


# =====================================================================
# 2. ACTIVATION FEATURE HOOK EXTRACTOR (FIXED KEY NAMES & RESHAPING)
# =====================================================================

class LayerFeatureExtractor:
    """
    Attaches forward hooks to specified layers (Conv2d, Linear, MaskedConv2d, MaskedLinear)
    and stores feature representations using uniform layer identifiers.
    """
    def __init__(self, model: nn.Module, target_layer_types=(nn.Conv2d, nn.Linear)):
        self.model = model
        self.hooks = []
        self.features: Dict[str, torch.Tensor] = {}
        
        layer_idx = 0
        for name, module in model.named_modules():
            if isinstance(module, target_layer_types):
                # FIX: Use clean, class-agnostic layer keys so Baseline and DADP keys match perfectly
                hook_name = f"Layer_{layer_idx:02d}_{name}"
                hook = module.register_forward_hook(self._get_hook(hook_name))
                self.hooks.append(hook)
                layer_idx += 1

    def _get_hook(self, layer_name: str):
        def hook(module, input, output):
            # Flatten feature dimensions for Conv outputs: (N, C, H, W) -> (N, C * H * W)
            if output.dim() == 4:
                flattened = output.detach().flatten(start_dim=1)
                self.features[layer_name] = flattened
            elif output.dim() == 2: # (N, D) for Linear layers
                self.features[layer_name] = output.detach()
        return hook

    def clear(self):
        self.features.clear()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


# =====================================================================
# 3. EVALUATION PIPELINE FOR MODEL COMPARISON
# =====================================================================

def evaluate_layerwise_entropy(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    device: torch.device,
    num_batches: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    Passes evaluation dataset through model and calculates layer-wise metrics.
    """
    model.eval()
    model.to(device)
    
    extractor = LayerFeatureExtractor(model)
    layer_metrics = {}
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            images = images.to(device)
            _ = model(images) # Forward pass triggers hooks
            
            for layer_name, feature_tensor in extractor.features.items():
                entropy, norm_entropy, eff_rank = compute_matrix_entropy(feature_tensor)
                
                if layer_name not in layer_metrics:
                    layer_metrics[layer_name] = {
                        "entropy": [],
                        "norm_entropy": [],
                        "eff_rank": []
                    }
                
                layer_metrics[layer_name]["entropy"].append(entropy)
                layer_metrics[layer_name]["norm_entropy"].append(norm_entropy)
                layer_metrics[layer_name]["eff_rank"].append(eff_rank)
            
            extractor.clear()
            
    extractor.remove_hooks()
    
    # Average metrics across batches
    summary = {}
    for layer_name, metrics in layer_metrics.items():
        summary[layer_name] = {
            "entropy": float(np.mean(metrics["entropy"])),
            "norm_entropy": float(np.mean(metrics["norm_entropy"])),
            "eff_rank": float(np.mean(metrics["eff_rank"]))
        }
        
    return summary


# =====================================================================
# 4. PLOTTING & VISUALIZATION FUNCTION (FIXED KEY MATCHING)
# =====================================================================

def plot_dadp_vs_baseline(
    baseline_metrics: Dict[str, Dict[str, float]], 
    dadp_metrics: Dict[str, Dict[str, float]],
    model_name: str,
    output_dir: str
):
    """
    Generates comparison curves of Baseline vs DADP across layer depth percentage.
    """
    # Intersect keys to guarantee matching layer order
    layer_names = [k for k in baseline_metrics.keys() if k in dadp_metrics]
    num_layers = len(layer_names)
    
    if num_layers == 0:
        print("⚠️ Warning: No matching layer names found between Baseline and DADP models.")
        return

    denom = (num_layers - 1) if num_layers > 1 else 1
    depth_percentages = [i / denom * 100 for i in range(num_layers)]
    
    base_norm_entropy = [baseline_metrics[l]["norm_entropy"] for l in layer_names]
    dadp_norm_entropy = [dadp_metrics[l]["norm_entropy"] for l in layer_names]
    
    base_eff_rank = [baseline_metrics[l]["eff_rank"] for l in layer_names]
    dadp_eff_rank = [dadp_metrics[l]["eff_rank"] for l in layer_names]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    
    # Subplot 1: Normalized Dataset Entropy
    axes[0].plot(depth_percentages, base_norm_entropy, 'o-', label='Baseline (Dense)', color='#1f77b4', linewidth=2.2, markersize=6)
    axes[0].plot(depth_percentages, dadp_norm_entropy, 's--', label='DADP (Sparse)', color='#ff7f0e', linewidth=2.2, markersize=6)
    axes[0].set_title(f"{model_name}: Normalized Dataset Entropy ($S_1$) Across Layers", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Layer Depth Percentage (%)", fontsize=10)
    axes[0].set_ylabel("Normalized Entropy $S_1(Z) / \\log_2(N)$", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(fontsize=9, frameon=True)
    
    # Subplot 2: Effective Rank
    axes[1].plot(depth_percentages, base_eff_rank, 'o-', label='Baseline (Dense)', color='#1f77b4', linewidth=2.2, markersize=6)
    axes[1].plot(depth_percentages, dadp_eff_rank, 's--', label='DADP (Sparse)', color='#ff7f0e', linewidth=2.2, markersize=6)
    axes[1].set_title(f"{model_name}: Effective Rank ($\\exp(S_1)$) Across Layers", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Layer Depth Percentage (%)", fontsize=10)
    axes[1].set_ylabel("Effective Rank", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(fontsize=9, frameon=True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'plots', f"{model_name}_representation_entropy.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Representation analysis plot saved to: {plot_path}")


# =====================================================================
# 5. MAIN EXECUTION ENTRY
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description='DADP Representation Analysis (Dataset Entropy & Effective Rank)')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'resnet18'])
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['MNIST', 'CIFAR10'])
    parser.add_argument('--baseline_model', type=str, required=True, help='Path to baseline dense model checkpoint (.pth)')
    parser.add_argument('--dadp_model', type=str, required=True, help='Path to pruned DADP model checkpoint (.pth)')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results/representation_analysis')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_batches', type=int, default=10, help='Number of batches to evaluate over')
    parser.add_argument('--colab', action='store_true', help='running in Google Colab')
    
    args = parser.parse_args()
    
    if args.colab:
        drive_path = '/content/drive/MyDrive/hebbian_learning'
        if args.data_dir == './data':
            args.data_dir = os.path.join(drive_path, 'data')
        args.output_dir = os.path.join(drive_path, 'results/representation_analysis')
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Data
    _, test_loader = get_data_loaders(args.dataset, args.batch_size, args.data_dir)
    
    # 2. Instantiate and load Baseline model
    print(f"Loading Baseline model ({args.arch})...")
    if args.arch == 'vgg16':
        baseline_model = BaselineVGG16(input_channels=3 if args.dataset == 'CIFAR10' else 1, num_classes=10)
    else:
        baseline_model = get_resnet18(num_classes=10, masked=False)
        
    state_base = load_sparse_checkpoint(args.baseline_model, device)
    baseline_model.load_state_dict(state_base['model_state_dict'])
    
    # 3. Instantiate and load DADP model
    print(f"Loading DADP model ({args.arch})...")
    if args.arch == 'vgg16':
        dadp_model = BaselineVGG16(input_channels=3 if args.dataset == 'CIFAR10' else 1, num_classes=10)
    else:
        dadp_model = get_resnet18(num_classes=10, masked=False)
        
    dadp_model = convert_to_masked_model(dadp_model)
    state_dadp = load_sparse_checkpoint(args.dadp_model, device)
    dadp_model.load_state_dict(state_dadp['model_state_dict'])
    
    # 4. Run Evaluation
    print("Evaluating Baseline layer-wise entropy metrics...")
    baseline_metrics = evaluate_layerwise_entropy(baseline_model, test_loader, device, args.num_batches)
    
    print("Evaluating DADP layer-wise entropy metrics...")
    dadp_metrics = evaluate_layerwise_entropy(dadp_model, test_loader, device, args.num_batches)
    
    # 5. Generate plots and save JSON statistics
    plot_dadp_vs_baseline(baseline_metrics, dadp_metrics, args.arch.upper(), args.output_dir)
    
    json_path = os.path.join(args.output_dir, 'results', f"{args.arch}_representation_entropy.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump({
            "baseline": baseline_metrics,
            "dadp": dadp_metrics
        }, f, indent=4)
    print(f"✅ Representation metrics JSON saved to: {json_path}")
    print("Done!")

if __name__ == '__main__':
    main()