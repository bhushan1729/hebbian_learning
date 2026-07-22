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
    dadp_metrics_dict: Dict[str, Dict[str, Dict[str, float]]],
    model_name: str,
    output_dir: str
):
    """
    Generates comparison curves of Baseline vs multiple DADP models across actual layer names.
    """
    # Intersect keys to guarantee matching layer order
    layer_names = list(baseline_metrics.keys())
    for label, metrics in dadp_metrics_dict.items():
        layer_names = [k for k in layer_names if k in metrics]
        
    num_layers = len(layer_names)
    
    if num_layers == 0:
        print("⚠️ Warning: No matching layer names found between all compared models.")
        return

    # Clean layer names for the x-axis ticks
    clean_layer_names = ["_".join(k.split("_")[2:]) for k in layer_names]
    x_coords = np.arange(num_layers)
    
    base_norm_entropy = [baseline_metrics[l]["norm_entropy"] for l in layer_names]
    base_eff_rank = [baseline_metrics[l]["eff_rank"] for l in layer_names]
    
    # Increase height slightly to accommodate the layer labels
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.0), dpi=150)
    
    # Define color palette & marker styles
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf']
    markers = ['s', 'v', '^', 'D', 'o', '*', 'x', '+']
    
    # Subplot 1: Normalized Dataset Entropy
    axes[0].plot(x_coords, base_norm_entropy, 'o-', label='Baseline (Dense)', color='#1f77b4', linewidth=2.5, markersize=7)
    
    for idx, (label, metrics) in enumerate(dadp_metrics_dict.items()):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        norm_entropy = [metrics[l]["norm_entropy"] for l in layer_names]
        axes[0].plot(x_coords, norm_entropy, f'{marker}--', label=label, color=color, linewidth=2.0, markersize=5)
        
    axes[0].set_title(f"{model_name}: Normalized Dataset Entropy ($S_1$) Across Layers", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Layer Name", fontsize=10)
    axes[0].set_ylabel("Normalized Entropy $S_1(Z) / \\log_2(N)$", fontsize=10)
    axes[0].set_xticks(x_coords)
    axes[0].set_xticklabels(clean_layer_names, rotation=90, fontsize=8, ha='center')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(fontsize=9, frameon=True, loc='best')
    
    # Subplot 2: Effective Rank
    axes[1].plot(x_coords, base_eff_rank, 'o-', label='Baseline (Dense)', color='#1f77b4', linewidth=2.5, markersize=7)
    
    for idx, (label, metrics) in enumerate(dadp_metrics_dict.items()):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        eff_rank = [metrics[l]["eff_rank"] for l in layer_names]
        axes[1].plot(x_coords, eff_rank, f'{marker}--', label=label, color=color, linewidth=2.0, markersize=5)
        
    axes[1].set_title(f"{model_name}: Effective Rank ($\\exp(S_1)$) Across Layers", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Layer Name", fontsize=10)
    axes[1].set_ylabel("Effective Rank", fontsize=10)
    axes[1].set_xticks(x_coords)
    axes[1].set_xticklabels(clean_layer_names, rotation=90, fontsize=8, ha='center')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(fontsize=9, frameon=True, loc='best')
    
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
    parser.add_argument('--dadp_models', type=str, nargs='+', required=True, help='Paths to pruned DADP model checkpoints (.pth)')
    parser.add_argument('--labels', type=str, nargs='+', help='Custom labels for each DADP model')
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
    
    # 3. Setup labels for DADP models
    if not args.labels:
        args.labels = []
        for path in args.dadp_models:
            filename = os.path.basename(path)
            if 'thr' in filename:
                parts = filename.split('thr')
                thr_val = parts[1].split('_')[0]
                args.labels.append(f"DADP (thr={thr_val})")
            elif 'sp' in filename:
                parts = filename.split('sp')
                sp_val = parts[1].split('_')[0].replace('.pth', '')
                args.labels.append(f"DADP (sp={sp_val})")
            else:
                args.labels.append(filename.replace('.pth', ''))
                
    # Validate labels length matches models length
    if len(args.labels) != len(args.dadp_models):
        print("⚠️ Warning: Length of labels does not match models. Auto-generating labels.")
        args.labels = [f"DADP {i}" for i in range(len(args.dadp_models))]
        
    # 4. Evaluate Baseline model
    print("Evaluating Baseline layer-wise entropy metrics...")
    baseline_metrics = evaluate_layerwise_entropy(baseline_model, test_loader, device, args.num_batches)
    
    # 5. Evaluate all DADP models
    dadp_metrics_dict = {}
    for path, label in zip(args.dadp_models, args.labels):
        print(f"Evaluating DADP model '{label}' from path: {path}...")
        if args.arch == 'vgg16':
            dadp_model = BaselineVGG16(input_channels=3 if args.dataset == 'CIFAR10' else 1, num_classes=10)
        else:
            dadp_model = get_resnet18(num_classes=10, masked=False)
            
        dadp_model = convert_to_masked_model(dadp_model)
        state_dadp = load_sparse_checkpoint(path, device)
        dadp_model.load_state_dict(state_dadp['model_state_dict'])
        
        metrics = evaluate_layerwise_entropy(dadp_model, test_loader, device, args.num_batches)
        dadp_metrics_dict[label] = metrics
    
    # 6. Generate plots and save JSON statistics
    plot_dadp_vs_baseline(baseline_metrics, dadp_metrics_dict, args.arch.upper(), args.output_dir)
    
    json_path = os.path.join(args.output_dir, 'results', f"{args.arch}_representation_entropy.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump({
            "baseline": baseline_metrics,
            "dadp_sweeps": dadp_metrics_dict
        }, f, indent=4)
    print(f"✅ Representation metrics JSON saved to: {json_path}")
    print("Done!")

if __name__ == '__main__':
    main()