import sys
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

# Ensure the scripts directory is in sys.path so sister imports (model, structured_pruning) work from any context
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from model import MaskedLinear, HebbianMLP

def plot_mlp_pruning(model, save_path="pruned_mlp_visualization.png", max_neurons_per_layer=15, title="MLP Structural Pruning Visualization"):
    """
    Visualizes the dense vs. DADP-pruned structure of an MLP model.
    Supports both downsampled layouts and full network visualization (by passing None or large values).
    Dynamically scales sizes and spacing, and optimizes performance for large networks.
    """
    # Find all MaskedLinear layers in the model
    layers = [m for m in model.modules() if isinstance(m, MaskedLinear) or isinstance(m, nn.Linear)]
    
    if not layers:
        print("Error: No Linear or MaskedLinear layers found in model.")
        return
        
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.axis('off')
    
    # Layer sizes
    layer_sizes = []
    layer_sizes.append(layers[0].in_features)
    for layer in layers:
        layer_sizes.append(layer.out_features)
        
    num_layers = len(layer_sizes)
    
    # Determine which indices of neurons to show for each layer
    sampled_indices = []
    for l_idx, size in enumerate(layer_sizes):
        if max_neurons_per_layer is not None and max_neurons_per_layer > 0 and size > max_neurons_per_layer:
            # Sample evenly
            step = size / max_neurons_per_layer
            indices = [int(i * step) for i in range(max_neurons_per_layer)]
            sampled_indices.append(indices)
        else:
            sampled_indices.append(list(range(size)))
            
    # Calculate spacing and vertical positions. Normalize total height so all columns match.
    total_height = 20.0
    node_positions = {} # maps (layer, node_index) to (x, y)
    node_spacings = {}
    
    for l_idx, indices in enumerate(sampled_indices):
        # Stretch columns slightly more if drawing full size to give space for connections
        x = l_idx * 4.5
        n_nodes = len(indices)
        if n_nodes > 1:
            spacing = total_height / (n_nodes - 1)
        else:
            spacing = 0.0
        node_spacings[l_idx] = spacing
        start_y = total_height / 2.0
        
        for draw_idx, orig_idx in enumerate(indices):
            y = start_y - draw_idx * spacing
            node_positions[(l_idx, orig_idx)] = (x, y)
            
    # Check size of connection grid. If too large, skip drawing pruned paths to keep it readable and fast.
    total_drawn_connections = sum(len(sampled_indices[l_idx]) * len(sampled_indices[l_idx+1]) for l_idx in range(num_layers - 1))
    draw_pruned_paths = total_drawn_connections < 15000
    
    # Connection line properties: dynamically scale down thickness & alpha for large nets
    max_layer_drawn_size = max(len(indices) for indices in sampled_indices)
    scale_factor = min(1.0, 30.0 / max_layer_drawn_size) if max_layer_drawn_size > 0 else 1.0
    
    # First, draw connections (so they appear behind the nodes)
    for l_idx in range(num_layers - 1):
        layer = layers[l_idx]
        weight = layer.weight.data
        mask = getattr(layer, 'mask', None)
        if mask is None:
            mask = torch.ones_like(weight)
            
        src_indices = sampled_indices[l_idx]
        dst_indices = sampled_indices[l_idx + 1]
        
        # Max weight magnitude for scaling line widths
        max_w = torch.abs(weight).max().item()
        if max_w == 0:
            max_w = 1.0
            
        for s_idx in src_indices:
            for d_idx in dst_indices:
                x1, y1 = node_positions[(l_idx, s_idx)]
                x2, y2 = node_positions[(l_idx + 1, d_idx)]
                
                # Weight index in PyTorch is weight[dst, src]
                w_val = weight[d_idx, s_idx].item()
                m_val = mask[d_idx, s_idx].item()
                
                if m_val > 0:
                    # Active connection: draw solid colored line
                    color = "#1f77b4" if w_val >= 0 else "#d62728" # Blue for positive, Red for negative
                    linewidth = (0.5 + 2.5 * (abs(w_val) / max_w)) * scale_factor
                    alpha = 0.7 * scale_factor
                    ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, alpha=alpha, zorder=1)
                elif draw_pruned_paths:
                    # Pruned connection: draw faint, dotted gray line
                    ax.plot([x1, x2], [y1, y2], color="gray", linewidth=0.2, alpha=0.15 * scale_factor, linestyle="dotted", zorder=0)

    # Next, draw nodes (neurons)
    for l_idx, indices in enumerate(sampled_indices):
        spacing = node_spacings[l_idx]
        # Dynamically size the circles based on layer vertical density
        radius = min(0.22, spacing * 0.45) if spacing > 0.0 else 0.22
        
        for orig_idx in indices:
            x, y = node_positions[(l_idx, orig_idx)]
            
            is_active = False
            # Check active status based on masks, restricted to only the drawn/sampled nodes to prevent visual mismatch
            if l_idx == 0:
                # Input layer: active if any outgoing mask to drawn hidden neurons is 1
                mask = getattr(layers[0], 'mask', None)
                if mask is not None:
                    dst_indices = sampled_indices[1]
                    is_active = (mask[dst_indices, orig_idx].sum() > 0).item()
                else:
                    is_active = True
            elif l_idx == num_layers - 1:
                # Output layer: active if any incoming mask from drawn hidden neurons is 1
                mask = getattr(layers[-1], 'mask', None)
                if mask is not None:
                    src_indices = sampled_indices[-2]
                    is_active = (mask[orig_idx, src_indices].sum() > 0).item()
                else:
                    is_active = True
            else:
                # Hidden layer: active if it has both drawn incoming and drawn outgoing active connections
                mask_in = getattr(layers[l_idx - 1], 'mask', None)
                mask_out = getattr(layers[l_idx], 'mask', None)
                
                src_indices = sampled_indices[l_idx - 1]
                dst_indices = sampled_indices[l_idx + 1]
                
                in_active = (mask_in[orig_idx, src_indices].sum() > 0).item() if mask_in is not None else True
                out_active = (mask_out[dst_indices, orig_idx].sum() > 0).item() if mask_out is not None else True
                is_active = in_active and out_active
                
            # Styling nodes
            if is_active:
                facecolor = "#2ca02c" if l_idx in [0, num_layers-1] else "#1f77b4" # Green for input/output, Blue for hidden
                edgecolor = "black"
                alpha = 1.0
                zorder = 3
            else:
                facecolor = "#e0e0e0"
                edgecolor = "#a0a0a0"
                alpha = 0.5
                zorder = 2
                
            circle = plt.Circle((x, y), radius=radius, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, zorder=zorder)
            ax.add_patch(circle)
            
            # Label layers (only once per layer)
            if orig_idx == indices[0]:
                label = "Input Layer" if l_idx == 0 else ("Output Layer" if l_idx == num_layers - 1 else f"Hidden {l_idx}")
                # Place label above the column top
                ax.text(x, (total_height / 2.0) + 0.8, label, ha='center', va='bottom', fontsize=12, fontweight='bold')
                # Subtitle under layer labels for original features size context
                ax.text(x, (total_height / 2.0) + 0.3, f"({len(indices)} shown / {layer_sizes[l_idx]} total)", ha='center', va='bottom', fontsize=8, style='italic')

    plt.title(title, fontsize=14, fontweight='bold', pad=30)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] Network architecture plot saved to {save_path}")

def generate_toy_visualization(save_path="results/toy_mlp_pruned.png"):
    """
    Creates a small toy HebbianMLP, applies some random sparsity to simulate DADP, 
    and saves its structure visualization.
    """
    print("Generating toy MLP visualization...")
    model = HebbianMLP(input_size=10, hidden_size=8, num_classes=4)
    
    # Manually mask out some connections to simulate pruning
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, MaskedLinear):
                # Set mask to 80% sparsity
                m.mask.copy_((torch.rand_like(m.mask) > 0.8).float())
                # Apply mask to weight
                m.weight.mul_(m.mask)
                
    plot_mlp_pruning(
        model, save_path=save_path, max_neurons_per_layer=12,
        title="DADP Pruned ANN Subnetwork (Surviving vs. Pruned Connections)"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize pruned MLP network")
    parser.add_argument('--toy', action='store_true', help="Generate toy MLP visualization")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to PyTorch model checkpoint (.pth)")
    parser.add_argument('--output', type=str, default="results/toy_mlp_pruned.png", help="Output visualization path")
    parser.add_argument('--input_size', type=int, default=784, help="Input features size (default: 784 for MNIST)")
    parser.add_argument('--hidden_size', type=int, default=512, help="Hidden layers feature size (default: 512)")
    parser.add_argument('--num_classes', type=int, default=10, help="Output classes count (default: 10)")
    parser.add_argument('--max_neurons', type=int, default=15, help="Max neurons to show per layer. Set to -1 to show all neurons (no downsampling).")
    args = parser.parse_args()
    
    if args.toy:
        generate_toy_visualization(args.output)
    elif args.checkpoint:
        from structured_pruning import load_sparse_checkpoint
        
        print(f"Loading checkpoint from: {args.checkpoint}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load sparse checkpoint
        try:
            checkpoint = load_sparse_checkpoint(args.checkpoint, device)
        except Exception as e:
            print(f"Error loading sparse checkpoint: {e}. Trying standard torch.load...")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            
        # Instantiate model structure
        model = HebbianMLP(input_size=args.input_size, hidden_size=args.hidden_size, num_classes=args.num_classes)
        model.to(device)
        
        # Extract model state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        print("Model checkpoint loaded successfully.")
        
        # Calculate actual connection sparsity
        total_conn = 0
        pruned_conn = 0
        for m in model.modules():
            if isinstance(m, MaskedLinear):
                total_conn += m.mask.numel()
                pruned_conn += (m.mask == 0).sum().item()
        
        sparsity_pct = (pruned_conn / total_conn * 100) if total_conn > 0 else 0.0
        
        # Create output plot title
        checkpoint_name = os.path.basename(args.checkpoint)
        title = f"DADP Pruned MLP ({checkpoint_name})\nSparsity: {sparsity_pct:.2f}% ({pruned_conn}/{total_conn} connections pruned)"
        
        print(f"Plotting model with shape: Input={args.input_size}, Hidden={args.hidden_size}, Output={args.num_classes}")
        print(f"Current sparsity: {sparsity_pct:.2f}%")
        
        max_neurons = args.max_neurons if args.max_neurons > 0 else None
        plot_mlp_pruning(
            model,
            save_path=args.output,
            max_neurons_per_layer=max_neurons,
            title=title
        )
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python scripts/visualize_network.py --checkpoint results/mnist/mlp_mnist_epoch20/models/hebbian_mlp_MNIST_thr1e-05_dt500.pth --output results/mnist_mlp_pruned.png")

