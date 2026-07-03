import torch
import torch.nn as nn
import os

def save_sparse_checkpoint(state, path):
    """
    Saves a PyTorch state dict using sparse representations for masked weights
    to minimize file size on disk.
    """
    if 'model_state_dict' not in state:
        torch.save(state, path)
        return

    model_state = state['model_state_dict']
    sparse_model_state = {}
    
    # Identify keys that correspond to masks
    mask_keys = {k for k in model_state.keys() if k.endswith('.mask')}
    
    for k, v in model_state.items():
        if isinstance(v, torch.Tensor):
            # Check if this is a weight layer with a mask
            if k.endswith('.weight'):
                mask_name = k.replace('.weight', '.mask')
                if mask_name in mask_keys:
                    mask = model_state[mask_name]
                    # Convert to sparse COO tensor
                    sparse_model_state[k] = (v * mask).detach().cpu().to_sparse_coo()
                    continue
            sparse_model_state[k] = v.detach().cpu()
        else:
            sparse_model_state[k] = v
            
    state['model_state_dict'] = sparse_model_state
    
    # Make mask_dict sparse if present
    if 'mask_dict' in state and state['mask_dict']:
        sparse_mask_dict = {}
        for k, v in state['mask_dict'].items():
            if isinstance(v, torch.Tensor):
                sparse_mask_dict[k] = v.detach().cpu().to_sparse_coo()
            else:
                sparse_mask_dict[k] = v
        state['mask_dict'] = sparse_mask_dict

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"[Sparse Save] Sparsified checkpoint saved to {path} (Size reduced on disk)")

def load_sparse_checkpoint(path, device):
    """
    Loads a sparsified checkpoint and expands sparse tensors back into dense format
    for PyTorch model state initialization.
    """
    state = torch.load(path, map_location=device)
    if 'model_state_dict' not in state:
        return state

    model_state = state['model_state_dict']
    dense_model_state = {}
    for k, v in model_state.items():
        if isinstance(v, torch.Tensor) and v.is_sparse:
            dense_model_state[k] = v.to_dense().to(device)
        elif isinstance(v, torch.Tensor):
            dense_model_state[k] = v.to(device)
        else:
            dense_model_state[k] = v
            
    state['model_state_dict'] = dense_model_state
    
    if 'mask_dict' in state and state['mask_dict']:
        dense_mask_dict = {}
        for k, v in state['mask_dict'].items():
            if isinstance(v, torch.Tensor) and v.is_sparse:
                dense_mask_dict[k] = v.to_dense().to(device)
            elif isinstance(v, torch.Tensor):
                dense_mask_dict[k] = v.to(device)
            else:
                dense_mask_dict[k] = v
        state['mask_dict'] = dense_mask_dict
        
    return state


def compress_model_structured(model):
    """
    Physically prunes completely dead neurons (in Linear layers) and channels 
    (in Conv2d layers) along with their corresponding input connections in 
    subsequent layers and BatchNorm adjustments.
    Works sequentially through model modules.
    """
    print("\n--- Running Structured Model Compression ---")
    
    # 1. Collect all layers that we need to prune sequentially
    layers = []
    names = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            layers.append(module)
            names.append(name)
            
    if not layers:
        print("No prunable layers found.")
        return model

    # Keep track of active indices from previous layer
    # None indicates all inputs are active (e.g. input layer)
    active_indices = None
    prev_layer_type = None
    
    with torch.no_grad():
        for i in range(len(layers)):
            layer = layers[i]
            name = names[i]
            
            # --- BN LAYER ---
            if isinstance(layer, nn.BatchNorm2d):
                if active_indices is not None:
                    # Filter BatchNorm parameters
                    num_features = len(active_indices)
                    device = layer.weight.device
                    
                    new_bn = nn.BatchNorm2d(
                        num_features, eps=layer.eps, momentum=layer.momentum, 
                        affine=layer.affine, track_running_stats=layer.track_running_stats
                    ).to(device)
                    
                    if layer.affine:
                        new_bn.weight.copy_(layer.weight[active_indices])
                        new_bn.bias.copy_(layer.bias[active_indices])
                    if layer.track_running_stats:
                        new_bn.running_mean.copy_(layer.running_mean[active_indices])
                        new_bn.running_var.copy_(layer.running_var[active_indices])
                        new_bn.num_batches_tracked.copy_(layer.num_batches_tracked)
                        
                    # Find parent module and replace
                    _replace_module(model, name, new_bn)
                    layers[i] = new_bn
                    print(f"Pruned BatchNorm '{name}': channels {layer.num_features} -> {num_features}")
                prev_layer_type = 'Conv2d'
                continue

            # --- CONV2D LAYER ---
            elif isinstance(layer, nn.Conv2d):
                # Filter input channels first based on previous layer's output
                weight = layer.weight
                bias = layer.bias
                device = weight.device
                
                # Apply previous layer's channel pruning to input dimension (dim 1)
                if active_indices is not None:
                    weight = weight[:, active_indices, :, :]

                # Identify active output channels in current layer
                mask = getattr(layer, 'mask', None)
                if mask is not None:
                    # An output channel is active if it has at least one active connection
                    # mask shape: (out_channels, in_channels, kh, kw)
                    out_sums = mask.view(mask.size(0), -1).sum(dim=1)
                    current_active = torch.where(out_sums > 0)[0]
                    # Never prune everything, keep at least 1 channel
                    if len(current_active) == 0:
                        current_active = torch.tensor([0], device=device)
                else:
                    current_active = torch.arange(weight.size(0), device=device)
                
                # Check if this is the last convolutional layer or output classification layer
                # We do not prune the final output classes (e.g. classification head outputs)
                is_last_conv_or_linear = (i == len(layers) - 1) or \
                                         (i == len(layers) - 2 and isinstance(layers[i+1], nn.Linear) and layers[i+1].out_features == 10)
                                         
                if is_last_conv_or_linear:
                    # Keep all output classes
                    current_active = torch.arange(weight.size(0), device=device)
                
                # Filter weight and bias along output dimension (dim 0)
                weight = weight[current_active, :, :, :]
                if bias is not None:
                    bias = bias[current_active]
                    
                # Create replacement Conv2d
                # Check if it was MaskedConv2d
                from model import MaskedConv2d
                if isinstance(layer, MaskedConv2d):
                    new_conv = MaskedConv2d(
                        weight.size(1), weight.size(0), layer.kernel_size,
                        stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
                        groups=layer.groups, bias=bias is not None
                    ).to(device)
                    # Also crop the mask
                    new_mask = layer.mask[current_active, :, :, :]
                    if active_indices is not None:
                        new_mask = new_mask[:, active_indices, :, :]
                    new_conv.mask.copy_(new_mask)
                else:
                    new_conv = nn.Conv2d(
                        weight.size(1), weight.size(0), layer.kernel_size,
                        stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
                        groups=layer.groups, bias=bias is not None
                    ).to(device)
                    
                new_conv.weight.copy_(weight)
                if bias is not None:
                    new_conv.bias.copy_(bias)
                    
                _replace_module(model, name, new_conv)
                layers[i] = new_conv
                print(f"Pruned Conv2d '{name}': shape {list(layer.weight.shape)} -> {list(weight.shape)}")
                active_indices = current_active
                prev_layer_type = 'Conv2d'

            # --- LINEAR LAYER ---
            elif isinstance(layer, nn.Linear):
                weight = layer.weight
                bias = layer.bias
                device = weight.device
                
                # Handle input dimension pruning
                if active_indices is not None:
                    if prev_layer_type == 'Conv2d':
                        # Handle transitions from Conv to Linear (flattening)
                        num_flat_features = weight.size(1)
                        num_prev_channels = len(active_indices)
                        
                        if num_flat_features % num_prev_channels == 0:
                            spatial_dim = num_flat_features // num_prev_channels
                            # Reconstruct indices for flattened features
                            expanded_indices = []
                            for idx in active_indices:
                                expanded_indices.extend(range(idx.item() * spatial_dim, (idx.item() + 1) * spatial_dim))
                            expanded_indices = torch.tensor(expanded_indices, device=device)
                            weight = weight[:, expanded_indices]
                        else:
                            # Fallback / mismatch: skip input filtering
                            print(f"Warning: size mismatch in Conv->Linear transition for '{name}'")
                    else:
                        # Linear -> Linear transition
                        # Ensure active_indices fits within weight size
                        if max(active_indices).item() < weight.size(1):
                            weight = weight[:, active_indices]
                        else:
                            print(f"Warning: size mismatch in Linear->Linear transition for '{name}'")

                # Identify active output neurons
                mask = getattr(layer, 'mask', None)
                if mask is not None:
                    out_sums = mask.sum(dim=1)
                    current_active = torch.where(out_sums > 0)[0]
                    if len(current_active) == 0:
                        current_active = torch.tensor([0], device=device)
                else:
                    current_active = torch.arange(weight.size(0), device=device)
                
                # Do not prune output classes of final layer
                is_last_layer = (i == len(layers) - 1)
                if is_last_layer:
                    current_active = torch.arange(weight.size(0), device=device)
                    
                weight = weight[current_active, :]
                if bias is not None:
                    bias = bias[current_active]
                    
                # Create replacement Linear
                from model import MaskedLinear
                if isinstance(layer, MaskedLinear):
                    new_linear = MaskedLinear(weight.size(1), weight.size(0), bias=bias is not None).to(device)
                    new_mask = layer.mask[current_active, :]
                    if active_indices is not None:
                        if prev_layer_type == 'Conv2d':
                            # Expand indices
                            num_flat_features = new_mask.size(1)
                            num_prev_channels = len(active_indices)
                            if num_flat_features % num_prev_channels == 0:
                                spatial_dim = num_flat_features // num_prev_channels
                                expanded_indices = []
                                for idx in active_indices:
                                    expanded_indices.extend(range(idx.item() * spatial_dim, (idx.item() + 1) * spatial_dim))
                                expanded_indices = torch.tensor(expanded_indices, device=device)
                                new_mask = new_mask[:, expanded_indices]
                        else:
                            # Linear -> Linear
                            if max(active_indices).item() < new_mask.size(1):
                                new_mask = new_mask[:, active_indices]
                    new_linear.mask.copy_(new_mask)
                else:
                    new_linear = nn.Linear(weight.size(1), weight.size(0), bias=bias is not None).to(device)
                    
                new_linear.weight.copy_(weight)
                if bias is not None:
                    new_linear.bias.copy_(bias)
                    
                _replace_module(model, name, new_linear)
                layers[i] = new_linear
                print(f"Pruned Linear '{name}': shape {list(layer.weight.shape)} -> {list(weight.shape)}")
                active_indices = current_active
                prev_layer_type = 'Linear'

    print("Structured compression complete!\n")
    return model

def _replace_module(model, name, new_module):
    """
    Helper function to dynamically replace a sub-module by name in PyTorch.
    """
    parts = name.split('.')
    curr = model
    for p in parts[:-1]:
        if p.isdigit():
            curr = curr[int(p)]
        else:
            curr = getattr(curr, p)
            
    last = parts[-1]
    if last.isdigit():
        curr[int(last)] = new_module
    else:
        setattr(curr, last, new_module)
