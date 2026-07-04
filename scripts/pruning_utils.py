import torch

def snip_prune(model, loss_fn, dataloader, device, sparsity=0.9):
    model.to(device)
    model.train()

    # Use ONLY ONE batch
    x, y = next(iter(dataloader))
    x, y = x.to(device), y.to(device)

    # Forward + backward
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()

    scores = []
    params = []

    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None and "weight" in name:
            score = torch.abs(p.grad * p)
            scores.append(score.view(-1))
            params.append((name, p, score))

    if not scores:
        return {}

    all_scores = torch.cat(scores)
    k = int((1 - sparsity) * all_scores.numel())
    if k == 0:
        threshold = all_scores.max() + 1 # Prune everything
    else:
        threshold = torch.topk(all_scores, k)[0][-1]

    mask_dict = {}
    total_elements = 0
    active_elements = 0
    
    for name, p, score in params:
        mask = (score >= threshold).float()
        mask_dict[name] = mask.to(p.device)
        p.data.mul_(mask)
        
        total_elements += mask.numel()
        active_elements += mask.sum().item()
        
    print(f"[SNIP] Sparsity initialized. Active: {int(active_elements)} / {total_elements} ({(1 - active_elements/total_elements)*100:.2f}% sparsity)")

    return mask_dict

def magnitude_prune(model, sparsity=0.9):
    scores = []
    params = []

    for name, p in model.named_parameters():
        if p.requires_grad and "weight" in name:
            score = torch.abs(p.data)
            scores.append(score.view(-1))
            params.append((name, p, score))

    if not scores:
        return {}

    all_scores = torch.cat(scores)
    k = int((1 - sparsity) * all_scores.numel())
    if k == 0:
        threshold = all_scores.max() + 1
    else:
        threshold = torch.topk(all_scores, k)[0][-1]

    mask_dict = {}
    total_elements = 0
    active_elements = 0

    for name, p, score in params:
        mask = (score >= threshold).float()
        mask_dict[name] = mask.to(p.device)
        p.data.mul_(mask)
        
        total_elements += mask.numel()
        active_elements += mask.sum().item()

    print(f"[Magnitude] Sparsity initialized. Active: {int(active_elements)} / {total_elements} ({(1 - active_elements/total_elements)*100:.2f}% sparsity)")

    return mask_dict

def rigl_step(model, mask_dict, prune_fraction=0.2):
    for name, p in model.named_parameters():
        if name not in mask_dict or p.grad is None:
            continue

        mask = mask_dict[name]

        # PRUNE (small active weights)
        weights = torch.abs(p.data)
        num_active = int(mask.sum().item())
        k = int(prune_fraction * num_active)

        if k < 1:
            continue

        flat_mask = mask.view(-1)
        
        # Find k smallest active weights
        active_scores = weights.clone()
        active_scores[mask == 0] = float('inf')
        prune_indices = torch.topk(active_scores.view(-1), k, largest=False)[1]
        flat_mask[prune_indices] = 0

        # REGROW (large gradients where mask=0)
        grad = torch.abs(p.grad)
        
        # Find k largest inactive gradients
        regrow_scores = grad.clone()
        regrow_scores[mask == 1] = -1.0
        
        grow_indices = torch.topk(regrow_scores.view(-1), k, largest=True)[1]
        flat_mask[grow_indices] = 1

        # Apply updated mask
        p.data.mul_(mask)

    return mask_dict

def apply_mask(model, mask_dict):
    with torch.no_grad():
        # Synchronize mask_dict with layer .mask buffers
        for name, module in model.named_modules():
            if hasattr(module, 'mask'):
                weight_name = f"{name}.weight"
                if weight_name in mask_dict:
                    module.mask.copy_(mask_dict[weight_name])
                    
        # Apply mask to parameters to keep weights zeroed out
        for name, p in model.named_parameters():
            if name in mask_dict:
                p.mul_(mask_dict[name])

def init_random_mask(model, sparsity=0.9):
    # Initializes an EXACT sparsity random mask avoiding torch.rand_like direct > threshold.
    scores = []
    params = []

    for name, p in model.named_parameters():
        if p.requires_grad and "weight" in name:
            score = torch.rand_like(p.data)
            scores.append(score.view(-1))
            params.append((name, p, score))

    if not scores:
        return {}
        
    all_scores = torch.cat(scores)
    k = int((1 - sparsity) * all_scores.numel())
    if k == 0:
        threshold = all_scores.max() + 1
    else:
        threshold = torch.topk(all_scores, k)[0][-1]

    mask_dict = {}
    total_elements = 0
    active_elements = 0

    for name, p, score in params:
        mask = (score >= threshold).float()
        mask_dict[name] = mask.to(p.device)
        p.data.mul_(mask)
        
        total_elements += mask.numel()
        active_elements += mask.sum().item()

    print(f"[RigL Init] Exact random mask applied. Active: {int(active_elements)} / {total_elements} ({(1 - active_elements/total_elements)*100:.2f}% sparsity)")
    
    return mask_dict
