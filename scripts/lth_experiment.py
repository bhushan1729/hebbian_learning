import argparse
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add scripts directory to path for clean imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from data_loader import get_data_loaders
from model import (
    BaselineMLP, BaselineCNN, BaselineVGG16, get_resnet18, 
    convert_to_masked_model, get_model_sparsity
)
from engine import Trainer

class DualLogger:
    def __init__(self, filepath, mode="w"):
        self.terminal = sys.stdout
        self.log = open(filepath, mode, encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_architecture(arch, num_classes, device):
    if arch == 'mlp':
        model = BaselineMLP(input_size=784, num_classes=num_classes)
    elif arch == 'cnn':
        model = BaselineCNN(num_classes=num_classes)
    elif arch == 'vgg16':
        model = BaselineVGG16(num_classes=num_classes)
    elif arch == 'resnet18':
        model = get_resnet18(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    
    # Convert to masked model to support mask tracking
    model = convert_to_masked_model(model)
    return model.to(device)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description='Lottery Ticket Hypothesis Verification with DADP')
    parser.add_argument('--arch', type=str, default='mlp', choices=['mlp', 'cnn', 'vgg16', 'resnet18'])
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10'])
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--prune_threshold', type=float, default=0.0001, help='DADP threshold for Run A')
    parser.add_argument('--prune_interval', type=int, default=500, help='DADP interval for Run A')
    parser.add_argument('--seed_a', type=int, default=42, help='Original seed for Run A and Run B')
    parser.add_argument('--seed_c', type=int, default=2024, help='Random re-initialization seed for Run C')
    parser.add_argument('--colab', action='store_true')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--resume_from', type=str2bool, default=False, help='resume from checkpoint if exists (True or False)')
    
    args = parser.parse_args()
    
    # Handle Colab specific paths
    if args.colab:
        drive_path = '/content/drive/MyDrive/hebbian_learning'
        if args.data_dir == './data':
            args.data_dir = os.path.join(drive_path, 'data')
        args.output_dir = os.path.join(drive_path, 'results')
        
    # Configure logs directory and start dual logging
    logs_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_mode = "a" if args.resume_from else "w"
    log_name = f"lth_experiment_{args.arch}_{args.dataset}"
    sys.stdout = DualLogger(os.path.join(logs_dir, f"{log_name}.log"), mode=log_mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Data
    train_loader, test_loader = get_data_loaders(args.dataset, args.batch_size, args.data_dir)
    num_classes = 10
    
    # =========================================================================
    # RUN A: DADP Baseline
    # =========================================================================
    print("\n" + "="*50)
    print("RUN A: Training DADP Baseline (Discovering Winning Mask)")
    print("="*50)
    
    history_a = None
    final_sparsity = None
    dadp_mask = None
    
    history_path_a = os.path.join(args.output_dir, 'results', f"history_lth_runA_{args.arch}.json")
    checkpoint_path_a = os.path.join(args.output_dir, 'models', f"lth_runA_{args.arch}_best.pth")
    
    if args.resume_from and os.path.exists(history_path_a) and os.path.exists(checkpoint_path_a):
        try:
            import json
            with open(history_path_a, 'r') as f:
                history_a = json.load(f)
            if history_a.get('test_acc') and len(history_a['test_acc']) >= args.epochs:
                final_sparsity = history_a['sparsity'][-1]
                print(f"--> Found cached Run A results: Sparsity={final_sparsity*100:.2f}%, Test Acc={history_a['test_acc'][-1]:.2f}%")
                
                # Load mask from checkpoint
                from structured_pruning import load_sparse_checkpoint
                model_a_temp = get_architecture(args.arch, num_classes, device)
                state_a = load_sparse_checkpoint(checkpoint_path_a, device)
                model_a_temp.load_state_dict(state_a['model_state_dict'])
                
                dadp_mask = {}
                for name, module in model_a_temp.named_modules():
                    if hasattr(module, 'mask'):
                        dadp_mask[name] = module.mask.clone()
        except Exception as e:
            print(f"--> Failed to load cached Run A: {e}. Re-running Run A...")
            history_a = None
            final_sparsity = None
            dadp_mask = None

    if history_a is None or final_sparsity is None or dadp_mask is None:
        set_seed(args.seed_a)
        model_a = get_architecture(args.arch, num_classes, device)
        
        trainer_a = Trainer(
            model=model_a,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            mode='hebbian',
            lr=args.lr,
            prune_interval=args.prune_interval,
            prune_threshold=args.prune_threshold,
            output_dir=args.output_dir,
            base_name=f"lth_runA_{args.arch}"
        )
        
        history_a = trainer_a.run(args.epochs)
        final_sparsity = get_model_sparsity(model_a)
        print(f"--> Run A finished. Global Sparsity: {final_sparsity*100:.2f}%, Test Acc: {history_a['test_acc'][-1]:.2f}%")
        
        # Extract the final DADP mask dictionary
        dadp_mask = {}
        for name, module in model_a.named_modules():
            if hasattr(module, 'mask'):
                dadp_mask[name] = module.mask.clone()
            
    # =========================================================================
    # RUN B: The Winning Ticket Test
    # =========================================================================
    print("\n" + "="*50)
    print("RUN B: Training Winning Ticket (Reset to original W0)")
    print("="*50)
    
    history_b = None
    history_path_b = os.path.join(args.output_dir, 'results', f"history_lth_runB_{args.arch}.json")
    
    if args.resume_from and os.path.exists(history_path_b):
        try:
            import json
            with open(history_path_b, 'r') as f:
                history_b = json.load(f)
            if history_b.get('test_acc') and len(history_b['test_acc']) >= args.epochs:
                print(f"--> Found cached Run B results: Test Acc={history_b['test_acc'][-1]:.2f}%")
        except Exception as e:
            print(f"--> Failed to load cached Run B: {e}. Re-running Run B...")
            history_b = None

    if history_b is None:
        # Use exact same seed_a to recreate same W0 initialization
        set_seed(args.seed_a)
        model_b = get_architecture(args.arch, num_classes, device)
        
        # Apply mask and zero out weights
        for name, module in model_b.named_modules():
            if name in dadp_mask:
                module.mask.copy_(dadp_mask[name])
                with torch.no_grad():
                    module.weight.data *= module.mask.data
                
                # Register backward hook to zero out gradients of pruned weights (late-binding safe)
                module.weight.register_hook(lambda grad, m=module.mask.clone(): grad * m)
                    
        trainer_b = Trainer(
            model=model_b,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            mode='hebbian',
            lr=args.lr,
            prune_interval=0,  # Disable progressive pruning (fixed mask)
            prune_threshold=0.0,
            output_dir=args.output_dir,
            base_name=f"lth_runB_{args.arch}"
        )
        
        history_b = trainer_b.run(args.epochs)
        print(f"--> Run B finished. Test Acc: {history_b['test_acc'][-1]:.2f}%")
    
    # =========================================================================
    # RUN C: The Control (Random Re-initialization)
    # =========================================================================
    print("\n" + "="*50)
    print("RUN C: Training Random Re-init (W'0 with new seed)")
    print("="*50)
    
    history_c = None
    history_path_c = os.path.join(args.output_dir, 'results', f"history_lth_runC_{args.arch}.json")
    
    if args.resume_from and os.path.exists(history_path_c):
        try:
            import json
            with open(history_path_c, 'r') as f:
                history_c = json.load(f)
            if history_c.get('test_acc') and len(history_c['test_acc']) >= args.epochs:
                print(f"--> Found cached Run C results: Test Acc={history_c['test_acc'][-1]:.2f}%")
        except Exception as e:
            print(f"--> Failed to load cached Run C: {e}. Re-running Run C...")
            history_c = None

    if history_c is None:
        # Use different seed_c to get completely new random initialization
        set_seed(args.seed_c)
        model_c = get_architecture(args.arch, num_classes, device)
        
        # Apply same mask and zero out weights
        for name, module in model_c.named_modules():
            if name in dadp_mask:
                module.mask.copy_(dadp_mask[name])
                with torch.no_grad():
                    module.weight.data *= module.mask.data
                    
                # Register backward hook to zero out gradients of pruned weights (late-binding safe)
                module.weight.register_hook(lambda grad, m=module.mask.clone(): grad * m)
                    
        trainer_c = Trainer(
            model=model_c,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            mode='hebbian',
            lr=args.lr,
            prune_interval=0,  # Disable progressive pruning (fixed mask)
            prune_threshold=0.0,
            output_dir=args.output_dir,
            base_name=f"lth_runC_{args.arch}"
        )
        
        history_c = trainer_c.run(args.epochs)
        print(f"--> Run C finished. Test Acc: {history_c['test_acc'][-1]:.2f}%")
    
    # =========================================================================
    # Plotting & Comparison Table
    # =========================================================================
    epochs_range = np.arange(1, args.epochs + 1)
    
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(epochs_range, history_a['test_acc'], color='#d62728', marker='o', linewidth=2, label='Run A: DADP Progressive (Dynamic)')
    plt.plot(epochs_range, history_b['test_acc'], color='#1f77b4', marker='s', linewidth=2, label='Run B: Winning Ticket (Reset to W0)')
    plt.plot(epochs_range, history_c['test_acc'], color='#2ca02c', marker='^', linewidth=2, label="Run C: Random Re-init (W'0)")
    
    plt.title(f"Lottery Ticket Hypothesis Verification ({args.arch.upper()} on {args.dataset})\nSparsity: {final_sparsity*100:.2f}%", fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Test Accuracy (%)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', fontsize=10)
    
    plot_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"lth_validation_{args.arch}_{args.dataset}.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print("\n" + "="*50)
    print("EXPERIMENTAL SUMMARY TABLE")
    print("="*50)
    print(f"| Run | Configuration | Sparsity (%) | Final Test Acc (%) |")
    print(f"| :--- | :--- | :---: | :---: |")
    print(f"| **Run A** | DADP Baseline (Dynamic Pruning) | {final_sparsity*100:.2f}% | {history_a['test_acc'][-1]:.2f}% |")
    print(f"| **Run B** | Winning Ticket (Reset to W0) | {final_sparsity*100:.2f}% | {history_b['test_acc'][-1]:.2f}% |")
    print(f"| **Run C** | Random Re-init (W'0 seed={args.seed_c}) | {final_sparsity*100:.2f}% | {history_c['test_acc'][-1]:.2f}% |")
    print("="*50)
    print(f"Validation plot successfully saved to: {plot_path}")

if __name__ == '__main__':
    main()
