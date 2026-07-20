import argparse
import os
import sys
import torch
import torch.nn as nn
import json
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
        return BaselineMLP(input_size=784, num_classes=num_classes).to(device)
    elif arch == 'cnn':
        return BaselineCNN(num_classes=num_classes).to(device)
    elif arch == 'vgg16':
        return BaselineVGG16(num_classes=num_classes).to(device)
    elif arch == 'resnet18':
        # Don't mask here yet, we will mask it after initializing standard weights
        return get_resnet18(num_classes=num_classes, masked=False).to(device)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

def apply_initialization(model, init_type):
    """
    Applies the specified weight initialization method to the Conv2d and Linear layers.
    """
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                w = module.weight
                if init_type == 'kaiming_normal':
                    nn.init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu')
                elif init_type == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
                elif init_type == 'xavier_normal':
                    nn.init.xavier_normal_(w)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(w)
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(w)
                elif init_type == 'normal_0.02':
                    nn.init.normal_(w, mean=0.0, std=0.02)
                elif init_type == 'normal_0.1':
                    nn.init.normal_(w, mean=0.0, std=0.1)
                else:
                    raise ValueError(f"Unknown initialization method: {init_type}")
                
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

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
    parser = argparse.ArgumentParser(description='DADP Weight Initialization Sensitivity Benchmark')
    parser.add_argument('--arch', type=str, default='mlp', choices=['mlp', 'cnn', 'vgg16', 'resnet18'])
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--prune_threshold', type=float, default=0.0001, help='Fixed DADP threshold for pruning')
    parser.add_argument('--prune_interval', type=int, default=500)
    parser.add_argument('--inits', type=str, nargs='+', 
                        default=['kaiming_normal', 'kaiming_uniform', 'xavier_normal', 'xavier_uniform', 'orthogonal', 'normal_0.02', 'normal_0.1'],
                        help='List of initialization schemes to sweep')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results/init_ablation')
    parser.add_argument('--colab', action='store_true')
    parser.add_argument('--resume_from', type=str2bool, default=False, help='resume from checkpoint if exists (True or False)')
    
    args = parser.parse_args()
    
    if args.colab:
        drive_path = '/content/drive/MyDrive/hebbian_learning'
        if args.data_dir == './data':
            args.data_dir = os.path.join(drive_path, 'data')
        args.output_dir = os.path.join(drive_path, 'results/init_ablation')
        
    # Configure logs directory and start dual logging
    logs_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_mode = "a" if args.resume_from else "w"
    log_name = f"init_ablation_{args.arch}_{args.dataset}"
    sys.stdout = DualLogger(os.path.join(logs_dir, f"{log_name}.log"), mode=log_mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Sweeping initializations: {args.inits}")
    print(f"Fixed Pruning Threshold: {args.prune_threshold}")
    
    train_loader, test_loader = get_data_loaders(args.dataset, args.batch_size, args.data_dir)
    num_classes = 10
    
    # Storage structure for ablation results
    ablation_results = {}
    
    # Load existing results if resuming
    json_path = os.path.join(args.output_dir, 'results', f"init_ablation_{args.arch}_{args.dataset}.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    if args.resume_from and os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                ablation_results = json.load(f)
            print(f"--> Loaded existing ablation results from {json_path} for resumption.")
        except Exception as e:
            print(f"--> Could not load existing results JSON: {e}. Starting fresh.")
            ablation_results = {}
    elif not args.resume_from:
        if os.path.exists(json_path):
            os.remove(json_path)
            print(f"Removed old results JSON at {json_path} to start from scratch.")

    for init_name in args.inits:
        print("\n" + "="*80)
        print(f"Executing: Initialization = {init_name}")
        print("="*80)
        
        # Check if already completed
        if args.resume_from and init_name in ablation_results:
            print(f"--> Skipping {init_name} (Already completed in cached results)")
            continue
            
        # 1. Initialize and apply weights
        set_seed(42)  # Use seed 42 to keep the data loaders and model structure aligned
        model = get_architecture(args.arch, num_classes, device)
        apply_initialization(model, init_name)
        
        # 2. Convert to Masked Model for DADP
        model = convert_to_masked_model(model)
        
        # 3. Train with DADP
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            mode='hebbian',
            lr=args.lr,
            prune_interval=args.prune_interval,
            prune_threshold=args.prune_threshold,
            output_dir=args.output_dir,
            base_name=f"init_run_{args.arch}_{init_name}"
        )
        
        history = trainer.run(args.epochs)
        final_acc = history['test_acc'][-1]
        final_sparsity = get_model_sparsity(model) * 100
        
        ablation_results[init_name] = {
            'accuracy': final_acc,
            'sparsity': final_sparsity,
            'train_loss': history['train_loss'][-1],
            'test_loss': history['test_loss'][-1]
        }
        
        # Save results dynamically
        with open(json_path, 'w') as f:
            json.dump(ablation_results, f, indent=4)
        
        print(f"--> Done {init_name}. Accuracy: {final_acc:.2f}%, Sparsity: {final_sparsity:.2f}%")

    # =========================================================================
    # Summary Table and Plotting
    # =========================================================================
    print("\n" + "="*80)
    print("WEIGHT INITIALIZATION ABLATION RESULTS SUMMARY")
    print("="*80)
    print(f"| Initialization Method | Final Sparsity (%) | Final Test Acc (%) |")
    print(f"| :--- | :---: | :---: |")
    
    inits_list = list(ablation_results.keys())
    accuracies = [ablation_results[name]['accuracy'] for name in inits_list]
    sparsities = [ablation_results[name]['sparsity'] for name in inits_list]
    
    for name in inits_list:
        sp = ablation_results[name]['sparsity']
        acc = ablation_results[name]['accuracy']
        print(f"| {name} | {sp:.2f}% | {acc:.2f}% |")
    print("="*80)
    
    # 1x2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    
    # Subplot 1: Accuracy
    colors_acc = ['#1f77b4' if 'normal_0.1' not in name else '#d62728' for name in inits_list]
    y_pos = np.arange(len(inits_list))
    ax1.barh(y_pos, accuracies, color=colors_acc, height=0.55, edgecolor='black', alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(inits_list, fontsize=10, fontweight='bold')
    ax1.invert_yaxis()  # top-down
    ax1.set_xlabel('Final Test Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5, axis='x')
    
    # Add values on the bars
    for i, v in enumerate(accuracies):
        ax1.text(v + 1.0, i + .05, f"{v:.2f}%", color='black', fontweight='bold', fontsize=9)
        
    # Subplot 2: Sparsity
    colors_sp = ['#2ca02c' if 'normal_0.1' not in name else '#d62728' for name in inits_list]
    ax2.barh(y_pos, sparsities, color=colors_sp, height=0.55, edgecolor='black', alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])  # Hide y-labels on second plot to avoid redundancy
    ax2.invert_yaxis()
    ax2.set_xlabel('Emergent Sparsity (%)', fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5, axis='x')
    
    for i, v in enumerate(sparsities):
        ax2.text(v + 1.0, i + .05, f"{v:.2f}%", color='black', fontweight='bold', fontsize=9)
        
    fig.suptitle(f"DADP Weight Initialization Sensitivity Ablation ({args.arch.upper()} on {args.dataset})\nFixed Threshold: {args.prune_threshold}", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    plot_path = os.path.join(args.output_dir, 'plots', f"init_ablation_{args.arch}_{args.dataset}.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Results JSON saved to: {json_path}")
    print(f"Shaded validation plot saved to: {plot_path}")
    print("="*80)

if __name__ == '__main__':
    main()
