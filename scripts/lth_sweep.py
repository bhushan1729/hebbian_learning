import argparse
import os
import sys
import torch
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
    
    model = convert_to_masked_model(model)
    return model.to(device)

def main():
    parser = argparse.ArgumentParser(description='DADP Lottery Ticket Hypothesis Sweep')
    parser.add_argument('--arch', type=str, default='mlp', choices=['mlp', 'cnn', 'vgg16', 'resnet18'])
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--thresholds', type=float, nargs='+', default=[1e-5, 5e-5, 1e-4], 
                        help='Space-separated list of DADP thresholds to sweep')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44], 
                        help='Space-separated list of seeds to run for standard deviation')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results/lth_sweep')
    parser.add_argument('--colab', action='store_true')
    
    args = parser.parse_args()
    
    if args.colab:
        drive_path = '/content/drive/MyDrive/hebbian_learning'
        if args.data_dir == './data':
            args.data_dir = os.path.join(drive_path, 'data')
        args.output_dir = os.path.join(drive_path, 'results/lth_sweep')
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Sweeping thresholds: {args.thresholds}")
    print(f"Seeds: {args.seeds}")
    
    train_loader, test_loader = get_data_loaders(args.dataset, args.batch_size, args.data_dir)
    num_classes = 10
    
    # Storage structure for sweep results
    # threshold -> { 'sparsities': [], 'acc_a': [], 'acc_b': [], 'acc_c': [] }
    sweep_results = {}
    
    for thr in args.thresholds:
        sweep_results[thr] = {
            'sparsities': [],
            'acc_a': [],
            'acc_b': [],
            'acc_c': []
        }
        
        for seed in args.seeds:
            print("\n" + "="*80)
            print(f"Executing: Threshold = {thr} | Seed = {seed}")
            print("="*80)
            
            # ----------------------------------------------------
            # RUN A: DADP Baseline
            # ----------------------------------------------------
            print(f"\n[Run A] DADP baseline training...")
            set_seed(seed)
            model_a = get_architecture(args.arch, num_classes, device)
            
            trainer_a = Trainer(
                model=model_a,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                mode='hebbian',
                lr=args.lr,
                prune_interval=500,
                prune_threshold=thr,
                output_dir=args.output_dir,
                base_name=f"sweep_runA_{args.arch}_thr{thr}_seed{seed}"
            )
            history_a = trainer_a.run(args.epochs)
            sparsity = get_model_sparsity(model_a)
            acc_a = history_a['test_acc'][-1]
            
            sweep_results[thr]['sparsities'].append(sparsity)
            sweep_results[thr]['acc_a'].append(acc_a)
            
            # Extract winning mask
            dadp_mask = {}
            for name, module in model_a.named_modules():
                if hasattr(module, 'mask'):
                    dadp_mask[name] = module.mask.clone()
            
            # ----------------------------------------------------
            # RUN B: Winning Ticket
            # ----------------------------------------------------
            print(f"\n[Run B] Winning Ticket (reset to same seed)...")
            set_seed(seed)
            model_b = get_architecture(args.arch, num_classes, device)
            
            for name, module in model_b.named_modules():
                if name in dadp_mask:
                    module.mask.copy_(dadp_mask[name])
                    with torch.no_grad():
                        module.weight.data *= module.mask.data
                    # Late-binding safe gradient hooks
                    module.weight.register_hook(lambda grad, m=module.mask.clone(): grad * m)
                    
            trainer_b = Trainer(
                model=model_b,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                mode='hebbian',
                lr=args.lr,
                prune_interval=0,
                prune_threshold=0.0,
                output_dir=args.output_dir,
                base_name=f"sweep_runB_{args.arch}_thr{thr}_seed{seed}"
            )
            history_b = trainer_b.run(args.epochs)
            acc_b = history_b['test_acc'][-1]
            sweep_results[thr]['acc_b'].append(acc_b)
            
            # ----------------------------------------------------
            # RUN C: Random Re-initialization
            # ----------------------------------------------------
            print(f"\n[Run C] Random Re-initialization (new seed)...")
            set_seed(seed + 1000)  # Use offset to guarantee different seed
            model_c = get_architecture(args.arch, num_classes, device)
            
            for name, module in model_c.named_modules():
                if name in dadp_mask:
                    module.mask.copy_(dadp_mask[name])
                    with torch.no_grad():
                        module.weight.data *= module.mask.data
                    # Late-binding safe gradient hooks
                    module.weight.register_hook(lambda grad, m=module.mask.clone(): grad * m)
                    
            trainer_c = Trainer(
                model=model_c,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                mode='hebbian',
                lr=args.lr,
                prune_interval=0,
                prune_threshold=0.0,
                output_dir=args.output_dir,
                base_name=f"sweep_runC_{args.arch}_thr{thr}_seed{seed}"
            )
            history_c = trainer_c.run(args.epochs)
            acc_c = history_c['test_acc'][-1]
            sweep_results[thr]['acc_c'].append(acc_c)
            
            print(f"Results for seed {seed}: Sparsity={sparsity*100:.2f}%, Run A={acc_a:.2f}%, Run B={acc_b:.2f}%, Run C={acc_c:.2f}%")

    # =========================================================================
    # Process Statistics
    # =========================================================================
    sorted_thrs = sorted(args.thresholds)
    
    mean_sparsities = []
    
    mean_a, std_a = [], []
    mean_b, std_b = [], []
    mean_c, std_c = [], []
    
    print("\n" + "="*80)
    print("SWEEP STATISTICAL RESULTS SUMMARY")
    print("="*80)
    print(f"| Threshold | Avg Sparsity (%) | Run A (DADP) (%) | Run B (Winning Ticket) (%) | Run C (Random Re-init) (%) |")
    print(f"| :--- | :---: | :---: | :---: | :---: |")
    
    for thr in sorted_thrs:
        res = sweep_results[thr]
        avg_sp = np.mean(res['sparsities']) * 100
        
        m_a, s_a = np.mean(res['acc_a']), np.std(res['acc_a'])
        m_b, s_b = np.mean(res['acc_b']), np.std(res['acc_b'])
        m_c, s_c = np.mean(res['acc_c']), np.std(res['acc_c'])
        
        mean_sparsities.append(avg_sp)
        mean_a.append(m_a); std_a.append(s_a)
        mean_b.append(m_b); std_b.append(s_b)
        mean_c.append(m_c); std_c.append(s_c)
        
        print(f"| {thr} | {avg_sp:.2f}% | {m_a:.2f} ± {s_a:.2f}% | {m_b:.2f} ± {s_b:.2f}% | {m_c:.2f} ± {s_c:.2f}% |")
    
    # Save raw stats dictionary to output folder
    json_path = os.path.join(args.output_dir, f"lth_sweep_{args.arch}_{args.dataset}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(sweep_results, f, indent=4)
        
    # =========================================================================
    # Plotting Curve with Standard Deviation Shading
    # =========================================================================
    plt.figure(figsize=(10, 6), dpi=150)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    mean_sparsities = np.array(mean_sparsities)
    
    # Plot curves with shaded fill representing standard deviation
    # Run A
    plt.plot(mean_sparsities, mean_a, color='#d62728', marker='o', linewidth=2.2, label='Run A: DADP Baseline (Dynamic)')
    plt.fill_between(mean_sparsities, np.array(mean_a) - np.array(std_a), np.array(mean_a) + np.array(std_a), color='#d62728', alpha=0.15)
    
    # Run B
    plt.plot(mean_sparsities, mean_b, color='#1f77b4', marker='s', linewidth=2.2, label='Run B: Winning Ticket (Reset to W0)')
    plt.fill_between(mean_sparsities, np.array(mean_b) - np.array(std_b), np.array(mean_b) + np.array(std_b), color='#1f77b4', alpha=0.15)
    
    # Run C
    plt.plot(mean_sparsities, mean_c, color='#2ca02c', marker='^', linewidth=2.2, label="Run C: Random Re-init (W'0)")
    plt.fill_between(mean_sparsities, np.array(mean_c) - np.array(std_c), np.array(mean_c) + np.array(std_c), color='#2ca02c', alpha=0.15)
    
    plt.title(f"Lottery Ticket Hypothesis Validation Matrix ({args.arch.upper()} on {args.dataset})\nTest Accuracy vs. Global Sparsity Level (N={len(args.seeds)} Seeds)", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Pruned Sparsity (%)', fontsize=11)
    plt.ylabel('Final Test Accuracy (%)', fontsize=11)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x:.1f}%'))
    
    plt.legend(loc='lower left', fontsize=10, frameon=True)
    
    plot_path = os.path.join(args.output_dir, f"lth_sweep_{args.arch}_{args.dataset}.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print("="*80)
    print(f"Sweep stats saved to: {json_path}")
    print(f"Shaded validation plot saved to: {plot_path}")
    print("="*80)

if __name__ == '__main__':
    main()
