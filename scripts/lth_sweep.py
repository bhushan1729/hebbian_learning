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
    parser = argparse.ArgumentParser(description='DADP Lottery Ticket Hypothesis Sweep')
    parser.add_argument('--arch', type=str, default='mlp', choices=['mlp', 'cnn', 'vgg16', 'resnet18'])
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--prune_interval', type=int, default=500, help='pruning step interval')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--thresholds', type=float, nargs='+', default=[1e-5, 5e-5, 1e-4], 
                        help='Space-separated list of DADP thresholds to sweep')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44], 
                        help='Space-separated list of seeds to run for standard deviation')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results/lth_sweep')
    parser.add_argument('--colab', action='store_true')
    parser.add_argument('--kaggle', action='store_true', help='running in Kaggle environment')
    parser.add_argument('--resume_from', type=str2bool, default=False, help='resume from checkpoint if exists (True or False)')
    
    args = parser.parse_args()

    # Auto-detect environments
    is_colab = 'google.colab' in sys.modules or args.colab
    is_kaggle = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or args.kaggle

    # Handle environment specific paths
    if is_colab:
        try:
            drive_path = '/content/drive/MyDrive/hebbian_learning'
            if args.data_dir == './data':
                args.data_dir = os.path.join(drive_path, 'data')
            if args.output_dir == './results/lth_sweep':
                args.output_dir = os.path.join(drive_path, 'results/lth_sweep')
            print(f"Colab environment detected. Data: {args.data_dir}, Results: {args.output_dir}")
        except Exception as e:
            print(f"Colab pathing issue: {e}. Using local paths.")
    elif is_kaggle:
        try:
            if args.data_dir == './data':
                args.data_dir = '/tmp/data'  # Use fast local scratch /tmp on Kaggle
            if args.output_dir == './results/lth_sweep':
                args.output_dir = '/kaggle/working/hebbian_learning/results/lth_sweep'
            print(f"Kaggle environment detected. Data: {args.data_dir}, Results: {args.output_dir}")
        except Exception as e:
            print(f"Kaggle pathing issue: {e}. Using local paths.")
        
    # Configure logs directory and start dual logging
    logs_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_mode = "a" if args.resume_from else "w"
    log_name = f"lth_sweep_{args.arch}_{args.dataset}"
    sys.stdout = DualLogger(os.path.join(logs_dir, f"{log_name}.log"), mode=log_mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Sweeping thresholds: {args.thresholds}")
    print(f"Seeds: {args.seeds}")
    
    train_loader, test_loader = get_data_loaders(args.dataset, args.batch_size, args.data_dir)
    num_classes = 10
    
    # Storage structure for sweep results
    # threshold -> { 'sparsities': [], 'acc_a': [], 'acc_b': [], 'acc_c': [] }
    sweep_results = {}
    
    # Load existing sweep results if resuming
    json_path = os.path.join(args.output_dir, 'results', f"lth_sweep_{args.arch}_{args.dataset}.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    if args.resume_from and os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                loaded = json.load(f)
            # JSON keys are always strings; convert threshold keys back to float
            for k, v in loaded.items():
                sweep_results[float(k)] = v
            print(f"--> Loaded existing sweep results from {json_path} for resumption.")
        except Exception as e:
            print(f"--> Could not load existing sweep results JSON: {e}. Starting fresh.")
            sweep_results = {}
    elif not args.resume_from:
        if os.path.exists(json_path):
            os.remove(json_path)
            print(f"Removed old sweep results JSON at {json_path} to start from scratch.")

    for thr in args.thresholds:
        if thr not in sweep_results:
            sweep_results[thr] = {
                'sparsities': [],
                'acc_a': [],
                'acc_b': [],
                'acc_c': []
            }
        
        for idx, seed in enumerate(args.seeds):
            print("\n" + "="*80)
            print(f"Executing: Threshold = {thr} | Seed = {seed} (Index {idx})")
            print("="*80)
            
            # Check if this specific index was already completed and saved
            if (args.resume_from and 
                idx < len(sweep_results[thr]['acc_a']) and 
                idx < len(sweep_results[thr]['acc_b']) and 
                idx < len(sweep_results[thr]['acc_c']) and
                idx < len(sweep_results[thr]['sparsities'])):
                print(f"--> Skipping Seed = {seed} (Already fully completed in cached results)")
                continue

            # ----------------------------------------------------
            # RUN A: DADP Baseline
            # ----------------------------------------------------
            acc_a = None
            sparsity = None
            dadp_mask = None
            
            history_path_a = os.path.join(args.output_dir, 'results', f"history_sweep_runA_{args.arch}_thr{thr}_seed{seed}.json")
            checkpoint_path_a = os.path.join(args.output_dir, 'models', f"sweep_runA_{args.arch}_thr{thr}_seed{seed}_best.pth")
            
            if args.resume_from and os.path.exists(history_path_a) and os.path.exists(checkpoint_path_a):
                try:
                    with open(history_path_a, 'r') as f:
                        h_a = json.load(f)
                    if h_a.get('test_acc') and len(h_a['test_acc']) >= args.epochs:
                        acc_a = h_a['test_acc'][-1]
                        sparsity = h_a['sparsity'][-1]
                        print(f"--> Found cached Run A results: Sparsity={sparsity*100:.2f}%, Test Acc={acc_a:.2f}%")
                        
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
                    acc_a = None
                    sparsity = None
                    dadp_mask = None

            if acc_a is None or sparsity is None or dadp_mask is None:
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
                    prune_interval=args.prune_interval,
                    prune_threshold=thr,
                    output_dir=args.output_dir,
                    base_name=f"sweep_runA_{args.arch}_thr{thr}_seed{seed}"
                )
                history_a = trainer_a.run(args.epochs)
                sparsity = get_model_sparsity(model_a)
                acc_a = history_a['test_acc'][-1]
                
                # Extract winning mask
                dadp_mask = {}
                for name, module in model_a.named_modules():
                    if hasattr(module, 'mask'):
                        dadp_mask[name] = module.mask.clone()
            
            if len(sweep_results[thr]['sparsities']) > idx:
                sweep_results[thr]['sparsities'][idx] = sparsity
                sweep_results[thr]['acc_a'][idx] = acc_a
            else:
                sweep_results[thr]['sparsities'].append(sparsity)
                sweep_results[thr]['acc_a'].append(acc_a)
            
            # ----------------------------------------------------
            # RUN B: Winning Ticket
            # ----------------------------------------------------
            acc_b = None
            history_path_b = os.path.join(args.output_dir, 'results', f"history_sweep_runB_{args.arch}_thr{thr}_seed{seed}.json")
            
            if args.resume_from and os.path.exists(history_path_b):
                try:
                    with open(history_path_b, 'r') as f:
                        h_b = json.load(f)
                    if h_b.get('test_acc') and len(h_b['test_acc']) >= args.epochs:
                        acc_b = h_b['test_acc'][-1]
                        print(f"--> Found cached Run B results: Test Acc={acc_b:.2f}%")
                except Exception as e:
                    print(f"--> Failed to load cached Run B: {e}. Re-running Run B...")
                    acc_b = None

            if acc_b is None:
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
                
            if len(sweep_results[thr]['acc_b']) > idx:
                sweep_results[thr]['acc_b'][idx] = acc_b
            else:
                sweep_results[thr]['acc_b'].append(acc_b)
            
            # ----------------------------------------------------
            # RUN C: Random Re-initialization
            # ----------------------------------------------------
            acc_c = None
            history_path_c = os.path.join(args.output_dir, 'results', f"history_sweep_runC_{args.arch}_thr{thr}_seed{seed}.json")
            
            if args.resume_from and os.path.exists(history_path_c):
                try:
                    with open(history_path_c, 'r') as f:
                        h_c = json.load(f)
                    if h_c.get('test_acc') and len(h_c['test_acc']) >= args.epochs:
                        acc_c = h_c['test_acc'][-1]
                        print(f"--> Found cached Run C results: Test Acc={acc_c:.2f}%")
                except Exception as e:
                    print(f"--> Failed to load cached Run C: {e}. Re-running Run C...")
                    acc_c = None

            if acc_c is None:
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
                
            if len(sweep_results[thr]['acc_c']) > idx:
                sweep_results[thr]['acc_c'][idx] = acc_c
            else:
                sweep_results[thr]['acc_c'].append(acc_c)
            
            # Save raw stats dictionary dynamically after each seed completes to support real-time checkpoints
            os.makedirs(args.output_dir, exist_ok=True)
            with open(json_path, 'w') as f:
                json.dump(sweep_results, f, indent=4)
            
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
    json_path = os.path.join(args.output_dir, 'results', f"lth_sweep_{args.arch}_{args.dataset}.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
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
    
    plot_path = os.path.join(args.output_dir, 'plots', f"lth_sweep_{args.arch}_{args.dataset}.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print("="*80)
    print(f"Sweep stats saved to: {json_path}")
    print(f"Shaded validation plot saved to: {plot_path}")
    print("="*80)

if __name__ == '__main__':
    main()
