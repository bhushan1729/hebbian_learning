import os
import sys
import subprocess
import argparse

def run_cmd(cmd):
    print(f"\n=======================================================")
    print(f"Executing: {' '.join(cmd)}")
    print(f"=======================================================\n")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"⚠️ Command failed with returncode {res.returncode}")

def main():
    parser = argparse.ArgumentParser(description='Run ResNet-18 Tiny-ImageNet Benchmark Suite')
    parser.add_argument('--colab', action='store_true', help='running in Google Colab')
    parser.add_argument('--kaggle', action='store_true', help='running in Kaggle')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs per experiment')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results/tiny_imagenet_experiments')
    
    args = parser.parse_args()
    
    # Environment auto-detection
    is_colab = 'google.colab' in sys.modules or args.colab
    is_kaggle = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or args.kaggle
    
    if is_colab:
        drive_path = '/content/drive/MyDrive/hebbian_learning'
        if args.data_dir == './data':
            args.data_dir = os.path.join(drive_path, 'data')
        args.output_dir = os.path.join(drive_path, 'results/tiny_imagenet_experiments')
    elif is_kaggle:
        if args.data_dir == './data':
            args.data_dir = '/tmp/data'
        args.output_dir = '/kaggle/working/hebbian_learning/results/tiny_imagenet_experiments'
        
    python_cmd = sys.executable
    
    # 1. Baseline Dense ResNet-18 on Tiny-ImageNet
    run_cmd([
        python_cmd, 'scripts/main.py',
        '--mode', 'baseline',
        '--arch', 'resnet18',
        '--dataset', 'TinyImageNet',
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--data_dir', args.data_dir,
        '--output_dir', args.output_dir
    ])
    
    # 2. DADP (Hebbian) Threshold Sweeps on Tiny-ImageNet
    dadp_thresholds = [1e-5, 5e-5, 1e-4, 5e-4]
    for thr in dadp_thresholds:
        run_cmd([
            python_cmd, 'scripts/main.py',
            '--mode', 'hebbian',
            '--arch', 'resnet18',
            '--dataset', 'TinyImageNet',
            '--prune_threshold', str(thr),
            '--prune_interval', '500',
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--lr', str(args.lr),
            '--data_dir', args.data_dir,
            '--output_dir', args.output_dir
        ])
        
    # 3. Standard Pruning Baselines (Magnitude, SNIP, RigL) at ~90% and ~95% Sparsity
    baseline_methods = ['magnitude', 'snip', 'rigl']
    sparsities = [0.90, 0.95]
    
    for mode in baseline_methods:
        for sp in sparsities:
            run_cmd([
                python_cmd, 'scripts/main.py',
                '--mode', mode,
                '--arch', 'resnet18',
                '--dataset', 'TinyImageNet',
                '--sparsity', str(sp),
                '--epochs', str(args.epochs),
                '--batch_size', str(args.batch_size),
                '--lr', str(args.lr),
                '--data_dir', args.data_dir,
                '--output_dir', args.output_dir
            ])
            
    print("\n✅ ResNet-18 Tiny-ImageNet Benchmark Suite execution complete!")

if __name__ == '__main__':
    main()
