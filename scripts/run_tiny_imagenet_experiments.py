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
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--data_dir', type=str, default='/content/data')
    parser.add_argument('--output_dir', type=str, default='/content/drive/MyDrive/hebbian_learning/results/tiny_imagenet_experiments')
    
    args = parser.parse_args()
    
    # Environment auto-detection
    is_colab = 'google.colab' in sys.modules or args.colab
    is_kaggle = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or args.kaggle
    
    if is_colab:
        drive_path = '/content/drive/MyDrive/hebbian_learning'
        if args.output_dir == './results':
            args.output_dir = os.path.join(drive_path, 'results/tiny_imagenet_experiments')
    elif is_kaggle:
        if args.data_dir == './data':
            args.data_dir = '/tmp/data'
        args.output_dir = '/kaggle/working/hebbian_learning/results/tiny_imagenet_experiments'
        
    python_cmd = sys.executable
    
    # 1. Baseline Dense ResNet-18
    print("--- Running DENSE ResNet-18 Baseline ---")
    run_cmd([
        python_cmd, 'scripts/main.py',
        '--batch_size', str(args.batch_size),
        '--epochs', str(args.epochs),
        '--lr', str(args.lr),
        '--mode', 'baseline',
        '--arch', 'resnet18',
        '--dataset', 'TinyImageNet',
        '--data_dir', args.data_dir,
        '--output_dir', args.output_dir,
        '--colab' if is_colab else '',
        '--early_stopping', 'False',
        '--resume_from', 'False'
    ])
    
    # 2. DADP (Hebbian) ResNet-18 Sweeps
    print("\n--- Running DADP (Hebbian) ResNet-18 Sweeps ---")
    hebbian_thresholds = ["1e-6", "5e-6", "1e-5", "5e-5", "1e-4"]
    for thr in hebbian_thresholds:
        print(f"\n--- Running DADP (Hebbian) ResNet-18 (Threshold: {thr}) ---")
        run_cmd([
            python_cmd, 'scripts/main.py',
            '--batch_size', str(args.batch_size),
            '--epochs', str(args.epochs),
            '--lr', str(args.lr),
            '--mode', 'hebbian',
            '--arch', 'resnet18',
            '--prune_interval', '500',
            '--prune_threshold', thr,
            '--dataset', 'TinyImageNet',
            '--data_dir', args.data_dir,
            '--output_dir', args.output_dir,
            '--colab' if is_colab else '',
            '--early_stopping', 'False',
            '--resume_from', 'False'
        ])
        
    # 3. SNIP ResNet-18 Sweeps
    print("\n--- Running SNIP ResNet-18 Sweeps ---")
    snip_sparsities = ["0.70", "0.80", "0.90", "0.95"]
    for sp in snip_sparsities:
        print(f"\n--- Running SNIP ResNet-18 (Sparsity: {sp}) ---")
        run_cmd([
            python_cmd, 'scripts/main.py',
            '--batch_size', str(args.batch_size),
            '--epochs', str(args.epochs),
            '--lr', str(args.lr),
            '--mode', 'snip',
            '--arch', 'resnet18',
            '--prune_interval', '0',
            '--prune_threshold', '0.0',
            '--sparsity', sp,
            '--dataset', 'TinyImageNet',
            '--data_dir', args.data_dir,
            '--output_dir', args.output_dir,
            '--colab' if is_colab else '',
            '--early_stopping', 'False',
            '--resume_from', 'False'
        ])
        
    # 4. MAGNITUDE ResNet-18 Sweeps
    print("\n--- Running MAGNITUDE ResNet-18 Sweeps ---")
    magnitude_sparsities = ["0.70", "0.80", "0.90", "0.95"]
    for sp in magnitude_sparsities:
        print(f"\n--- Running MAGNITUDE ResNet-18 (Sparsity: {sp}) ---")
        run_cmd([
            python_cmd, 'scripts/main.py',
            '--batch_size', str(args.batch_size),
            '--epochs', str(args.epochs),
            '--lr', str(args.lr),
            '--mode', 'magnitude',
            '--arch', 'resnet18',
            '--prune_interval', '0',
            '--prune_threshold', '0.0',
            '--sparsity', sp,
            '--dataset', 'TinyImageNet',
            '--data_dir', args.data_dir,
            '--output_dir', args.output_dir,
            '--colab' if is_colab else '',
            '--early_stopping', 'False',
            '--resume_from', 'False'
        ])
        
    # 5. RIGL ResNet-18 Sweeps
    print("\n--- Running RIGL ResNet-18 Sweeps ---")
    rigl_sparsities = ["0.70", "0.80", "0.90", "0.95"]
    for sp in rigl_sparsities:
        print(f"\n--- Running RIGL ResNet-18 (Sparsity: {sp}) ---")
        run_cmd([
            python_cmd, 'scripts/main.py',
            '--batch_size', str(args.batch_size),
            '--epochs', str(args.epochs),
            '--lr', str(args.lr),
            '--mode', 'rigl',
            '--arch', 'resnet18',
            '--prune_interval', '0',
            '--prune_threshold', '0.0',
            '--sparsity', sp,
            '--rigl_interval', '100',
            '--rigl_prune_fraction', '0.2',
            '--dataset', 'TinyImageNet',
            '--data_dir', args.data_dir,
            '--output_dir', args.output_dir,
            '--colab' if is_colab else '',
            '--early_stopping', 'False',
            '--resume_from', 'False'
        ])
            
    print("\n✅ ResNet-18 Tiny-ImageNet Benchmark Suite execution complete!")

if __name__ == '__main__':
    main()
