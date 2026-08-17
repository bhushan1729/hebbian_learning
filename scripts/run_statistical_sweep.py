import argparse
import os
import sys
import json
import numpy as np
import torch

# Ensure repository root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from data_loader import get_data_loaders
from model import (
    BaselineMLP, BaselineCNN, BaselineVGG16, get_resnet18, 
    BiLSTM_CRF, get_mini_transformer, convert_to_masked_model,
    get_model_sparsity
)
from engine import Trainer

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_architecture(arch, num_classes, input_channels=3, input_size=3072, fc_input_dim=4096):
    if arch == 'mlp':
        model = BaselineMLP(input_size=input_size, num_classes=num_classes)
    elif arch == 'cnn':
        model = BaselineCNN(input_channels=input_channels, num_classes=num_classes, fc_input_dim=fc_input_dim)
    elif arch == 'vgg16':
        model = BaselineVGG16(input_channels=input_channels, num_classes=num_classes)
    elif arch == 'resnet18':
        model = get_resnet18(num_classes=num_classes, masked=False)
    elif arch == 'transformer':
        model = get_mini_transformer(vocab_size=5000, num_classes=num_classes, masked=False)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model

def main():
    parser = argparse.ArgumentParser(description='Multi-Seed DADP Statistical Sweep Runner')
    parser.add_argument('--arch', type=str, default='resnet18', choices=['mlp', 'cnn', 'vgg16', 'resnet18', 'bilstm_crf', 'transformer', 'bert'])
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['MNIST', 'CIFAR10', 'TinyImageNet', 'Tiny-ImageNet', 'CoNLL2003', 'SST2', 'IMDB'])
    parser.add_argument('--epochs', type=int, default=20, help='epochs to train per run')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--prune_interval', type=int, default=500, help='pruning interval steps')
    parser.add_argument('--thresholds', type=str, nargs='+', default=['1e-6', '5e-6', '1e-5', '5e-5', '1e-4', '5e-4'],
                        help='List of pruning thresholds to sweep')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 512, 1729],
                        help='List of seeds to execute')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results/statistical_sweeps')
    parser.add_argument('--colab', action='store_true')
    parser.add_argument('--kaggle', action='store_true')
    parser.add_argument('--save_model', type=str2bool, default=False, help='save heavy model checkpoints (.pth) (default: False)')
    parser.add_argument('--use_amp', type=str2bool, default=False, help='use AMP FP16 on GPU (default: False for exact FP32)')
    parser.add_argument('--transformer_model', type=str, default='prajjwal1/bert-mini', help='pre-trained HuggingFace transformer model')

    args = parser.parse_args()

    # Auto-detect environments
    is_colab = 'google.colab' in sys.modules or args.colab
    is_kaggle = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or args.kaggle

    if is_colab:
        drive_path = '/content/drive/MyDrive/hebbian_learning'
        if args.data_dir == './data':
            args.data_dir = os.path.join(drive_path, 'data')
        if args.output_dir == './results/statistical_sweeps':
            args.output_dir = os.path.join(drive_path, 'results/statistical_sweeps')
        print(f"Colab environment detected. Data: {args.data_dir}, Results: {args.output_dir}")
    elif is_kaggle:
        if args.data_dir == './data':
            args.data_dir = '/tmp/data'
        if args.output_dir == './results/statistical_sweeps':
            args.output_dir = '/kaggle/working/hebbian_learning/results/statistical_sweeps'
        print(f"Kaggle environment detected. Data: {args.data_dir}, Results: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    results_dir = os.path.join(args.output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Arch: {args.arch} ({args.transformer_model if args.arch == 'bert' else ''}) | Dataset: {args.dataset}")
    print(f"Sweeping thresholds: {args.thresholds}")
    print(f"Seeds: {args.seeds}")
    print(f"Save models (.pth): {args.save_model}\n")

    # Load Data once
    train_loader, test_loader = get_data_loaders(args.dataset, args.batch_size, args.data_dir, args.transformer_model)

    num_classes = 10
    input_channels = 1
    input_size = 784
    fc_input_dim = 3136

    if args.dataset == 'CIFAR10':
        num_classes = 10
        input_channels = 3
        input_size = 3072
        fc_input_dim = 4096
    elif args.dataset in ['TinyImageNet', 'Tiny-ImageNet', 'tiny_imagenet']:
        num_classes = 200
        input_channels = 3
        input_size = 12288
        fc_input_dim = 16384
    elif args.dataset == 'CoNLL2003':
        num_classes = 9
    elif args.dataset in ['SST2', 'IMDB']:
        num_classes = 2

    # Container for all statistical results
    stats_data = {}

    for thr_str in args.thresholds:
        thr = float(thr_str)
        stats_data[thr_str] = {
            'final_sparsity': [],
            'final_acc': [],
            'peak_acc': []
        }

        print("\n" + "="*80)
        print(f"🚀 SWEEPING THRESHOLD τ = {thr_str} across Seeds {args.seeds}")
        print("="*80)

        for seed in args.seeds:
            print(f"\n---> Running Seed = {seed} (Threshold τ = {thr_str})")
            set_seed(seed)

            base_name = f"hebbian_{args.arch}_{args.dataset}_thr{thr_str}_seed{seed}"

            # Instantiate model
            if args.arch == 'bilstm_crf':
                vocab_size = getattr(train_loader, 'vocab_size', 5000)
                tag_to_ix = getattr(train_loader, 'tag_to_ix', None)
                model = BiLSTM_CRF(vocab_size=vocab_size, tag_to_ix=tag_to_ix, embedding_dim=128, hidden_dim=128, masked=True)
            elif args.arch == 'bert':
                from transformers import AutoModelForSequenceClassification, BertForSequenceClassification
                if "bert-mini" in args.transformer_model or "bert-tiny" in args.transformer_model:
                    model = BertForSequenceClassification.from_pretrained(args.transformer_model, num_labels=num_classes)
                else:
                    model = AutoModelForSequenceClassification.from_pretrained(args.transformer_model, num_labels=num_classes)
                model = convert_to_masked_model(model)
            else:
                raw_model = get_architecture(args.arch, num_classes, input_channels, input_size, fc_input_dim)
                model = convert_to_masked_model(raw_model)

            config_dict = vars(args).copy()
            config_dict['seed'] = seed
            config_dict['prune_threshold'] = thr

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                mode='hebbian',
                lr=args.lr,
                prune_interval=args.prune_interval,
                prune_threshold=thr,
                output_dir=args.output_dir,
                base_name=base_name,
                config=config_dict,
                early_stopping=False,
                use_amp=args.use_amp,
                save_model=args.save_model
            )

            history = trainer.run(args.epochs)

            final_sp = history['sparsity'][-1] * 100.0
            final_acc = history['test_acc'][-1]
            peak_acc = max(history['test_acc'])

            stats_data[thr_str]['final_sparsity'].append(final_sp)
            stats_data[thr_str]['final_acc'].append(final_acc)
            stats_data[thr_str]['peak_acc'].append(peak_acc)

            print(f"✅ Completed Seed {seed} | Sparsity: {final_sp:.2f}% | Final Acc: {final_acc:.2f}% | Peak Acc: {peak_acc:.2f}%")

    # =========================================================================
    # CALCULATE MEAN & STD AND PRINT STATISTICAL SUMMARY TABLE
    # =========================================================================
    print("\n" + "="*90)
    print(f"📊 STATISTICAL SUMMARY RESULTS ({args.arch.upper()} on {args.dataset}) Across {len(args.seeds)} Seeds: {args.seeds}")
    print("="*90)

    table_header = f"{'Threshold (τ)':^18} | {'Final Sparsity (%)':^22} | {'Final Test Acc (%)':^22} | {'Peak Test Acc (%)':^22}"
    print(table_header)
    print("-" * 90)

    markdown_rows = []
    summary_report = {}

    for thr_str in args.thresholds:
        sp_arr = np.array(stats_data[thr_str]['final_sparsity'])
        final_acc_arr = np.array(stats_data[thr_str]['final_acc'])
        peak_acc_arr = np.array(stats_data[thr_str]['peak_acc'])

        sp_mean, sp_std = np.mean(sp_arr), np.std(sp_arr)
        final_mean, final_std = np.mean(final_acc_arr), np.std(final_acc_arr)
        peak_mean, peak_std = np.mean(peak_acc_arr), np.std(peak_acc_arr)

        row_str = f"τ = {thr_str:<14} | {sp_mean:6.2f} ± {sp_std:4.2f}%         | {final_mean:6.2f} ± {final_std:4.2f}%         | {peak_mean:6.2f} ± {peak_std:4.2f}%"
        print(row_str)

        markdown_rows.append(f"| `τ = {thr_str}` | {sp_mean:.2f} ± {sp_std:.2f}% | {final_mean:.2f} ± {final_std:.2f}% | {peak_mean:.2f} ± {peak_std:.2f}% |")

        summary_report[thr_str] = {
            'sparsity_mean': float(sp_mean), 'sparsity_std': float(sp_std),
            'final_acc_mean': float(final_mean), 'final_acc_std': float(final_std),
            'peak_acc_mean': float(peak_mean), 'peak_acc_std': float(peak_std),
            'raw_seeds': stats_data[thr_str]
        }

    print("="*90 + "\n")

    # Save summary files
    json_summary_path = os.path.join(args.output_dir, f"summary_{args.arch}_{args.dataset}.json")
    with open(json_summary_path, 'w') as f:
        json.dump(summary_report, f, indent=4)

    md_summary_path = os.path.join(args.output_dir, f"summary_{args.arch}_{args.dataset}.md")
    with open(md_summary_path, 'w') as f:
        f.write(f"# Statistical Sweep Summary: {args.arch.upper()} on {args.dataset}\n\n")
        f.write(f"Executed across {len(args.seeds)} seeds: `{args.seeds}` over `{args.epochs}` epochs.\n\n")
        f.write("| Threshold (τ) | Final Sparsity (%) | Final Test Acc (%) | Peak Test Acc (%) |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for mrow in markdown_rows:
            f.write(mrow + "\n")

    print(f"✅ Saved statistical JSON summary to: {json_summary_path}")
    print(f"✅ Saved statistical Markdown summary to: {md_summary_path}")

if __name__ == '__main__':
    main()
