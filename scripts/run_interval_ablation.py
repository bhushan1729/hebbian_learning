import argparse
import os
import sys
import json
import numpy as np
import torch

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

def main():
    parser = argparse.ArgumentParser(description='Prune Interval (dt) Ablation Study Runner')
    parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'vgg16', 'bert', 'mlp', 'bilstm_crf'])
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'MNIST', 'SST2', 'CoNLL2003', 'TinyImageNet'])
    parser.add_argument('--threshold', type=float, default=1e-5, help='Fixed pruning threshold for approx 90 percent target sparsity')
    parser.add_argument('--intervals', type=int, nargs='+', default=[100, 250, 500, 1000, 1500, 2000, 5000],
                        help='List of prune intervals dt (in batches) to sweep')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42], help='Seeds to run for ablation')
    parser.add_argument('--epochs', type=int, default=20, help='epochs to train per run')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--transformer_model', type=str, default='prajjwal1/bert-mini', help='transformer model ID for bert')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results/ablation_studies')
    parser.add_argument('--colab', action='store_true')
    parser.add_argument('--kaggle', action='store_true')
    parser.add_argument('--save_model', type=str2bool, default=False, help='save heavy model checkpoints (.pth)')
    parser.add_argument('--use_amp', type=str2bool, default=False, help='use AMP FP16 on GPU (default: False for exact FP32)')

    args = parser.parse_args()

    # Auto-detect environments
    is_colab = 'google.colab' in sys.modules or args.colab
    is_kaggle = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or args.kaggle

    if is_colab:
        drive_path = '/content/drive/MyDrive/hebbian_learning'
        if args.data_dir == './data':
            args.data_dir = os.path.join(drive_path, 'data')
        if args.output_dir == './results/ablation_studies':
            args.output_dir = os.path.join(drive_path, 'results/ablation_studies')
    elif is_kaggle:
        if args.data_dir == './data':
            args.data_dir = '/tmp/data'
        if args.output_dir == './results/ablation_studies':
            args.output_dir = '/kaggle/working/hebbian_learning/results/ablation_studies'

    os.makedirs(args.output_dir, exist_ok=True)
    results_dir = os.path.join(args.output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"--- PRUNE INTERVAL (Δt) ABLATION STUDY ---")
    print(f"Arch: {args.arch} | Dataset: {args.dataset} | Fixed Threshold τ = {args.threshold}")
    print(f"Evaluating Prune Intervals Δt: {args.intervals}")
    print(f"Seeds: {args.seeds} | Epochs: {args.epochs} | Batch Size: {args.batch_size}\n")

    # Load Data
    train_loader, test_loader = get_data_loaders(args.dataset, args.batch_size, args.data_dir, args.transformer_model)

    num_classes = 10
    input_channels = 3
    input_size = 3072
    fc_input_dim = 4096

    if args.dataset == 'CIFAR10':
        num_classes = 10
    elif args.dataset == 'MNIST':
        num_classes = 10
        input_channels = 1
        input_size = 784
    elif args.dataset in ['SST2', 'IMDB']:
        num_classes = 2

    # Storage for ablation results
    # dt -> { 'final_sparsity': [], 'final_acc': [], 'peak_acc': [] }
    ablation_summary = {}

    for dt in args.intervals:
        dt_str = str(dt)
        ablation_summary[dt_str] = {
            'final_sparsity': [],
            'final_acc': [],
            'peak_acc': []
        }

        print("\n" + "="*80)
        print(f"🚀 TESTING PRUNE INTERVAL Δt = {dt} batches (Threshold τ = {args.threshold})")
        print("="*80)

        for seed in args.seeds:
            print(f"\n---> Running Seed = {seed} | Δt = {dt} batches")
            set_seed(seed)

            base_name = f"ablation_dt{dt}_{args.arch}_{args.dataset}_thr{args.threshold}_seed{seed}"

            if args.arch == 'resnet18':
                raw_model = get_resnet18(num_classes=num_classes, masked=False)
                model = convert_to_masked_model(raw_model)
            elif args.arch == 'vgg16':
                raw_model = BaselineVGG16(input_channels=input_channels, num_classes=num_classes)
                model = convert_to_masked_model(raw_model)
            elif args.arch == 'bert':
                from transformers import AutoModelForSequenceClassification, BertForSequenceClassification
                if "bert-mini" in args.transformer_model or "bert-tiny" in args.transformer_model:
                    model = BertForSequenceClassification.from_pretrained(args.transformer_model, num_labels=num_classes)
                else:
                    model = AutoModelForSequenceClassification.from_pretrained(args.transformer_model, num_labels=num_classes)
                model = convert_to_masked_model(model)
            elif args.arch == 'mlp':
                raw_model = BaselineMLP(input_size=input_size, num_classes=num_classes)
                model = convert_to_masked_model(raw_model)
            elif args.arch == 'bilstm_crf':
                vocab_size = getattr(train_loader, 'vocab_size', 5000)
                tag_to_ix = getattr(train_loader, 'tag_to_ix', None)
                model = BiLSTM_CRF(vocab_size=vocab_size, tag_to_ix=tag_to_ix, embedding_dim=128, hidden_dim=128, masked=True)

            config_dict = vars(args).copy()
            config_dict['seed'] = seed
            config_dict['prune_interval'] = dt
            config_dict['prune_threshold'] = args.threshold

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                mode='hebbian',
                lr=args.lr,
                prune_interval=dt,
                prune_threshold=args.threshold,
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

            ablation_summary[dt_str]['final_sparsity'].append(final_sp)
            ablation_summary[dt_str]['final_acc'].append(final_acc)
            ablation_summary[dt_str]['peak_acc'].append(peak_acc)

            print(f"✅ Finished Δt = {dt} | Seed {seed} | Sparsity: {final_sp:.2f}% | Final Acc: {final_acc:.2f}% | Peak Acc: {peak_acc:.2f}%")

    # =========================================================================
    # SUMMARY TABLE & REPORT
    # =========================================================================
    print("\n" + "="*90)
    print(f"📊 ABLATION SUMMARY: Effect of Prune Interval (Δt) on {args.arch.upper()} ({args.dataset})")
    print(f"Fixed Threshold τ = {args.threshold} | Seeds: {args.seeds}")
    print("="*90)

    header = f"{'Prune Interval (Δt)':^22} | {'Final Sparsity (%)':^22} | {'Final Test Acc (%)':^22} | {'Peak Test Acc (%)':^22}"
    print(header)
    print("-" * 90)

    markdown_rows = []
    summary_report = {}

    for dt in args.intervals:
        dt_str = str(dt)
        sp_arr = np.array(ablation_summary[dt_str]['final_sparsity'])
        f_acc_arr = np.array(ablation_summary[dt_str]['final_acc'])
        p_acc_arr = np.array(ablation_summary[dt_str]['peak_acc'])

        sp_mean, sp_std = np.mean(sp_arr), np.std(sp_arr)
        f_mean, f_std = np.mean(f_acc_arr), np.std(f_acc_arr)
        p_mean, p_std = np.mean(p_acc_arr), np.std(p_acc_arr)

        batches_per_epoch = len(train_loader)
        prune_freq_per_epoch = batches_per_epoch / dt

        print(f"Δt = {dt:4d} ({prune_freq_per_epoch:.2f}/ep) | {sp_mean:6.2f} ± {sp_std:4.2f}%         | {f_mean:6.2f} ± {f_std:4.2f}%         | {p_mean:6.2f} ± {p_std:4.2f}%")

        markdown_rows.append(f"| `Δt = {dt}` ({prune_freq_per_epoch:.2f} prunes/epoch) | {sp_mean:.2f} ± {sp_std:.2f}% | {f_mean:.2f} ± {f_std:.2f}% | {p_mean:.2f} ± {p_std:.2f}% |")

        summary_report[dt_str] = {
            'batches_per_epoch': batches_per_epoch,
            'prune_frequency_per_epoch': float(prune_freq_per_epoch),
            'sparsity_mean': float(sp_mean), 'sparsity_std': float(sp_std),
            'final_acc_mean': float(f_mean), 'final_acc_std': float(f_std),
            'peak_acc_mean': float(p_mean), 'peak_acc_std': float(p_std),
            'raw_seeds': ablation_summary[dt_str]
        }

    print("="*90 + "\n")

    # Save output files
    json_path = os.path.join(args.output_dir, f"ablation_interval_{args.arch}_{args.dataset}.json")
    with open(json_path, 'w') as f:
        json.dump(summary_report, f, indent=4)

    md_path = os.path.join(args.output_dir, f"ablation_interval_{args.arch}_{args.dataset}.md")
    with open(md_path, 'w') as f:
        f.write(f"# Ablation Study: Effect of Pruning Interval (Δt) on {args.arch.upper()} ({args.dataset})\n\n")
        f.write(f"Fixed threshold $\\tau = {args.threshold}$ | Trained for {args.epochs} epochs over seeds {args.seeds}.\n\n")
        f.write("| Prune Interval (Δt) | Final Sparsity (%) | Final Test Acc (%) | Peak Test Acc (%) |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for mrow in markdown_rows:
            f.write(mrow + "\n")

    print(f"✅ Saved ablation JSON summary to: {json_path}")
    print(f"✅ Saved ablation Markdown summary to: {md_path}")

if __name__ == '__main__':
    main()
