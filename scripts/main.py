import argparse
import os
import torch
import json
import sys
from data_loader import get_data_loaders
from model import (
    BaselineMLP, BaselineCNN, BaselineVGG16, get_resnet18, BiLSTM_CRF, 
    get_mini_transformer, convert_to_masked_model
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

def main():
    parser = argparse.ArgumentParser(description='DADP & Baseline Pruning Benchmarking Pipeline')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--mode', type=str, default='hebbian', choices=['baseline', 'hebbian', 'snip', 'magnitude', 'rigl'], help='mode: baseline, hebbian (DADP), snip, magnitude, rigl')
    parser.add_argument('--arch', type=str, default='mlp', choices=['mlp', 'cnn', 'vgg16', 'resnet18', 'bilstm_crf', 'transformer'], help='architecture')
    parser.add_argument('--colab', action='store_true', help='running in Google Colab environment')
    parser.add_argument('--kaggle', action='store_true', help='running in Kaggle environment')
    parser.add_argument('--prune_interval', type=int, default=500, help='interval for pruning')
    parser.add_argument('--prune_threshold', type=float, default=0.0001, help='threshold for pruning')
    parser.add_argument('--sparsity', type=float, default=0.9, help='target sparsity for snip/magnitude/rigl')
    parser.add_argument('--rigl_prune_fraction', type=float, default=0.2, help='fraction of weights to prune/regrow in RigL step')
    parser.add_argument('--rigl_interval', type=int, default=100, help='RigL step interval')
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10', 'CoNLL2003', 'SST2', 'IMDB'], help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for datasets')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory for results')
    parser.add_argument('--exp_name', type=str, default=None, help='Custom name for this experiment run')
    parser.add_argument('--structured_prune', action='store_true', help='apply physical structured pruning to compress the network after training')
    parser.add_argument('--early_stopping', type=str2bool, default=False, help='enable early stopping (True or False)')
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
            if args.output_dir == './results':
                args.output_dir = os.path.join(drive_path, 'results')
            print(f"Colab environment detected. Data: {args.data_dir}, Results: {args.output_dir}")
        except Exception as e:
            print(f"Colab pathing issue: {e}. Using local paths.")
    elif is_kaggle:
        try:
            if args.data_dir == './data':
                args.data_dir = '/tmp/data'  # Use fast local scratch /tmp on Kaggle
            if args.output_dir == './results':
                args.output_dir = '/kaggle/working/hebbian_learning/results'
            print(f"Kaggle environment detected. Data: {args.data_dir}, Results: {args.output_dir}")
        except Exception as e:
            print(f"Kaggle pathing issue: {e}. Using local paths.")

    # For baseline mode, ensure pruning params are zeroed in metadata
    if args.mode == 'baseline':
        args.prune_interval = 0
        args.prune_threshold = 0.0

    # Naming logic
    if args.exp_name:
        base_name = args.exp_name
    else:
        base_name = f"{args.mode}_{args.arch}_{args.dataset}"
        if args.mode == 'hebbian':
            base_name += f"_thr{args.prune_threshold}_dt{args.prune_interval}"
        elif args.mode in ['snip', 'magnitude', 'rigl']:
            base_name += f"_sp{args.sparsity}"

    # Set up dual logging to both console and file (append if resuming, else write fresh)
    logs_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_mode = "a" if args.resume_from else "w"
    sys.stdout = DualLogger(os.path.join(logs_dir, f"{base_name}.log"), mode=log_mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    train_loader, test_loader = get_data_loaders(args.dataset, args.batch_size, args.data_dir)

    # Determine dimensions based on dataset
    num_classes = 10
    input_channels = 1
    input_size = 784
    fc_input_dim = 3136

    if args.dataset == 'CIFAR10':
        num_classes = 10
        input_channels = 3
        input_size = 3072
        fc_input_dim = 4096
    elif args.dataset == 'CoNLL2003':
        num_classes = 9 # standard NER classes
    elif args.dataset in ['SST2', 'IMDB']:
        num_classes = 2 # binary sentiment classification

    # Initialize Base Model
    if args.arch == 'mlp':
        model = BaselineMLP(input_size=input_size, num_classes=num_classes)
    elif args.arch == 'cnn':
        model = BaselineCNN(input_channels=input_channels, num_classes=num_classes, fc_input_dim=fc_input_dim)
    elif args.arch == 'vgg16':
        model = BaselineVGG16(input_channels=input_channels, num_classes=num_classes)
    elif args.arch == 'resnet18':
        model = get_resnet18(num_classes=num_classes, masked=False)
    elif args.arch == 'bilstm_crf':
        vocab_size = getattr(train_loader, 'vocab_size', 5000)
        tag_to_ix = getattr(train_loader, 'tag_to_ix', None)
        model = BiLSTM_CRF(vocab_size=vocab_size, tag_to_ix=tag_to_ix, embedding_dim=128, hidden_dim=128, masked=(args.mode != 'baseline'))
    elif args.arch == 'transformer':
        model = get_mini_transformer(vocab_size=5000, num_classes=num_classes, masked=False)

    # Convert to masked version if running Hebbian (DADP) or other pruning methods
    # This enables unified metric extraction and masking.
    if args.mode != 'baseline':
        model = convert_to_masked_model(model)
            
    # Prepare complete dictionary of CLI hyperparameters and output paths
    config_dict = vars(args).copy()
    config_dict['model_checkpoint_path'] = os.path.join(args.output_dir, 'models', f"{base_name}.pth")
    config_dict['history_json_path'] = os.path.join(args.output_dir, 'results', f"history_{base_name}.json")
    if args.structured_prune:
        config_dict['structured_compressed_checkpoint_path'] = os.path.join(args.output_dir, 'models', f"{base_name}_structured_compressed.pth")

    # Initialize Trainer with output splitting and configuration logging
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        mode=args.mode,
        lr=args.lr,
        prune_interval=args.prune_interval,
        prune_threshold=args.prune_threshold,
        sparsity=args.sparsity,
        rigl_prune_fraction=args.rigl_prune_fraction,
        rigl_interval=args.rigl_interval,
        output_dir=args.output_dir,
        base_name=base_name,
        config=config_dict,
        early_stopping=args.early_stopping
    )

    # Auto-resume if checkpoint exists and requested, else clean start
    if args.resume_from:
        trainer.load_checkpoint()
    else:
        # Overwrite: delete existing checkpoints and history for this run to start fresh
        if os.path.exists(trainer.checkpoint_path):
            os.remove(trainer.checkpoint_path)
            print(f"Removed old checkpoint at {trainer.checkpoint_path} to start from scratch.")
        if os.path.exists(trainer.checkpoint_best_path):
            os.remove(trainer.checkpoint_best_path)
            print(f"Removed old best checkpoint at {trainer.checkpoint_best_path} to start from scratch.")
        if os.path.exists(trainer.history_path):
            os.remove(trainer.history_path)
            print(f"Removed old history file at {trainer.history_path} to start from scratch.")

    # Run Training
    history = trainer.run(args.epochs)

    # Run Physical Structured Pruning if requested post-training
    if args.structured_prune and args.mode != 'baseline':
        from structured_pruning import compress_model_structured, safe_torch_save
        print("\n--- Performing Post-Training Physical Structured Pruning ---")
        try:
            compressed_model = compress_model_structured(model)
            # Save physical weights state dict
            compressed_checkpoint_path = os.path.join(args.output_dir, 'models', f"{base_name}_structured_compressed.pth")
            safe_torch_save(compressed_model.state_dict(), compressed_checkpoint_path)
            print(f"Physically compressed model state dict saved to {compressed_checkpoint_path}")
        except Exception as e:
            print(f"\n⚠️ Structured pruning skipped or failed: {e}")
            print("Note: Structured pruning is only supported for sequential architectures (like MLP, CNN, VGG16) and does not support branching/residual connections in ResNet-18.")

    print("Experiment run completed successfully.")

if __name__ == '__main__':
    main()
