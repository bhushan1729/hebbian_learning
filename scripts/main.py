import argparse
import os
import torch
import json
from data_loader import get_data_loaders
from model import BaselineMLP, HebbianMLP
from engine import Trainer

def main():
    parser = argparse.ArgumentParser(description='Hebbian-Inspired Pruning Experiment')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--mode', type=str, default='hebbian', choices=['baseline', 'hebbian'], help='mode: baseline, hebbian')
    parser.add_argument('--colab', action='store_true', help='running in Google Colab environment')
    parser.add_argument('--prune_interval', type=int, default=500, help='interval for pruning')
    parser.add_argument('--prune_threshold', type=float, default=0.0001, help='threshold for pruning')
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10'], help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for datasets')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory for results')
    parser.add_argument('--exp_name', type=str, default=None, help='Custom name for this experiment run')
    
    args = parser.parse_args()

    # Handle Colab specific paths
    if args.colab:
        try:
            # We skip interactive mount here as requested, user will mount manually
            drive_path = '/content/drive/MyDrive/hebbian_learning'
            args.data_dir = os.path.join(drive_path, 'data')
            args.output_dir = os.path.join(drive_path, 'results')
            print(f"Colab mode active. Data: {args.data_dir}, Results: {args.output_dir}")
        except Exception as e:
            print(f"Colab pathing issue: {e}. Using local paths.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    train_loader, test_loader = get_data_loaders(args.dataset, args.batch_size, args.data_dir)

    # Initialize Model
    input_size = 784 if args.dataset == 'MNIST' else 3072
    if args.mode == 'baseline':
        model = BaselineMLP(input_size=input_size)
    else:
        model = HebbianMLP(input_size=input_size)

    # Naming logic
    base_name = args.exp_name if args.exp_name else f"{args.mode}_{args.dataset}"
    checkpoint_name = f"{base_name}.pth"
    checkpoint_path = os.path.join(args.output_dir, checkpoint_name)
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        lr=args.lr,
        prune_interval=args.prune_interval,
        prune_threshold=args.prune_threshold,
        checkpoint_path=checkpoint_path
    )

    # Auto-resume
    trainer.load_checkpoint()

    # Run Training
    history = trainer.run(args.epochs)

    # Save History
    history_file = os.path.join(args.output_dir, f"history_{base_name}.json")
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Final history saved to {history_file}")

if __name__ == '__main__':
    main()
