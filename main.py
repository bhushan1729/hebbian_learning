import argparse
import torch
import os
import json
from data_loader import get_data_loaders
from model import BaselineMLP, HebbianMLP
from engine import Trainer

def main():
    parser = argparse.ArgumentParser(description='Hebbian-Inspired Pruning Experiments')
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10'], help='Dataset to use')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--mode', type=str, default='hebbian', choices=['baseline', 'hebbian'], help='Experiment mode')
    parser.add_argument('--prune_interval', type=int, default=500, help='Steps between pruning actions')
    parser.add_argument('--prune_threshold', type=float, default=0.0001, help='Threshold for importance pruning')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for datasets')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory for results and checkpoints')
    parser.add_argument('--colab', action='store_true', help='Mount Google Drive for Colab usage')
    
    args = parser.parse_args()

    # Handle Google Drive mounting if requested
    if args.colab:
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            args.data_dir = '/content/drive/MyDrive/hebbian_learning/data'
            args.output_dir = '/content/drive/MyDrive/hebbian_learning/results'
        except ImportError:
            print("Google Colab environment not detected. Proceeding with local paths.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Data
    train_loader, test_loader = get_data_loaders(args.dataset, args.batch_size, args.data_dir)

    # Initialize Model
    input_size = 784 if args.dataset == 'MNIST' else 3072 # 32*32*3
    if args.mode == 'baseline':
        model = BaselineMLP(input_size=input_size)
        ckpt_name = f"baseline_{args.dataset}.pth"
    else:
        model = HebbianMLP(input_size=input_size)
        ckpt_name = f"hebbian_{args.dataset}.pth"

    checkpoint_path = os.path.join(args.output_dir, ckpt_name)
    
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

    # Auto-resume if checkpoint exists
    trainer.load_checkpoint()

    # Run Training
    history = trainer.run(args.epochs)

    # Save final history
    history_file = os.path.join(args.output_dir, f"history_{args.mode}_{args.dataset}.json")
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Final history saved to {history_file}")

if __name__ == '__main__':
    main()
