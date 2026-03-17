import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
# from google.colab import drive # Commented out to avoid interactive prompt issues

# Define a simple Hebbian learning model
class HebbianNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(HebbianNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def hebbian_update(self, input_data, output_data, lr):
        # Simplified Hebbian rule: dw_ij = eta * y_i * x_j
        # For fc1: W = W + lr * (output_fc1.T @ input_data)
        # For fc2: W = W + lr * (output_fc2.T @ output_fc1)
        pass # Placeholder for actual Hebbian update logic

def main():
    parser = argparse.ArgumentParser(description='Hebbian Learning Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--mode', type=str, default='baseline',
                        help='mode of operation: baseline, hebbian')
    parser.add_argument('--colab', action='store_true', default=False,
                        help='running in Google Colab environment')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # if args.colab:
    #     drive.mount('/content/drive')
    #     # Assuming data will be in /content/drive/MyDrive/data
    #     data_path = '/content/drive/MyDrive/data'
    #     if not os.path.exists(data_path):
    #         os.makedirs(data_path)
    # else:
    #     data_path = './data'

    # Using a consistent data_path for Colab environment to avoid drive mounting issues
    data_path = '/content/data'
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = HebbianNet(input_size=784, hidden_size=256, output_size=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        # Training loop
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        # Test loop
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
              f'({100. * correct / len(test_loader.dataset):.0f}%)\n')

if __name__ == '__main__':
    main()
