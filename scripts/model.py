import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def forward(self, input):
        masked_weight = self.weight * self.mask
        return F.linear(input, masked_weight, self.bias)

    def prune(self, importance, threshold):
        if not hasattr(self, 'mask'):
            return
        with torch.no_grad():
            new_mask = (importance > threshold).float()
            self.mask.data *= new_mask
            self.weight.data *= self.mask.data

class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.register_buffer('mask', torch.ones(self.weight.shape))

    def forward(self, input):
        masked_weight = self.weight * self.mask
        return F.conv2d(input, masked_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def prune(self, importance, threshold):
        if not hasattr(self, 'mask'):
            return
        with torch.no_grad():
            new_mask = (importance > threshold).float()
            self.mask.data *= new_mask
            self.weight.data *= self.mask.data

class BaselineMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=512, num_classes=10):
        super(BaselineMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_sparsity(self):
        return 0.0

    def get_pruned_count(self):
        return 0

    def get_total_connections(self):
        total = 0
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                total += m.weight.numel()
        return total

    def get_active_connections(self):
        return self.get_total_connections()

    def get_active_neurons(self):
        active_neurons = 0
        for m in self.modules():
            if hasattr(m, 'out_features'):
                active_neurons += m.out_features
            elif hasattr(m, 'out_channels'):
                active_neurons += m.out_channels
        return active_neurons

class HebbianMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=512, num_classes=10):
        super(HebbianMLP, self).__init__()
        self.fc1 = MaskedLinear(input_size, hidden_size)
        self.fc2 = MaskedLinear(hidden_size, hidden_size)
        self.fc3 = MaskedLinear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_sparsity(self):
        total = self.get_total_connections()
        pruned = self.get_pruned_count()
        return pruned / total if total > 0 else 0

    def get_pruned_count(self):
        pruned_connections = 0
        for m in self.modules():
            if hasattr(m, 'mask'):
                pruned_connections += (m.mask == 0).sum().item()
        return pruned_connections

    def get_total_connections(self):
        total_connections = 0
        for m in self.modules():
            if hasattr(m, 'mask'):
                total_connections += m.mask.numel()
        return total_connections

    def get_active_connections(self):
        return self.get_total_connections() - self.get_pruned_count()

    def get_active_neurons(self):
        active_neurons = 0
        for m in self.modules():
            if hasattr(m, 'mask'):
                # For both Linear and Conv2d, mask.sum(dim=1...) works if we flatten dims > 0
                # But it's easier to just check if the entire output unit's weights are zero
                # For Linear: (out_features, in_features) -> dim=1
                # For Conv2d: (out_channels, in_channels, k, k) -> dims=(1,2,3)
                mask_flat = m.mask.view(m.mask.size(0), -1)
                active_rows = (mask_flat.sum(dim=1) > 0).sum().item()
                active_neurons += active_rows
        return active_neurons

class BaselineCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, fc_input_dim=3136):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(fc_input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_sparsity(self):
        return 0.0

    def get_pruned_count(self):
        return 0

    def get_total_connections(self):
        total = 0
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                total += m.weight.numel()
        return total

    def get_active_connections(self):
        return self.get_total_connections()

    def get_active_neurons(self):
        active_neurons = 0
        for m in self.modules():
            if hasattr(m, 'out_features'):
                active_neurons += m.out_features
            elif hasattr(m, 'out_channels'):
                active_neurons += m.out_channels
        return active_neurons

class HebbianCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, fc_input_dim=3136):
        super(HebbianCNN, self).__init__()
        self.conv1 = MaskedConv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = MaskedConv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = MaskedLinear(fc_input_dim, 128)
        self.fc2 = MaskedLinear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_sparsity(self):
        total = self.get_total_connections()
        pruned = self.get_pruned_count()
        return pruned / total if total > 0 else 0

    def get_pruned_count(self):
        pruned = 0
        for m in self.modules():
            if hasattr(m, 'mask'):
                pruned += (m.mask == 0).sum().item()
        return pruned

    def get_total_connections(self):
        total = 0
        for m in self.modules():
            if hasattr(m, 'mask'):
                total += m.mask.numel()
        return total

    def get_active_connections(self):
        return self.get_total_connections() - self.get_pruned_count()

    def get_active_neurons(self):
        active_neurons = 0
        for m in self.modules():
            if hasattr(m, 'mask'):
                mask_flat = m.mask.view(m.mask.size(0), -1)
                active_rows = (mask_flat.sum(dim=1) > 0).sum().item()
                active_neurons += active_rows
        return active_neurons

