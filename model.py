import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def forward(self, input):
        # Apply the mask to the weights before the forward pass
        masked_weight = self.weight * self.mask
        return F.linear(input, masked_weight, self.bias)

    def prune(self, importance, threshold):
        """
        Update the mask based on importance and threshold.
        Connections with importance <= threshold are permanently pruned (set to 0).
        """
        with torch.no_grad():
            new_mask = (importance > threshold).float()
            # Ensure once pruned, it stays pruned
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
        """Returns the fraction of pruned connections."""
        total = self.get_total_connections()
        pruned = self.get_pruned_count()
        return pruned / total if total > 0 else 0

    def get_pruned_count(self):
        """Returns the total number of pruned connections."""
        pruned_connections = 0
        for m in self.modules():
            if isinstance(m, MaskedLinear):
                pruned_connections += (m.mask == 0).sum().item()
        return pruned_connections

    def get_total_connections(self):
        """Returns the total number of potential connections."""
        total_connections = 0
        for m in self.modules():
            if isinstance(m, MaskedLinear):
                total_connections += m.mask.numel()
        return total_connections

    def get_active_connections(self):
        """Returns the number of connections that are NOT pruned."""
        return self.get_total_connections() - self.get_pruned_count()

    def get_active_neurons(self):
        """
        Returns the number of 'active' neurons in each layer.
        A neuron is active if it has at least one non-zero connection coming into it.
        """
        active_neurons = 0
        for m in self.modules():
            if isinstance(m, MaskedLinear):
                # m.mask has shape (out_features, in_features)
                # A neuron (row) is active if any weight in its row is non-zero
                active_rows = (m.mask.sum(dim=1) > 0).sum().item()
                active_neurons += active_rows
        return active_neurons
