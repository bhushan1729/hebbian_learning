import torch
import torch.nn as nn
import torch.optim as optim
import os
import time

class Trainer:
    def __init__(self, model, train_loader, test_loader, device, lr=0.01, 
                 prune_interval=100, prune_threshold=0.01, checkpoint_path='checkpoint.pth'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        
        self.prune_interval = prune_interval
        self.prune_threshold = prune_threshold
        self.checkpoint_path = checkpoint_path
        
        self.step = 0
        self.epoch = 0
        self.history = {
            'train_loss': [], 'train_acc': [],
            'test_loss': [], 'test_acc': [],
            'sparsity': [], 'pruned_count': [],
            'active_connections': [], 'active_neurons': []
        }
        
        # Importance tracking
        self.importance_scores = {}
        self.hooks = []
        self._setup_hooks()

    def _setup_hooks(self):
        def hook_fn(module, input, output):
            module._current_input = input[0].detach()

        def backward_hook_fn(module, grad_input, grad_output):
            x = module._current_input
            dy = grad_output[0].detach()
            batch_importance = torch.matmul(dy.abs().t(), x.abs()) / x.size(0)
            
            name = module._layer_name
            if name not in self.importance_scores:
                self.importance_scores[name] = torch.zeros_like(batch_importance)
            self.importance_scores[name] += batch_importance

        for name, module in self.model.named_modules():
            # Use hasattr for robustness
            if hasattr(module, 'mask'):
                module._layer_name = name
                self.hooks.append(module.register_forward_hook(hook_fn))
                self.hooks.append(module.register_full_backward_hook(backward_hook_fn))

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def save_checkpoint(self):
        state = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'importance_scores': self.importance_scores
        }
        torch.save(state, self.checkpoint_path)
        print(f"Checkpoint saved to {self.checkpoint_path}")

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            state = torch.load(self.checkpoint_path, map_location=self.device)
            self.epoch = state['epoch']
            self.step = state['step']
            self.model.load_state_dict(state['model_state_dict'])
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            self.history = state['history']
            self.importance_scores = state.get('importance_scores', {})
            print(f"Resumed from epoch {self.epoch}, step {self.step}")
            return True
        return False

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            self.step += 1
            if self.prune_interval > 0 and self.step % self.prune_interval == 0:
                self.prune()
            
        avg_loss = total_loss / len(self.train_loader)
        acc = 100. * correct / total
        return avg_loss, acc

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(self.test_loader)
        acc = 100. * correct / total
        return avg_loss, acc

    def prune(self):
        # Only prune if there are Masked layers
        has_masked = any(hasattr(m, 'mask') for m in self.model.modules())
        if not has_masked:
            return

        print(f"\nStep {self.step}: Pruning connections...")
        for name, module in self.model.named_modules():
            if hasattr(module, 'mask') and name in self.importance_scores:
                avg_importance = self.importance_scores[name] / self.prune_interval
                before_pruned = (module.mask == 0).sum().item()
                if hasattr(module, 'prune'):
                    module.prune(avg_importance, self.prune_threshold)
                after_pruned = (module.mask == 0).sum().item()
                print(f"Layer {name}: Pruned {after_pruned - before_pruned} connections. Total pruned: {after_pruned}")
                self.importance_scores[name].zero_()

    def run(self, num_epochs):
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch + 1 # Set to next epoch for potential resume
            start_time = time.time()
            
            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc = self.evaluate()
            
            sparsity = 0
            pruned_count = 0
            active_connections = 0
            active_neurons = 0
            
            # Robust extraction of metrics
            if hasattr(self.model, 'get_sparsity'):
                sparsity = self.model.get_sparsity()
                pruned_count = self.model.get_pruned_count()
                active_connections = self.model.get_active_connections()
                active_neurons = self.model.get_active_neurons()
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['sparsity'].append(sparsity)
            self.history['pruned_count'].append(pruned_count)
            self.history.setdefault('active_connections', []).append(active_connections)
            self.history.setdefault('active_neurons', []).append(active_neurons)
            
            duration = time.time() - start_time
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
            print(f"Metrics | Sparsity: {sparsity:.4f} | Pruned: {pruned_count} | "
                  f"Active Conns: {active_connections} | Active Neurons: {active_neurons} | Time: {duration:.2f}s")
            
            self.save_checkpoint()
        
        return self.history
