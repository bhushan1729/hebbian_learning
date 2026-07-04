import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import json
from pruning_utils import snip_prune, magnitude_prune, rigl_step, apply_mask, init_random_mask
from structured_pruning import save_sparse_checkpoint, load_sparse_checkpoint, compress_model_structured
from model import (
    get_model_sparsity, get_model_pruned_count, get_model_total_connections,
    get_model_active_connections, get_model_active_neurons
)

class Trainer:
    def __init__(self, model, train_loader, test_loader, device, mode='hebbian', lr=0.01, 
                 prune_interval=100, prune_threshold=0.01, sparsity=0.9, rigl_prune_fraction=0.2, 
                 rigl_interval=100, output_dir='./results', base_name='experiment', config=None,
                 early_stopping=False):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.mode = mode
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # BiLSTM_CRF manages its own loss
        self.is_ner = hasattr(self.model, 'tag_to_ix')
        if not self.is_ner:
            self.criterion = nn.CrossEntropyLoss()
        
        self.prune_interval = prune_interval
        self.prune_threshold = prune_threshold
        self.sparsity = sparsity
        self.rigl_prune_fraction = rigl_prune_fraction
        self.rigl_interval = rigl_interval
        self.early_stopping = early_stopping
        
        # Configure output directories separating models (checkpoints) and results (histories)
        self.output_dir = output_dir
        self.base_name = base_name
        self.checkpoint_dir = os.path.join(output_dir, 'models')
        self.results_dir = os.path.join(output_dir, 'results')
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.checkpoint_path = os.path.join(self.checkpoint_dir, f"{base_name}.pth")
        self.checkpoint_best_path = os.path.join(self.checkpoint_dir, f"{base_name}_best.pth")
        self.history_path = os.path.join(self.results_dir, f"history_{base_name}.json")
        
        self.mask_dict = {}
        self.step = 0
        self.epoch = 0
        self.best_test_acc = 0.0
        self.epochs_no_improve = 0
        self.history = {
            'train_loss': [], 'train_acc': [],
            'test_loss': [], 'test_acc': [],
            'sparsity': [], 'pruned_count': [],
            'active_connections': [], 'active_neurons': [],
            'layer_sparsity': {}, # detailed structural logging
            'config': {
                'mode': self.mode,
                'prune_threshold': self.prune_threshold,
                'prune_interval': self.prune_interval,
                'sparsity': self.sparsity,
                'lr': self.lr,
                'dataset_samples': len(train_loader.dataset)
            }
        }
        if config is not None:
            self.history['config'].update(config)
        
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
            
            if isinstance(module, nn.Conv2d):
                batch_importance = torch.nn.grad.conv2d_weight(
                    x.abs(), module.weight.shape, dy.abs(),
                    stride=module.stride, padding=module.padding, 
                    dilation=module.dilation, groups=module.groups
                ) / x.size(0)
            else:
                # Linear layers & custom LSTM cell projections
                # Flatten batch and sequence dimensions if input/grad are 3D (e.g. in Transformers)
                x_flat = x.reshape(-1, x.size(-1))
                dy_flat = dy.reshape(-1, dy.size(-1))
                batch_importance = torch.matmul(dy_flat.abs().t(), x_flat.abs()) / x_flat.size(0)
            
            name = module._layer_name
            if name not in self.importance_scores:
                self.importance_scores[name] = torch.zeros_like(batch_importance)
            self.importance_scores[name] += batch_importance

        for name, module in self.model.named_modules():
            if hasattr(module, 'mask'):
                module._layer_name = name
                self.hooks.append(module.register_forward_hook(hook_fn))
                self.hooks.append(module.register_full_backward_hook(backward_hook_fn))

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def save_checkpoint(self, is_best=False):
        state = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'importance_scores': self.importance_scores,
            'mask_dict': self.mask_dict
        }
        path = self.checkpoint_best_path if is_best else self.checkpoint_path
        # Use our new sparsified serializer to save disk space
        save_sparse_checkpoint(state, path)

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            try:
                # Use our new sparse deserializer
                state = load_sparse_checkpoint(self.checkpoint_path, self.device)
                self.epoch = state['epoch']
                self.step = state['step']
                self.model.load_state_dict(state['model_state_dict'])
                self.optimizer.load_state_dict(state['optimizer_state_dict'])
                self.history = state['history']
                self.importance_scores = state.get('importance_scores', {})
                self.mask_dict = state.get('mask_dict', {})
                print(f"Resumed from epoch {self.epoch}, step {self.step}")
                return True
            except Exception as e:
                print(f"Failed to load checkpoint: {e}. Starting fresh.")
        return False

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.is_ner:
                loss = self.model(data, target)
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                
            loss.backward()
            self.optimizer.step()
            
            if self.mode in ['snip', 'magnitude', 'rigl'] and self.mask_dict:
                apply_mask(self.model, self.mask_dict)
            
            total_loss += loss.item()
            
            if self.is_ner:
                predicted = self.model.predict(data)
                correct += predicted.eq(target.cpu()).sum().item()
                total += target.numel()
            else:
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            self.step += 1
            if self.mode == 'hebbian' and self.prune_interval > 0 and self.step % self.prune_interval == 0:
                self.prune()
            elif self.mode == 'rigl' and self.rigl_interval > 0 and self.step % self.rigl_interval == 0:
                self.mask_dict = rigl_step(self.model, self.mask_dict, self.rigl_prune_fraction)
            
            if batch_idx % 100 == 0:
                print(".", end="", flush=True)
            
        avg_loss = total_loss / len(self.train_loader)
        acc = 100. * correct / total
        print(" done.")
        return avg_loss, acc

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                if self.is_ner:
                    loss = self.model(data, target)
                    predicted = self.model.predict(data)
                    correct += predicted.eq(target.cpu()).sum().item()
                    total += target.numel()
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
                    
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.test_loader)
        acc = 100. * correct / total
        return avg_loss, acc

    def prune(self):
        has_masked = any(hasattr(m, 'mask') for m in self.model.modules())
        if not has_masked:
            return

        summary = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'mask') and name in self.importance_scores:
                avg_importance = self.importance_scores[name] / self.prune_interval
                before_pruned = (module.mask == 0).sum().item()
                if hasattr(module, 'prune'):
                    module.prune(avg_importance, self.prune_threshold)
                after_pruned = (module.mask == 0).sum().item()
                summary.append(f"{name}: +{after_pruned - before_pruned}")
                self.importance_scores[name].zero_()
        
        if summary:
            print(f"\n[Pruning] {' | '.join(summary)}")

    def run(self, num_epochs):
        if self.epoch == 0:
            if self.mode == 'snip':
                self.mask_dict = snip_prune(self.model, self.criterion, self.train_loader, self.device, self.sparsity)
                apply_mask(self.model, self.mask_dict)
            elif self.mode == 'rigl':
                self.mask_dict = init_random_mask(self.model, self.sparsity)
                apply_mask(self.model, self.mask_dict)

        for epoch in range(self.epoch, num_epochs):
            if self.mode == 'magnitude' and epoch == max(1, num_epochs - 3) and not self.mask_dict:
                self.mask_dict = magnitude_prune(self.model, self.sparsity)
                apply_mask(self.model, self.mask_dict)
                
            self.epoch = epoch + 1
            start_time = time.time()
            
            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc = self.evaluate()
            
            # Retrieve global metrics using generic helper function
            sparsity = get_model_sparsity(self.model)
            pruned_count = get_model_pruned_count(self.model)
            active_connections = get_model_active_connections(self.model)
            active_neurons = get_model_active_neurons(self.model)
            
            # --- DETAILED LAYER-WISE SPARSITY LOGGING ---
            epoch_key = f"epoch_{self.epoch}"
            self.history['layer_sparsity'][epoch_key] = {}
            for name, module in self.model.named_modules():
                if hasattr(module, 'mask'):
                    m_mask = module.mask
                    m_total = m_mask.numel()
                    m_active = int(m_mask.sum().item())
                    m_pruned = m_total - m_active
                    m_sparsity = m_pruned / m_total if m_total > 0 else 0.0
                    
                    # Estimate active neurons/filters in this layer
                    mask_flat = m_mask.view(m_mask.size(0), -1)
                    m_active_neurons = int((mask_flat.sum(dim=1) > 0).sum().item())
                    m_total_neurons = m_mask.size(0)
                    
                    self.history['layer_sparsity'][epoch_key][name] = {
                        'sparsity': m_sparsity,
                        'active_weights': m_active,
                        'total_weights': m_total,
                        'active_neurons': m_active_neurons,
                        'total_neurons': m_total_neurons
                    }
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['sparsity'].append(sparsity)
            self.history['pruned_count'].append(pruned_count)
            self.history.setdefault('active_connections', []).append(active_connections)
            self.history.setdefault('active_neurons', []).append(active_neurons)
            
            # Print epoch summary table
            if epoch == 0:
                header = f"{'Epoch':^7} | {'Tr Loss':^8} | {'Tr Acc':^7} | {'Te Loss':^8} | {'Te Acc':^7} | {'Sparsity':^8} | {'Active':^10}"
                print("\n" + "="*75)
                print(header)
                print("-" * 75)
            
            row = (f"{epoch+1:^7} | {train_loss:^8.4f} | {train_acc:^7.2f}% | "
                   f"{test_loss:^8.4f} | {test_acc:^7.2f}% | {sparsity:^8.4f} | {active_connections:^10}")
            print(row)
            
            if epoch == num_epochs - 1:
                print("="*75 + "\n")
            
            self.save_checkpoint(is_best=False)
            
            # Check for best test accuracy
            if test_acc > self.best_test_acc:
                self.best_test_acc = test_acc
                self.epochs_no_improve = 0
                self.save_checkpoint(is_best=True)
                print(f"--> New best test accuracy: {self.best_test_acc:.2f}% (Best model saved)")
            else:
                self.epochs_no_improve += 1
                
            # Early stopping (patience = 5)
            if self.early_stopping and self.epochs_no_improve >= 5:
                print(f"\n[Early Stopping] Triggered after 5 epochs without test accuracy improvement.")
                print(f"Best Test Accuracy: {self.best_test_acc:.2f}%")
                if epoch != num_epochs - 1:
                    print("="*75 + "\n")
                break
            
            # Save small history JSON file dynamically
            with open(self.history_path, 'w') as f:
                json.dump(self.history, f, indent=4)
        
        return self.history
