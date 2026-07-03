import unittest
import torch
import torch.nn as nn
import os
import shutil
import json

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import get_data_loaders
from model import (
    BaselineMLP, HebbianMLP, BaselineCNN, HebbianCNN, 
    BaselineVGG16, HebbianVGG16, get_resnet18, 
    BiLSTM_CRF, get_mini_transformer, convert_to_masked_model,
    get_model_sparsity, get_model_active_connections, get_model_active_neurons
)
from structured_pruning import save_sparse_checkpoint, load_sparse_checkpoint, compress_model_structured
from engine import Trainer

class TestDADPPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.test_dir = "./local_test_out"
        os.makedirs(cls.test_dir, exist_ok=True)
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_data_loaders(self):
        """Test that data loaders for all five datasets successfully initialize and yield batches."""
        datasets = ['MNIST', 'CIFAR10', 'SST2', 'IMDB', 'CoNLL2003']
        for ds in datasets:
            train_loader, test_loader = get_data_loaders(ds, batch_size=4, data_dir=self.test_dir)
            self.assertIsNotNone(train_loader)
            self.assertIsNotNone(test_loader)
            
            # Check shape of a batch
            x, y = next(iter(train_loader))
            self.assertEqual(x.size(0), 4)
            
            # For sequence tasks, verify dimensions
            if ds in ['SST2', 'IMDB']:
                self.assertEqual(len(x.shape), 2) # (batch, seq_len)
                self.assertEqual(len(y.shape), 1) # (batch,)
            elif ds == 'CoNLL2003':
                self.assertEqual(len(x.shape), 2) # (batch, seq_len)
                self.assertEqual(len(y.shape), 2) # (batch, seq_len) NER tags
            elif ds == 'MNIST':
                self.assertEqual(x.shape[1], 1) # 1 channel
            elif ds == 'CIFAR10':
                self.assertEqual(x.shape[1], 3) # 3 channels

    def test_model_masking_conversion(self):
        """Test converting a dense standard model to a masked model."""
        base_model = get_mini_transformer(vocab_size=100, num_classes=2, masked=False)
        # Verify no layers are masked initially
        has_masked_init = any(hasattr(m, 'mask') for m in base_model.modules())
        self.assertFalse(has_masked_init)
        
        # Convert model
        masked_model = convert_to_masked_model(base_model)
        has_masked_after = any(hasattr(m, 'mask') for m in masked_model.modules())
        self.assertTrue(has_masked_after)
        
        # Verify metrics helpers
        sparsity = get_model_sparsity(masked_model)
        self.assertEqual(sparsity, 0.0)
        active_neurons = get_model_active_neurons(masked_model)
        self.assertGreater(active_neurons, 0)

    def test_lstm_and_crf(self):
        """Test BiLSTM-CRF model forward, loss, and tag predict outputs."""
        model = BiLSTM_CRF(vocab_size=1000, embedding_dim=32, hidden_dim=32, masked=True)
        model.to(self.device)
        
        inputs = torch.randint(0, 1000, (4, 10)).to(self.device)
        targets = torch.randint(0, 9, (4, 10)).to(self.device)
        
        # Loss calculation
        loss = model(inputs, targets)
        self.assertTrue(torch.is_tensor(loss))
        self.assertGreater(loss.item(), 0.0)
        
        # Inference prediction
        predictions = model.predict(inputs)
        self.assertEqual(predictions.shape, (4, 10))

    def test_sparse_checkpointing(self):
        """Test sparse checkpoint saving and loading functionality."""
        model = HebbianMLP(input_size=10, hidden_size=8, num_classes=2).to(self.device)
        # Apply structured 50% mask
        with torch.no_grad():
            for m in model.modules():
                if hasattr(m, 'mask'):
                    m.mask.copy_((torch.rand_like(m.mask) > 0.5).float())
                    m.weight.mul_(m.mask)
                    
        sparsity_before = get_model_sparsity(model)
        
        state = {
            'epoch': 1,
            'step': 10,
            'model_state_dict': model.state_dict(),
            'mask_dict': {}
        }
        
        chk_path = os.path.join(self.test_dir, 'sparse_chk.pth')
        save_sparse_checkpoint(state, chk_path)
        
        # Verify sparse saving on disk by inspecting loaded state
        loaded_state = load_sparse_checkpoint(chk_path, self.device)
        self.assertEqual(loaded_state['epoch'], 1)
        self.assertEqual(loaded_state['step'], 10)
        
        # Load state into a fresh model and assert matches original
        model_new = HebbianMLP(input_size=10, hidden_size=8, num_classes=2).to(self.device)
        model_new.load_state_dict(loaded_state['model_state_dict'])
        sparsity_after = get_model_sparsity(model_new)
        self.assertAlmostEqual(sparsity_before, sparsity_after, places=5)

    def test_structured_pruning(self):
        """Test structured compression algorithm physically trims weight shapes."""
        model = HebbianMLP(input_size=16, hidden_size=8, num_classes=4).to(self.device)
        
        # Force pruning: set all outgoing connections of neuron 2 and 3 in fc1 to zero
        with torch.no_grad():
            model.fc1.mask[2, :] = 0.0
            model.fc1.mask[3, :] = 0.0
            model.fc1.weight.mul_(model.fc1.mask)
            
        fc1_orig_shape = list(model.fc1.weight.shape)
        fc2_orig_shape = list(model.fc2.weight.shape)
        
        # Run structured pruning
        compressed = compress_model_structured(model)
        
        fc1_new_shape = list(compressed.fc1.weight.shape)
        fc2_new_shape = list(compressed.fc2.weight.shape)
        
        # fc1 out_features should drop from 8 to 6
        self.assertEqual(fc1_new_shape[0], 6)
        # fc2 in_features should drop from 8 to 6
        self.assertEqual(fc2_new_shape[1], 6)

    def test_trainer_execution(self):
        """Verify that Trainer runs single training steps successfully across multiple architectures."""
        configs = [
            ('mlp', 'MNIST'),
            ('cnn', 'MNIST'),
            ('transformer', 'SST2'),
            ('bilstm_crf', 'CoNLL2003')
        ]
        
        for arch, dataset in configs:
            print(f"\nVerifying Trainer step for {arch} on {dataset}...")
            train_loader, test_loader = get_data_loaders(dataset, batch_size=4, data_dir=self.test_dir)
            
            num_classes = 10 if dataset == 'MNIST' else (2 if dataset == 'SST2' else 9)
            input_channels = 1
            input_size = 784
            fc_input_dim = 3136
            
            if arch == 'mlp':
                model = BaselineMLP(input_size=input_size, num_classes=num_classes)
            elif arch == 'cnn':
                model = BaselineCNN(input_channels=input_channels, num_classes=num_classes, fc_input_dim=fc_input_dim)
            elif arch == 'transformer':
                model = get_mini_transformer(vocab_size=5000, num_classes=num_classes, masked=False)
            elif arch == 'bilstm_crf':
                model = BiLSTM_CRF(vocab_size=5000, embedding_dim=16, hidden_dim=16, masked=False)
                
            model = convert_to_masked_model(model)
            
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                device=self.device,
                mode='hebbian',
                lr=0.001,
                prune_interval=1,
                prune_threshold=100.0, # force pruning to run
                output_dir=self.test_dir,
                base_name=f"test_run_{arch}"
            )
            
            # Run for 1 epoch
            history = trainer.run(1)
            self.assertIsNotNone(history)
            self.assertIn('layer_sparsity', history)
            self.assertEqual(len(history['train_loss']), 1)

if __name__ == '__main__':
    unittest.main()
