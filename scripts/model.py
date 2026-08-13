import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, mask, disable_mask_backward=False):
        ctx.save_for_backward(mask)
        ctx.disable_mask_backward = disable_mask_backward
        return weight * mask

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        if ctx.disable_mask_backward:
            return grad_output, None, None
        else:
            return grad_output * mask, None, None

class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))
        self.disable_mask_backward = False

    def forward(self, input):
        disable_mask_backward = getattr(self, 'disable_mask_backward', False)
        masked_weight = MaskedOp.apply(self.weight, self.mask, disable_mask_backward)
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
        self.disable_mask_backward = False

    def forward(self, input):
        disable_mask_backward = getattr(self, 'disable_mask_backward', False)
        masked_weight = MaskedOp.apply(self.weight, self.mask, disable_mask_backward)
        return F.conv2d(input, masked_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def prune(self, importance, threshold):
        if not hasattr(self, 'mask'):
            return
        with torch.no_grad():
            new_mask = (importance > threshold).float()
            self.mask.data *= new_mask
            self.weight.data *= self.mask.data

# --- Dynamic Module Conversion ---

def convert_to_masked_model(model):
    """
    Recursively replaces all nn.Linear and nn.Conv2d modules in a model with
    MaskedLinear and MaskedConv2d to make it compatible with DADP, SNIP, etc.
    """
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            # Keep classification head dense if it maps to classes
            if 'classifier' in name or 'fc' in name or 'output' in name:
                # Still support mask if it's not the final classes (but we can prune it safely too)
                pass
            masked_layer = MaskedLinear(child.in_features, child.out_features, bias=child.bias is not None)
            masked_layer.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                masked_layer.bias.data.copy_(child.bias.data)
            setattr(model, name, masked_layer)
        elif isinstance(child, nn.Conv2d):
            masked_layer = MaskedConv2d(
                child.in_channels, child.out_channels, child.kernel_size,
                stride=child.stride, padding=child.padding, dilation=child.dilation,
                groups=child.groups, bias=child.bias is not None
            )
            masked_layer.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                masked_layer.bias.data.copy_(child.bias.data)
            setattr(model, name, masked_layer)
        else:
            convert_to_masked_model(child)
    return model

# --- Generic Model Metrics Estimators ---

def get_model_sparsity(model):
    total = get_model_total_connections(model)
    pruned = get_model_pruned_count(model)
    return pruned / total if total > 0 else 0.0

def get_model_pruned_count(model):
    pruned = 0
    for m in model.modules():
        if hasattr(m, 'mask'):
            pruned += (m.mask == 0).sum().item()
    return pruned

def get_model_total_connections(model):
    total = 0
    for m in model.modules():
        if hasattr(m, 'mask'):
            total += m.mask.numel()
        elif isinstance(m, (nn.Linear, nn.Conv2d)):
            total += m.weight.numel()
    return total

def get_model_active_connections(model):
    return get_model_total_connections(model) - get_model_pruned_count(model)

def get_model_active_neurons(model):
    active_neurons = 0
    for m in model.modules():
        if hasattr(m, 'mask'):
            mask_flat = m.mask.view(m.mask.size(0), -1)
            active_rows = (mask_flat.sum(dim=1) > 0).sum().item()
            active_neurons += active_rows
        elif isinstance(m, (nn.Linear, nn.Conv2d)):
            active_neurons += m.weight.size(0)
    return active_neurons


# --- Standard Vision baselines ---

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

class BaselineVGG16(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(BaselineVGG16, self).__init__()
        self.features = self._make_layers(input_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(False),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(False),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def _make_layers(self, input_channels):
        layers = []
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        in_channels = input_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class HebbianVGG16(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(HebbianVGG16, self).__init__()
        self.features = self._make_layers(input_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            MaskedLinear(512, 512),
            nn.ReLU(False),
            nn.Dropout(),
            MaskedLinear(512, 512),
            nn.ReLU(False),
            nn.Dropout(),
            MaskedLinear(512, num_classes),
        )

    def _make_layers(self, input_channels):
        layers = []
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        in_channels = input_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = MaskedConv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# --- ResNet18 (Pure PyTorch + torchvision wrapper fallback) ---

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class NativeResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(NativeResNet18, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def get_resnet18(num_classes=10, masked=False):
    try:
        import torchvision.models as models
        model = models.resnet18(num_classes=num_classes)
    except Exception:
        model = NativeResNet18(num_classes=num_classes)
        
    if masked:
        model = convert_to_masked_model(model)
    return model


# --- Masked LSTM Cells & Layers (Sequential) ---

class MaskedLSTMCell(nn.Module):
    """
    An LSTM cell where input-to-hidden and hidden-to-hidden projections
    can be masked by DADP / standard pruning tools.
    """
    def __init__(self, input_size, hidden_size):
        super(MaskedLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc_ih = MaskedLinear(input_size, 4 * hidden_size, bias=True)
        self.fc_hh = MaskedLinear(hidden_size, 4 * hidden_size, bias=True)

    def forward(self, x, hx):
        h, c = hx
        gates = self.fc_ih(x) + self.fc_hh(h)
        i, f, g, o = gates.chunk(4, 1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)
        return new_h, new_c

class MaskedLSTM(nn.Module):
    """
    Multilayer bidirectional LSTM using MaskedLSTMCells.
    """
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False):
        super(MaskedLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.forward_cells = nn.ModuleList()
        self.backward_cells = nn.ModuleList() if bidirectional else None
        
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else (hidden_size * 2 if bidirectional else hidden_size)
            self.forward_cells.append(MaskedLSTMCell(layer_input_size, hidden_size))
            if bidirectional:
                self.backward_cells.append(MaskedLSTMCell(layer_input_size, hidden_size))

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        current_input = x
        for layer in range(self.num_layers):
            h_f = torch.zeros(batch_size, self.hidden_size, device=device)
            c_f = torch.zeros(batch_size, self.hidden_size, device=device)
            outputs_f = []
            for t in range(seq_len):
                h_f, c_f = self.forward_cells[layer](current_input[:, t, :], (h_f, c_f))
                outputs_f.append(h_f.unsqueeze(1))
            layer_output = torch.cat(outputs_f, dim=1)
            
            if self.bidirectional:
                h_b = torch.zeros(batch_size, self.hidden_size, device=device)
                c_b = torch.zeros(batch_size, self.hidden_size, device=device)
                outputs_b = []
                for t in reversed(range(seq_len)):
                    h_b, c_b = self.backward_cells[layer](current_input[:, t, :], (h_b, c_b))
                    outputs_b.append(h_b.unsqueeze(1))
                outputs_b.reverse()
                layer_output_b = torch.cat(outputs_b, dim=1)
                layer_output = torch.cat([layer_output, layer_output_b], dim=-1)
                
            current_input = layer_output
        return current_input

# --- BiLSTM-CRF for NER sequence labeling ---

class CRF(nn.Module):
    """
    Self-contained CRF layer implementing Viterbi and Forward algorithms
    with support for sequence masking and start/end tags.
    """
    def __init__(self, num_tags):
        super(CRF, self).__init__()
        self.num_tags = num_tags
        # Transition parameters: transitions[i, j] = score of transitioning from tag j to tag i
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        
    def forward_alg(self, emissions, mask=None):
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
            
        batch_size, seq_len, num_tags = emissions.shape
        alpha = self.start_transitions + emissions[:, 0]
        
        for t in range(1, seq_len):
            emit_scores = emissions[:, t].unsqueeze(1) # [batch, 1, num_tags]
            trans_scores = self.transitions.unsqueeze(0) # [1, num_tags, num_tags]
            alpha_broadcast = alpha.unsqueeze(2) # [batch, num_tags, 1]
            
            next_alpha = alpha_broadcast + trans_scores + emit_scores
            next_alpha = torch.logsumexp(next_alpha, dim=1)
            
            # Apply mask
            alpha = next_alpha * mask[:, t].unsqueeze(1) + alpha * (~mask[:, t]).unsqueeze(1)
            
        alpha = alpha + self.end_transitions
        return torch.logsumexp(alpha, dim=1)

    def score_sentence(self, emissions, tags, mask=None):
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
            
        batch_size, seq_len = tags.shape
        score = self.start_transitions[tags[:, 0]]
        
        for t in range(seq_len):
            score += emissions[:, t].gather(1, tags[:, t].unsqueeze(1)).squeeze(1) * mask[:, t]
            
        for t in range(1, seq_len):
            prev_tags = tags[:, t-1]
            curr_tags = tags[:, t]
            score += self.transitions[curr_tags, prev_tags] * mask[:, t]
            
        last_tag_indices = mask.sum(1) - 1
        last_tags = tags.gather(1, last_tag_indices.long().unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]
        return score

    def decode(self, emissions, mask=None):
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
            
        batch_size, seq_len, num_tags = emissions.shape
        viterbi = self.start_transitions + emissions[:, 0]
        backpointers = []
        
        for t in range(1, seq_len):
            broadcast_viterbi = viterbi.unsqueeze(2) # [batch, num_tags, 1]
            broadcast_transitions = self.transitions.unsqueeze(0) # [1, num_tags, num_tags]
            
            next_tag_var = broadcast_viterbi + broadcast_transitions
            best_tag_ids = torch.argmax(next_tag_var, dim=1)
            backpointers.append(best_tag_ids)
            
            viterbi_max = next_tag_var.max(dim=1)[0]
            viterbi = viterbi_max + emissions[:, t]
            
            viterbi = viterbi * mask[:, t].unsqueeze(1) + viterbi * (~mask[:, t]).unsqueeze(1)
            
        viterbi = viterbi + self.end_transitions
        best_last_tag = torch.argmax(viterbi, dim=1)
        best_tags = [best_last_tag]
        
        for backpointer in reversed(backpointers):
            best_last_tag = backpointer.gather(1, best_last_tag.unsqueeze(1)).squeeze(1)
            best_tags.append(best_last_tag)
            
        best_tags = torch.stack(list(reversed(best_tags)), dim=1)
        
        # Apply mask to output tags (setting padded tokens to 0)
        best_tags = best_tags * mask
        
        scores = viterbi.max(dim=1)[0]
        return scores, best_tags

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size=5000, tag_to_ix=None, embedding_dim=128, hidden_dim=128, masked=False):
        super(BiLSTM_CRF, self).__init__()
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix if tag_to_ix else {str(i): i for i in range(9)}
        self.num_tags = len(self.tag_to_ix)
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        if masked:
            self.lstm = MaskedLSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
            self.hidden2tag = MaskedLinear(hidden_dim, self.num_tags)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
            self.hidden2tag = nn.Linear(hidden_dim, self.num_tags)
            
        self.crf = CRF(self.num_tags)

    def _get_lstm_features(self, sentence):
        embeds = self.embedding(sentence)
        if isinstance(self.lstm, MaskedLSTM):
            lstm_out = self.lstm(embeds)
        else:
            lstm_out, _ = self.lstm(embeds)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def forward(self, sentence, tags, lengths=None):
        feats = self._get_lstm_features(sentence)
        if lengths is not None:
            batch_size, seq_len = sentence.shape
            mask = torch.arange(seq_len, device=sentence.device).unsqueeze(0) < lengths.unsqueeze(1)
        else:
            mask = None
            
        forward_score = self.crf.forward_alg(feats, mask=mask)
        gold_score = self.crf.score_sentence(feats, tags, mask=mask)
        return (forward_score - gold_score).mean()

    def predict(self, sentence, lengths=None):
        feats = self._get_lstm_features(sentence)
        if lengths is not None:
            batch_size, seq_len = sentence.shape
            mask = torch.arange(seq_len, device=sentence.device).unsqueeze(0) < lengths.unsqueeze(1)
        else:
            mask = None
            
        scores, paths = self.crf.decode(feats, mask=mask)
        # Ensure it returns a tensor
        return paths.clone().detach()


# --- Native PyTorch Mini-Transformer Baseline (BERT-Mini alternative) ---

class MiniTransformer(nn.Module):
    """
    A lightweight Transformer model built on PyTorch modules.
    Can be dynamically converted to DADP masked version.
    """
    def __init__(self, vocab_size=5000, d_model=128, nhead=4, num_layers=2, num_classes=2):
        super(MiniTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Standard PyTorch TransformerEncoderLayer (contains projections and feedforwards)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, 
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len)
        emb = self.embedding(x)
        out = self.transformer(emb)
        # Global average pooling over token dimensions
        out = out.mean(dim=1)
        return self.fc(out)

def get_mini_transformer(vocab_size=5000, num_classes=2, masked=False):
    model = MiniTransformer(vocab_size=vocab_size, num_classes=num_classes)
    if masked:
        model = convert_to_masked_model(model)
    return model
