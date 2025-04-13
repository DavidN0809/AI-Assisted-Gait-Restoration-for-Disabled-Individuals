import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
##############################################################################
# Utility Layers
##############################################################################
class Chomp1d(nn.Module):
    """
    Utility layer to ensure 'causal' convolution by chopping off extra elements
    introduced by padding.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        positional_encoding = self.pe[:, :x.size(1), :]
        x = x + positional_encoding
        return self.dropout(x)

##############################################################################
# TCNModel – Temporal Convolutional Network
##############################################################################
# Updated TemporalBlock using GroupNorm instead of BatchNorm.
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, num_groups=8):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=n_outputs)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=n_outputs)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        nn.init.constant_(self.conv2.bias, 0)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity='relu')

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.gn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNModel(nn.Module):
    """
    A standard TCN that can output either the full sequence or only the last n_ahead steps.
    """
    def __init__(self, input_channels=1, num_channels=[32, 32], kernel_size=3,
                 dropout=0.2, num_classes=6, n_ahead=None):
        """
        Args:
          input_channels: Number of input channels.
          num_channels:   List with number of filters per TCN layer.
          kernel_size:    Convolution kernel size.
          dropout:        Dropout rate.
          num_classes:    Final output dimension.
          n_ahead:        If provided, output only the last n_ahead time steps.
        """
        super(TCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size=kernel_size,
                                     stride=1, dilation=dilation_size, padding=padding,
                                     dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.final_fc = nn.Linear(num_channels[-1], num_classes)  # num_classes should be passed as 9 when instantiating
        self.n_ahead = n_ahead
        
    def forward(self, x):
        # x: (batch, seq_len, input_channels)
        x = x.transpose(1, 2)  # -> (batch, input_channels, seq_len)
        out = self.network(x)
        out = out.transpose(1, 2)  # -> (batch, seq_len, channels)
        out = self.final_fc(out)   # -> (batch, seq_len, num_classes)
        if self.n_ahead is not None:
            out = out[:, -self.n_ahead:, :]
        return out


##############################################################################
# Recurrent Models: RNN, LSTM, GRU
##############################################################################
class RNNModel(nn.Module):
    """
    Vanilla multi-layer RNN.
    If n_ahead is provided, outputs predictions for the last n_ahead time steps.
    Otherwise, returns predictions for all time steps.
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                 num_classes=6, bidirectional=False, n_ahead=None):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.n_ahead = n_ahead
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=bidirectional)
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction_factor, num_classes)  # Ensure num_classes equals 9

    def init_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1),
                         batch_size, self.hidden_size, device=x.device)
        out, _ = self.rnn(x, h0)  # out: (batch, seq_len, hidden_size * direction_factor)
        out = self.fc(out)        # (batch, seq_len, num_classes)
        if self.n_ahead is not None:
            out = out[:, -self.n_ahead:, :]
        return out

class GRUModel(nn.Module):
    """
    Multi-layer GRU.
    If n_ahead is provided, outputs predictions for the last n_ahead time steps.
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                 num_classes=6, bidirectional=False, n_ahead=None):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.n_ahead = n_ahead
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=bidirectional)
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction_factor, num_classes)

    def init_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1),
                         batch_size, self.hidden_size, device=x.device)
        out, _ = self.gru(x, h0)  # (batch, seq_len, hidden_size * direction_factor)
        
        # Apply ReLU activation before final projection to prevent flat outputs
        out = torch.relu(out)
        
        out = self.fc(out)        # (batch, seq_len, num_classes)
        if self.n_ahead is not None:
            out = out[:, -self.n_ahead:, :]
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, n_ahead=None):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = num_classes
        self.n_ahead = n_ahead if n_ahead is not None else 3  # Default to 3 if not specified
        self.input_size = input_size
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Create separate fully connected layers for each time step prediction
        self.fc_layers = nn.ModuleList([
            nn.Linear(hidden_size, num_classes) for _ in range(self.n_ahead)
        ])
        
        # Add a second LSTM layer to model temporal dependencies in the output sequence
        self.decoder_lstm = nn.LSTM(num_classes, hidden_size, 1, batch_first=True)
        
        # Final projection layer
        self.final_fc = nn.Linear(hidden_size, num_classes)
        
        self.Drop = nn.Dropout(0.3)
        
        # Debug flag
        self.debug = False
        
    def set_debug(self, debug=True):
        """Enable or disable debug printing of tensor shapes"""
        self.debug = debug
        return self
        
    def forward(self, x):
        batch_size = x.size(0)
        
        if self.debug:
            print(f"Batch size: {batch_size}, Input shape: {x.shape}")
        
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Initialize cell state with zeros
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Process input sequence with encoder LSTM
        out, (h_n, c_n) = self.lstm(x, (h0, c0))
        
        if self.debug:
            print(f"LSTM output shape: {out.shape}")
        
        # Apply dropout
        out = self.Drop(out)
        
        # Get the last time step output
        last_hidden = out[:, -1, :]
        
        if self.debug:
            print(f"Last hidden state shape: {last_hidden.shape}")
        
        # Method 1: Use different FC layers for each time step
        # Initialize the output tensor
        result = torch.zeros(batch_size, self.n_ahead, self.output_size).to(x.device)
        
        # Generate initial prediction for first time step
        first_pred = self.fc_layers[0](last_hidden)
        result[:, 0, :] = first_pred
        
        # Initialize decoder input with first prediction
        decoder_input = first_pred.unsqueeze(1)  # [batch_size, 1, output_size]
        
        # Initialize decoder hidden state with encoder's final state
        decoder_h0 = h_n[-1:, :, :]  # Use only the last layer's hidden state
        decoder_c0 = c_n[-1:, :, :]
        
        # Autoregressive generation of subsequent time steps
        for i in range(1, self.n_ahead):
            # Pass previous prediction through decoder LSTM
            decoder_out, (decoder_h0, decoder_c0) = self.decoder_lstm(
                decoder_input, (decoder_h0, decoder_c0)
            )
            
            # Generate prediction for current time step
            current_pred = self.final_fc(decoder_out.squeeze(1))
            result[:, i, :] = current_pred
            
            # Update decoder input for next iteration
            decoder_input = current_pred.unsqueeze(1)
        
        if self.debug:
            print(f"Final output shape: {result.shape}")
        
        return result

class RBM(nn.Module):
    """Restricted Boltzmann Machine (used in DBN)."""
    def __init__(self, visible_dim, hidden_dim, k=1, learning_rate=1e-3):
        super(RBM, self).__init__()
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.lr = learning_rate
        self.W = nn.Parameter(torch.randn(hidden_dim, visible_dim) * 0.01)
        self.bv = nn.Parameter(torch.zeros(visible_dim))
        self.bh = nn.Parameter(torch.zeros(hidden_dim))

    def sample_h(self, v):
        wx = F.linear(v, self.W, self.bh)
        p_h_given_v = torch.sigmoid(wx)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, h):
        wx = F.linear(h, self.W.t(), self.bv)
        p_v_given_h = torch.sigmoid(wx)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def contrastive_divergence(self, v):
        pos_p_h, pos_h_sample = self.sample_h(v)
        h_ = pos_h_sample
        for _ in range(self.k):
            neg_p_v, neg_v_sample = self.sample_v(h_)
            neg_p_h, neg_h_sample = self.sample_h(neg_v_sample)
            h_ = neg_h_sample
        pos_grad = torch.bmm(pos_h_sample.unsqueeze(2), v.unsqueeze(1))
        neg_grad = torch.bmm(neg_p_h.unsqueeze(2), neg_v_sample.unsqueeze(1))
        dW  = torch.mean(pos_grad - neg_grad, dim=0)
        dbv = torch.mean(v - neg_v_sample, dim=0)
        dbh = torch.mean(pos_h_sample - neg_p_h, dim=0)
        self.W.data += self.lr * dW
        self.bv.data += self.lr * dbv
        self.bh.data += self.lr * dbh
        loss = torch.mean((v - neg_v_sample) ** 2)
        return loss
    
class DBN(nn.Module):
    """
    Deep Belief Network with layerwise RBM pretraining.
    Inputs are expected to be 2D (batch, features). If a 3D tensor is provided,
    it will be flattened. If n_ahead is provided, the final output is expanded 
    along the forecast horizon.
    """
    def __init__(self, sizes, output_dim=6, k=1, rbm_lr=1e-3, n_ahead=None):
        super(DBN, self).__init__()
        self.sizes = sizes
        self.num_rbm_layers = len(sizes) - 1
        self.rbms = nn.ModuleList()
        for i in range(self.num_rbm_layers):
            rbm = RBM(sizes[i], sizes[i+1], k=k, learning_rate=rbm_lr)
            self.rbms.append(rbm)
        self.final_layer = nn.Linear(sizes[-1], output_dim)
        self.n_ahead = n_ahead

    def forward(self, x):
        # If x is 3D, flatten it so that x becomes (batch, lag * channels)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        h = x
        for rbm in self.rbms:
            p_h, _ = rbm.sample_h(h)
            h = p_h
        out = self.final_layer(h)
        if self.n_ahead is not None:
            # Expand output along forecast horizon
            out = out.unsqueeze(1).repeat(1, self.n_ahead, 1)
        return out

    def pretrain(self, data_loader, num_epochs=10, batch_size=128, lr_override=None, verbose=True):
        """
        Performs layerwise pretraining on the DBN using contrastive divergence.
        
        Args:
          data_loader: DataLoader yielding input data. If the data is 3D, it will be flattened.
          num_epochs: Number of epochs to train each RBM layer.
          batch_size: Batch size for each pretraining step.
          lr_override: Optional learning rate to override the RBM's default learning rate.
          verbose: If True, prints progress.
        """
        # Accumulate all data from the data loader.
        all_data = []
        for batch in data_loader:
            # If batch is a tuple, assume the first element is the input.
            X = batch[0] if isinstance(batch, (tuple, list)) else batch
            if X.dim() > 2:
                X = X.view(X.size(0), -1)
            all_data.append(X)
        current_data = torch.cat(all_data, dim=0).to(next(self.parameters()).device)
        
        for i, rbm in enumerate(self.rbms):
            if verbose:
                print(f"Pretraining RBM layer {i+1}/{self.num_rbm_layers}")
            optimizer = torch.optim.SGD(rbm.parameters(), lr=lr_override if lr_override is not None else rbm.lr)
            for epoch in range(num_epochs):
                permutation = torch.randperm(current_data.size(0))
                epoch_loss = 0.0
                for j in range(0, current_data.size(0), batch_size):
                    indices = permutation[j:j+batch_size]
                    batch_data = current_data[indices]
                    loss = rbm.contrastive_divergence(batch_data)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * batch_data.size(0)
                epoch_loss /= current_data.size(0)
                if verbose:
                    print(f"  Epoch {epoch+1}/{num_epochs}: Loss = {epoch_loss:.6f}")
            # After training this RBM, update current_data to its hidden representation.
            with torch.no_grad():
                p_h, _ = rbm.sample_h(current_data)
                current_data = p_h


##############################################################################
# Transformer-based Models
##############################################################################
# 4. TimeSeriesTransformer – Uses nn.Transformer with sinusoidal positional encoding.
class TemporalSelfAttention(nn.Module):
    """Multi-head attention with skip connection and layer normalization."""
    def __init__(self, d_model, nhead, dropout=0.1):
        super(TemporalSelfAttention, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mem=None, attn_mask=None, key_padding_mask=None):
        # Enforce contiguity: try both .contiguous() and specifying a contiguous memory format.
        x = x.contiguous(memory_format=torch.contiguous_format)
        if mem is not None:
            mem = mem.contiguous(memory_format=torch.contiguous_format)
        if attn_mask is not None:
            attn_mask = attn_mask.contiguous()
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.contiguous()

        # Optionally log the shapes (only in debug) to check for any unexpected changes.
        # print("TemporalSelfAttention input x:", x.shape)

        # Handle attention mask shape if needed.
        batch_size, seq_len, _ = x.shape
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                if attn_mask.shape[0] != seq_len or attn_mask.shape[1] != seq_len:
                    print(f"Warning: attn_mask shape {attn_mask.shape} does not match seq_len {seq_len}. Disabling mask.")
                    attn_mask = None
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask[0]

        try:
            # Use the multihead attention call. Because we are in eval mode during validation,
            # no gradient tracking should be needed.
            if mem is not None:
                attn_output, _ = self.self_attn(x, mem, mem, attn_mask=attn_mask,
                                                  key_padding_mask=key_padding_mask)
            else:
                attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask,
                                                  key_padding_mask=key_padding_mask)
        except RuntimeError as e:
            print(f"Error in self-attention: {e}")
            print(f"x shape: {x.shape}, attn_mask: {None if attn_mask is None else attn_mask.shape}")
            # As a last resort, try without any mask.
            attn_output, _ = self.self_attn(x, x, x)

        out = x + self.dropout(attn_output)
        out = self.norm(out)
        return out


class PositionwiseFeedForwardWithSkip(nn.Module):
    """Position-wise feed-forward network with skip connection and layer normalization."""
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super(PositionwiseFeedForwardWithSkip, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # Store the original input for the skip connection
        residual = x
        
        # Apply layer normalization first (pre-norm architecture)
        x = self.norm(x)
        
        # Feed-forward
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        
        # Add skip connection
        out = residual + self.dropout(x)
        
        return out

class EnhancedTransformerEncoderLayer(nn.Module):
    """
    Enhanced transformer encoder layer with skip connections and pre-norm architecture.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(EnhancedTransformerEncoderLayer, self).__init__()
        self.self_attn = TemporalSelfAttention(d_model, nhead, dropout)
        self.feed_forward = PositionwiseFeedForwardWithSkip(d_model, dim_feedforward, dropout)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention block with skip connection
        src = self.self_attn(src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        
        # Feed-forward block with skip connection
        src = self.feed_forward(src)
        
        return src

class EnhancedTransformerDecoderLayer(nn.Module):
    """
    Enhanced transformer decoder layer with skip connections and pre-norm architecture.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(EnhancedTransformerDecoderLayer, self).__init__()
        self.self_attn = TemporalSelfAttention(d_model, nhead, dropout)
        self.cross_attn = TemporalSelfAttention(d_model, nhead, dropout)
        self.feed_forward = PositionwiseFeedForwardWithSkip(d_model, dim_feedforward, dropout)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Self-attention block with skip connection
        tgt = self.self_attn(tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        
        # Cross-attention block with skip connection
        tgt = self.cross_attn(tgt, mem=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        
        # Feed-forward block with skip connection
        tgt = self.feed_forward(tgt)
        
        return tgt

##############################################################################
# N-Beats Model
##############################################################################
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# Generic Block (NBeatsGenericBlock)
# --------------------------
class NBeatsGenericBlock(nn.Module):
    """
    Generic block with fully connected stack and two parallel output branches.
    """
    def __init__(self, input_size, num_layers, hidden_size, output_size, n_ahead=1, 
                 stack_types=('generic',), share_weights_in_stack=False):
        super(NBeatsGenericBlock, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_ahead = n_ahead
        self.hidden_size = hidden_size
        self.stack_types = stack_types
        self.share_weights_in_stack = share_weights_in_stack
        
        # Fully connected layers shared across block types
        self.fc_layers = self._build_fc_stack(input_size, hidden_size, num_layers)
        
        # Backcast and forecast branches
        self.backcast_layer = nn.Linear(hidden_size, input_size)
        self.forecast_layer = nn.Linear(hidden_size, n_ahead * output_size)
        
        # Initialize weights
        self.init_weights()
    
    def _build_fc_stack(self, input_size, hidden_size, num_layers):
        """Build a stack of fully connected layers with ReLU activations."""
        layers = []
        current_size = input_size
        for _ in range(num_layers - 1):
            fc = nn.Linear(current_size, hidden_size)
            layers.extend([fc, nn.ReLU()])
            current_size = hidden_size
        
        # Add final layer
        layers.append(nn.Linear(current_size, hidden_size))
        layers.append(nn.ReLU())
        
        return nn.Sequential(*layers)
        
    def init_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, input_size]
            
        Returns:
            Tuple of (backcast, forecast) where:
            - backcast has shape [batch_size, input_size]
            - forecast has shape [batch_size, n_ahead, output_size]
        """
        # Main FC stack
        hidden = self.fc_layers(x)
        
        # Backcast branch - used for residual connection
        backcast = self.backcast_layer(hidden)
        
        # Forecast branch
        forecast = self.forecast_layer(hidden)
        forecast = forecast.reshape(-1, self.n_ahead, self.output_size)
        
        return backcast, forecast

class NBeatsSeasonalityBlock(nn.Module):
    """
    Block specialized for modeling seasonality components using Fourier basis.
    """
    def __init__(self, input_size, num_layers, hidden_size, output_size, n_ahead=1, 
                 num_harmonics=10):
        super(NBeatsSeasonalityBlock, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_ahead = n_ahead
        self.num_harmonics = num_harmonics
        
        # Compute total number of Fourier coefficients (sines and cosines)
        self.num_coefficients = 2 * num_harmonics
        
        # Main fully connected stack to learn Fourier coefficients
        self.fc_layers = self._build_fc_stack(input_size, hidden_size, num_layers)
        
        # Linear layer to map from hidden to coefficients
        self.theta_layer = nn.Linear(hidden_size, self.num_coefficients)
        
        # Precompute basis for backcast and forecast
        self._init_basis(input_size, n_ahead)
    
    def _build_fc_stack(self, input_size, hidden_size, num_layers):
        layers = []
        current_size = input_size
        for _ in range(num_layers - 1):
            fc = nn.Linear(current_size, hidden_size)
            layers.extend([fc, nn.ReLU()])
            current_size = hidden_size
        
        # Add final layer
        layers.append(nn.Linear(current_size, hidden_size))
        layers.append(nn.ReLU())
        
        return nn.Sequential(*layers)
    
    def _init_basis(self, backcast_length, forecast_length):
        """Initialize Fourier basis functions."""
        # Create time index for backcast (historical data)
        backcast_time = torch.arange(backcast_length).unsqueeze(0)  # [1, backcast_length]
        
        # Create time index for forecast (future data)
        forecast_time = torch.arange(forecast_length).unsqueeze(0)  # [1, forecast_length]
        
        # Create Fourier basis for backcast and forecast
        backcast_basis = []
        forecast_basis = []
        
        for i in range(1, self.num_harmonics + 1):
            # Sine components
            backcast_basis.append(torch.sin(2 * math.pi * i * backcast_time))
            forecast_basis.append(torch.sin(2 * math.pi * i * forecast_time))
            
            # Cosine components
            backcast_basis.append(torch.cos(2 * math.pi * i * backcast_time))
            forecast_basis.append(torch.cos(2 * math.pi * i * forecast_time))
        
        # Stack basis functions
        self.register_buffer('backcast_basis', torch.cat(backcast_basis, dim=0).transpose(0, 1))  # [backcast_length, 2*num_harmonics]
        self.register_buffer('forecast_basis', torch.cat(forecast_basis, dim=0).transpose(0, 1))  # [forecast_length, 2*num_harmonics]
    
    def forward(self, x):
        # Main FC stack
        hidden = self.fc_layers(x)
        
        # Get Fourier coefficients
        theta = self.theta_layer(hidden)  # [batch_size, 2*num_harmonics]
        
        # Generate backcast using Fourier basis
        backcast = torch.matmul(self.backcast_basis, theta.unsqueeze(-1)).squeeze(-1)
        
        # Generate forecast using Fourier basis
        forecast = torch.matmul(self.forecast_basis, theta.unsqueeze(-1)).squeeze(-1)
        
        # Reshape forecast to [batch_size, n_ahead, output_size]
        # For seasonality, we repeat the output for each feature
        forecast = forecast.unsqueeze(-1).expand(-1, -1, self.output_size)
        
        return backcast, forecast

class NBeatsTrendBlock(nn.Module):
    """
    Block specialized for modeling trend components using polynomial basis.
    """
    def __init__(self, input_size, n_ahead, output_size=1, num_layers=4, hidden_size=256, degree=2):
        super(NBeatsTrendBlock, self).__init__()
        self.input_size = input_size
        self.n_ahead = n_ahead
        self.output_size = output_size
        self.degree = degree
        
        # Build fully-connected layers
        self.fc_layers = self._build_fc_stack(input_size, hidden_size, num_layers)
        
        # Coefficient projection layer for polynomial basis
        # We need degree+1 coefficients for a polynomial of degree 'degree'
        self.theta_layer = nn.Linear(hidden_size, degree + 1)
        
        # Precompute polynomial basis for backcast and forecast
        self._init_basis(input_size, n_ahead)
        
    def _build_fc_stack(self, input_size, hidden_size, num_layers):
        layers = []
        current_size = input_size
        for _ in range(num_layers - 1):
            fc = nn.Linear(current_size, hidden_size)
            layers.extend([fc, nn.ReLU()])
            current_size = hidden_size
        
        # Add final layer
        layers.append(nn.Linear(current_size, hidden_size))
        layers.append(nn.ReLU())
        
        return nn.Sequential(*layers)
        
    def _init_basis(self, backcast_length, forecast_length):
        """Initialize polynomial basis functions."""
        # Create normalized time indices
        backcast_time = torch.arange(backcast_length) / backcast_length
        backcast_time = backcast_time.unsqueeze(1)  # [backcast_length, 1]
        
        forecast_time = torch.arange(forecast_length) / forecast_length
        forecast_time = forecast_time.unsqueeze(1)  # [forecast_length, 1]
        
        # Create polynomial basis
        # For each power i from 0 to degree, we compute t^i
        backcast_basis = [backcast_time ** i for i in range(self.degree + 1)]
        forecast_basis = [forecast_time ** i for i in range(self.degree + 1)]
        
        # Stack basis functions as columns
        self.register_buffer('backcast_basis', torch.cat(backcast_basis, dim=1))  # [backcast_length, degree+1]
        self.register_buffer('forecast_basis', torch.cat(forecast_basis, dim=1))  # [forecast_length, degree+1]
    
    def forward(self, x):
        # Main FC stack
        hidden = self.fc_layers(x)
        
        # Get polynomial coefficients
        theta = self.theta_layer(hidden)  # [batch_size, degree+1]
        
        # Generate backcast using polynomial basis
        backcast = torch.matmul(self.backcast_basis, theta.t()).t()
        
        # Generate forecast using polynomial basis
        forecast = torch.matmul(self.forecast_basis, theta.t()).t()
        
        # Reshape forecast to [batch_size, n_ahead, output_size]
        # For trend, we repeat the output for each feature
        forecast = forecast.unsqueeze(-1).expand(-1, -1, self.output_size)
        
        return backcast, forecast

class NBeats(nn.Module):
    """
    N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting.
    
    Implementation based on the paper:
    Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2020, April).
    N-BEATS: Neural basis expansion analysis for interpretable time series forecasting.
    In International Conference on Learning Representations.
    """
    def __init__(self, input_size, num_stacks=2, num_blocks_per_stack=3, num_layers=4,
                 hidden_size=256, output_size=1, n_ahead=1, stack_types=None, 
                 share_weights_in_stack=False, trend_degree=2, seasonality_num_harmonics=8):
        """
        Args:
            input_size: the flattened input dimension (lag * number_of_channels)
            num_stacks: number of stacks (groups) of blocks
            num_blocks_per_stack: number of blocks in each stack
            num_layers: number of FC layers in each block
            hidden_size: size of the FC layers in the blocks
            output_size: output dimension per time step
            n_ahead: forecast horizon
            stack_types: list of stack types, one of 'generic', 'trend', 'seasonality'
                        If None, defaults to all 'generic'
            share_weights_in_stack: whether to share weights across blocks in a stack
            trend_degree: degree of the polynomial for trend blocks
            seasonality_num_harmonics: number of harmonics for seasonality blocks
        """
        super(NBeats, self).__init__()
        self.n_ahead = n_ahead
        self.output_size = output_size
        
        # Default stack types if not provided
        if stack_types is None:
            stack_types = ['generic'] * num_stacks
        
        # Ensure we have the right number of stack types
        assert len(stack_types) == num_stacks, "Must provide stack type for each stack"
        
        # Build stacks of blocks with double residual architecture
        self.stacks = nn.ModuleList()
        for stack_id, stack_type in enumerate(stack_types):
            blocks = nn.ModuleList()
            for block_id in range(num_blocks_per_stack):
                # Reuse block if sharing weights
                if share_weights_in_stack and block_id > 0:
                    block = blocks[0]
                else:
                    if stack_type == 'trend':
                        block = NBeatsTrendBlock(
                            input_size=input_size,
                            n_ahead=n_ahead,
                            output_size=output_size,
                            num_layers=num_layers,
                            hidden_size=hidden_size,
                            degree=trend_degree
                        )
                    elif stack_type == 'seasonality':
                        block = NBeatsSeasonalityBlock(
                            input_size=input_size,
                            num_layers=num_layers,
                            hidden_size=hidden_size,
                            output_size=output_size,
                            n_ahead=n_ahead,
                            num_harmonics=seasonality_num_harmonics
                        )
                    else:  # 'generic'
                        block = NBeatsGenericBlock(
                            input_size=input_size,
                            num_layers=num_layers,
                            hidden_size=hidden_size,
                            output_size=output_size,
                            n_ahead=n_ahead
                        )
                blocks.append(block)
            self.stacks.append(blocks)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        for stack in self.stacks:
            for block in stack:
                if hasattr(block, 'init_weights'):
                    block.init_weights()
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, lag, channels] or [batch_size, input_size]
            
        Returns:
            Forecast tensor of shape [batch_size, n_ahead, output_size]
        """
        # Flatten input if it's sequential
        if x.dim() > 2:
            batch_size, lag, channels = x.shape
            x = x.reshape(batch_size, lag * channels)
        
        # Stack-level connections
        residuals = x
        forecast = 0
        
        # Process through stacks with double residual connections
        for stack_id, stack in enumerate(self.stacks):
            # Stack-level residual
            stack_residual = residuals
            stack_forecast = 0
            
            # Process through blocks in the stack
            for block_id, block in enumerate(stack):
                # Block forward pass
                backcast, block_forecast = block(stack_residual)
                
                # Update residual for next block
                stack_residual = stack_residual - backcast
                
                # Accumulate forecast
                stack_forecast = stack_forecast + block_forecast
            
            # Update global residual and forecast
            residuals = stack_residual
            forecast = forecast + stack_forecast
        
        return forecast

class TemporalTransformer(nn.Module):
    """
    Temporal Transformer model that uses self-attention mechanisms to model time series data.
    """
    def __init__(self, input_size, num_classes, d_model=128, nhead=8, 
                 num_layers=4, dim_feedforward=512, dropout=0.1, n_ahead=10, activation='relu'):
        super(TemporalTransformer, self).__init__()
        
        self.n_ahead = n_ahead
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        # Create a stack of encoder layers
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                TemporalSelfAttention(d_model, nhead, dropout),
                PositionwiseFeedForwardWithSkip(d_model, dim_feedforward, dropout)
            ) for _ in range(num_layers)
        ])
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Initialize output layer bias to zero
        nn.init.zeros_(self.output_layer[1].bias)
    
    def forward(self, src):
        """
        Args:
            src: Input sequence [batch_size, seq_len, input_size]
            
        Returns:
            out: Forecast sequence [batch_size, n_ahead, num_classes]
        """
        batch_size, seq_len, _ = src.size()
        
        # Project input to d_model dimension
        x = self.input_projection(src)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Select the last time step for forecasting
        decoder_input = x[:, -1:, :].repeat(1, self.n_ahead, 1)
        
        # Apply positional encoding on decoder input
        decoder_input = self.positional_encoding(decoder_input)
        
        # Apply output layer
        output = self.output_layer(decoder_input)
        
        return output



##############################################################################
# Informer: Transformer-based Model for Time Series Forecasting (Informer2020)
##############################################################################
# This section adds the Informer model code from the Informer2020 repository.
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        # Convert PyTorch tensor dimensions to Python integers before applying NumPy functions
        L_K_int = L_K.item() if isinstance(L_K, torch.Tensor) else int(L_K)
        L_Q_int = L_Q.item() if isinstance(L_Q, torch.Tensor) else int(L_Q)
        
        # Now use Python integers with NumPy functions
        U_part = self.factor * np.ceil(np.log(L_K_int)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q_int)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K_int else L_K_int
        u = u if u<L_Q_int else L_Q_int
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2,1).contiguous(), attn

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        d_ff = d_ff or 4*d_model
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm3(x+y)

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        # Handle None input gracefully
        if x is None:
            return None
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        # Handle None input values
        if x is None:
            # Return zero tensor with proper shape and device
            # We'll need to determine proper batch size and sequence length
            # from context, so we'll return None and handle it in DataEmbedding
            return None
            
        x = x.long()
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, x, x_mark):
        # Handle case where input x is None
        if x is None:
            # If we know the expected shape from the context, create a zero tensor
            # For now, return None and handle it at the next level
            return None
            
        # Compute the value and position embeddings
        value_embed = self.value_embedding(x)
        position_embed = self.position_embedding(x)
        
        # Only add temporal embedding if x_mark is not None
        if x_mark is not None:
            temporal_embed = self.temporal_embedding(x_mark)
            x = value_embed + position_embed + temporal_embed
        else:
            # If no temporal marks available, just use value and position embeddings
            x = value_embed + position_embed
        
        return self.dropout(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,2)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm2(x+y), attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class EncoderStack(nn.Module):
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        x_stack = []; attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1]//(2**i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s); attns.append(attn)
        x_stack = torch.cat(x_stack, -2)
        
        return x_stack, attns

class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]


class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers))) # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # Check if inputs are None and provide default zero tensors if needed
        if x_enc is None:
            # Create a default zero tensor with appropriate shape
            # Taking standard dimensions as a fallback
            batch_size = x_dec.shape[0] if x_dec is not None else 1
            seq_len = 96  # Typical encoder sequence length
            feature_dim = self.enc_embedding.value_embedding.tokenConv.in_channels
            x_enc = torch.zeros((batch_size, seq_len, feature_dim), device=next(self.parameters()).device)
            
        if x_dec is None:
            # Create a default zero tensor with appropriate shape
            batch_size = x_enc.shape[0]
            seq_len = self.pred_len + 24  # label_len + pred_len is typical
            feature_dim = self.dec_embedding.value_embedding.tokenConv.in_channels
            x_dec = torch.zeros((batch_size, seq_len, feature_dim), device=next(self.parameters()).device)
                
        # Get embeddings
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        # If embedding returns None (due to None inputs), create a default tensor
        if enc_out is None:
            batch_size = x_enc.shape[0]
            seq_len = x_enc.shape[1]
            d_model = self.enc_embedding.d_model
            enc_out = torch.zeros((batch_size, seq_len, d_model), device=next(self.parameters()).device)
            
        # Proceed with encoding
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # Get decoder embedding
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        
        # If embedding returns None (due to None inputs), create a default tensor
        if dec_out is None:
            batch_size = x_dec.shape[0]
            seq_len = x_dec.shape[1]
            d_model = self.dec_embedding.d_model
            dec_out = torch.zeros((batch_size, seq_len, d_model), device=next(self.parameters()).device)
        
        # Proceed with decoding and projection
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]

# --- Time Series Transformer ---
class TimeSeriesTransformer(nn.Module):
    """
    A vanilla Transformer model for time series forecasting.
    
    This model is based on the original Transformer architecture [Vaswani et al., 2017]
    and adapted to forecast future time steps given historical data.
    
    Args:
        input_size: Number of features in the input time series.
        output_size: Number of features in the forecasted output.
        d_model: Model dimensionality for the Transformer.
        nhead: Number of attention heads.
        num_encoder_layers: Number of encoder layers.
        num_decoder_layers: Number of decoder layers.
        dim_feedforward: Dimensionality of the feedforward network.
        dropout: Dropout rate.
        n_ahead: Forecast horizon (number of future time steps to predict).
    """
    def __init__(self, input_size, output_size, d_model=64, nhead=8,
                 num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=256,
                 dropout=0.1, n_ahead=10):
        super(TimeSeriesTransformer, self).__init__()
        
        self.n_ahead = n_ahead
        self.d_model = d_model

        # Project input features to Transformer dimension
        self.input_projection = nn.Linear(input_size, d_model)
        # Positional encoding is used for both encoder and decoder inputs.
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Use PyTorch's built-in Transformer with batch_first=True for clarity.
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout, batch_first=True)

        # Learnable decoder input tokens (one per forecast step)
        # This parameter is of shape (n_ahead, d_model) and will be repeated for each batch.
        self.decoder_input = nn.Parameter(torch.zeros(n_ahead, d_model))
        # Optionally, initialize the decoder tokens (e.g., with small random values)
        nn.init.uniform_(self.decoder_input, -0.1, 0.1)

        # Final projection layer to map from model dimension to desired output size.
        self.output_layer = nn.Linear(d_model, output_size)

    def forward(self, src):
        """
        Args:
            src: Input tensor of shape [batch_size, seq_len, input_size]
        
        Returns:
            out: Forecast tensor of shape [batch_size, n_ahead, output_size]
        """
        batch_size, seq_len, _ = src.size()

        # Project input and add positional encoding
        src_proj = self.input_projection(src)  # shape: [batch_size, seq_len, d_model]
        src_proj = self.positional_encoding(src_proj)

        # Pass through the Transformer encoder
        # Note: Using the built-in encoder from nn.Transformer.
        encoder_output = self.transformer.encoder(src_proj)

        # Prepare decoder inputs by repeating the learnable tokens over the batch dimension.
        # Shape: [batch_size, n_ahead, d_model]
        dec_input = self.decoder_input.unsqueeze(0).expand(batch_size, -1, -1)
        dec_input = self.positional_encoding(dec_input)

        # Transformer decoder: uses encoder outputs to attend to for forecasting.
        decoder_output = self.transformer.decoder(dec_input, encoder_output)

        # Project the Transformer output to get the final forecast.
        out = self.output_layer(decoder_output)
        return out


##############################################################################
# New Models: PatchTST, CrossFormer, and DLinear
##############################################################################

###########################################
# 1. PatchTST Model
###########################################
class PatchTST(nn.Module):
    """
    PatchTST: A Transformer model for Time Series Forecasting using Patch-Level Embedding.
    
    This model divides the input time series into patches (via a 1D convolution) and then applies
    a Transformer encoder on the patch embeddings to produce a forecast.
    
    Args:
        input_channels: Number of input features (channels).
        patch_size: Length of each patch (number of time steps per patch).
        d_model: Dimension of the patch embeddings and the Transformer model.
        nhead: Number of attention heads in the Transformer.
        num_layers: Number of Transformer encoder layers.
        dim_feedforward: Hidden dimension of the feed-forward network in each encoder layer.
        dropout: Dropout rate.
        forecast_horizon: Number of future time steps to forecast.
        output_size: Number of output features.
    """
    def __init__(self, input_channels, patch_size=16, d_model=512, nhead=8, 
                 num_layers=3, dim_feedforward=2048, dropout=0.1, 
                 forecast_horizon=10, output_size=1):
        super(PatchTST, self).__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon
        
        # Patch embedding: using a Conv1d (stride==kernel_size means non-overlapping patches)
        self.patch_embedding = nn.Conv1d(in_channels=input_channels, 
                                         out_channels=d_model, 
                                         kernel_size=patch_size, 
                                         stride=patch_size)
        # Positional encoding (reuse your sinusoidal implementation)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout, max_len=1000)
        
        # Transformer encoder: batch_first=True so input shape is (B, L, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Forecast head: maps the last patch representation to forecast_horizon * output_size.
        self.fc = nn.Linear(d_model, forecast_horizon * output_size)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_channels]
        Returns:
            Forecast tensor of shape [batch_size, forecast_horizon, output_size]
        """
        # Transpose to (B, input_channels, seq_len) for Conv1d
        x = x.transpose(1, 2)
        # Apply patch embedding to get shape (B, d_model, n_patches)
        x = self.patch_embedding(x)
        # Transpose to (B, n_patches, d_model)
        x = x.transpose(1, 2)
        # Add positional encoding
        x = self.positional_encoding(x)
        # Pass through Transformer encoder
        x = self.transformer_encoder(x)
        # Use the representation of the last patch (or consider alternative aggregation)
        x_last = x[:, -1, :]  # shape: (B, d_model)
        # Forecast head: project to (forecast_horizon * output_size) then reshape
        out = self.fc(x_last)  # shape: (B, forecast_horizon * output_size)
        out = out.view(x.size(0), self.forecast_horizon, -1)
        return out

###########################################
# 2. CrossFormer Model
###########################################
##############################################################################
# Revised CrossFormer with Self-Supervised Pretraining,
# Attention Pooling, and Multi-Head Forecasting
##############################################################################
class CrossFormer(nn.Module):
    """
    Revised CrossFormer: A dual-branch Transformer for time series forecasting enhanced with:
      - Self-supervised pretraining via an extra projection head.
      - Attention pooling (in addition to mean and max pooling) to weight aggregated tokens.
      - Multiple forecast heads (one per output channel) for finer modeling.
    
    Args:
        input_channels (int): Number of input features.
        seq_len (int): Length of the input time series.
        d_model (int, optional): Dimension of the model embeddings. Default is 64.
        nhead (int, optional): Number of attention heads in Transformer layers. Default is 8.
        num_layers (int, optional): Number of Transformer encoder layers per branch. Default is 3.
        dim_feedforward (int, optional): Hidden dimension of the encoder layers. Default is 256.
        dropout (float, optional): Dropout rate. Default is 0.1.
        forecast_horizon (int, optional): Number of future time steps to forecast. Default is 10.
        output_size (int, optional): Number of output features/channels. Default is 1.
        pooling_method (str, optional): Pooling type to use ("mean", "max", or "attention"). Default is "mean".
        self_supervised (bool, optional): Whether to enable self-supervised pretraining head. Default is False.
        pretrain_dim (int, optional): Projection dimension for the self-supervised head. Default is d_model.
    """
    def __init__(self, input_channels, seq_len, d_model=64, nhead=8, num_layers=3, 
                 dim_feedforward=256, dropout=0.1, forecast_horizon=10, output_size=1, 
                 pooling_method="mean", self_supervised=False, pretrain_dim=None):
        super(CrossFormer, self).__init__()
        self.seq_len = seq_len
        self.input_channels = input_channels
        self.forecast_horizon = forecast_horizon
        self.output_size = output_size
        self.d_model = d_model
        self.pooling_method = pooling_method.lower()
        self.self_supervised = self_supervised
        if pretrain_dim is None:
            pretrain_dim = d_model

        # ----------------------------
        # Temporal Branch
        # ----------------------------
        self.temporal_projection = nn.Linear(input_channels, d_model)
        self.temporal_pos_encoding = PositionalEncoding(d_model, dropout=dropout, max_len=seq_len)
        temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, activation="relu", batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_encoder_layer, num_layers=num_layers)
        
        # ----------------------------
        # Channel Branch
        # ----------------------------
        # Transpose input so that each channel's entire time series is a token.
        self.channel_projection = nn.Linear(seq_len, d_model)
        self.channel_pos_encoding = PositionalEncoding(d_model, dropout=dropout, max_len=input_channels)
        channel_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, activation="relu", batch_first=True
        )
        self.channel_encoder = nn.TransformerEncoder(channel_encoder_layer, num_layers=num_layers)
        
        # ----------------------------
        # Attention Pooling Parameters
        # ----------------------------
        # If attention pooling is selected, use learnable vectors.
        if self.pooling_method == "attention":
            self.temporal_attn_vector = nn.Parameter(torch.randn(d_model, 1))
            self.channel_attn_vector = nn.Parameter(torch.randn(d_model, 1))
        
        # ----------------------------
        # Fusion and Forecast Head
        # ----------------------------
        self.fusion_fc = nn.Linear(d_model * 2, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        self.fusion_dropout = nn.Dropout(dropout)
        # Create one forecast head (linear layer) per output channel.
        self.fc = nn.ModuleList([nn.Linear(d_model, forecast_horizon) for _ in range(output_size)])
        
        # ----------------------------
        # Self-supervised Pretraining Head
        # ----------------------------
        if self_supervised:
            self.pretrain_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, pretrain_dim)
            )
        
        # ----------------------------
        # Weight Initialization
        # ----------------------------
        self._init_weights()
    
    def _init_weights(self):
        """Initialize all linear layers with Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _pooling(self, x, branch):
        """
        Pool over the sequence dimension using the specified pooling method.
        
        Args:
            x (Tensor): Input tensor of shape [B, L, d_model] (if branch=="temporal")
                        or [B, C, d_model] (if branch=="channel").
            branch (str): "temporal" or "channel", used to select the appropriate attention vector.
        
        Returns:
            Tensor: Aggregated tensor of shape [B, d_model].
        """
        if self.pooling_method == "mean":
            return x.mean(dim=1)
        elif self.pooling_method == "max":
            return x.max(dim=1)[0]
        elif self.pooling_method == "attention":
            if branch == "temporal":
                # x: [B, L, d_model], attn: [d_model, 1]
                scores = torch.matmul(x, self.temporal_attn_vector)  # [B, L, 1]
            elif branch == "channel":
                scores = torch.matmul(x, self.channel_attn_vector)  # [B, C, 1]
            else:
                raise ValueError("Invalid branch type. Use 'temporal' or 'channel'.")
            weights = F.softmax(scores, dim=1)  # normalize across time or channels
            pooled = torch.sum(x * weights, dim=1)
            return pooled
        else:
            raise ValueError("Unsupported pooling method. Choose 'mean', 'max', or 'attention'.")
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (Tensor): Input tensor with shape [B, seq_len, input_channels].
            
        Returns:
            If self_supervised is False:
                Tensor: Forecast tensor with shape [B, forecast_horizon, output_size].
            Else:
                Tuple: (forecast tensor, self-supervised representation)
                where self-supervised representation has shape [B, pretrain_dim].
        """
        B, L, C = x.shape
        # -------- Temporal Branch --------
        # Project and encode over time.
        temporal_input = self.temporal_projection(x)               # (B, L, d_model)
        temporal_input = self.temporal_pos_encoding(temporal_input)   # (B, L, d_model)
        temporal_out = self.temporal_encoder(temporal_input)          # (B, L, d_model)
        temporal_repr = self._pooling(temporal_out, branch="temporal")  # (B, d_model)
        
        # -------- Channel Branch --------
        # Transpose so that each channel becomes a token.
        channel_input = x.transpose(1, 2)                            # (B, input_channels, seq_len)
        channel_input = self.channel_projection(channel_input)       # (B, input_channels, d_model)
        channel_input = self.channel_pos_encoding(channel_input)       # (B, input_channels, d_model)
        channel_out = self.channel_encoder(channel_input)            # (B, input_channels, d_model)
        channel_repr = self._pooling(channel_out, branch="channel")    # (B, d_model)
        
        # -------- Fusion --------
        fusion = torch.cat([temporal_repr, channel_repr], dim=-1)     # (B, 2*d_model)
        fusion = self.fusion_fc(fusion)                              # (B, d_model)
        fusion = F.relu(fusion)
        fusion = self.fusion_norm(fusion)
        fusion = self.fusion_dropout(fusion)
        
        # -------- Forecast Heads --------
        forecasts = []
        for head in self.fc:
            out = head(fusion)      # (B, forecast_horizon)
            forecasts.append(out.unsqueeze(-1))  # (B, forecast_horizon, 1)
        forecast_output = torch.cat(forecasts, dim=-1)  # (B, forecast_horizon, output_size)
        
        if self.self_supervised:
            # Compute self-supervised representation from fused feature.
            pretrain_rep = self.pretrain_head(fusion)  # (B, pretrain_dim)
            return forecast_output, pretrain_rep
        else:
            return forecast_output

###########################################
# 3. DLinear Model
###########################################
class DLinear(nn.Module):
    """
    DLinear: A decoupled linear model for time series forecasting.
    
    This model implements a simple yet effective forecasting architecture by decomposing the input
    time series into trend and seasonal components via a moving average filter and then applying 
    linear mappings to each component. Two variants are supported:
      - Shared: The same linear layers are applied across channels.
      - Individual: Different linear layers are applied per channel.
      
    Args:
        seq_len: Length of the input historical window.
        forecast_horizon: Number of future time steps to forecast.
        num_channels: Number of input (and output) channels.
        individual: If True, uses separate (individual) linear layers per channel.
                    Otherwise, parameters are shared across channels.
        moving_avg_kernel: Kernel size for the moving average filter used for decomposition.
    """
    def __init__(self, seq_len, forecast_horizon, num_channels=1, individual=False, moving_avg_kernel=25):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        self.num_channels = num_channels
        self.individual = individual
        self.moving_avg_kernel = moving_avg_kernel
        self.padding = (moving_avg_kernel - 1) // 2
        
        # Average pooling as a moving average filter along the temporal dimension.
        self.avg_pool = nn.AvgPool1d(kernel_size=moving_avg_kernel, stride=1, padding=self.padding)
        
        if individual:
            # Separate linear layers for each channel.
            self.linear_trend = nn.ModuleList([nn.Linear(seq_len, forecast_horizon) for _ in range(num_channels)])
            self.linear_seasonal = nn.ModuleList([nn.Linear(seq_len, forecast_horizon) for _ in range(num_channels)])
        else:
            # Shared linear layers across all channels.
            self.linear_trend = nn.Linear(seq_len, forecast_horizon)
            self.linear_seasonal = nn.Linear(seq_len, forecast_horizon)
    
    def moving_average(self, x):
        """
        Apply a moving average filter along the time dimension.
        
        Args:
            x: Tensor of shape [B, seq_len, C]
        Returns:
            Tensor of the same shape containing the moving average values.
        """
        # Permute to (B, C, seq_len) for nn.AvgPool1d and then back.
        x = x.transpose(1, 2)
        ma = self.avg_pool(x)
        ma = ma.transpose(1, 2)
        return ma
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, num_channels]
        Returns:
            Forecast tensor of shape [batch_size, forecast_horizon, num_channels]
        """
        # Decompose the input into trend and seasonal components.
        moving_mean = self.moving_average(x)
        seasonal = x - moving_mean
        
        if self.individual:
            # Apply separate linear layers per channel.
            B, L, C = x.size()
            trend_out_list = []
            seasonal_out_list = []
            for c in range(C):
                trend_out_list.append(self.linear_trend[c](moving_mean[:, :, c]))  # (B, forecast_horizon)
                seasonal_out_list.append(self.linear_seasonal[c](seasonal[:, :, c]))
            # Stack outputs along the channel dimension.
            trend_out = torch.stack(trend_out_list, dim=-1)     # (B, forecast_horizon, C)
            seasonal_out = torch.stack(seasonal_out_list, dim=-1) # (B, forecast_horizon, C)
        else:
            # Shared linear layers: reshape to merge batch and channel dimensions.
            B, L, C = x.size()
            moving_mean_shared = moving_mean.permute(0, 2, 1).contiguous().view(B * C, L)
            seasonal_shared = seasonal.permute(0, 2, 1).contiguous().view(B * C, L)
            trend_shared = self.linear_trend(moving_mean_shared)    # (B*C, forecast_horizon)
            seasonal_shared = self.linear_seasonal(seasonal_shared)   # (B*C, forecast_horizon)
            # Reshape back to (B, C, forecast_horizon) and then permute.
            trend_out = trend_shared.view(B, C, self.forecast_horizon).permute(0, 2, 1)
            seasonal_out = seasonal_shared.view(B, C, self.forecast_horizon).permute(0, 2, 1)
        
        # Combine trend and seasonal components.
        out = trend_out + seasonal_out  # (B, forecast_horizon, C)
        return out

# PositionalEncoding as used in transformer architectures.
class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        pe[:, 0::2] = torch.sin(position * div_term)  # apply sine on even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # apply cosine on odd indices
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# Hybrid model combining LSTM and Transformer
class HybridLSTMTransformer(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, lstm_layers, 
                 d_model, nhead, transformer_layers,
                 forecast_horizon, output_size, dropout=0.1):
        """
        Args:
            input_size (int): Number of features in the input time series.
            lstm_hidden_size (int): Hidden state dimension for LSTM.
            lstm_layers (int): Number of LSTM layers.
            d_model (int): Dimension used inside the Transformer.
            nhead (int): Number of attention heads in Transformer layers.
            transformer_layers (int): Number of Transformer encoder layers.
            forecast_horizon (int): Number of future timesteps to predict.
            output_size (int): Number of outputs per timestep.
            dropout (float): Dropout probability.
        """
        super(HybridLSTMTransformer, self).__init__()
        # LSTM component to capture local temporal dependencies.
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, num_layers=lstm_layers, 
                            batch_first=True, dropout=dropout)
        
        # Project LSTM output (hidden state from each timestep) to transformer dimension.
        self.input_projection = nn.Linear(lstm_hidden_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer component: process the projected sequence to capture global context.
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                    dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # For fusion, we use both the last LSTM hidden state and a global (e.g. mean-pooled) transformer feature.
        self.fusion_fc = nn.Linear(lstm_hidden_size + d_model, lstm_hidden_size)
        self.fusion_activation = nn.ReLU()
        
        # Forecast head: Predict forecast_horizon * output_size from the fused features.
        self.fc_out = nn.Linear(lstm_hidden_size, forecast_horizon * output_size)
        self.forecast_horizon = forecast_horizon
        self.output_size = output_size

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        # LSTM passes the entire sequence.
        lstm_out, (hn, cn) = self.lstm(x)  # lstm_out: [batch, seq_len, lstm_hidden_size]
        # Use the final time step hidden state of LSTM (hn[-1]) as local feature.
        lstm_last = hn[-1]  # [batch, lstm_hidden_size]
        
        # For the Transformer, we project the LSTM outputs for all timesteps.
        proj = self.input_projection(lstm_out)  # [batch, seq_len, d_model]
        proj = self.positional_encoding(proj)
        transformer_out = self.transformer(proj)  # [batch, seq_len, d_model]
        # Aggregate transformer outputs via mean pooling.
        transformer_feat = transformer_out.mean(dim=1)  # [batch, d_model]
        
        # Concatenate LSTM local features and transformer global feature.
        fusion = torch.cat([lstm_last, transformer_feat], dim=-1)  # [batch, lstm_hidden_size + d_model]
        fusion = self.fusion_activation(self.fusion_fc(fusion))  # [batch, lstm_hidden_size]
        
        # Generate forecast with the forecast head.
        out = self.fc_out(fusion)  # [batch, forecast_horizon * output_size]
        out = out.view(-1, self.forecast_horizon, self.output_size)  # [batch, forecast_horizon, output_size]
        return out