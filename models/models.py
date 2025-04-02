import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


##############################################################################
# TCNModel – Temporal Convolutional Network
##############################################################################
class TemporalBlock(nn.Module):
    """
    One block in the TCN: dilated convolution + residual connection.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)

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
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
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
        self.final_fc = nn.Linear(num_channels[-1], num_classes)
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
        self.fc = nn.Linear(hidden_size * direction_factor, num_classes)

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

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, n_ahead):
        super(LSTMModel, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(num_classes, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.n_ahead = n_ahead
        
        # Apply Xavier and orthogonal initialization to the weights.
        self.init_weights()
        
    def init_weights(self):
        # Initialize encoder weights
        for name, param in self.encoder.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        # Initialize decoder weights
        for name, param in self.decoder.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        # Initialize the fully connected layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, x, target=None):
        batch_size = x.size(0)
        # Encode the input sequence
        _, (hidden, cell) = self.encoder(x)
        
        # Use the last time step of the input as the initial decoder input.
        decoder_input = self.fc(hidden[-1]).unsqueeze(1)  # shape: [batch, 1, num_classes]
        outputs = []
        # Generate predictions for n_ahead time steps
        for t in range(self.n_ahead):
            out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            pred = self.fc(out)  # shape: [batch, 1, num_classes]
            outputs.append(pred)
            decoder_input = pred  # autoregressive input for next time step
        
        outputs = torch.cat(outputs, dim=1)  # shape: [batch, n_ahead, num_classes]
        return outputs


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

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1),
                         batch_size, self.hidden_size, device=x.device)
        out, _ = self.gru(x, h0)  # (batch, seq_len, hidden_size * direction_factor)
        out = self.fc(out)        # (batch, seq_len, num_classes)
        if self.n_ahead is not None:
            out = out[:, -self.n_ahead:, :]
        return out


##############################################################################
# AutoEncoder – Fully-connected autoencoder
##############################################################################
class AutoEncoder(nn.Module):
    """
    A simple fully-connected autoencoder.
    If the input is sequential (3D), it applies the encoder and decoder to each time step.
    If n_ahead is provided and the input is sequential, returns only the last n_ahead steps.
    """
    def __init__(self, input_dim=256, latent_dim=64, n_ahead=None):
        super(AutoEncoder, self).__init__()
        self.n_ahead = n_ahead
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.dim() > 2:
            b, t, d = x.size()
            x = x.view(b * t, d)
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            decoded = decoded.view(b, t, -1)
            if self.n_ahead is not None:
                decoded = decoded[:, -self.n_ahead:, :]
            return decoded
        else:
            return self.decoder(self.encoder(x))


##############################################################################
# RBM and DBN
##############################################################################
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
    If the input is sequential (3D) and n_ahead is provided, outputs only the last n_ahead steps.
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
        # If input is sequential, process each time step independently.
        if x.dim() > 2:
            b, t, d = x.size()
            x = x.view(b * t, d)
            h = x
            for rbm in self.rbms:
                p_h, _ = rbm.sample_h(h)
                h = p_h
            out = self.final_layer(h)
            out = out.view(b, t, -1)
            if self.n_ahead is not None:
                out = out[:, -self.n_ahead:, :]

# 1. LSTMFullSequence – Many-to-many LSTM that applies a fully connected layer to each time step.
class LSTMFullSequence(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, n_ahead=None):
        super(LSTMFullSequence, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.n_ahead = n_ahead

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)   # (batch, seq_len, hidden_size)
        out = self.fc(out)      # (batch, seq_len, num_classes)
        if self.n_ahead is not None:
            out = out[:, -self.n_ahead:, :]
        return out

# 2. LSTMAutoregressive – Encoder with an autoregressive decoder. Optionally uses teacher forcing.
class LSTMAutoregressive(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, n_ahead, teacher_forcing_ratio=0.5):
        super(LSTMAutoregressive, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(num_classes, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.n_ahead = n_ahead
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, x, target=None):
        # x: (batch, seq_len, input_size)
        batch_size = x.size(0)
        _, (hidden, cell) = self.encoder(x)
        # Initialize decoder input with zeros (or use a learned token)
        decoder_input = torch.zeros(batch_size, 1, self.fc.out_features, device=x.device)
        outputs = []
        for t in range(self.n_ahead):
            out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            pred = self.fc(out)  # (batch, 1, num_classes)
            outputs.append(pred)
            # Teacher forcing: if target provided and random chance meets ratio, use ground truth
            if target is not None and torch.rand(1).item() < self.teacher_forcing_ratio:
                decoder_input = target[:, t:t+1, :]
            else:
                decoder_input = pred
        outputs = torch.cat(outputs, dim=1)  # (batch, n_ahead, num_classes)
        return outputs

# 3. LSTMDecoder – Uses a separate decoder with a simple attention mechanism.
class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, n_ahead):
        super(LSTMDecoder, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTMCell(num_classes, hidden_size)
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.n_ahead = n_ahead

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.size()
        encoder_outputs, (hidden, cell) = self.encoder(x)  # encoder_outputs: (batch, seq_len, hidden_size)
        # Use the last encoder output as the initial decoder hidden state
        decoder_hidden = hidden[-1]  # (batch, hidden_size)
        decoder_cell = cell[-1]      # (batch, hidden_size)
        # Start token for decoder input (zeros)
        decoder_input = torch.zeros(batch_size, self.fc.out_features, device=x.device)
        outputs = []
        for t in range(self.n_ahead):
            # Compute attention weights over encoder outputs
            attn_weights = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)  # (batch, seq_len)
            attn_weights = F.softmax(attn_weights, dim=1)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch, hidden_size)
            # Combine context and decoder input
            decoder_input_combined = decoder_input + context  # simple additive combination
            decoder_hidden, decoder_cell = self.decoder(decoder_input_combined, (decoder_hidden, decoder_cell))
            pred = self.fc(decoder_hidden)  # (batch, num_classes)
            outputs.append(pred.unsqueeze(1))
            decoder_input = pred  # autoregressive
        outputs = torch.cat(outputs, dim=1)  # (batch, n_ahead, num_classes)
        return outputs

##############################################################################
# Transformer-based Models
##############################################################################
# 4. TimeSeriesTransformer – Uses nn.Transformer with sinusoidal positional encoding.
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, num_classes, d_model=64, nhead=8,
                 num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=128,
                 dropout=0.1, n_ahead=10):
        super(TimeSeriesTransformer, self).__init__()
        self.n_ahead = n_ahead
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.output_projection = nn.Linear(d_model, num_classes)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers,
                                          num_decoder_layers, dim_feedforward, dropout, batch_first=True)
        
    def forward(self, src):
        # src: (batch, seq_len, input_size)
        batch_size, seq_len, _ = src.size()
        src = self.input_projection(src)  # (batch, seq_len, d_model)
        src = self.positional_encoding(src)
        # Prepare target sequence with zeros
        tgt = torch.zeros(batch_size, self.n_ahead, self.d_model, device=src.device)
        tgt = self.positional_encoding(tgt)
        out = self.transformer(src, tgt)  # (batch, n_ahead, d_model)
        out = self.output_projection(out)  # (batch, n_ahead, num_classes)
        return out

# 5. TemporalTransformer – Similar to TimeSeriesTransformer but uses learned positional embeddings.
class TemporalTransformer(nn.Module):
    def __init__(self, input_size, num_classes, d_model=64, nhead=8,
                 num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=128,
                 dropout=0.1, n_ahead=1, max_len=5000):
        super(TemporalTransformer, self).__init__()
        self.n_ahead = n_ahead
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.output_projection = nn.Linear(d_model, num_classes)
        self.positional_embedding = nn.Embedding(max_len, d_model)
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers,
                                          num_decoder_layers, dim_feedforward, dropout, batch_first=True)
        
    def forward(self, src):
        batch_size, seq_len, _ = src.size()
        src = self.input_projection(src)  # (batch, seq_len, d_model)
        positions = torch.arange(0, seq_len, device=src.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_embed = self.positional_embedding(positions)
        src = src + pos_embed
        src = self.dropout(src)
        # Create target positions for forecast
        tgt_positions = torch.arange(0, self.n_ahead, device=src.device).unsqueeze(0).expand(batch_size, self.n_ahead)
        tgt = self.positional_embedding(tgt_positions)
        tgt = self.dropout(tgt)
        out = self.transformer(src, tgt)  # (batch, n_ahead, d_model)
        out = self.output_projection(out)  # (batch, n_ahead, num_classes)
        return out

# 6. Informer – A version using standard transformer components.
class Informer(nn.Module):
    def __init__(self, input_size, num_classes, d_model=64, nhead=8,
                 num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=128,
                 dropout=0.1, n_ahead=1):
        super(Informer, self).__init__()
        self.n_ahead = n_ahead
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.output_projection = nn.Linear(d_model, num_classes)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
    def forward(self, src):
        batch_size, seq_len, _ = src.size()
        src = self.input_projection(src)  # (batch, seq_len, d_model)
        src = self.positional_encoding(src)
        memory = self.encoder(src)
        # Prepare target sequence for forecast
        tgt = torch.zeros(batch_size, self.n_ahead, self.d_model, device=src.device)
        tgt = self.positional_encoding(tgt)
        out = self.decoder(tgt, memory)
        out = self.output_projection(out)
        return out

##############################################################################
# N-Beats Model
##############################################################################
# 7. NBeats – A fully-connected block-based model for time series forecasting.
class NBeatsBlock(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, output_size, expansion_coefficient_dim=4):
        super(NBeatsBlock, self).__init__()
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(num_layers - 1):
            self.fc_layers.append(nn.Linear(hidden_size, hidden_size))
        # Theta predicts both backcast and forecast components
        self.theta = nn.Linear(hidden_size, output_size + expansion_coefficient_dim)
        self.output_size = output_size
        self.expansion_coefficient_dim = expansion_coefficient_dim

    def forward(self, x):
        # x: (batch, input_size)
        for layer in self.fc_layers:
            x = F.relu(layer(x))
        theta = self.theta(x)
        backcast = theta[:, :self.output_size]
        forecast = theta[:, self.output_size:]
        return backcast, forecast
    
class NBeats(nn.Module):
    def __init__(self, input_size, num_stacks=3, num_blocks_per_stack=3, num_layers=4,
                 hidden_size=128, output_size=1):
        super(NBeats, self).__init__()
        self.stacks = nn.ModuleList()
        for _ in range(num_stacks):
            blocks = nn.ModuleList([
                NBeatsBlock(input_size, num_layers, hidden_size, output_size)
                for _ in range(num_blocks_per_stack)
            ])
            self.stacks.append(blocks)
            
    def forward(self, x):
        # If x has more than 2 dimensions, flatten the non-batch dimensions.
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        residual = x
        forecast = 0
        for blocks in self.stacks:
            for block in blocks:
                backcast, block_forecast = block(residual)
                residual = residual - backcast
                forecast = forecast + block_forecast
        # Adjust forecast shape if needed.
        return forecast.unsqueeze(1)


# Hybrid Transformer + LSTM model with auxiliary branch for temporal differences.
class HybridTransformerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, n_ahead=1,
                 d_model=64, nhead=8, num_encoder_layers=3):
        """
        Args:
          input_size: Number of input channels.
          hidden_size: Hidden size for the LSTM decoder.
          num_layers: Number of layers for the LSTM decoder.
          num_classes: Output channels.
          n_ahead: Forecast horizon.
          d_model: Embedding dimension for transformer.
          nhead: Number of attention heads.
          num_encoder_layers: Number of transformer encoder layers.
        """
        super(HybridTransformerLSTM, self).__init__()
        self.n_ahead = n_ahead

        # Transformer encoder to capture global context.
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=0.1)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # LSTM decoder to produce the forecast.
        self.lstm_decoder = nn.LSTM(d_model, hidden_size, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, num_classes)
        
        # Auxiliary branch: Predict temporal differences (trend)
        self.auxiliary_fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.size()
        # Transformer encoder:
        proj = self.input_projection(x)  # (batch, seq_len, d_model)
        proj = self.positional_encoding(proj)
        enc_out = self.transformer_encoder(proj)  # (batch, seq_len, d_model)
        
        # Use the last encoder output as the context vector for the decoder.
        context = enc_out[:, -1:, :]  # (batch, 1, d_model)
        # Repeat context for each forecast step.
        decoder_input = context.repeat(1, self.n_ahead, 1)  # (batch, n_ahead, d_model)
        lstm_out, _ = self.lstm_decoder(decoder_input)
        main_output = self.fc_out(lstm_out)  # (batch, n_ahead, num_classes)
        
        # Auxiliary branch: Use the last n_ahead encoder outputs to predict the "trend" (or temporal difference)
        aux_input = enc_out[:, -self.n_ahead:, :]  # (batch, n_ahead, d_model)
        aux_output = self.auxiliary_fc(aux_input)    # (batch, n_ahead, num_classes)
        
        return main_output, aux_output
