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
# LSTMFULL
##############################################################################

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
    
##############################################################################
# LSTMAuto
##############################################################################
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, n_ahead):
        """
        A sequence-to-sequence LSTM autoencoder for time series forecasting.
        This model is non-autoregressive: the decoder decodes all forecast steps in parallel.
        
        Args:
          input_size: Dimensionality of the input features.
          hidden_size: Size of the LSTM hidden state.
          num_layers: Number of LSTM layers for both encoder and decoder.
          num_classes: Output dimension per forecast time step.
          n_ahead: Number of forecast steps.
        """
        super(LSTMAutoencoder, self).__init__()
        self.n_ahead = n_ahead
        
        # Encoder: process the input sequence and produce a latent representation.
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Decoder: decode the latent representation in one shot without autoregression.
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer to map the decoder output to the forecast.
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        """
        Args:
          x: Input sequence of shape (batch, seq_len, input_size)
          
        Returns:
          Forecast of shape (batch, n_ahead, num_classes)
        """
        batch_size, seq_len, _ = x.size()
        # Encode the input sequence.
        _, (hidden, cell) = self.encoder(x)
        
        # Create a fixed decoder input: here we use zeros repeated n_ahead times.
        decoder_input = torch.zeros(batch_size, self.n_ahead, hidden.size(-1), device=x.device)
        
        # Decode the forecast in parallel.
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))
        
        # Map the decoder output to the final forecast.
        out = self.fc(decoder_output)
        return out

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
##############################################################################
# Temporal Transformer Model
##############################################################################
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
##############################################################################
# Informer Model
##############################################################################
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
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# Generic Block (NBeatsGenericBlock)
# --------------------------
class NBeatsGenericBlock(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, output_size, n_ahead=1):
        """
        A generic N‑BEATS block consisting of a stack of fully-connected layers
        with ReLU activations, followed by a linear layer outputting a theta vector.
        The theta vector is split into a backcast (explaining the input) and a forecast.
        """
        super(NBeatsGenericBlock, self).__init__()
        self.backcast_size = input_size  # original (flattened) input dimension
        self.n_ahead = n_ahead
        self.output_size = output_size
        
        layers = []
        # First layer maps input_size to hidden_size, then hidden_size layers
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        self.fc_layers = nn.Sequential(*layers)
        # Final linear layer outputs theta, of size = backcast_size + output_size * n_ahead
        self.theta = nn.Linear(hidden_size, self.backcast_size + output_size * n_ahead)

    def forward(self, x):
        # x shape: (batch, input_size)
        x = self.fc_layers(x)
        theta = self.theta(x)
        # First part is the backcast, second part is the forecast
        backcast = theta[:, :self.backcast_size]
        forecast = theta[:, self.backcast_size:]
        forecast = forecast.view(x.size(0), self.n_ahead, self.output_size)
        return backcast, forecast

# --------------------------
# Trend Block (Interpretable Block)
# --------------------------
class NBeatsTrendBlock(nn.Module):
    def __init__(self, input_size, n_ahead, degree):
        """
        A trend block that uses a polynomial basis expansion.
        The block learns the coefficients (theta) for a polynomial of a given degree.
        The backcast is produced by projecting theta onto a basis defined on the input time indices,
        and the forecast similarly uses a basis defined on the forecast horizon.
        """
        super(NBeatsTrendBlock, self).__init__()
        self.input_size = input_size
        self.n_ahead = n_ahead
        self.degree = degree
        self.theta_layer = nn.Linear(input_size, degree + 1)
        
        # Precompute the basis for backcast and forecast (these are registered as buffers so they are moved with the model)
        backcast_time = torch.linspace(0, 1, input_size).unsqueeze(1)  # shape: (input_size, 1)
        forecast_time = torch.linspace(0, 1, n_ahead).unsqueeze(1)       # shape: (n_ahead, 1)
        self.register_buffer('backcast_basis', torch.cat([backcast_time ** i for i in range(degree + 1)], dim=1))  # (input_size, degree+1)
        self.register_buffer('forecast_basis', torch.cat([forecast_time ** i for i in range(degree + 1)], dim=1))  # (n_ahead, degree+1)

    def forward(self, x):
        # x: (batch, input_size)
        theta = self.theta_layer(x)  # (batch, degree+1)
        # The backcast is given by projecting the polynomial coefficients on the backcast basis
        backcast = torch.matmul(theta, self.backcast_basis.t())  # (batch, input_size)
        forecast = torch.matmul(theta, self.forecast_basis.t())  # (batch, n_ahead)
        # Reshape forecast to have output_size dimension = 1 (or you can modify if multivariate)
        forecast = forecast.unsqueeze(2)  # (batch, n_ahead, 1)
        return backcast, forecast

# --------------------------
# NBeats Model Combining Blocks
# --------------------------
class NBeats(nn.Module):
    def __init__(self, input_size, num_stacks=3, num_blocks_per_stack=3, num_layers=4,
                 hidden_size=128, output_size=1, n_ahead=1, block_type='generic', trend_degree=2):
        """
        N‑BEATS model.
        - input_size: the flattened input dimension (lag * number_of_channels)
        - num_stacks: number of stacks (groups) of blocks
        - num_blocks_per_stack: number of blocks in each stack
        - block_type: either 'generic' or 'trend' (you can add seasonality similarly)
        - trend_degree: degree of the polynomial for trend blocks
        """
        super(NBeats, self).__init__()
        self.n_ahead = n_ahead
        self.stacks = nn.ModuleList()
        for _ in range(num_stacks):
            blocks = nn.ModuleList()
            for _ in range(num_blocks_per_stack):
                if block_type == 'trend':
                    block = NBeatsTrendBlock(input_size, n_ahead, degree=trend_degree)
                else:  # default to generic block
                    block = NBeatsGenericBlock(input_size, num_layers, hidden_size, output_size, n_ahead=n_ahead)
                blocks.append(block)
            self.stacks.append(blocks)

    def forward(self, x):
        # x: if sequential, shape (batch, lag, channels) and we flatten it to (batch, lag*channels)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        residual = x
        forecast = 0
        for blocks in self.stacks:
            for block in blocks:
                backcast, block_forecast = block(residual)
                residual = residual - backcast  # update residual
                forecast = forecast + block_forecast  # aggregate forecast from blocks
        return forecast


class HybridTransformerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, n_ahead=1,
                 d_model=64, nhead=8, num_encoder_layers=3):
        """
        Hybrid Transformer + LSTM model for forecasting.
        
        Args:
          input_size: Number of input channels.
          hidden_size: Hidden size for the LSTM decoder.
          num_layers: Number of layers for the LSTM decoder.
          num_classes: Output channels.
          n_ahead: Forecast horizon.
          d_model: Embedding dimension for the transformer.
          nhead: Number of attention heads.
          num_encoder_layers: Number of transformer encoder layers.
        """
        super(HybridTransformerLSTM, self).__init__()
        self.n_ahead = n_ahead

        # Transformer encoder to capture global context.
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=0.1)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, 
                                                    dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # LSTM decoder to produce the forecast.
        self.lstm_decoder = nn.LSTM(d_model, hidden_size, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.size()
        # Transformer encoder:
        proj = self.input_projection(x)         # (batch, seq_len, d_model)
        proj = self.positional_encoding(proj)
        enc_out = self.transformer_encoder(proj)  # (batch, seq_len, d_model)
        
        # Use the last encoder output as the context vector for the decoder.
        context = enc_out[:, -1:, :]              # (batch, 1, d_model)
        # Repeat context for each forecast step.
        decoder_input = context.repeat(1, self.n_ahead, 1)  # (batch, n_ahead, d_model)
        lstm_out, _ = self.lstm_decoder(decoder_input)
        main_output = self.fc_out(lstm_out)       # (batch, n_ahead, num_classes)
        
        return main_output
