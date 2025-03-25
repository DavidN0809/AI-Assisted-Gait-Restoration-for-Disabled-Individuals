import torch
import torch.nn as nn
import torch.nn.functional as F

#############################################
# LSTM-based Models
#############################################

class BasicLSTM(nn.Module):
    """
    Basic stacked LSTM for regression.
    Predicts a sequence of outputs (n_ahead timesteps) from the last LSTM output.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, n_ahead, dropout=0.3):
        super(BasicLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_ahead = n_ahead
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size * n_ahead)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        out = out.view(-1, self.n_ahead, out.size(1) // self.n_ahead)
        return out

class BidirectionalLSTM(nn.Module):
    """
    LSTM with bidirectional layers.
    Combines the last hidden state from both forward and backward directions.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, n_ahead, dropout=0.3):
        super(BidirectionalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_ahead = n_ahead
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size * n_ahead)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)
        out, (h_n, _) = self.lstm(x, (h0, c0))
        forward_last = h_n[-2, :, :]
        backward_last = h_n[-1, :, :]
        h_combined = torch.cat((forward_last, backward_last), dim=1)
        h_combined = self.dropout(h_combined)
        out = self.fc(h_combined)
        out = out.view(-1, self.n_ahead, out.size(1) // self.n_ahead)
        return out

class ResidualLSTM(nn.Module):
    """
    Residual LSTM adds a skip connection from the input (last timestep) to 
    the LSTM's final output (after dropout) to help gradient flow.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, n_ahead,dropout=0.3):
        super(ResidualLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_ahead = n_ahead
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size * n_ahead)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        last_out = out[:, -1, :]
        x_proj = self.input_projection(x[:, -1, :])
        residual = last_out + x_proj
        out = self.fc(residual)
        out = out.view(-1, self.n_ahead, out.size(1) // self.n_ahead)
        return out

class AttentionLSTM(nn.Module):
    """
    Attention-based LSTM uses a learned attention mechanism over the LSTM outputs.
    Computes a weighted sum (context vector) of all timesteps before the final prediction.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, n_ahead,dropout=0.3):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_ahead = n_ahead
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attn = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size * n_ahead)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        attn_weights = torch.softmax(self.attn(out), dim=1)
        context = torch.sum(attn_weights * out, dim=1)
        context = self.dropout(context)
        out = self.fc(context)
        out = out.view(-1, self.n_ahead, out.size(1) // self.n_ahead)
        return out

#############################################
# Transformer-based Models
#############################################

class TimeSeriesTransformer(nn.Module):
    """
    Updated Transformer model inspired by Temporal Fusion Transformers (Lim et al., 2021)
    and DeepAR (Salinas et al., 2019). Uses an input projection, a transformer encoder,
    and a gating mechanism to better capture peaks and sharp transitions.
    """
    def __init__(self, input_size, num_heads, num_layers, hidden_dim, output_size, dropout=0.3, n_ahead=1):
        super(TimeSeriesTransformer, self).__init__()
        self.n_ahead = n_ahead
        self.output_size = output_size
        self.input_projection = nn.Linear(input_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                    nhead=num_heads,
                                                    dropout=dropout,
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.output_projection = nn.Linear(hidden_dim, output_size * n_ahead)

    def forward(self, x):
        x = self.input_projection(x)
        encoded = self.transformer_encoder(x)
        last_encoding = encoded[:, -1, :]
        gate = self.gate(last_encoding)
        gated_output = last_encoding * gate
        out = self.output_projection(gated_output)
        out = out.view(-1, self.n_ahead, self.output_size)
        return out

class TemporalFusionTransformer(nn.Module):
    """
    A simplified Temporal Fusion Transformer (TFT) inspired by Lim et al. (2021).
    Uses variable selection gating and attention to capture both long-term trends
    and high-frequency dynamics.
    """
    def __init__(self, input_size, hidden_dim, num_heads, num_layers, output_size, dropout=0.3, n_ahead=1):
        super(TemporalFusionTransformer, self).__init__()
        self.n_ahead = n_ahead
        self.input_projection = nn.Linear(input_size, hidden_dim)
        self.variable_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                    nhead=num_heads,
                                                    dropout=dropout,
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention = nn.Linear(hidden_dim, 1)
        self.output_projection = nn.Linear(hidden_dim, output_size * n_ahead)

    def forward(self, x):
        x = self.input_projection(x)
        x = x * self.variable_gate(x)
        encoded = self.transformer_encoder(x)
        attn_weights = torch.softmax(self.attention(encoded), dim=1)
        context = torch.sum(attn_weights * encoded, dim=1)
        out = self.output_projection(context)
        out = out.view(-1, self.n_ahead, out.size(1) // self.n_ahead)
        return out

#############################################
# Additional Transformer-inspired Models
#############################################

# Minimal Informer Implementation
class ProbSparseSelfAttention(nn.Module):
    """
    Simplified ProbSparse Self-Attention (placeholder for Informer).
    """
    def __init__(self, d_model, n_heads, dropout=0.3):
        super(ProbSparseSelfAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        q = self.query(x).view(B, L, self.n_heads, self.d_k).transpose(1,2)
        k = self.key(x).view(B, L, self.n_heads, self.d_k).transpose(1,2)
        v = self.value(x).view(B, L, self.n_heads, self.d_k).transpose(1,2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.d_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1,2).contiguous().view(B, L, D)
        out = self.out(context)
        return out

class InformerEncoderLayer(nn.Module):
    """
    Informer Encoder Layer: ProbSparse Self-Attention + FeedForward.
    """
    def __init__(self, d_model, n_heads, dropout=0.3):
        super(InformerEncoderLayer, self).__init__()
        self.self_attn = ProbSparseSelfAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out = self.self_attn(x)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        ff_out = self.feedforward(x)
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)
        return x

class InformerEncoder(nn.Module):
    """
    Stack of Informer Encoder Layers.
    """
    def __init__(self, d_model, n_heads, num_layers, dropout=0.3):
        super(InformerEncoder, self).__init__()
        self.layers = nn.ModuleList([InformerEncoderLayer(d_model, n_heads, dropout=dropout)
                                     for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Informer(nn.Module):
    """
    Informer-based model for multistep regression.
    """
    def __init__(self, input_size, d_model, n_heads, num_layers, output_size, n_ahead=1, dropout=0.3):
        super(Informer, self).__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.encoder = InformerEncoder(d_model, n_heads, num_layers, dropout=dropout)
        self.fc_out = nn.Linear(d_model, output_size * n_ahead)
        self.n_ahead = n_ahead
        self.output_size = output_size

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        out = x[:, -1, :]
        out = self.fc_out(out)
        out = out.view(-1, self.n_ahead, self.output_size)
        return out

# Minimal N-BEATS Implementation
class NBeatsBlock(nn.Module):
    """
    One block of the N-BEATS architecture.
    """
    def __init__(self, input_size, hidden_dim, output_size, n_ahead):
        super(NBeatsBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_size * n_ahead)
        self.relu = nn.ReLU()
        self.n_ahead = n_ahead
        self.output_size = output_size

    def forward(self, x):
        # x: [batch, input_size]
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        out = self.fc4(x)
        out = out.view(-1, self.n_ahead, self.output_size)
        return out

class NBeats(nn.Module):
    """
    Minimal N-BEATS model: stacks several blocks.
    """
    def __init__(self, input_size, hidden_dim, output_size, n_ahead, num_blocks=3):
        super(NBeats, self).__init__()
        self.blocks = nn.ModuleList([NBeatsBlock(input_size, hidden_dim, output_size, n_ahead)
                                     for _ in range(num_blocks)])

    def forward(self, x):
        # x: [batch, seq_len, input_dim] => flatten last two dims
        batch_size, seq_len, input_dim = x.size()
        x = x.view(batch_size, -1)
        # Here, we simply use the last block's output as the prediction.
        for block in self.blocks:
            out = block(x)
        return out

#############################################
# End of models.py
#############################################
