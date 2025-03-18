import torch
import torch.nn as nn

class BasicLSTM(nn.Module):
    """
    Basic stacked LSTM for regression.
    Predicts a sequence of outputs (n_ahead timesteps) from the last LSTM output.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, n_ahead):
        super(BasicLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_ahead = n_ahead
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, output_size * n_ahead)
        
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        # Take the last timestep output
        out = self.fc(out[:, -1, :])
        # Reshape to [batch, n_ahead, output_size]
        out = out.view(-1, self.n_ahead, out.size(1) // self.n_ahead)
        return out


class BidirectionalLSTM(nn.Module):
    """
    LSTM with bidirectional layers.
    Combines the last hidden state from both forward and backward directions.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, n_ahead):
        super(BidirectionalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_ahead = n_ahead
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)
        # Because of bidirectionality, the effective hidden size doubles.
        self.fc = nn.Linear(hidden_size * 2, output_size * n_ahead)
        
    def forward(self, x):
        # For bidirectional LSTM, initialize hidden states for both directions
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        out, (h_n, _) = self.lstm(x, (h0, c0))
        # h_n shape: [num_layers*2, batch, hidden_size]
        # Retrieve the last layer's hidden states for both directions
        forward_last = h_n[-2, :, :]  # Forward direction
        backward_last = h_n[-1, :, :] # Backward direction
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
    def __init__(self, input_size, hidden_size, num_layers, output_size, n_ahead):
        super(ResidualLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_ahead = n_ahead
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        # Project the raw input to match the hidden size for the skip connection
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size * n_ahead)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        # Last timestep output from LSTM
        last_out = out[:, -1, :]
        # Project the last input timestep for the residual connection
        x_proj = self.input_projection(x[:, -1, :])
        residual = last_out + x_proj
        out = self.fc(residual)
        out = out.view(-1, self.n_ahead, out.size(1) // self.n_ahead)
        return out


class AttentionLSTM(nn.Module):
    """
    Attention-based LSTM uses a learned attention mechanism over the LSTM outputs.
    This computes a weighted sum (context vector) of all timesteps before the final prediction.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, n_ahead):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_ahead = n_ahead
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Attention layer: computes a scalar score for each timestep's hidden state
        self.attn = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, output_size * n_ahead)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))  # out: [batch, seq_len, hidden_size]
        # Compute attention weights over timesteps
        attn_weights = torch.softmax(self.attn(out), dim=1)  # [batch, seq_len, 1]
        # Compute context vector as the weighted sum of LSTM outputs
        context = torch.sum(attn_weights * out, dim=1)  # [batch, hidden_size]
        context = self.dropout(context)
        out = self.fc(context)
        out = out.view(-1, self.n_ahead, out.size(1) // self.n_ahead)
        return out
