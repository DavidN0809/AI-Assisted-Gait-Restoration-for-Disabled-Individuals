import torch
import torch.nn as nn

class BasicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, n_ahead):
        super(BasicLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.n_ahead = n_ahead
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Use n_ahead instead of hardcoding 4:
        self.fc = nn.Linear(hidden_size, output_size * n_ahead)
        self.Drop = nn.Dropout(0.3)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.Drop(out)
        out = self.fc(out[:, -1, :])
        # Reshape the output to [batch_size, n_ahead, output_size]:
        out = out.view(-1, self.n_ahead, self.output_size)
        return out
