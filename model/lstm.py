import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, num_classes, bidirectional, input_size, hidden_size=128, dropout=0.1, num_layers=2):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout) 
        self.bidirectional = bidirectional

        if bidirectional:
            self.output_layer = nn.Linear(hidden_size * 2, num_classes) 

        else: 
            self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids):
        input_ids = input_ids.unsqueeze(1)  
        _, (hidden_state, _) = self.lstm(input_ids)

        if self.bidirectional:
            last_sequential_backward = hidden_state[-2]
            last_sequential_forward = hidden_state[-1]
            lstm_output = torch.cat([last_sequential_backward, last_sequential_forward], dim=-1)

        else:
            lstm_output = hidden_state[-1]

        preds = self.output_layer(self.dropout(lstm_output))

        return preds