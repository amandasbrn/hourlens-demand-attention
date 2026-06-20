import torch
from torch import nn

'''

model architecture of LSTM

input sequence
    ↓
LSTM
    ↓
use only last hidden state
    ↓
forecast

input shape = (batch_size, sequence_length, input_features)

'''

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, output_size) # converts the last hidden state into a single output

    def forward(self, x):
        lstm_out, _ = self.lstm(x) # output shape = (batch_size, 24 (sequence length), hidden_size)
         
        '''
            last_hidden
            :    = take all samples in the batch
            -1   = take the last time step
            :    = take all hidden features

            why -1? because the sequence is t-24 → t-23 → ... → t-1
            so we take t-1 as the representation after seeing all 24h
            that's why LSTM only see h_t-1 only.
        '''
        
        last_hidden = lstm_out[:, -1, :] # use final hidden state
        prediction = self.fc(last_hidden)
        return prediction