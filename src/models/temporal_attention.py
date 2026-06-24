import torch
from torch import nn

'''
Temporal attention:
When predicting demand_t, how much did the model attend to:
t-24, t-23, ..., t-1?

            X: past 24 hours
                ↓
            LSTM
                ↓
            get hidden state for every hour
                ↓
            attention layer chooses important hours
                ↓
            weighted average context vector
                ↓
            Linear layer
                ↓
            prediction

attention here is a weighted average of the 24 LSTM hidden states

LSTM baseline:
forecast = Linear(last hidden state)

LSTM + attention:
forecast = Linear(weighted average of all hidden states)

model return = prediction, attention_weights
prediction = used for loss/training, shape 
attention_weights = used for visualization & dashboard

output shape
context_vector:     (batch_size, hidden_size)
attention_weights:  (batch_size, sequence_length)

'''
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()

        self.attention_score = nn.Linear(hidden_size, 1)

    def forward(self, lstm_outputs: torch.Tensor):
        '''
        lstm_outputs:
            from LSTM model earlier, shape: (batch_size, 24, hidden_size)
            (batch_size, 24, 32)
                    ↓ nn.Linear(32, 1)
            (batch_size, 24, 1)

        scores:
            for each timestep, produce one importance score
            (batch_size, 24, 32) → (batch_size, 24, 1)
            from 32 numbers to 1 score. Each 24 hours has one raw attention score.

        attention_weights:
            Turns raw scores into weights that sum to 1 (probability) across the 24 time steps.
            for each sample:

            attention over 24 hours = [0.02, 0.04, ..., 0.20]
            sum = 1.0
        
        context_vector:
            This creates a weighted average of all LSTM hidden states.
            If one hour has high attention, its hidden state contributes more.

        attention_weights:
            changes shape (batch_size, 24, 1) → (batch_size, 24)

            Easier to save and visualize.
        '''
        scores = self.attention_score(lstm_outputs)
        attention_weights = torch.softmax(scores,dim=1)

        context_vector = torch.sum(attention_weights * lstm_outputs, dim=1)

        attention_weights = attention_weights.squeeze(-1)

        return context_vector, attention_weights

class LSTMAttentionForecaster(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, num_layers:int, output_size:int, dropout: float = 0.1):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.attention = TemporalAttention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size) # # converts attention context into prediction

    def forward(self, x):
        lstm_out, _ = self.lstm(x) # output shape = (batch_size, 24 (sequence length), hidden_size)

        lstm_out = self.dropout(lstm_out)

        context_vector, attention_weights = self.attention(lstm_out)

        context_vector = self.dropout(context_vector)

        prediction = self.fc(context_vector)

        return prediction, attention_weights