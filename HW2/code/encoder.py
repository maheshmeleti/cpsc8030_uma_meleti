import torch.nn as nn


class Encoder_lstm(nn.Module):
    def __init__(self):
        super(Encoder_lstm, self).__init__()

        self.Embedding = nn.Linear(4096, 512)
        self.dropout = nn.Dropout(0.35)
        self.lstm = nn.LSTM(512, 512, batch_first=True)

    def forward(self, input):
        batch_size, seq_len, feat_n = input.size()
        input = input.view(-1, feat_n)
        input = self.Embedding(input)
        input = self.dropout(input)
        input = input.view(batch_size, seq_len, 512)

        output, t = self.lstm(input)
        hidden_state, context = t[0], t[1]
        return output, hidden_state