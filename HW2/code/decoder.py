import torch.nn as nn
import torch
from attention import attention
from torch.autograd import Variable
import random
device = 0


class Decoder_lstm_attn(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_size, word_dim, dropout_percentage=0.35):
        super(Decoder_lstm_attn, self).__init__()

        self.hidden_size = 512
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.word_dim = word_dim

        self.embedding = nn.Embedding(output_size, 1024)
        self.dropout = nn.Dropout(0.35)
        self.lstm = nn.LSTM(hidden_size + word_dim, hidden_size, batch_first=True)
        self.attention = attention(hidden_size)
        self.to_final_output = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_last_hidden_state, encoder_output, targets=None, mode='train', tr_steps=None):
        _, batch_size, _ = encoder_last_hidden_state.size()

        decoder_current_hidden_state = None if encoder_last_hidden_state is None else encoder_last_hidden_state
        decoder_cxt = torch.zeros(decoder_current_hidden_state.size()) #.to(device)
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)) #.long().to(device)
        seq_logProb = []
        seq_predictions = []

        targets = self.embedding(targets)
        _, seq_len, _ = targets.size()

        for i in range(seq_len - 1):
            threshold = self.teacher_forcing_ratio(training_steps=tr_steps)
            if random.uniform(0.05, 0.995) > threshold:  # returns a random float value between 0.05 and 0.995
                current_input_word = targets[:, i]
            else:
                current_input_word = self.embedding(decoder_current_input_word).squeeze(1)

            context = self.attention(decoder_current_hidden_state, encoder_output)
            lstm_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            lstm_output, t = self.lstm(lstm_input, (decoder_current_hidden_state, decoder_cxt))
            decoder_current_hidden_state = t[0]
            logprob = self.to_final_output(lstm_output.squeeze(1))
            seq_logProb.append(logprob.unsqueeze(1))
            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]

        seq_logProb = torch.cat(seq_logProb, dim=1)
        seq_predictions = seq_logProb.max(2)[1]
        return seq_logProb, seq_predictions

    def infer(self, encoder_last_hidden_state, encoder_output):
        _, batch_size, _ = encoder_last_hidden_state.size()
        decoder_current_hidden_state = None if encoder_last_hidden_state is None else encoder_last_hidden_state
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long()
        decoder_c = torch.zeros(decoder_current_hidden_state.size())
        seq_logProb = []
        seq_predictions = []
        assumption_seq_len = 28

        for i in range(assumption_seq_len - 1):
            current_input_word = self.embedding(decoder_current_input_word).squeeze(1)
            context = self.attention(decoder_current_hidden_state, encoder_output)
            lstm_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            lstm_output, t = self.lstm(lstm_input, (decoder_current_hidden_state, decoder_c))
            decoder_current_hidden_state = t[0]
            logprob = self.to_final_output(lstm_output.squeeze(1))
            seq_logProb.append(logprob.unsqueeze(1))
            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]

        seq_logProb = torch.cat(seq_logProb, dim=1)
        seq_predictions = seq_logProb.max(2)[1]
        return seq_logProb, seq_predictions

    def teacher_forcing_ratio(self, training_steps):
        return (expit(training_steps / 20 + 0.85))  # inverse of the logit function
