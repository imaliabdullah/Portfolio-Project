import torch
import torch.nn as nn


class EncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EncoderDecoder, self).__init__()
        self.hidden_size = hidden_size

        # Encoder (LSTM)
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        # Decoder (LSTM)
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        # Fully connected layer to map the decoder output to class logits
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Encoder step
        x = x.unsqueeze(1)  # Add seq_length dimension (batch_size, seq_length, input_size)
        encoder_output, (hidden, cell) = self.encoder_lstm(x)

        # Decoder step, we use the encoder hidden state as the first decoder input
        decoder_output, _ = self.decoder_lstm(encoder_output, (hidden, cell))

        # Pass decoder output through the fully connected layer
        output = self.fc(decoder_output[:, -1, :])  # Only take the last time step for classification
        return output