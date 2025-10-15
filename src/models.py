"""LSTM Encoder-Decoder models for bond yield prediction"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange


class LSTMEncoder(nn.Module):
    """LSTM Encoder for sequence-to-sequence prediction."""

    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.5
        )

    def forward(self, x_input):
        """Forward pass through encoder."""
        lstm_out, self.hidden = self.lstm(
            x_input.view(x_input.shape[0], x_input.shape[1], self.input_size)
        )
        return lstm_out, self.hidden

    def init_hidden(self, batch_size):
        """Initialize hidden states."""
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size),
            torch.zeros(self.num_layers, batch_size, self.hidden_size)
        )


class LSTMDecoder(nn.Module):
    """LSTM Decoder for sequence-to-sequence prediction."""

    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.5
        )
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x_input, encoder_hidden_states):
        """Forward pass through decoder."""
        lstm_out, self.hidden = self.lstm(x_input, encoder_hidden_states)
        output = self.linear(lstm_out)
        return output, self.hidden


class LSTMDecoder2(nn.Module):
    """Alternative LSTM Decoder with different input size for autoregressive decoding."""

    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMDecoder2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x_input, encoder_hidden_states):
        """Forward pass through alternative decoder."""
        lstm_out, self.hidden = self.lstm(x_input, encoder_hidden_states)
        output = self.linear(lstm_out)
        return output, self.hidden


class Discriminator(nn.Module):
    """Discriminator for Professor Forcing training."""

    def __init__(self, input_size, hidden_size, linear_size, lin_dropout):
        super(Discriminator, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )

        self.linears = nn.Sequential(
            nn.Linear(hidden_size * 2, linear_size),
            nn.ReLU(),
            nn.Dropout(lin_dropout),
            nn.ReLU(),
            nn.Dropout(lin_dropout),
            nn.Linear(linear_size, 1),
            nn.Sigmoid()
        )

    def forward(self, hidden_states):
        """Forward pass through discriminator."""
        batch_size = hidden_states.size(0)
        initial_hidden = self.init_hidden(batch_size)
        _, rnn_final_hidden = self.lstm(hidden_states, initial_hidden)

        rnn_final_hidden = (
            rnn_final_hidden[0].view(batch_size, -1),
            rnn_final_hidden[1].view(batch_size, -1)
        )

        scores = self.linears(rnn_final_hidden[0])
        return scores

    def init_hidden(self, batch_size):
        """Initialize hidden states for discriminator."""
        hidden_1 = torch.zeros(2, batch_size, self.hidden_size)
        hidden_2 = torch.zeros(2, batch_size, self.hidden_size)
        return (hidden_1, hidden_2)


class LSTMSeq2Seq(nn.Module):
    """Complete LSTM Encoder-Decoder model with training and prediction capabilities."""

    def __init__(self, input_size, hidden_size):
        super(LSTMSeq2Seq, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder = LSTMEncoder(input_size=input_size, hidden_size=hidden_size, num_layers=2)
        self.decoder = LSTMDecoder(input_size=input_size, hidden_size=hidden_size, num_layers=2)
        self.decoder2 = LSTMDecoder2(input_size=1, hidden_size=hidden_size, num_layers=2)

    def train_model(self, input_tensor, target_tensor, n_epochs, target_len, batch_size,
                   training_prediction="recursive", teacher_forcing_ratio=0.5,
                   learning_rate=0.01, dynamic_tf=False):
        """Train the model with specified parameters."""
        losses = np.full(n_epochs, np.nan)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        n_batches = int(input_tensor.shape[1] // batch_size)

        print(f"Training with {n_batches} batches")

        with trange(n_epochs) as tr:
            for it in tr:
                batch_loss = 0

                for b in range(n_batches):
                    # Get batch data
                    input_batch = input_tensor[:, b: b + batch_size, :]
                    target_batch = target_tensor[:, b: b + batch_size, :]
                    outputs = torch.zeros(target_len, batch_size, 1)

                    # Initialize encoder
                    encoder_hidden = self.encoder.init_hidden(batch_size=batch_size)
                    optimizer.zero_grad()

                    # Encode
                    encoder_output, encoder_hidden = self.encoder(input_batch)
                    decoder_input = input_batch[-1, :, :]

                    # Prepare decoder hidden state
                    hidden_state = encoder_hidden[0]
                    cell_state = encoder_hidden[1]
                    decoder_hidden = (hidden_state[:, 0, :], cell_state[:, 0, :])

                    # Decode based on training strategy
                    if training_prediction == "recursive":
                        for t in range(target_len):
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            outputs[t] = decoder_output
                            decoder_input = decoder_output

                    elif training_prediction == "teacher_forcing":
                        for t in range(target_len):
                            if t == 0:
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                outputs[t] = decoder_output
                                decoder_input = target_batch[t, :, :]
                            else:
                                decoder_output, decoder_hidden = self.decoder2(decoder_input, decoder_hidden)
                                outputs[t] = decoder_output
                                decoder_input = target_batch[t, :, :]

                    # Calculate loss and backpropagate
                    target_batch = target_batch.reshape(target_batch.shape[0], target_batch.shape[1], 1)
                    loss = criterion(outputs, target_batch)
                    batch_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                losses[it] = batch_loss
                if dynamic_tf and teacher_forcing_ratio > 0:
                    teacher_forcing_ratio = teacher_forcing_ratio - 0.02

                tr.set_postfix(loss=f"{batch_loss:.3f}")

        # Save model
        model_save_path = os.getenv('MODEL_SAVE_PATH', 'trained_model.pth')
        torch.save(self.state_dict(), model_save_path)
        return sum(losses) / len(losses)

    def predict(self, input_tensor, target_len):
        """Generate predictions using the trained model."""
        encoder_output, encoder_hidden = self.encoder(input_tensor)
        outputs = torch.zeros(target_len, input_tensor.shape[1], 1)

        # Prepare decoder
        decoder_input = input_tensor[-1, :, :]
        hidden_state = encoder_hidden[0]
        cell_state = encoder_hidden[1]
        decoder_hidden = (hidden_state[:, 0, :], cell_state[:, 0, :])

        # Generate predictions
        for t in range(target_len):
            if t == 0:
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[t] = decoder_output
                decoder_input = decoder_output
            else:
                decoder_output, decoder_hidden = self.decoder2(decoder_input, decoder_hidden)
                outputs[t] = decoder_output
                decoder_input = decoder_output

        return outputs.detach().numpy()


class ModelAlternate(LSTMSeq2Seq):
    """Alternative model with Professor Forcing training."""

    def __init__(self, input_size, hidden_size):
        super(ModelAlternate, self).__init__(hidden_size=hidden_size, input_size=input_size)
        self.discriminator = Discriminator(
            input_size=1,
            hidden_size=hidden_size,
            linear_size=64,
            lin_dropout=0.5
        )
        self.other_params = [
            {'params': self.encoder.parameters(), 'lr': 0.0001},
            {'params': self.decoder.parameters(), 'lr': 0.0002},
            {'params': self.decoder2.parameters(), 'lr': 0.0003, 'weight_decay': 1e-4}
        ]

    def adversarial_train(self, learning_rate, input_tensor, target_tensor,
                         n_epochs, target_len, batch_size):
        """Train model using adversarial approach with discriminator (Professor Forcing)."""
        losses = np.full(n_epochs, np.nan)
        gen_optimizer = optim.SGD(self.other_params)
        disc_optimizer = optim.SGD(self.discriminator.parameters(), lr=0.003)
        binary_cross_entropy = nn.BCELoss()

        n_batches = int(input_tensor.shape[1] // batch_size)

        with trange(n_epochs) as tr:
            for it in tr:
                for b in range(n_batches):
                    input_batch = input_tensor[:, b:b + batch_size, :]
                    target_batch = target_tensor[:, b:b + batch_size, :]
                    outputs = torch.zeros(target_len, batch_size, 1).to(input_tensor.device)
                    labels = torch.zeros(target_len, batch_size, 1).to(input_tensor.device)

                    encoder_hidden = self.encoder.init_hidden(batch_size=batch_size)
                    gen_optimizer.zero_grad()
                    disc_optimizer.zero_grad()

                    encoder_output, encoder_hidden = self.encoder(input_batch)
                    decoder_input = input_batch[-1, :, :]

                    hidden_state = encoder_hidden[0]
                    cell_state = encoder_hidden[1]
                    decoder_hidden = (hidden_state[:, 0, :], cell_state[:, 0, :])

                    for t in range(target_len):
                        if t == 0:
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            outputs[t] = decoder_output
                            decoder_input = target_batch[t, :, :]
                        else:
                            decoder_output, decoder_hidden = self.decoder2(decoder_input, decoder_hidden)
                            outputs[t] = decoder_output
                            decoder_input = torch.cat([
                                decoder_output[0:25, :],
                                target_batch[t, 0:25, :]
                            ], dim=0)

                        labels[t] = torch.cat([torch.ones(25, 1), torch.zeros(25, 1)], dim=0)

                    labels = labels.transpose(1, 0)
                    outputs = outputs.transpose(1, 0)

                    preds = self.discriminator(outputs)
                    indices = torch.randperm(preds.size(0))
                    preds = preds[indices]
                    labels = labels[indices][:, :, 0, :]

                    discriminator_loss = binary_cross_entropy(preds, labels)
                    generator_loss = -discriminator_loss

                    if b % 2 == 0:
                        generator_loss.backward()
                        gen_optimizer.step()
                    else:
                        discriminator_loss.backward()
                        disc_optimizer.step()

        # Save model
        model_save_path = os.getenv('MODEL_SAVE_PATH', 'trained_model.pth')
        torch.save(self.state_dict(), model_save_path)
