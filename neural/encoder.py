"""
Neural Encoder Module
LSTM-based encoder for node observation sequences.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class LSTMEncoder(nn.Module):
    """
    LSTM encoder that compresses observation sequences into fixed-size embeddings.
    """
    
    def __init__(
        self,
        input_size: int = 16,
        hidden_size: int = 64,
        latent_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Projection to latent space
        lstm_output_size = hidden_size * self.num_directions
        self.latent_proj = nn.Sequential(
            nn.Linear(lstm_output_size, latent_size),
            nn.Tanh()
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode observation sequence.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            hidden: Optional initial hidden state
            
        Returns:
            latent: Latent representation (batch, latent_size)
            hidden: Final hidden state
        """
        batch_size = x.size(0)
        
        # Project input
        x = self.input_proj(x)
        x = self.dropout(x)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self._init_hidden(batch_size, x.device)
        
        # LSTM forward
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Take last output (or concatenate if bidirectional)
        if self.bidirectional:
            last_out = torch.cat([lstm_out[:, -1, :self.hidden_size],
                                  lstm_out[:, 0, self.hidden_size:]], dim=1)
        else:
            last_out = lstm_out[:, -1, :]
        
        # Project to latent space
        latent = self.latent_proj(last_out)
        
        return latent, hidden
    
    def _init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state with zeros."""
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=device)
        return (h0, c0)


class LSTMDecoder(nn.Module):
    """
    LSTM decoder for reconstructing observation sequences from latent space.
    """
    
    def __init__(
        self,
        latent_size: int = 32,
        hidden_size: int = 64,
        output_size: int = 16,
        num_layers: int = 2,
        seq_len: int = 20,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        
        # Project latent to initial hidden state
        self.latent_to_hidden = nn.Linear(latent_size, hidden_size * num_layers)
        self.latent_to_cell = nn.Linear(latent_size, hidden_size * num_layers)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=latent_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, output_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to observation sequence.
        
        Args:
            latent: Latent tensor of shape (batch, latent_size)
            
        Returns:
            Reconstructed sequence of shape (batch, seq_len, output_size)
        """
        batch_size = latent.size(0)
        device = latent.device
        
        # Initialize hidden state from latent
        h = self.latent_to_hidden(latent)
        c = self.latent_to_cell(latent)
        
        # Reshape to (num_layers, batch, hidden)
        h = h.view(batch_size, self.num_layers, self.hidden_size).permute(1, 0, 2).contiguous()
        c = c.view(batch_size, self.num_layers, self.hidden_size).permute(1, 0, 2).contiguous()
        
        hidden = (h, c)
        
        # Repeat latent as input for each timestep
        decoder_input = latent.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # LSTM forward
        lstm_out, _ = self.lstm(decoder_input, hidden)
        lstm_out = self.dropout(lstm_out)
        
        # Project to output
        output = self.output_proj(lstm_out)
        
        return output


class LSTMAutoencoder(nn.Module):
    """
    Complete LSTM autoencoder for anomaly detection.
    
    High reconstruction error indicates anomaly (potential failure).
    """
    
    def __init__(
        self,
        input_size: int = 16,
        hidden_size: int = 64,
        latent_size: int = 32,
        num_layers: int = 2,
        seq_len: int = 20,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.seq_len = seq_len
        
        self.encoder = LSTMEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            latent_size=latent_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.decoder = LSTMDecoder(
            latent_size=latent_size,
            hidden_size=hidden_size,
            output_size=input_size,
            num_layers=num_layers,
            seq_len=seq_len,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            
        Returns:
            reconstructed: Reconstructed input (batch, seq_len, input_size)
            latent: Latent representation (batch, latent_size)
        """
        latent, _ = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation only."""
        latent, _ = self.encoder(x)
        return latent
    
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction error (anomaly score).
        
        Returns:
            Per-sample reconstruction error (batch,)
        """
        reconstructed, _ = self.forward(x)
        error = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
        return error
    
    def get_anomaly_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for reconstruction_error."""
        return self.reconstruction_error(x)