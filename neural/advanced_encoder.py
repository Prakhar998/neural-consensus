# neural/advanced_encoder.py
"""
Advanced Neural Encoder with ResNet blocks, Attention, and multi-scale features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-style attention."""
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ResidualBlock(nn.Module):
    """1D Residual block for temporal data."""
    
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, seq_len)
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + residual
        out = self.relu(out)
        
        return out


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, seq_len)
        batch, channels, _ = x.size()
        
        squeeze = self.squeeze(x).view(batch, channels)
        excitation = self.excitation(squeeze).view(batch, channels, 1)
        
        return x * excitation


class TemporalAttention(nn.Module):
    """Multi-head self-attention for temporal sequences."""
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, embed_dim)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        return x


class MultiScaleConv(nn.Module):
    """Multi-scale convolution to capture patterns at different time scales."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # Different kernel sizes for different temporal scales
        self.conv1 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=7, padding=3)
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, in_channels, seq_len)
        out1 = self.conv1(x)
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        out7 = self.conv7(x)
        
        out = torch.cat([out1, out3, out5, out7], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        
        return out


class AdvancedEncoder(nn.Module):
    """
    Advanced encoder combining:
    - Multi-scale convolutions
    - ResNet blocks with SE attention
    - Bidirectional LSTM
    - Transformer attention
    """
    
    def __init__(
        self,
        input_size: int = 32,
        hidden_size: int = 128,
        latent_size: int = 64,
        num_res_blocks: int = 3,
        num_attention_heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-scale convolution
        self.multi_scale = MultiScaleConv(hidden_size, hidden_size)
        
        # Residual blocks with SE attention
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(hidden_size, dropout=dropout),
                SqueezeExcitation(hidden_size)
            )
            for _ in range(num_res_blocks)
        ])
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Positional encoding for attention
        self.pos_encoding = PositionalEncoding(hidden_size, dropout=dropout)
        
        # Self-attention layers
        self.attention_layers = nn.ModuleList([
            TemporalAttention(hidden_size, num_attention_heads, dropout)
            for _ in range(2)
        ])
        
        # Final projection to latent space
        self.latent_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, latent_size),
            nn.Tanh()
        )
        
        # Auxiliary projection for reconstruction
        self.aux_proj = nn.Linear(hidden_size, latent_size)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, input_size)
            
        Returns:
            latent: Latent representation (batch, latent_size)
            features: Intermediate features for auxiliary tasks
        """
        batch_size, seq_len, _ = x.size()
        
        # Input projection: (batch, seq_len, hidden)
        x = self.input_proj(x)
        
        # Conv processing: need (batch, hidden, seq_len)
        x_conv = x.permute(0, 2, 1)
        x_conv = self.multi_scale(x_conv)
        
        for res_block in self.res_blocks:
            x_conv = res_block(x_conv)
        
        # Back to (batch, seq_len, hidden)
        x = x_conv.permute(0, 2, 1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Attention processing
        x_attn = self.pos_encoding(x)
        for attn_layer in self.attention_layers:
            x_attn = attn_layer(x_attn)
        
        # Combine LSTM and attention outputs
        combined = torch.cat([lstm_out[:, -1, :], x_attn[:, -1, :]], dim=1)
        
        # Project to latent space
        latent = self.latent_proj(combined)
        
        # Auxiliary features (mean pooling)
        aux_features = self.aux_proj(x.mean(dim=1))
        
        return latent, aux_features


class AdvancedDecoder(nn.Module):
    """Decoder for reconstruction."""
    
    def __init__(
        self,
        latent_size: int = 64,
        hidden_size: int = 128,
        output_size: int = 32,
        seq_len: int = 50,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        self.output_proj = nn.Linear(hidden_size, output_size)
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        batch_size = latent.size(0)
        
        # Project latent
        x = self.latent_proj(latent)
        
        # Repeat for sequence
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # LSTM decode
        x, _ = self.lstm(x)
        
        # Output projection
        output = self.output_proj(x)
        
        return output


class AdvancedAutoencoder(nn.Module):
    """Complete advanced autoencoder."""
    
    def __init__(
        self,
        input_size: int = 32,
        hidden_size: int = 128,
        latent_size: int = 64,
        seq_len: int = 50,
        num_res_blocks: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.encoder = AdvancedEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            latent_size=latent_size,
            num_res_blocks=num_res_blocks,
            dropout=dropout
        )
        
        self.decoder = AdvancedDecoder(
            latent_size=latent_size,
            hidden_size=hidden_size,
            output_size=input_size,
            seq_len=seq_len,
            dropout=dropout
        )
        
        self.latent_size = latent_size
        self.seq_len = seq_len
        self.input_size = input_size
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, aux_features = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent, aux_features
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        latent, _ = self.encoder(x)
        return latent
    
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        reconstructed, _, _ = self.forward(x)
        error = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
        return error


class AdvancedClassifier(nn.Module):
    """
    Advanced classifier with:
    - Deep residual MLP
    - Dropout and batch norm
    - Multiple output heads
    """
    
    def __init__(
        self,
        latent_size: int = 64,
        hidden_sizes: List[int] = [256, 128, 64],
        num_classes: int = 6,
        dropout: float = 0.3
    ):
        super().__init__()
        
        layers = []
        prev_size = latent_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            
            # Add residual connection if sizes match
            if prev_size == hidden_size:
                layers.append(ResidualMLP(hidden_size, dropout))
            
            prev_size = hidden_size
        
        self.features = nn.Sequential(*layers)
        
        # Main classification head
        self.classifier = nn.Linear(prev_size, num_classes)
        
        # Auxiliary head for binary failure detection
        self.binary_head = nn.Linear(prev_size, 2)
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(prev_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.features(latent)
        
        logits = self.classifier(features)
        binary_logits = self.binary_head(features)
        confidence = self.confidence_head(features)
        
        return logits, binary_logits, confidence


class ResidualMLP(nn.Module):
    """Residual MLP block."""
    
    def __init__(self, size: int, dropout: float = 0.1):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(size, size)
        )
        self.norm = nn.LayerNorm(size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.layers(x))


class AdvancedCombinedModel(nn.Module):
    """
    Complete advanced model combining:
    - Advanced autoencoder
    - Advanced classifier
    - Multi-task learning
    """
    
    def __init__(
        self,
        input_size: int = 32,
        hidden_size: int = 128,
        latent_size: int = 64,
        seq_len: int = 50,
        num_classes: int = 6,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.autoencoder = AdvancedAutoencoder(
            input_size=input_size,
            hidden_size=hidden_size,
            latent_size=latent_size,
            seq_len=seq_len,
            dropout=dropout
        )
        
        self.classifier = AdvancedClassifier(
            latent_size=latent_size,
            hidden_sizes=[256, 128, 64],
            num_classes=num_classes,
            dropout=dropout
        )
        
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.input_size = input_size
        self.seq_len = seq_len
    
    def forward(self, x: torch.Tensor):
        # Autoencoder forward
        reconstructed, latent, aux_features = self.autoencoder(x)
        
        # Classifier forward
        logits, binary_logits, confidence = self.classifier(latent)
        
        return {
            'reconstructed': reconstructed,
            'latent': latent,
            'aux_features': aux_features,
            'logits': logits,
            'binary_logits': binary_logits,
            'confidence': confidence
        }
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(x)
        return torch.argmax(outputs['logits'], dim=1)
    
    def predict_failure(self, x: torch.Tensor):
        outputs = self.forward(x)
        
        probs = F.softmax(outputs['logits'], dim=1)
        confidence, predictions = torch.max(probs, dim=1)
        
        anomaly_score = self.autoencoder.reconstruction_error(x)
        
        # Binary failure prediction
        binary_probs = F.softmax(outputs['binary_logits'], dim=1)
        is_failure = binary_probs[:, 1] > 0.5
        
        return {
            'class_id': predictions,
            'confidence': confidence,
            'anomaly_score': anomaly_score,
            'probabilities': probs,
            'is_failure': is_failure,
            'latent': outputs['latent']
        }