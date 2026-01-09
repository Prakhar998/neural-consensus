"""
Failure Classifier Module
Classifies node failures based on latent representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from enum import IntEnum


class FailureClass(IntEnum):
    """Failure classification labels."""
    HEALTHY = 0
    PRE_FAILURE = 1
    CRASHED = 2
    BYZANTINE = 3
    PARTITIONED = 4
    SLOW = 5


FAILURE_NAMES = {
    FailureClass.HEALTHY: "healthy",
    FailureClass.PRE_FAILURE: "pre_failure",
    FailureClass.CRASHED: "crashed",
    FailureClass.BYZANTINE: "byzantine",
    FailureClass.PARTITIONED: "partitioned",
    FailureClass.SLOW: "slow"
}


class FailureClassifier(nn.Module):
    """
    MLP classifier for failure type prediction.
    
    Takes latent representation from encoder and predicts failure class.
    """
    
    def __init__(
        self,
        latent_size: int = 32,
        hidden_sizes: List[int] = [64, 32],
        num_classes: int = 6,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.latent_size = latent_size
        self.num_classes = num_classes
        
        # Build MLP layers
        layers = []
        prev_size = latent_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Classify failure type from latent representation.
        
        Args:
            latent: Latent tensor of shape (batch, latent_size)
            
        Returns:
            logits: Class logits (batch, num_classes)
        """
        return self.classifier(latent)
    
    def predict(self, latent: torch.Tensor) -> torch.Tensor:
        """Get predicted class indices."""
        logits = self.forward(latent)
        return torch.argmax(logits, dim=1)
    
    def predict_proba(self, latent: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        logits = self.forward(latent)
        return F.softmax(logits, dim=1)
    
    def predict_with_confidence(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get predictions with confidence scores."""
        probs = self.predict_proba(latent)
        confidence, predictions = torch.max(probs, dim=1)
        return predictions, confidence


class CombinedModel(nn.Module):
    """
    Combined autoencoder + classifier model.
    
    Uses joint training:
    - Autoencoder learns good representations via reconstruction
    - Classifier learns to predict failures from representations
    """
    
    def __init__(
        self,
        input_size: int = 16,
        hidden_size: int = 64,
        latent_size: int = 32,
        num_layers: int = 2,
        seq_len: int = 20,
        classifier_hidden: List[int] = [64, 32],
        num_classes: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        
        from .encoder import LSTMAutoencoder
        
        self.autoencoder = LSTMAutoencoder(
            input_size=input_size,
            hidden_size=hidden_size,
            latent_size=latent_size,
            num_layers=num_layers,
            seq_len=seq_len,
            dropout=dropout
        )
        
        self.classifier = FailureClassifier(
            latent_size=latent_size,
            hidden_sizes=classifier_hidden,
            num_classes=num_classes,
            dropout=dropout
        )
        
        self.latent_size = latent_size
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.
        
        Returns:
            reconstructed: Reconstructed input
            latent: Latent representation
            logits: Classification logits
        """
        reconstructed, latent = self.autoencoder(x)
        logits = self.classifier(latent)
        return reconstructed, latent, logits
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation."""
        return self.autoencoder.encode(x)
    
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Get classification predictions."""
        latent = self.encode(x)
        return self.classifier.predict(latent)
    
    def get_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Get anomaly score (reconstruction error)."""
        return self.autoencoder.reconstruction_error(x)
    
    def predict_failure(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Complete failure prediction.
        
        Returns dict with:
            - class_id: Predicted failure class
            - class_name: Name of failure class
            - confidence: Prediction confidence
            - anomaly_score: Reconstruction error
            - probabilities: Per-class probabilities
        """
        reconstructed, latent = self.autoencoder(x)
        logits = self.classifier(latent)
        probs = F.softmax(logits, dim=1)
        confidence, predictions = torch.max(probs, dim=1)
        anomaly = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
        
        return {
            'class_id': predictions,
            'confidence': confidence,
            'anomaly_score': anomaly,
            'probabilities': probs,
            'latent': latent
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Puts more weight on hard-to-classify examples.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss