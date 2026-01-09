"""
Transfer Learning Module
Enables model transfer across different network deployments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from .encoder import LSTMAutoencoder, LSTMEncoder
from .classifier import CombinedModel, FailureClassifier
from .features import FeatureExtractor, EnvironmentNormalizer


@dataclass
class TransferConfig:
    """Configuration for transfer learning."""
    freeze_encoder: bool = True
    fine_tune_epochs: int = 20
    fine_tune_lr: float = 0.0001
    adaptation_samples: int = 100
    domain_adaptation: bool = True


class DomainAdapter(nn.Module):
    """
    Domain adaptation layer for transfer learning.
    
    Learns to map source domain features to target domain.
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int = 32):
        super().__init__()
        
        self.adapter = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Initialize as identity-like
        self._init_identity()
    
    def _init_identity(self):
        """Initialize to approximate identity function."""
        for layer in self.adapter:
            if isinstance(layer, nn.Linear):
                nn.init.eye_(layer.weight[:min(layer.weight.shape)])
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adapt features."""
        return x + self.adapter(x)  # Residual connection


class TransferLearner:
    """
    Handles transfer learning for neural failure detector.
    
    Strategies:
    1. Feature normalization: Normalize features relative to environment baseline
    2. Encoder freezing: Keep pretrained encoder, fine-tune classifier
    3. Domain adaptation: Learn mapping between source and target domains
    """
    
    def __init__(
        self,
        source_model: CombinedModel,
        config: TransferConfig,
        device: str = 'cpu'
    ):
        self.source_model = source_model
        self.config = config
        self.device = torch.device(device)
        
        # Target model (copy of source)
        self.target_model = self._copy_model(source_model)
        self.target_model.to(self.device)
        
        # Domain adapter
        if config.domain_adaptation:
            self.adapter = DomainAdapter(
                feature_dim=source_model.latent_size
            ).to(self.device)
        else:
            self.adapter = None
        
        # Environment normalizer for target
        self.target_normalizer = EnvironmentNormalizer()
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor()
    
    def _copy_model(self, model: CombinedModel) -> CombinedModel:
        """Create a copy of the model."""
        new_model = CombinedModel(
            input_size=model.autoencoder.input_size,
            hidden_size=model.autoencoder.hidden_size,
            latent_size=model.latent_size,
            num_layers=model.autoencoder.encoder.num_layers,
            seq_len=model.autoencoder.seq_len,
            num_classes=model.num_classes
        )
        new_model.load_state_dict(model.state_dict())
        return new_model
    
    def adapt(
        self,
        target_healthy_data: List[np.ndarray],
        target_labeled_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> CombinedModel:
        """
        Adapt model to target environment.
        
        Args:
            target_healthy_data: List of feature windows from healthy operation
            target_labeled_data: Optional (features, labels) for fine-tuning
            
        Returns:
            Adapted model
        """
        logger.info("Starting transfer learning adaptation...")
        
        # Step 1: Fit environment normalizer on healthy baseline
        logger.info("Fitting environment normalizer...")
        self.target_normalizer.fit(target_healthy_data)
        
        # Step 2: Calibrate feature extractor
        logger.info("Calibrating feature extractor...")
        self.feature_extractor.fit(target_healthy_data)
        
        # Step 3: Optionally freeze encoder
        if self.config.freeze_encoder:
            logger.info("Freezing encoder weights...")
            for param in self.target_model.autoencoder.encoder.parameters():
                param.requires_grad = False
        
        # Step 4: Train domain adapter if enabled
        if self.adapter and len(target_healthy_data) > 0:
            logger.info("Training domain adapter...")
            self._train_adapter(target_healthy_data)
        
        # Step 5: Fine-tune on labeled data if available
        if target_labeled_data is not None:
            features, labels = target_labeled_data
            if len(features) >= self.config.adaptation_samples:
                logger.info(f"Fine-tuning on {len(features)} labeled samples...")
                self._fine_tune(features, labels)
            else:
                logger.warning(f"Only {len(features)} samples, skipping fine-tuning")
        
        return self.target_model
    
    def _train_adapter(self, healthy_data: List[np.ndarray], epochs: int = 50):
        """Train domain adapter to match source distribution."""
        if self.adapter is None:
            return
        
        self.adapter.train()
        optimizer = optim.Adam(self.adapter.parameters(), lr=0.001)
        
        # Get source embeddings (what healthy should look like)
        source_embeddings = self._get_healthy_embeddings(self.source_model, healthy_data[:100])
        source_mean = torch.tensor(np.mean(source_embeddings, axis=0), dtype=torch.float32).to(self.device)
        source_std = torch.tensor(np.std(source_embeddings, axis=0), dtype=torch.float32).to(self.device)
        
        for epoch in range(epochs):
            total_loss = 0
            
            for data in healthy_data:
                # Normalize features
                normalized = self.target_normalizer.transform(data)
                x = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Get target embedding
                with torch.no_grad():
                    target_emb = self.target_model.encode(x)
                
                # Adapt
                adapted_emb = self.adapter(target_emb)
                
                # Loss: Match source distribution
                loss = torch.mean((adapted_emb - source_mean) ** 2 / (source_std ** 2 + 1e-6))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.debug(f"Adapter epoch {epoch+1}: loss={total_loss/len(healthy_data):.4f}")
    
    def _get_healthy_embeddings(self, model: CombinedModel, data: List[np.ndarray]) -> np.ndarray:
        """Get embeddings from healthy data."""
        model.eval()
        embeddings = []
        
        with torch.no_grad():
            for window in data:
                x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
                emb = model.encode(x)
                embeddings.append(emb.cpu().numpy())
        
        return np.concatenate(embeddings, axis=0)
    
    def _fine_tune(self, features: np.ndarray, labels: np.ndarray):
        """Fine-tune model on labeled target data."""
        from .training import Trainer, TrainingConfig
        
        config = TrainingConfig(
            epochs=self.config.fine_tune_epochs,
            learning_rate=self.config.fine_tune_lr,
            batch_size=min(32, len(features)),
            early_stopping_patience=5,
            device=str(self.device)
        )
        
        # Unfreeze classifier
        for param in self.target_model.classifier.parameters():
            param.requires_grad = True
        
        trainer = Trainer(self.target_model, config)
        trainer.train(features, labels)
    
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """Transform features for target environment."""
        return self.target_normalizer.transform(features)
    
    def predict(self, features: np.ndarray) -> Dict[str, torch.Tensor]:
        """Make prediction with adapted model."""
        self.target_model.eval()
        
        # Normalize
        normalized = self.target_normalizer.transform(features)
        x = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Encode
            latent = self.target_model.encode(x)
            
            # Apply domain adapter if available
            if self.adapter:
                latent = self.adapter(latent)
            
            # Classify
            logits = self.target_model.classifier(latent)
            probs = torch.softmax(logits, dim=1)
            confidence, prediction = torch.max(probs, dim=1)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probs,
            'latent': latent
        }
    
    def save(self, path: str):
        """Save adapted model and normalizer."""
        checkpoint = {
            'model_state_dict': self.target_model.state_dict(),
            'adapter_state_dict': self.adapter.state_dict() if self.adapter else None,
            'normalizer_means': self.target_normalizer.baseline_means,
            'normalizer_stds': self.target_normalizer.baseline_stds,
            'config': self.config
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved adapted model to {path}")
    
    def load(self, path: str):
        """Load adapted model and normalizer."""
        checkpoint = torch.load(path, map_location=self.device)
        self.target_model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.adapter and checkpoint['adapter_state_dict']:
            self.adapter.load_state_dict(checkpoint['adapter_state_dict'])
        
        if checkpoint['normalizer_means'] is not None:
            self.target_normalizer.baseline_means = checkpoint['normalizer_means']
            self.target_normalizer.baseline_stds = checkpoint['normalizer_stds']
            self.target_normalizer.fitted = True
        
        logger.info(f"Loaded adapted model from {path}")


def compute_domain_distance(
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray
) -> float:
    """
    Compute distance between source and target domains.
    
    Uses Maximum Mean Discrepancy (MMD) as metric.
    """
    def rbf_kernel(x, y, sigma=1.0):
        dist = np.sum((x[:, None] - y[None, :]) ** 2, axis=2)
        return np.exp(-dist / (2 * sigma ** 2))
    
    n = len(source_embeddings)
    m = len(target_embeddings)
    
    k_ss = rbf_kernel(source_embeddings, source_embeddings)
    k_tt = rbf_kernel(target_embeddings, target_embeddings)
    k_st = rbf_kernel(source_embeddings, target_embeddings)
    
    mmd = (np.sum(k_ss) / (n * n) + 
           np.sum(k_tt) / (m * m) - 
           2 * np.sum(k_st) / (n * m))
    
    return float(np.sqrt(max(0, mmd)))