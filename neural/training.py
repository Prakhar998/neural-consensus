"""
Training Module
Training loop and utilities for neural failure detector.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from loguru import logger

from .encoder import LSTMAutoencoder
from .classifier import CombinedModel, FocalLoss, FailureClass


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 100
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    
    # Loss weights
    reconstruction_weight: float = 1.0
    classification_weight: float = 1.0
    
    # Regularization
    weight_decay: float = 1e-5
    dropout: float = 0.1
    
    # Learning rate scheduling
    lr_scheduler: str = 'plateau'  # 'plateau', 'cosine', 'none'
    lr_patience: int = 5
    lr_factor: float = 0.5
    
    # Device
    device: str = 'cpu'
    
    # Checkpointing
    checkpoint_dir: str = 'models'
    save_best: bool = True


class FailureDataset(Dataset):
    """Dataset for failure detection training."""
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        transform=None
    ):
        """
        Args:
            features: Array of shape (n_samples, seq_len, n_features)
            labels: Array of shape (n_samples,) with class indices
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


class Trainer:
    """Trainer for neural failure detector."""
    
    def __init__(self, model: CombinedModel, config: TrainingConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        if config.lr_scheduler == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=config.lr_factor,
                patience=config.lr_patience
            )
        elif config.lr_scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.epochs
            )
        else:
            self.scheduler = None
        
        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        self.classification_loss = FocalLoss(gamma=2.0)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_recon_loss': [],
            'train_class_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_recon_loss': [],
            'val_class_loss': [],
            'val_accuracy': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def train(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        val_features: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_features: Training features (n_samples, seq_len, n_features)
            train_labels: Training labels (n_samples,)
            val_features: Validation features (optional)
            val_labels: Validation labels (optional)
            
        Returns:
            Training history
        """
        # Create datasets
        train_dataset = FailureDataset(train_features, train_labels)
        
        if val_features is None:
            # Split training data
            val_size = int(len(train_dataset) * self.config.validation_split)
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        else:
            val_dataset = FailureDataset(val_features, val_labels)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
        
        # Training loop
        for epoch in range(self.config.epochs):
            # Train epoch
            train_metrics = self._train_epoch(train_loader)
            
            # Validate
            val_metrics = self._validate(val_loader)
            
            # Update history
            for key in train_metrics:
                self.history[f'train_{key}'].append(train_metrics[key])
            for key in val_metrics:
                self.history[f'val_{key}'].append(val_metrics[key])
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Logging
            logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} - "
                f"train_loss: {train_metrics['loss']:.4f}, "
                f"train_acc: {train_metrics['accuracy']:.4f}, "
                f"val_loss: {val_metrics['loss']:.4f}, "
                f"val_acc: {val_metrics['accuracy']:.4f}"
            )
            
            # Early stopping check
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.epochs_without_improvement = 0
                
                if self.config.save_best:
                    self._save_checkpoint('best_model.pt')
            else:
                self.epochs_without_improvement += 1
                
                if self.epochs_without_improvement >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if self.config.save_best:
            self._load_checkpoint('best_model.pt')
        
        return self.history
    
    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_recon_loss = 0
        total_class_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            reconstructed, latent, logits = self.model(batch_x)
            
            # Compute losses
            recon_loss = self.reconstruction_loss(reconstructed, batch_x)
            class_loss = self.classification_loss(logits, batch_y)
            
            loss = (self.config.reconstruction_weight * recon_loss + 
                   self.config.classification_weight * class_loss)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item() * batch_x.size(0)
            total_recon_loss += recon_loss.item() * batch_x.size(0)
            total_class_loss += class_loss.item() * batch_x.size(0)
            
            _, predicted = torch.max(logits, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
        
        return {
            'loss': total_loss / total,
            'recon_loss': total_recon_loss / total,
            'class_loss': total_class_loss / total,
            'accuracy': correct / total
        }
    
    def _validate(self, loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0
        total_recon_loss = 0
        total_class_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                reconstructed, latent, logits = self.model(batch_x)
                
                recon_loss = self.reconstruction_loss(reconstructed, batch_x)
                class_loss = self.classification_loss(logits, batch_y)
                
                loss = (self.config.reconstruction_weight * recon_loss + 
                       self.config.classification_weight * class_loss)
                
                total_loss += loss.item() * batch_x.size(0)
                total_recon_loss += recon_loss.item() * batch_x.size(0)
                total_class_loss += class_loss.item() * batch_x.size(0)
                
                _, predicted = torch.max(logits, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
        
        return {
            'loss': total_loss / total,
            'recon_loss': total_recon_loss / total,
            'class_loss': total_class_loss / total,
            'accuracy': correct / total
        }
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = Path(self.config.checkpoint_dir) / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }, path)
    
    def _load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = Path(self.config.checkpoint_dir) / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def evaluate(
        self,
        test_features: np.ndarray,
        test_labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate model on test data.
        
        Returns:
            Dictionary with metrics and confusion matrix
        """
        self.model.eval()
        
        test_dataset = FailureDataset(test_features, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)
        
        all_preds = []
        all_labels = []
        all_probs = []
        all_anomaly_scores = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                
                result = self.model.predict_failure(batch_x)
                
                all_preds.extend(result['class_id'].cpu().numpy())
                all_labels.extend(batch_y.numpy())
                all_probs.extend(result['probabilities'].cpu().numpy())
                all_anomaly_scores.extend(result['anomaly_score'].cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Compute metrics
        accuracy = np.mean(all_preds == all_labels)
        
        # Per-class metrics
        num_classes = len(FailureClass)
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        
        for pred, label in zip(all_preds, all_labels):
            confusion_matrix[label, pred] += 1
        
        # Per-class accuracy
        per_class_acc = {}
        for i in range(num_classes):
            class_total = np.sum(all_labels == i)
            if class_total > 0:
                class_correct = confusion_matrix[i, i]
                per_class_acc[FailureClass(i).name] = class_correct / class_total
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': confusion_matrix,
            'per_class_accuracy': per_class_acc,
            'predictions': all_preds,
            'labels': all_labels,
            'anomaly_scores': np.array(all_anomaly_scores)
        }