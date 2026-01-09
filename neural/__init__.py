"""Neural Network Package for Failure Detection"""
from .features import FeatureExtractor, ObservationWindow
from .encoder import LSTMEncoder, LSTMAutoencoder
from .classifier import FailureClassifier
from .detector import NeuralFailureDetector
from .training import Trainer, TrainingConfig
from .transfer import TransferLearner, DomainAdapter

__all__ = [
    'FeatureExtractor', 'ObservationWindow',
    'LSTMEncoder', 'LSTMAutoencoder',
    'FailureClassifier',
    'NeuralFailureDetector',
    'Trainer', 'TrainingConfig',
    'TransferLearner', 'DomainAdapter'
]